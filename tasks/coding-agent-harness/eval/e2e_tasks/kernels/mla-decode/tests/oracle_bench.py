"""Sealed MLA decode oracle, measured live on the grading GPU.

Hopper uses the existing vendored FlashMLA extension. Blackwell selects between
FlashInfer TRT-LLM MLA and pinned FA4 edb5c76ee329 using its qv interface: the
512-dimensional latent dot product plus the 64-dimensional RoPE dot product,
with the original 512-dimensional values. All layout/allocation costs are timed.
"""
import math
import os
import platform
import re
import sys
import tarfile
import tempfile

import torch

_ORACLE = None
ORACLE_REVISION = "portfolio-v1"


class OracleUnavailable(RuntimeError):
    """The sealed oracle cannot run on this machine. VOID the cell -- never score it 0.

    The oracle is a COMPILED CPython extension built for one (interpreter ABI, CPU architecture) pair,
    and it is the DENOMINATOR of the reward. If it will not import there is no ceiling to divide by,
    and the failure is indistinguishable from a harness that wrote a broken kernel: a 0.000 with a
    traceback in details.json. Those two must stay distinguishable, so the grader declines to write a
    reward at all and the cell is void rather than libelled.
    """


_EXT_TAG = re.compile(r"\.(cpython-\d+-[^/]+)\.so$")


def _require_runnable_oracle(names):
    """Refuse to grade unless the vendored .so matches this interpreter and this CPU.

    The tarball ships e.g. `_C.cpython-312-x86_64-linux-gnu.so`, and that tag is the whole contract.
    It is checked HERE, before extraction, rather than being left to `import`: main() wraps the whole
    grade in `except Exception`, so an ImportError from the wrong architecture becomes a score instead
    of an error. A tarball with no compiled extension in it is pure Python and passes.
    """
    want = f"cpython-{sys.version_info.major}{sys.version_info.minor}-{platform.machine()}-linux-gnu"
    tags = sorted({m.group(1) for m in (_EXT_TAG.search(n) for n in names) if m})
    if tags and want not in tags:
        raise OracleUnavailable(
            f"vendored oracle is built for {', '.join(tags)}; this container is {want}. "
            "Re-vendor the oracle on the target machine, or grade "
            "this task on the architecture it was vendored for. Refusing to score.")


# Where the image keeps the oracle. FlashInfer is this task's ceiling on sm_10x, so environment/Dockerfile
# moves the package off the DEFAULT import path after installing it -- otherwise the agent's own container
# ships the kernel it is being asked to write, and `import flashinfer` scores 1.000. Nothing but this file
# adds the directory back.
ORACLE_LIBS = "/opt/oracle-libs"


def _with_oracle_libs() -> None:
    if os.path.isdir(ORACLE_LIBS) and ORACLE_LIBS not in sys.path:
        sys.path.insert(0, ORACLE_LIBS)


# Which oracle this process ended up using. The verifier records it in details.json, and it is the first
# thing to look at when a grade on new hardware comes out wrong.
ORACLE_BACKEND = None


def _is_blackwell():
    """cc 10.x — B200 (sm_100) and GB300 (sm_103), which share the 10.x binary family."""
    import torch
    return torch.cuda.is_available() and torch.cuda.get_device_capability()[0] == 10


def _blackwell_decode():
    """The sm_103 oracle: FlashInfer's MLA decode.

    FlashMLA's DENSE decode -- what this task's vendored oracle calls -- is SM90-only by upstream's own
    support matrix (sparse decode and dense *prefill* reach SM100; dense decode does not). So there is no
    re-vendoring that helps: the kernel does not exist for Blackwell. FlashInfer resolves
    `cc in ((10, 0), (10, 3))` explicitly and is the replacement.

    The paged layout is not new work: the FlashMLA wrapper already builds a block table and cache_seqlens
    over a (num_blocks, 64, 1, 576) view, and those are what `block_tables` / `seq_lens` want here.

    Against the task's own fp32 reference and its own eager baseline on Blackwell, over all five grader
    shapes, this route agrees to relerr ~2-3e-3 at every shape; it is slower than eager at the smallest
    graded shape (32, 2048) and over 4x faster at the largest. So `qk_nope_head_dim=128` and
    `bmm1_scale=scale` are right, and the one shape where the oracle loses is small enough not to matter
    to the geomean. A single-shape check would mislead here in either direction.
    """
    try:
        _with_oracle_libs()
        from flashinfer.mla import trtllm_batch_decode_with_kv_cache_mla
    except ImportError as e:
        raise OracleUnavailable(
            "this is a Blackwell GPU, whose oracle is flashinfer.mla.trtllm_batch_decode_with_kv_cache_mla, "
            f"but flashinfer is not importable here ({e}) -- expected it in "
            "/opt/oracle-libs or site-packages. Rebuild the image on this node. "
            "Refusing to score.") from e
    import torch

    ws = {}

    def mla_decode(q, kv, scale):
        B, H, DH = q.shape                                   # (B, H, 576)
        L = kv.shape[1]
        nblk = L // _PAGE
        buf = ws.get(q.device)
        if buf is None:
            buf = ws[q.device] = torch.empty(128 * 1024 * 1024, dtype=torch.uint8, device=q.device)
        out = trtllm_batch_decode_with_kv_cache_mla(
            query=q.unsqueeze(1),                            # (B, 1, H, 576) -- one decode step
            # (num_blocks, num_kv_heads=1, page_size, head_dim) -- the TRT-LLM-gen paged layout.
            # A FREE VIEW: kv is (B, L, DH) contiguous with L % 64 == 0, so the singleton head axis
            # costs nothing. The head axis must come before the page axis: (num_blocks, page_size, 1, DH)
            # would be read as num_kv_heads=64, page_size=1 and rejected with
            #   ValueError: Expected block_num % (128 / block_size) == 0, got block_num=32, block_size=1
            # because 128/1 does not divide the blocks-per-sequence.
            kv_cache=kv.reshape(B * nblk, _PAGE, DH).unsqueeze(1),
            workspace_buffer=buf,
            qk_nope_head_dim=128,
            kv_lora_rank=_DV,                                # 512
            qk_rope_head_dim=DH - _DV,                       # 64
            block_tables=torch.arange(B * nblk, device=q.device, dtype=torch.int32).view(B, nblk),
            seq_lens=torch.full((B,), L, device=q.device, dtype=torch.int32),
            max_seq_len=L,
            bmm1_scale=scale,
        )
        return out.squeeze(1)                                # (B, H, 512)

    return mla_decode
_PAGE = 64
_DV = 512




# ROUTES. On Blackwell, over all five graded shapes, `flashinfer.mla.BatchMLAPagedAttentionWrapper` is
# 0.43x the eager baseline on the `auto` and `fa2` backends -- several times slower than the trtllm route
# above -- and its `fa3` backend has no kernel image for sm_100 at all, so it is not a route. The routes are
# the trtllm MLA decode and, on Blackwell, the pinned FA4 CuTe MLA kernel (`_routes`); `_Dispatch` picks the
# faster correct one per shape. (32, 2048) is too small for a paged MLA kernel to beat the eager baseline.

_DISPATCH_TOL = 2e-2                     # this task's PERF_TOL
_ROUTE_NAME = "mla-decode"


# ---------------------------------------------------------------------------- per-shape oracle dispatch
# One winner for the whole grade is the wrong shape of answer: the Blackwell trtllm route is slower than
# the eager baseline at (32, 2048) and over 4x faster at (128, 16384). The reward is a ratio against this
# number, so a ceiling that depends on which shape the probe happened to use is not a ceiling.
#
# Dispatching per shape is also what the libraries this stands in for do: their backends are tile-size
# heuristics over the problem shape. The choice is made on the verifier's FIRST call for a shape, which
# is its correctness check -- outside the timed region, and identical for every submission.
SELECTION_LOG: list = []

# Total time allowed for selecting routes. After this, a NEW shape uses the
# first (broadly supported) route, never a specialized winner from another shape.
SELECT_BUDGET_S = 300.0


def _disagree(a, b):
    """Worst relative difference between two oracle outputs (tensor, or tuple of tensors)."""
    if isinstance(a, (tuple, list)):
        return max(_disagree(x, y) for x, y in zip(a, b))
    return float((a.float() - b.float()).norm() / (b.float().norm() + 1e-12))


class _Dispatch:
    """Choose outside the timed region, checking each candidate against the anchor."""

    def __init__(self, routes, key):
        self.routes = list(routes)               # [(name, fn)] -- each validated at the probe shapes
        self.key = key
        self.chosen = {}
        self.spent = 0.0                         # seconds burned choosing; see SELECT_BUDGET_S
        self.last = routes[0][1] if routes else None

    def __call__(self, *args, **kwargs):
        k = self.key(*args, **kwargs)
        fn = self.chosen.get(k)
        if fn is None:
            fn = self.chosen[k] = self._pick(k, args, kwargs)
        return fn(*args, **kwargs)

    def _pick(self, k, args, kwargs):
        if len(self.routes) == 1:
            return self.routes[0][1]
        if self.spent > SELECT_BUDGET_S:
            SELECTION_LOG.append(f"{k}: selection budget spent ({self.spent:.0f}s > "
                                 f"{SELECT_BUDGET_S:.0f}s) -- using first supported route")
            # The last winner may only support one head/state width. Recheck
            # support here; the verifier still checks the answer against FP32.
            for name, fn in self.routes:
                try:
                    out = fn(*args, **kwargs)
                    if not math.isfinite(_disagree(out, out)):
                        raise ValueError("nonfinite oracle output")
                    SELECTION_LOG.append(f"{k}: budget fallback -> {name}")
                    return fn
                except Exception as e:
                    SELECTION_LOG.append(f"{k}: fallback {name} unusable -- {type(e).__name__}: {str(e)[:90]}")
            raise OracleUnavailable(f"no supported budget fallback at {k}")
        import time as _t
        _t0 = _t.perf_counter()
        try:
            return self._choose(k, args, kwargs)
        finally:
            self.spent += _t.perf_counter() - _t0

    def _choose(self, k, args, kwargs):
        anchor, anchor_name, timed = None, None, []
        for name, fn in self.routes:
            try:
                out = fn(*args, **kwargs)
                if not math.isfinite(_disagree(out, out)):
                    raise ValueError("nonfinite oracle output")
                if anchor is None:
                    anchor, anchor_name = out, name
                else:
                    d = _disagree(out, anchor)
                    if not math.isfinite(d) or d > _DISPATCH_TOL:
                        SELECTION_LOG.append(
                            f"{k}: {name} disagrees with {anchor_name} HERE (rel {d:.2e}) -- not used "
                            f"at this shape")
                        continue
                timed.append((_measure(fn, args, kwargs), name, fn))
            except BaseException as e:                                   # noqa: BLE001
                SELECTION_LOG.append(f"{k}: {name} unusable here -- {type(e).__name__}: {str(e)[:90]}")
        if not timed:
            raise OracleUnavailable(
                f"no oracle route runs correctly at {k}: " + " | ".join(SELECTION_LOG[-4:]))
        timed.sort(key=lambda r: r[0])
        SELECTION_LOG.append(f"{k}: " + ", ".join(f"{n} {t * 1e3:.3f}ms" for t, n, _ in timed)
                             + f" -> {timed[0][1]}")
        self.last = timed[0][2]
        return self.last


def _measure(fn, args, kwargs=None, target_s=0.05, batches=6, warm_s=0.2):
    """Back-to-back launches, one sync per batch, minimum over batches -- the method verify_env grades
    with. A sync after every launch times one cold isolated launch instead, which can misread a
    block-scaled FP4 GEMM by 89%. See verify_env._bench.
    """
    import time
    kwargs = kwargs or {}
    t0, n = time.perf_counter(), 0
    while n < 5 or (time.perf_counter() - t0) < warm_s:
        fn(*args, **kwargs)
        torch.cuda.synchronize()
        n += 1
    per = max((time.perf_counter() - t0) / n, 1e-6)
    batch = max(5, min(200, int(target_s / per)))
    best = float("inf")
    for _ in range(batches):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(batch):
            fn(*args, **kwargs)
        torch.cuda.synchronize()
        best = min(best, (time.perf_counter() - t0) / batch)
    return best


def _validated(routes, cases, tol):
    """Keep every route that is correct WHERE IT RUNS, recording where it does not.

    Used by tasks that build probe cases of their own (the GEMM task). The others hand
    `_Dispatch` their routes directly with the incumbent first: at a graded shape every route is
    cross-checked against that anchor before it is timed, which is a tighter check than a probe at
    different shapes, and `verify_env._oracle_must_agree` remains the backstop either way.

    Correctness first, and per case -- a route rejected because it refuses one configuration would take
    with it every configuration where it is the best kernel available. A route that is wrong everywhere,
    or that runs nowhere, is dropped.
    """
    keep = []
    for name, fn in routes:
        ok, notes = 0, []
        for args, ref, label in cases:
            try:
                rel = _disagree(fn(*args), ref)
            except BaseException as e:                                   # noqa: BLE001
                notes.append(f"{label}: {type(e).__name__}: {str(e)[:70]}")
                continue
            if not math.isfinite(rel) or rel > tol:
                notes.append(f"{label}: relerr {rel:.2e}")
            else:
                ok += 1
        if ok:
            keep.append((ok, name, fn))
            SELECTION_LOG.append(f"{name}: correct at {ok}/{len(cases)} probe cases"
                                 + (" | " + "; ".join(notes) if notes else ""))
        else:
            SELECTION_LOG.append(f"{name}: REJECTED -- " + "; ".join(notes))
    # MOST-BROADLY-CORRECT FIRST, stably. `_Dispatch` uses the first route that runs at a shape as its
    # correctness anchor and drops the others if they disagree with it, so the anchor must be the route
    # least likely to be the wrong one. A route kept because it is correct at one probe case out of three
    # is exactly the wrong anchor -- and a stable sort still leaves a route that is correct everywhere
    # (the incumbent, which every caller lists first) exactly where it was.
    keep.sort(key=lambda r: -r[0])
    return [(name, fn) for _, name, fn in keep]


def _build_route():
    """The one MLA-decode oracle that runs here: FlashInfer on Blackwell, the vendored FlashMLA `.so` on
    sm_90. Returns the callable; `_routes()` wraps it and `oracle_forward()` hands it to `_Dispatch`."""
    global ORACLE_BACKEND
    if _is_blackwell():
        # _blackwell_decode names the route itself -- "flashinfer/sm10x" would hide which entry point won
        ORACLE_BACKEND = "flashinfer/trtllm_mla"
        return _blackwell_decode()

    ORACLE_BACKEND = "vendored/sm90"
    d = tempfile.mkdtemp(prefix="_flash_mla_oracle_")
    with tarfile.open(os.path.join(os.path.dirname(__file__), "_flash_mla_oracle.tar.gz")) as t:
        _require_runnable_oracle(t.getnames())
        t.extractall(d)
    if d not in sys.path:
        sys.path.insert(0, d)
    from flash_mla import flash_mla_with_kvcache, get_mla_metadata

    def mla_decode(q, kv, scale):
        # q (B,H,576), kv (B,L,576) contiguous, L % 64 == 0  ->  out (B,H,512)
        B, H, DH = q.shape
        L = kv.shape[1]
        nblk = L // _PAGE
        q4 = q.unsqueeze(1)                                  # (B,1,H,576) view
        k_cache = kv.reshape(B * nblk, _PAGE, 1, DH)         # FREE view (kv contiguous, L % 64 == 0)
        block_table = torch.arange(B * nblk, device=q.device, dtype=torch.int32).view(B, nblk)
        cache_seqlens = torch.full((B,), L, device=q.device, dtype=torch.int32)
        sched_meta, _ = get_mla_metadata()
        out, _ = flash_mla_with_kvcache(q4, k_cache, block_table, cache_seqlens, _DV,
                                        tile_scheduler_metadata=sched_meta, softmax_scale=scale,
                                        causal=False)
        return out.squeeze(1)                                # (B,H,512)

    return mla_decode


def _fa4_route():
    if not _is_blackwell():
        raise ValueError("the FA4 MLA portfolio route is enabled on Blackwell only")
    d = tempfile.mkdtemp(prefix="_fa4_mla_oracle_")
    with tarfile.open(os.path.join(os.path.dirname(__file__), "_fa4_mla_oracle.tar.gz")) as t:
        t.extractall(d, filter="data")
    sys.path.insert(0, d)
    from flash_attn.cute.interface import _flash_attn_fwd

    def _mla(q, kv, scale):
        return _flash_attn_fwd(q[..., 512:].unsqueeze(1), kv[..., 512:].unsqueeze(2),
                               kv[..., :512].unsqueeze(2), qv=q[..., :512].unsqueeze(1),
                               softmax_scale=scale, causal=False)[0].squeeze(1)

    return _mla


def _routes():
    out = []
    builders = [(_ROUTE_NAME, _build_route)]
    if _is_blackwell():
        builders.append(("fa4/mla@edb5c76ee329", _fa4_route))
    for name, build in builders:
        try:
            out.append((name, build()))
        except OracleUnavailable:
            raise
        except BaseException as e:                                   # noqa: BLE001
            SELECTION_LOG.append(f"{name}: unavailable -- {type(e).__name__}: {str(e)[:140]}")
    if not out:
        raise OracleUnavailable("no MLA decode oracle route loaded here: "
                                + " | ".join(SELECTION_LOG) + ". Refusing to score.")
    return out


def oracle_forward():
    """Return the oracle `mla_decode(q, kv, scale)` (cached after first extract)."""
    global _ORACLE, ORACLE_BACKEND
    if _ORACLE is None:
        routes = _routes()
        ORACLE_BACKEND = "+".join(n for n, _ in routes)
        _ORACLE = _Dispatch(routes, key=lambda q, kv, scale:
                            (tuple(q.shape), tuple(kv.shape), q.dtype, q.device, scale))
    return _ORACLE


def candidates():
    return _routes()
