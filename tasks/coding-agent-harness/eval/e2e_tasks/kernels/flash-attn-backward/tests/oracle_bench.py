"""Sealed oracle provider (tests/ is copied only at grade time — the agent never sees this).

TWO ROUTES, CHOSEN PER SHAPE: the vendored FlashAttention-4 (CuTe DSL) backward and PyTorch's **cuDNN**
attention backward. FlashAttention-4 alone is a ceiling a strong submission can beat, which would clamp the
reward at 1.000 and stop the task discriminating. On Blackwell, over four of the five graded shapes, the
cuDNN attention backward is the stronger kernel by a wide margin:

    (2, 8192, causal)      cuDNN 4.00x over FA-2
    (2, 4096, non-causal)  cuDNN 2.45x
    (4, 4096, causal)      cuDNN 8.49x
    (1, 8192, non-causal)  cuDNN 3.45x
    GEOMEAN                cuDNN 4.12x,  ln = 1.416      (EFFICIENT_ATTENTION: 0.44x — not a candidate)

Both are kept and `_Dispatch` picks per shape, because "faster on four shapes" is not "faster on five",
and because FA-4 is the route that needs no layout conversion.

NAMED FA-4, AND THAT IS THE POINT. The vendored package is Dao-AILab's CuTe-DSL FlashAttention -- its own
README reads "FlashAttention-4 (CuTeDSL) ... pip install flash-attn-4" -- and it ships flash_bwd_sm100.py
and flash_bwd_sm120.py beside flash_bwd_sm90.py, so it covers Blackwell natively.

FAIRNESS OF THE TWO TIMED REGIONS. FA-4 is handed `o` and `lse` and does the backward only. The cuDNN
route builds its forward graph once, on first sight of a shape, and times only `torch.autograd.grad` --
which is exactly how this task's own FA-2 *baseline* is timed (`verify_env._fa2_bwd_fn` also transposes
and builds the graph outside the closure). Both routes are backward-only inside the timed region, and
neither is handed anything the baseline is not.

SDPA is already banned for submissions — twice, by a source scan and by a TorchDispatchMode that watches
for `scaled_dot_product` / `cudnn_attention` aten ops — so promoting it to the oracle changes nothing
about what a submission may do. It only stops the ceiling being lower than the floor.
"""
import importlib.metadata
import os
import sys
import tarfile
import tempfile

import torch

_ORACLE = None
ORACLE_BACKEND = None
_DISPATCH_TOL = 2e-2                    # the task's own PERF_TOL: two correct backwards agree far inside it


class OracleUnavailable(RuntimeError):
    """The sealed oracle cannot run on this machine. VOID the cell -- never score it 0.

    Every reward here is a ratio against this kernel. If none of the routes runs there is no ceiling to
    divide by, and a 0.000 would be indistinguishable from a harness that wrote a broken kernel, so the
    cell is void instead.
    """


# ---------------------------------------------------------------------------- per-shape oracle dispatch
# One winner for the whole grade is the wrong shape of answer: the faster backward differs between graded
# shapes (see the module docstring). The reward is a ratio against this number, so a ceiling that depends
# on which shape the probe happened to use is not a ceiling.
#
# Dispatching per shape is also what the libraries this stands in for do: their backends are tile-size
# heuristics over the problem shape. The choice is made on the verifier's FIRST call for a shape, which
# is its correctness check -- outside the timed region, and identical for every submission.
SELECTION_LOG: list = []

# How long the oracle may spend CHOOSING between routes, across the whole grade. Not a benchmark budget:
# a route's first call at a new shape can be expensive (a tile-configuration search, a JIT compile), and
# the verifier's own timeout is a hang catcher. Past this, `_Dispatch` stops exploring and reuses its
# last choice -- a slightly lower ceiling is still a measurement; a verifier timeout writes no reward at
# all and loses the cell.
SELECT_BUDGET_S = 300.0


def _disagree(a, b):
    """Worst relative difference between two oracle outputs (tensor, or tuple of tensors)."""
    if isinstance(a, (tuple, list)):
        return max(_disagree(x, y) for x, y in zip(a, b))
    return float((a.float() - b.float()).norm() / (b.float().norm() + 1e-12))


class _Dispatch:
    """The fastest route that is CORRECT AT THIS SHAPE, decided the first time the shape is seen.

    Three behaviours that a single-winner probe cannot express:

      * A route that RAISES at a shape is skipped there, not discarded everywhere. A kernel can be correct
        and fast at some configurations and refuse others outright; that makes it a good candidate at
        some graded shapes and no candidate at all at others. A dispatch table, not a failure.
      * A route that DISAGREES with the others at a shape is dropped at that shape and said so. Every
        route here was checked against an independent reference at the probe configurations, but a
        backend can be right at every probe configuration and badly wrong at a graded one.
        verify_env._oracle_must_agree is still the backstop; this turns what would be a voided cell into
        a correct, slightly slower oracle.
      * Being fastest somewhere is not being fastest everywhere, which is the whole point.
    """

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
            # CHOOSING MUST NOT COST THE CELL. A route can be expensive the first time it sees a shape --
            # FlashInfer's cutile backend runs a full tile-configuration search -- and the verifier's own
            # budget is a hang catcher, not an allowance for oracle selection. Past the budget, stop
            # exploring and keep whatever was chosen last: a slightly lower ceiling is a measurement, a
            # verifier timeout writes no reward.json at all and the cell is lost.
            SELECTION_LOG.append(f"{k}: selection budget spent ({self.spent:.0f}s > "
                                 f"{SELECT_BUDGET_S:.0f}s) -- reusing the last chosen route")
            return self.last
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
                if anchor is None:
                    anchor, anchor_name = out, name
                else:
                    d = _disagree(out, anchor)
                    if d > _DISPATCH_TOL:
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


def _measure(fn, args, kwargs=None, target_s=0.05, batches=5, warm_s=0.2):
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
            if rel > tol:
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


def _check_fa4_runtime():
    """Do not enter a known non-returning SM103 backward on an image with a broken CuTe DSL runtime.

    https://github.com/Dao-AILab/flash-attention/issues/2797 reports this with
    released DSL 4.6.0/4.6.1 (the 4.6.0.dev0 pin and 4.6.2 work). Check the CUDA
    runtime distributions too: updating the meta-package alone is insufficient.
    Conservatively reject a mixed install containing either broken runtime.
    _routes() records the reason and retains the independently checked cuDNN route.
    SELECT_BUDGET_S cannot rescue a call that never returns.
    """
    if torch.cuda.get_device_capability() != (10, 3):
        return
    bad = []
    for package in ("nvidia-cutlass-dsl", "nvidia-cutlass-dsl-libs-base", "nvidia-cutlass-dsl-libs-core",
                    "nvidia-cutlass-dsl-libs-cu12", "nvidia-cutlass-dsl-libs-cu13"):
        try:
            version = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            continue
        if version in ("4.6.0", "4.6.1"):
            bad.append(f"{package}=={version}")
    if bad:
        raise OracleUnavailable(
            "FA-4 backward can hang on sm_103 with " + ", ".join(bad)
            + "; rebuild with quack-kernels[cu13]==0.6.4 (DSL/runtime 4.6.2). "
              "Using the cuDNN candidate instead; see flash-attention#2797.")


def _fa4_route():
    """FlashAttention-4's CuTe-DSL backward, vendored. JIT-compiles per shape, offline."""
    _check_fa4_runtime()
    d = tempfile.mkdtemp(prefix="_fa4_oracle_")
    with tarfile.open(os.path.join(os.path.dirname(__file__), "_fa4_oracle.tar.gz")) as t:
        t.extractall(d)                        # -> <d>/flash_attn/{__init__.py,cute}
    if d not in sys.path:
        sys.path.insert(0, d)
    from flash_attn.cute.interface import _flash_attn_bwd   # FA-4 backward-only (CuTe DSL, JITs offline)

    def attn_backward(q, k, v, o, lse, do, scale, causal):
        # q,k,v,o,do: (B,S,H,D) bf16 ; lse: (B,H,S) fp32  ->  dq,dk,dv: (B,S,H,D) bf16
        dq, dk, dv = _flash_attn_bwd(q, k, v, o, do, lse, softmax_scale=scale, causal=causal)
        return dq, dk, dv

    return attn_backward


def _cudnn_route():
    """PyTorch's cuDNN attention backward, reached through SDPA and timed backward-only.

    The forward graph is built once per shape and cached, so the timed region is `torch.autograd.grad`
    alone -- the same construction as the FA-2 baseline this task measures against.

    The cache holds ONE entry and keeps strong references to the tensors it is keyed on. Both details
    matter: keying on `id()` without a strong reference lets a freed tensor's id be reused by the next
    shape's tensors and returns the wrong graph, and keeping every shape's graph alive would hold
    several GB of activations for shapes the grader has finished with.
    """
    import torch.nn.functional as F
    from torch.nn.attention import SDPBackend, sdpa_kernel

    cache = {}

    def cudnn_backward(q, k, v, o, lse, do, scale, causal):
        key = (id(q), id(k), id(v), id(do), float(scale), bool(causal))
        entry = cache.get(key)
        if entry is None:
            qb, kb, vb = (t.transpose(1, 2).contiguous().detach().requires_grad_()
                          for t in (q, k, v))                       # (B,H,S,D), cuDNN's layout
            dob = do.transpose(1, 2).contiguous()
            with sdpa_kernel([SDPBackend.CUDNN_ATTENTION]):
                out = F.scaled_dot_product_attention(qb, kb, vb, is_causal=causal, scale=scale)
            cache.clear()                                           # one shape at a time; see docstring
            entry = cache[key] = (q, k, v, do, qb, kb, vb, dob, out)
        _, _, _, _, qb, kb, vb, dob, out = entry
        dq, dk, dv = torch.autograd.grad(out, (qb, kb, vb), dob, retain_graph=True)
        return dq.transpose(1, 2), dk.transpose(1, 2), dv.transpose(1, 2)

    return cudnn_backward


def _routes():
    """The incumbent FIRST: `_Dispatch` treats the first route that runs as the correctness anchor, and
    the vendored FA-4 is this task's primary oracle."""
    out = []
    for name, build in (("FA-4/cute", _fa4_route), ("torch/cuDNN", _cudnn_route)):
        try:
            out.append((name, build()))
        except BaseException as e:                                   # noqa: BLE001
            SELECTION_LOG.append(f"{name}: unavailable -- {type(e).__name__}: {str(e)[:140]}")
    if not out:
        raise OracleUnavailable(
            "neither attention-backward oracle route loaded on this machine: "
            + " | ".join(SELECTION_LOG) + ". Refusing to score.")
    return out


def oracle_backward():
    """Return the oracle `attn_backward(q, k, v, o, lse, do, scale, causal) -> (dq, dk, dv)`.

    A `_Dispatch` over every route that loaded, deciding per (B, S, causal) on first sight. Cached.
    """
    global _ORACLE, ORACLE_BACKEND
    if _ORACLE is None:
        routes = _routes()
        ORACLE_BACKEND = "+".join(n for n, _ in routes)
        _ORACLE = _Dispatch(routes, key=lambda q, k, v, o, lse, do, scale, causal:
                            (tuple(q.shape), bool(causal)))
    return _ORACLE


def candidates():
    """Every oracle route, for inspecting the whole table rather than just the winner."""
    return _routes()
