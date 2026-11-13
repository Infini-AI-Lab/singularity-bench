"""Sealed oracle provider (tests/ is copied only at grade time — the agent never sees this).

Exposes `nvfp4_gemm(A_fp4, A_scales, B_fp4, B_scales, alpha) -> C (M,N) bf16` so the grader can
RE-BENCHMARK the ceiling live on whatever GPU it runs on. Nothing about the oracle's speed is hardcoded:
the reward ceiling is measured fresh each grade, against that same machine's bf16 cuBLAS, so the task is
hardware-generic within the generation it needs.

The oracle is FlashInfer's `mm_fp4`, which dispatches to cuDNN / CUTLASS block-scaled FP4 kernels. Unlike
every other oracle in this suite there is no vendored tarball and no compiled extension to ABI-match: the
package is already in the image (moved to /opt/oracle-libs so only the grader can see it), the cuDNN and
CUTLASS backends need no pre-downloaded cubins, and the kernels are the same cuBLAS-class ones NVIDIA
ships. What DOES vary is which backend is fastest and which is even correct on a given card, so the
backend is chosen live, correctness first — see `_best_candidate`.

ON BLACKWELL (sm_100), geomean over the five graded shapes, against that machine's bf16 cuBLAS:

    mm_fp4, fastest correct backend per shape   ~3.1x   ln ~ 1.15    <- the ceiling
    torch F.scaled_mm (one call)                ~3x                  <- BANNED (verify_env._PrebuiltGemmSeal)
    tuned Triton tl.dot_scaled                  ~1.3x   reward ~0.3  <- the task's middle rung
    unpack -> bf16 matmul                       ~0.2x   reward 0.00

Neither backend wins everywhere, which is why `_Dispatch` picks per shape: cutlass wins everything up to
~900 GFLOP (3.21x vs 2.59x at 8192^3) and cudnn wins above it (3.79x vs 3.15x at 12288x12288x8192).
Committing to one backend from a probe would give up part of the ceiling and could change between grades.

The scale swizzle is inside the timed region on purpose. The agent is handed row-major 1x16 scales,
because that is the layout a four-line fp32 reference can specify; every FP4 tensor-core path wants
NVIDIA's 128x4 blocked order instead. Converting is part of the kernel, so the ceiling pays for it too,
on exactly the inputs the submission gets. (It costs the oracle ~14% at 8192^3 — an agent who fuses the
conversion into their own kernel can legitimately beat it.)
"""
import os
import sys

import torch

_ORACLE = None
ORACLE_BACKEND = None


class OracleUnavailable(RuntimeError):
    """The sealed oracle cannot run on this machine. VOID the cell -- never score it 0.

    Every reward on this task is a ratio against this kernel. If it will not run there is no ceiling to
    divide by, and a 0.000 would be indistinguishable from a harness that wrote a broken kernel. The
    grader declines to write a reward at all and the cell is void rather than libelled.
    """


# Where the image keeps the oracle. environment/Dockerfile moves the flashinfer package off the DEFAULT
# import path after installing it -- otherwise the agent's own container ships the kernel it is being
# asked to write, and `import flashinfer` scores 1.000. Nothing but this file adds the directory back.
ORACLE_LIBS = "/opt/oracle-libs"


def _with_oracle_libs() -> None:
    if os.path.isdir(ORACLE_LIBS) and ORACLE_LIBS not in sys.path:
        sys.path.insert(0, ORACLE_LIBS)


def _require_blackwell():
    """NVFP4 is an sm_100+ format. On anything older there is no ceiling, so there is no grade.

    Checked HERE rather than being left to the kernel launch: main() wraps the grade in `except
    Exception`, so a "no kernel image for device" from the wrong architecture would become a SCORE
    instead of an error. FP4 tensor cores arrived with Blackwell; Hopper can emulate the arithmetic but
    not at a speed that means anything, and a ceiling nobody can reach is not a ceiling.
    """
    if not torch.cuda.is_available():
        raise OracleUnavailable("no CUDA device: this task's ceiling is a GPU kernel.")
    cc = torch.cuda.get_device_capability()
    if cc[0] < 10:
        raise OracleUnavailable(
            f"NVFP4 needs FP4 tensor cores (sm_100 or newer); this device is sm_{cc[0]}{cc[1]} "
            f"({torch.cuda.get_device_name()}). Grade this task on Blackwell. Refusing to score.")


# ---------------------------------------------------------------------------- the scale swizzle
def swizzle_scales(sf_u8):
    """Row-major (R, K//16) block scales -> NVIDIA's 128x4 blocked order, same shape.

    This is the layout every FP4 MMA path reads, and the only documentation of it that matters is what
    the kernels accept: verified byte-for-byte against `flashinfer.nvfp4_quantize`'s own output, and end
    to end against this task's fp32 reference at every graded shape (relerr 1.66e-03, which is FP4
    accumulation noise, not a layout error).

    M and K are multiples of 128, so R % 128 == 0 and (K//16) % 4 == 0 and nothing is padded.
    """
    R, Kb = sf_u8.shape
    return sf_u8.view(R // 128, 4, 32, Kb // 4, 4).permute(0, 3, 2, 1, 4).reshape(R, Kb).contiguous()


# Probe shapes: wide, tall and square, deliberately NOT the graded ones. Selection must not be tuned to
# the sizes it is then measured at, and `verify_env._oracle_must_agree` re-checks the winner at every
# graded shape anyway.
_PROBE_MNKS = [(2048, 8192, 4096), (8192, 2048, 4096), (5120, 5120, 5120)]
_PROBE_TOL = 1e-2                       # the task's own PERF_TOL; a wrong layout misses it by 20x

_E2M1 = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0)


def _quantize(x):
    """bf16 (R, K) -> the task's own contract: packed e2m1, row-major 1x16 e4m3 scales, global scale."""
    R, K = x.shape
    lut = torch.tensor(_E2M1, device=x.device, dtype=torch.float32)
    gsf = (448.0 * 6.0) / x.float().abs().amax()
    blk = x.float().view(R, K // 16, 16) * gsf
    sf = (blk.abs().amax(-1) / 6.0).to(torch.float8_e4m3fn)
    q = blk / sf.float().clamp(min=1e-30).unsqueeze(-1)
    idx = (q.unsqueeze(-1) - lut.view(1, 1, 1, 16)).abs().argmin(-1).to(torch.uint8).view(R, K)
    return (idx[:, 0::2] | (idx[:, 1::2] << 4)).contiguous(), sf, gsf


def _unpack(packed, sf):
    R, half = packed.shape
    K = half * 2
    lut = torch.tensor(_E2M1, device=packed.device, dtype=torch.float32)
    nib = torch.stack([packed & 0xF, packed >> 4], -1).view(R, K).long()
    return (lut[nib].view(R, K // 16, 16) * sf.float().unsqueeze(-1)).view(R, K)


def _probe_cases(device="cuda"):
    """Inputs + an INDEPENDENT fp32 reference, built here rather than imported from the verifier.

    The point of a candidate probe is to catch a backend that returns the wrong answer. A reference
    computed by the thing under test cannot do that, so this recomputes the dequant from the format
    definition.
    """
    cases = []
    for i, (M, N, K) in enumerate(_PROBE_MNKS):
        g = torch.Generator(device=device).manual_seed(4000 + i)
        a = torch.randn(M, K, device=device, dtype=torch.bfloat16, generator=g)
        b = torch.randn(N, K, device=device, dtype=torch.bfloat16, generator=g)
        aq, asf, ga = _quantize(a)
        bq, bsf, gb = _quantize(b)
        alpha = torch.tensor(1.0 / float(ga * gb), device=device, dtype=torch.float32)
        ref = (_unpack(aq, asf) @ _unpack(bq, bsf).T) * float(alpha)
        cases.append(((aq, asf, bq, bsf, alpha), ref, (M, N, K),
                      float((a.float() @ b.float().T).norm())))
        del a, b
    return cases


def _candidates(fi):
    """Every FP4 route this FlashInfer exposes, as (name, callable). Ordered by expected speed; the
    order is a hint only — `_best_candidate` validates all of them and then times the survivors.

    `trtllm` is deliberately absent: it wants the *weights* pre-shuffled by `shuffle_matrix_a` as well
    as the scales, which is a different contract from the one the agent is graded on, so timing it
    would compare the ceiling against inputs the submission never gets.

    `cute-dsl` is the one to watch, and it is why `environment/Dockerfile` pins `nvidia-cutlass-dsl` to
    4.8.0.dev0. Under 4.7.1 it does not compile (an internal pass-option error: `no such option
    enable-pyir`, a duplicated `cubin-chip`) and is rejected here. Under 4.8.0.dev0 it is correct at every
    probe shape and is the FASTEST route at large graded shapes on Blackwell (8192³ and 6144×8192×4096).
    If it shows as REJECTED in `oracle_selection`, the DSL pin has moved and the ceiling has quietly
    dropped.
    """
    mm_fp4 = fi.mm_fp4

    def route(backend):
        def call(A_fp4, A_scales, B_fp4, B_scales, alpha):
            return mm_fp4(A_fp4, B_fp4.T,
                          swizzle_scales(A_scales.view(torch.uint8)),
                          swizzle_scales(B_scales.view(torch.uint8)).T,
                          alpha, torch.bfloat16, None, backend=backend)
        return call

    return [(f"mm_fp4[{bk}]", route(bk)) for bk in ("cudnn", "cutlass", "cute-dsl")]


_DISPATCH_TOL = _PROBE_TOL               # the task's own correctness tolerance

# ---------------------------------------------------------------------------- per-shape oracle dispatch
# One winner for the whole grade is the wrong shape of answer: on Blackwell `mm_fp4[cutlass]` is the
# faster FP4 backend at every probe size and 0.68x bf16 at 4096^3 -- slower than the baseline -- where
# `[cudnn]` is 2.2x. The reward is a ratio against this number, so a ceiling that depends on which shape
# the probe happened to use is not a ceiling.
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

    Used by tasks that build probe cases of their own (this GEMM task). The others hand
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


def _best_candidate(fi):
    """CORRECTNESS FIRST, then speed. Two passes, never one.

    Validating and timing a candidate in the same pass ranks them by how warm the GPU happened to be:
    the first candidate pays the JIT compile and the cold clocks, the last is timed at full boost, and
    the winner is whichever went last. Worse, a single-pass loop times a candidate before it has been
    proved right, and a wrong ceiling must never be timed. So: validate every candidate at every probe shape against an independent fp32 reference,
    keep the ones that are correct WHERE THEY RUN, and hand the survivors to `_Dispatch`, which chooses
    between them per shape.
    """
    cases = [(args, ref, str(shape)) for args, ref, shape, _ in _probe_cases()]
    routes = _validated(_candidates(fi), cases, _PROBE_TOL)
    if not routes:
        raise OracleUnavailable(
            "no FlashInfer FP4 backend is both available and correct on this device. Tried: "
            + "; ".join(SELECTION_LOG))
    del cases
    torch.cuda.empty_cache()
    return ("+".join(n for n, _ in routes),
            _Dispatch(routes, key=lambda A_fp4, A_scales, B_fp4, B_scales, alpha, *a, **kw:
                      (A_fp4.shape[0], B_fp4.shape[0], A_fp4.shape[1] * 2)))


def oracle_nvfp4_gemm():
    """Load FlashInfer from the grader-only path and return the fastest CORRECT FP4 route on this GPU."""
    global _ORACLE, ORACLE_BACKEND
    if _ORACLE is not None:
        return _ORACLE
    _require_blackwell()
    _with_oracle_libs()
    try:
        import flashinfer as fi
    except Exception as e:                                               # noqa: BLE001
        raise OracleUnavailable(
            "FlashInfer is this task's oracle and it will not import: "
            f"{type(e).__name__}: {str(e)[:200]}. The image installs it and moves it to "
            f"{ORACLE_LIBS}; check that step in environment/Dockerfile. Refusing to score.") from e
    if not hasattr(fi, "mm_fp4"):
        raise OracleUnavailable(
            f"FlashInfer {getattr(fi, '__version__', '?')} has no mm_fp4 — this build predates the "
            "NVFP4 GEMM entry point. Rebuild the image. Refusing to score.")
    ORACLE_BACKEND, _ORACLE = _best_candidate(fi)
    return _ORACLE


def candidates():
    """Every FP4 route, for inspecting the whole table rather than just the winner."""
    _require_blackwell()
    _with_oracle_libs()
    import flashinfer as fi
    return _candidates(fi)
