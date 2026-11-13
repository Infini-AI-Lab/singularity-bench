"""nvfp4-gemm verifier — MULTI-SHAPE, LIVE-ORACLE reward (nothing hardcoded).

The agent's /app/nvfp4_gemm.py:nvfp4_gemm(A_fp4, A_scales, B_fp4, B_scales, alpha) -> C (M,N) bf16 is graded
by its GEOMEAN speedup over a bfloat16 cuBLAS matmul (`a @ b.T` of the same size) across several UNDISCLOSED
(M,N,K) shapes — a big square, a medium square, and non-square M!=N!=K cases (tall + wide) — so a kernel
hyper-specialised to one size cannot win. The reward ceiling is the sealed FlashInfer FP4 oracle's geomean
speedup over the same bf16 baseline, **re-benchmarked live on this GPU every grade** (no hardcoded speedup
-> portable across hardware):

    reward = correct x clamp( ln(agent_geomean) / ln(oracle_geomean), 0, 1 )

`correct` gates on matching an independent fp32 dequant reference within relerr 1e-2 at every graded shape
(a fast wrong kernel scores 0). The exact grader shapes are NOT disclosed to the agent (measure.py uses a
different set).

On Blackwell, geomean over the five graded shapes: the oracle is ~3.1x bf16 (ln ~ 1.15), a tuned Triton
`tl.dot_scaled` kernel is ~1.3x, and unpacking to bf16 and calling cuBLAS is ~0.2x (reward 0.00), so the
reward band has room in the middle.
"""
import importlib.util
import json
import math
import os
import sys
sys.path.insert(0, "/opt/eval-runtime")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import time
import traceback

import torch  # noqa: E402

REWARD_DIR = "/logs/verifier"
MODULE_PATH = "/app/nvfp4_gemm.py"
PERF_TOL = 1e-2                 # FP4-appropriate correctness tolerance vs the fp32 dequant reference

# Undisclosed grader shapes (M, N, K), all multiples of 128: a big square, a medium square, and non-square
# M!=N!=K cases (tall + wide) so single-shape overfitting can't score 1.0.
# On Blackwell the best FP4 kernel available is 0.68x bf16 at 4096^3 -- BELOW the baseline, because
# 137 GFLOP is too little work to amortise a block-scaled MMA -- and 2.5x to 3.8x everywhere from ~400 GFLOP
# up. A graded shape where the best kernel in the world loses does not measure kernel quality, it measures
# how small the GEMM is, so 4096^3 is not one. What is here:
# a big square, a wide short-K case, a tall one, a wide one, and a medium non-square, spanning 412-1100
# GFLOP. None of them is a measure.py shape.
GRADER_SHAPES = [
    (8192, 8192, 8192),
    (6144, 8192, 4096),
    (16384, 4096, 8192),
    (4096, 16384, 8192),
    (12288, 6144, 6144),
]
# extra small shapes purely for correctness breadth (fast; agent vs fp32 reference only, NOT run through
# the oracle). M and K stay multiples of 128 — the format's own alignment, stated in instruction.md.
CORRECT_SHAPES = [
    (256, 256, 256),
    (512, 1024, 512),
    (1024, 512, 768),
    (1536, 2048, 1024),
]

# the complete e2m1 value table — 1 sign bit, 2 exponent bits, 1 mantissa bit, so 16 values and no more
E2M1_VALUES = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
               -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # tests/ — for oracle_bench (sealed)
from oracle_bench import OracleUnavailable  # noqa: E402  (sealed; see its docstring)


# Oracle code runs only in this privileged process. The submission runs in a
# separate UID/process with no access to /tests or /opt/oracle-libs. The
# import guard below is diagnostic only; it is not the security boundary.
# Oracle module names and a diagnostic import guard for the trusted grader.
# Candidate isolation is enforced by the worker UID and private filesystem paths.
ORACLE_LIBS = ('deep_gemm', 'flashinfer')


class _OracleSeal:
    """A sys.meta_path finder that makes ORACLE_LIBS unimportable outside the sealed grader."""

    def __init__(self, names):
        self.names = set(names)
        self.seen = set()

    def _grader_is_asking(self) -> bool:
        f = sys._getframe(1)
        while f is not None:
            name = f.f_globals.get("__name__") or ""
            if name == "oracle_bench" or name.split(".")[0] in self.names:
                return True
            f = f.f_back
        return False

    def find_spec(self, fullname, path=None, target=None):
        root = fullname.split(".")[0]
        if root in self.names and not self._grader_is_asking():
            self.seen.add(root)
            raise ModuleNotFoundError(
                repr(root) + " is this grader's own oracle and is not available to a submission: "
                "the task is to write the kernel yourself. See instruction.md.", name=fullname)
        return None                                  # everything else: not our business


_SEAL = _OracleSeal(ORACLE_LIBS)
sys.meta_path.insert(0, _SEAL)


# ---------------------------------------------------------------- the prebuilt-GEMM seal
# torch's own `_scaled_mm` / `_scaled_mm_v2` (and their public wrapper `torch.nn.functional.scaled_mm`)
# reach cuBLAS's block-scaled FP4 kernel, which on Blackwell is ~2.7x bf16 across the graded shapes --
# the same class as this grader's FlashInfer ceiling, in ONE CALL. Left available, the top of the reward
# band would be an API lookup rather than a kernel: every driver that knows the recipe would score 1.000
# and every driver that does not would score ~0.
#
# So it is closed, for the same stated reason `deep_gemm` and `flashinfer` are: THIS IS THE CEILING, and
# calling the ceiling is not writing a kernel. What the task hands over on purpose -- torch's ordinary ops,
# Triton (whose `tl.dot_scaled` speaks NVFP4 natively on Blackwell and is the intended middle rung),
# CUTLASS / the CuTe DSL, nvcc -- is untouched. `a @ b.T` still works; so does the oracle.
#
# Closed at the DISPATCHER, not on the python attribute. `torch._scaled_mm_v2 = raise` leaves
# `torch.ops.aten._scaled_mm_v2` and `torch._C._VariableFunctions._scaled_mm_v2` wide open, and a guard
# that the first search-and-replace walks around is worse than none: it would report "clean" for a
# submission that called it. Overriding the op's CUDA kernel catches every route into it, including ones
# nobody thought to patch -- verified against the public wrapper, the private binding and the op itself.
#
# Like the import seal, it RAISES rather than silently scoring: a submission that reaches for it sees the
# refusal in its own traceback and can route around it, instead of finding out four hours later. And like
# the import seal, the attempt is recorded and zeroes the reward, because "called the prebuilt kernel" and
# "wrote a fast one" must not arrive wearing the same number. environment/measure.py installs the same ban,
# so this is discovered while developing rather than at grade time.
BANNED_OPS = ("_scaled_mm", "_scaled_mm_v2")


class _PrebuiltGemmSeal:
    def __init__(self, names):
        self.names = names
        self.seen = set()
        self._lib = None

    def arm(self):
        try:
            self._lib = torch.library.Library("aten", "IMPL")       # kept alive: GC would deregister it
            for op in self.names:
                self._lib.impl(op, self._make(op), "CUDA")
        except Exception as e:                                       # noqa: BLE001
            # Not fatal. The ban is a guard on the reward, not a precondition for measuring one; a torch
            # that will not take the override still grades honestly for every submission that did not go
            # looking. Recorded so the cell says which regime it ran under.
            self.armed_error = f"{type(e).__name__}: {str(e)[:160]}"
            return False
        self.armed_error = ""
        return True

    def _make(self, op):
        def _banned(*args, **kwargs):
            self.seen.add(op)
            raise RuntimeError(
                "torch." + op + " reaches cuBLAS's prebuilt block-scaled GEMM, which is this grader's own "
                "ceiling -- the task is to write that kernel, not to call it. See instruction.md. "
                "torch's ordinary ops, Triton (tl.dot_scaled speaks NVFP4), CUTLASS/CuTe and nvcc are "
                "all available and are the task.")
        return _banned


_GEMM_SEAL = _PrebuiltGemmSeal(BANNED_OPS)
_GEMM_SEAL_ARMED = _GEMM_SEAL.arm()


class GraderBroken(BaseException):
    """The GRADER could not measure. Not the submission's doing, and not a score.

    Derives from BaseException on purpose, so it passes straight through the `except Exception` in
    main() -- which exists to turn a submission that raises into reward 0.0. Every reward here is a
    ratio against the sealed oracle, so an oracle that cannot load on this machine leaves nothing to
    measure; writing 0.0 would be indistinguishable from a slow kernel. Nothing is written instead,
    the process exits non-zero, and the cell is void: no reward at all.
    """

# The version of the grading contract, written into reward.json so that a result records which verifier
# produced it.
VERIFIER_VERSION = 4


def _grader_only(fn):
    """Wrap the sealed oracle so anything IT raises is a grader fault, never the submission's score.

    The oracle is the denominator of every reward here, and it is wrapped code we wrote around someone
    else's kernel -- a layout that a new library version reads differently, a cubin the offline image
    never downloaded, a compile cache limit the submission lowered. Every one of those raises from
    inside the oracle call, in the middle of the perf loop, where `except Exception` was waiting to
    turn it into `reward: 0.0`. That is the grader libelling the submission: the same number a genuinely
    slow kernel earns, for a fault the submission had no part in.

    GraderBroken derives from BaseException, so it passes straight through that handler, nothing is
    written, and the cell is VOID. Load-time failures are handled the same way.
    """
    def _call(*args, **kwargs):
        try:
            return fn(*args, **kwargs)
        except BaseException as e:                                   # noqa: BLE001
            raise GraderBroken(
                "the sealed oracle raised while benchmarking: "
                + type(e).__name__ + ": " + str(e)[:300]) from e
    return _call


def _oracle_must_agree(out, reference, shape, tol):
    """The oracle is checked at EVERY graded shape, on the same inputs the reward is computed from.

    Not a paranoid extra: an oracle backend can agree with the reference at every probe configuration
    and be badly wrong at a graded one -- while reporting a large "speedup" for producing garbage. A
    load-time probe only covers the configurations someone thought to probe. This covers the ones that are actually graded, which is the set that matters.

    Wrong oracle -> GraderBroken -> the cell is VOID. Never a score: every reward on this task is a
    ratio against this kernel, so if it is wrong the reward is not a small number, it is a meaningless
    one, and 0.0 would be indistinguishable from a submission that failed.
    """
    def _rel(a, b):
        if isinstance(a, (tuple, list)):
            return max(_rel(x, y) for x, y in zip(a, b))
        return float((a.float() - b.float()).norm() / (b.float().norm() + 1e-12))

    try:
        rel = _rel(out, reference)
    except BaseException as e:                                       # noqa: BLE001
        raise GraderBroken(f"could not compare the oracle against the reference at {shape}: "
                           f"{type(e).__name__}: {e}") from e
    if not (rel <= tol):
        raise GraderBroken(
            f"THE ORACLE IS WRONG at {shape}: relerr {rel:.3e} against this task's own reference "
            f"(tolerance {tol:.1e}). Every reward here would be a ratio against a kernel computing "
            f"the wrong answer, so nothing is scored.")
    return rel


def _void(exc) -> None:
    """A voided cell must still say why.

    No reward file -- that is the whole point, a ratio against an oracle that never ran is not a score
    -- but the reason is written beside the missing reward.
    """
    try:
        os.makedirs(REWARD_DIR, exist_ok=True)
        with open(f"{REWARD_DIR}/grader_error.json", "w") as f:
            json.dump({"voided": True, "verifier_version": VERIFIER_VERSION,
                       "reason": (type(exc).__name__ + ": " + str(exc))[:800],
                       "trace": traceback.format_exc()[-3000:],
                       "note": "No reward.json is written for a voided cell: the oracle is the "
                               "denominator of the reward, so a grade without it would be a number "
                               "with no meaning. This is a GRADER fault, not a submission result."},
                      f, indent=2)
        print("VOID:", type(exc).__name__, str(exc)[:300])
    except BaseException:                                            # noqa: BLE001
        pass


def quantize(x):
    """bf16 (R, K) -> the agent's contract: packed e2m1 (R, K//2), row-major 1x16 e4m3 scales, global scale.

    NVFP4 as NVIDIA defines it: one e4m3 scale per run of 16 elements, chosen so the block's largest
    magnitude maps onto e2m1's largest value (6.0), plus one per-tensor scale so those e4m3 block scales
    themselves stay in range. The reward's inputs are generated here and handed to BOTH the submission and
    the oracle, so neither can be advantaged by a quantiser of its own.
    """
    R, K = x.shape
    lut = torch.tensor(E2M1_VALUES, device=x.device, dtype=torch.float32)
    gsf = (448.0 * 6.0) / x.float().abs().amax()
    blk = x.float().view(R, K // 16, 16) * gsf
    sf = (blk.abs().amax(-1) / 6.0).to(torch.float8_e4m3fn)
    q = blk / sf.float().clamp(min=1e-30).unsqueeze(-1)
    idx = (q.unsqueeze(-1) - lut.view(1, 1, 1, 16)).abs().argmin(-1).to(torch.uint8).view(R, K)
    return (idx[:, 0::2] | (idx[:, 1::2] << 4)).contiguous(), sf, gsf


def unpack(packed, sf):                             # the dequant spec, as published in reference.py
    R, half = packed.shape
    K = half * 2
    lut = torch.tensor(E2M1_VALUES, device=packed.device, dtype=torch.float32)
    nib = torch.stack([packed & 0xF, packed >> 4], -1).view(R, K).long()
    return (lut[nib].view(R, K // 16, 16) * sf.float().unsqueeze(-1)).view(R, K)


def ref_fp32(A_fp4, A_scales, B_fp4, B_scales, alpha):   # independent ground truth (fp32 dequant + matmul)
    return (unpack(A_fp4, A_scales) @ unpack(B_fp4, B_scales).T) * float(alpha)


def _load():
    from kernel_client import RemoteKernel
    return RemoteKernel(MODULE_PATH, 'nvfp4_gemm', 'nvfp4-gemm')


def _mk(M, N, K, seed):
    g = torch.Generator(device="cuda").manual_seed(seed)
    a = torch.randn(M, K, device="cuda", dtype=torch.bfloat16, generator=g)
    b = torch.randn(N, K, device="cuda", dtype=torch.bfloat16, generator=g)
    Aq, As, ga = quantize(a)
    Bq, Bs, gb = quantize(b)
    alpha = torch.tensor(1.0 / float(ga * gb), device="cuda", dtype=torch.float32)
    return a, b, (Aq, As, Bq, Bs, alpha)


def _relerr(a, b):
    return float((a.float() - b.float()).norm() / (b.float().norm() + 1e-12))


def _bench(fn, target_s=0.05, batches=6, warm_s=0.2):
    """Back-to-back launches, one sync per batch, MINIMUM over batches. Not per-launch CUDA events.

    A sync after every launch times ONE COLD, ISOLATED launch: launch latency in, warm L2 out. At 8192^3
    that reads the FlashInfer ceiling as 0.424 ms when 200 back-to-back launches average 0.224 ms -- an
    89% error, on the number every reward on this task is divided by, and it is not even a consistent
    error: it depends on how long each kernel takes to launch. Steady-state throughput is both
    what the kernel does inside a model and what the bf16 baseline it is compared against is quoted at.

    Minimum over batches rather than mean, for the same reason the per-launch version took a min: on a
    shared GPU contention slows some batches and never the fastest one. The batch size adapts so each
    batch is ~`target_s`, which keeps a slow submission from turning the perf loop into a timeout and a
    fast one from being measured over four launches.
    """
    t0 = time.perf_counter()
    n = 0
    while n < 5 or (time.perf_counter() - t0) < warm_s:     # JIT is primed by the caller; this ramps clocks
        fn()
        torch.cuda.synchronize()
        n += 1
    per = max((time.perf_counter() - t0) / n, 1e-6)
    batch = max(5, min(200, int(target_s / per)))
    best = float("inf")
    for _ in range(batches):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(batch):
            fn()
        torch.cuda.synchronize()
        best = min(best, (time.perf_counter() - t0) / batch)
    return best  # seconds



# ---------------------------------------------------------------------------- the recomputation check
# THE PERF LOOP TIMES THE SAME TENSORS IT CHECKED. It has to: regenerating inputs inside the timed
# region would measure the generator, not the kernel. But that makes one cheat free -- cache the result
# on the identity of the inputs, and every timed call after the first is a dict lookup:
#
#     def kernel(*args):
#         key = (id(args[0]), id(args[2]))
#         if key not in CACHE: CACHE[key] = slow_but_correct(*args)
#         return CACHE[key]
#
# With the body of that cache calling the task's OWN SLOW REFERENCE, such a submission is correct, far
# faster than the oracle, and would score 1.000. It computes nothing and wins.
#
# So: negate one input, call again, and require the output to move. A kernel recomputes and its output
# moves (every operation in this suite is odd or nonlinear in the tensor chosen, so negating it changes
# the answer materially); a cache returns the identical tensor and moves by exactly zero. The reference
# output is cloned first, because a legitimate kernel is allowed to reuse its output buffer between
# calls -- reusing storage is not caching a result, and must not be confused with it.
#
# A GUARD, not a proof. A cache keyed on tensor CONTENT would survive this, but it has to read the
# inputs to hash them, which is most of what the kernel reads anyway. The point is that the free version
# is not free.
RECOMPUTE_MIN = 1e-3


def _must_recompute(fn, call_args, flip):
    """-> relative movement of the output when `flip` negates one input. ~2.0 for a kernel, 0.0 for a cache."""
    def snap(o):
        return tuple(t.detach().clone() for t in o) if isinstance(o, (tuple, list)) else o.detach().clone()

    def diff(a, b):
        if isinstance(a, (tuple, list)):
            return max(diff(x, y) for x, y in zip(a, b))
        return float((a.float() - b.float()).norm() / (b.float().norm() + 1e-12))

    try:
        before = snap(fn(*call_args))
        flip()
        try:
            after = fn(*call_args)
        finally:
            flip()                                   # negation is exact and self-inverse
        return diff(after, before)
    except Exception:                                # noqa: BLE001  a submission that raises here
        return float("inf")                          # fails the correctness gate anyway

def _geomean(xs):
    return math.exp(sum(math.log(max(v, 1e-9)) for v in xs) / len(xs)) if xs else 0.0


def main():
    details = {"kernel_execution": "restricted-worker-v1", "benchmark_method": "isolated-batch-wall-v1"}
    correct_gate = False
    cached_output = False
    agent_geo, oracle_geo = 0.0, 1.0
    corr_ok = False
    try:
        fn = _load()

        # ---- correctness breadth (small shapes; agent vs fp32 reference) ----
        corr_ok, worst, msg = True, 0.0, ""
        for i, (M, N, K) in enumerate(CORRECT_SHAPES):
            _, _, args = _mk(M, N, K, seed=10 + i)
            ref = ref_fp32(*args)
            out = fn(*args)
            if i == 0:          # once per grade: is the submission computing, or replaying?
                _moved = _must_recompute(fn, args, lambda: args[4].mul_(-1.0))
                details["recompute_movement"] = round(_moved, 6) if _moved != float('inf') else -1.0
                if not (_moved > RECOMPUTE_MIN):
                    cached_output = True

            if out is None or tuple(out.shape) != (M, N):
                corr_ok, msg = False, f"cfg{i} {(M, N, K)} bad/None shape"
                break
            re = _relerr(out, ref)
            worst = max(worst, re)
            if re > PERF_TOL:
                corr_ok, msg = False, f"cfg{i} {(M, N, K)} relerr {re:.3e}"
                break
        details.update(correct_worst_relerr=round(worst, 6), correct_msg=msg or "all within tol")

        # ---- per-shape perf: agent + oracle speedup over bf16 cuBLAS, correctness-checked at each perf shape ----
        # The perf-size output is ALSO correctness-checked, so "return fast garbage at the perf size" (a
        # shape-conditional reward hack) cannot pass — the timed kernel must actually be correct at each shape.
        oracle_gemm = None
        agent_sp, oracle_sp, per_shape, perf_ok = [], [], [], corr_ok
        for i, (M, N, K) in enumerate(GRADER_SHAPES):
            a, b, args = _mk(M, N, K, seed=100 + i)
            ref = ref_fp32(*args)
            out = fn(*args)                                  # correctness + primes the agent's JIT for this shape
            re = _relerr(out, ref) if (out is not None and tuple(out.shape) == (M, N)) else 1.0
            if re > PERF_TOL:
                perf_ok = False
            if oracle_gemm is None:                          # lazy-load the sealed FlashInfer oracle (LIVE)
                try:
                    from oracle_bench import oracle_nvfp4_gemm as _get_oracle
                    oracle_gemm = _grader_only(_get_oracle())
                except Exception as _e:   # GRADER-side failure -- see GraderBroken
                    raise GraderBroken("sealed oracle unavailable: "
                                       + type(_e).__name__ + ": " + str(_e)[:200]) from _e
            _oracle_must_agree(oracle_gemm(*args), ref, (M, N, K), PERF_TOL)
            # round-robin pre-warm all three together to pin the GPU at steady-state boost clocks, so the three
            # timings below are NOT biased by benching order (first kernel would otherwise see cooler clocks than
            # the last). JIT is already primed above, so this only ramps clocks.
            tw = time.perf_counter()
            while time.perf_counter() - tw < 0.2:
                a @ b.T
                fn(*args)
                oracle_gemm(*args)
                torch.cuda.synchronize()
            # time all three at steady-state clocks (each _bench re-ramps via warm_s; no compile stalls remain)
            bf16_t = _bench(lambda: a @ b.T)                 # the speedup baseline (bf16 cuBLAS)
            agt_t = fn.benchmark(*args)
            details.setdefault("isolated_batches", []).append(dict(fn.last_benchmark))
            o_t = _bench(lambda: oracle_gemm(*args))
            a_sp = bf16_t / agt_t if agt_t > 0 else 0.0
            o_sp = bf16_t / o_t if o_t > 0 else 0.0
            agent_sp.append(a_sp)
            oracle_sp.append(o_sp)
            per_shape.append({"shape": [M, N, K], "agent_x": round(a_sp, 3),
                              "oracle_x": round(o_sp, 3), "relerr": round(re, 5)})
            del a, b, args, ref, out
            torch.cuda.empty_cache()
        agent_geo, oracle_geo = _geomean(agent_sp), _geomean(oracle_sp)
        details.update(per_shape=per_shape, agent_geomean=round(agent_geo, 4),
                       oracle_geomean=round(oracle_geo, 4), perf_size_ok=perf_ok)
        try:
            from oracle_bench import ORACLE_BACKEND, SELECTION_LOG
            details.update(oracle_backend=ORACLE_BACKEND, oracle_selection=list(SELECTION_LOG))
        except Exception:                                    # noqa: BLE001
            pass
        correct_gate = corr_ok and perf_ok
    except OracleUnavailable:
        # The GRADER is broken here, not the submission. Write nothing: a cell with no
        # reward.json is void, which is the honest outcome.
        raise
    except Exception as e:
        details["error"] = f"{e.__class__.__name__}: {e}"[:220]
        details["trace"] = traceback.format_exc()[-800:]

    reward = 0.0
    if correct_gate and agent_geo > 1.0 and oracle_geo > 1.0:
        reward = min(1.0, math.log(agent_geo) / math.log(oracle_geo))
    # A submission that reached for the grader's own oracle library, or for cuBLAS's prebuilt block-scaled
    # GEMM, did not write a kernel. Zeroed, and SAID SO in both files, so it is never confused with a
    # correctness failure or a slow kernel.
    if _SEAL.seen:
        details["oracle_libraries_refused"] = sorted(_SEAL.seen)
        reward = 0.0
    # A submission that returned a cached tensor was TIMED ON A DICT LOOKUP. Not a slow kernel and not
    # a wrong one -- a number with nothing behind it, which is the one thing this suite exists to avoid
    # recording. Zeroed and said so, in both files, like the seals.
    if cached_output:
        details["cached_output"] = ("the output did not change when an input was negated: the submission "
                                    "is replaying a cached result, not computing one. See "
                                    "verify_env._must_recompute.")
        reward = 0.0
    if _GEMM_SEAL.seen:
        details["prebuilt_gemm_refused"] = sorted(_GEMM_SEAL.seen)
        reward = 0.0
    details["prebuilt_gemm_seal_armed"] = bool(_GEMM_SEAL_ARMED)
    if not _GEMM_SEAL_ARMED:
        details["prebuilt_gemm_seal_error"] = _GEMM_SEAL.armed_error
    os.makedirs(REWARD_DIR, exist_ok=True)
    result = {"reward": round(reward, 4), "correct": 1.0 if correct_gate else 0.0,
              "used_oracle_library": 1.0 if _SEAL.seen else 0.0,
              "cached_output": 1.0 if cached_output else 0.0,
              "used_prebuilt_gemm": 1.0 if _GEMM_SEAL.seen else 0.0,
              # THE CEILING'S OWN REPORT CARD. reward is clamped at 1.0, so a submission that
              # is twice the oracle's speed and one that merely matches it both record 1.000.
              # This is how you tell them apart -- and how you find out that the oracle needs
              # replacing before a whole grid of 1.000s gets read as "the task is solved".
              "agent_over_oracle": round(agent_geo / oracle_geo, 4) if oracle_geo > 0 else 0.0,
              "beat_oracle": 1.0 if agent_geo > oracle_geo else 0.0,
              "verifier_version": VERIFIER_VERSION,
              "agent_geomean": round(agent_geo, 4), "oracle_geomean": round(oracle_geo, 4)}
    json.dump(result, open(f"{REWARD_DIR}/reward.json", "w"), indent=2)
    open(f"{REWARD_DIR}/reward.txt", "w").write(str(round(reward, 4)))
    json.dump(details, open(f"{REWARD_DIR}/details.json", "w"), indent=2, default=str)
    print("reward:", round(reward, 4), "| correct:", result["correct"],
          "| agent_geo:", round(agent_geo, 2), "| oracle_geo:", round(oracle_geo, 2))


try:
    main()
except BaseException as _fatal:          # GraderBroken / OracleUnavailable / anything else terminal
    _void(_fatal)
    raise
