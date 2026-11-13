"""flash-attn-backward verifier — MULTI-SHAPE, LIVE-ORACLE reward (nothing hardcoded).

The agent's /app/attn_backward.py is graded by its GEOMEAN speedup over a FlashAttention-2 backward (PyTorch's
FLASH SDPA backend) across several UNDISCLOSED shapes (varied batch / seqlen / causal, incl. a big-S case and a
non-causal case) — so a kernel hyper-specialised to one shape cannot win. The reward ceiling is the sealed
oracle's geomean (per shape, the faster correct route of the vendored FlashAttention-4 CuTe DSL backward and
PyTorch's cuDNN attention backward), **re-benchmarked live on this GPU every grade** (no hardcoded speedup →
portable across hardware):

    reward = correct × clamp( ln(agent_geomean) / ln(oracle_geomean), 0, 1 )

`correct` gates on matching an independent fp32 AUTOGRAD ground truth within relerr 2e-2 at every graded shape (a
fast wrong kernel scores 0). The exact grader shapes are NOT disclosed to the agent (measure.py uses a different
set). Borrowing a prebuilt attention backward is blocked two ways: a source scan for banned APIs and a
TorchDispatchMode runtime guard that flags any SDPA/cuDNN attention op the submission dispatches — neither touches a
legit hand-written Triton/CuTe/CUDA kernel (nor the vendored FA-4 CuTe oracle).
"""
import faulthandler
import importlib.util
import json
import math
import os
import sys
sys.path.insert(0, "/opt/eval-runtime")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import traceback

import time

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from torch.nn.attention import SDPBackend, sdpa_kernel  # noqa: E402
from torch.utils._python_dispatch import TorchDispatchMode  # noqa: E402

REWARD_DIR = "/logs/verifier"
MODULE_PATH = "/app/attn_backward.py"
PERF_TOL = 2e-2            # timed-shape output must match within this rel-err, else the speedup is for garbage
H, D = 16, 128            # grader-fixed heads / head_dim
SCALE = 1.0 / math.sqrt(D)

# Undisclosed grader shapes (B, S, causal): the canonical case, a BIG-S case, a NON-causal case, a batch-4 case —
# H=16, D=128 fixed. FA-4 (CuTe) supports all of them; a kernel overfit to one (B,S,causal) can't win the geomean.
GRADER_SHAPES = [
    (2, 8192, True),      # canonical causal, large-ish
    (1, 16384, True),     # BIG S, causal
    (2, 4096, False),     # non-causal
    (4, 4096, True),      # batch 4, causal
    (1, 8192, False),     # non-causal, large-ish
]
# extra small shapes purely for correctness breadth (fast)
CORRECT_SHAPES = [
    (1, 1024, True),
    (2, 2048, False),
    (1, 2048, True),
    (2, 1024, True),
]

# banned: pre-built attention kernels reachable in the agent image — the task is to WRITE the backward, not borrow
# one. (flash_attn itself is NOT listed: it isn't installed and there is no network to fetch it, so it is sealed by
# the environment; listing it would also false-flag the vendored oracle. SDPA — incl. its raw aten
# flash/efficient/cuDNN backward ops — is the actual reachable cheat.)
BANNED = ["scaled_dot_product_attention", "_scaled_dot_product", "sdpa_kernel", "sdpbackend",
          "cudnn_attention", "xformers", "mem_efficient_attention"]


# Oracle code runs only in this privileged process. The submission runs in a
# separate UID/process with no access to /tests or /opt/oracle-libs. The
# import guard below is diagnostic only; it is not the security boundary.
# Oracle module names and a diagnostic import guard for the trusted grader.
# Candidate isolation is enforced by the worker UID and private filesystem paths.
ORACLE_LIBS = ('flash_attn', 'flashinfer')


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


class GraderBroken(BaseException):
    """The GRADER could not measure. Not the submission's doing, and not a score.

    Derives from BaseException on purpose, so it passes straight through the `except Exception` in
    main() -- which exists to turn a submission that raises into reward 0.0. Every reward here is a
    ratio against the sealed oracle, so an oracle that cannot load on this machine leaves nothing to
    measure; writing 0.0 would be indistinguishable from a slow kernel. Nothing is written instead,
    the process exits non-zero, and the cell is void: no reward at all.
    """

def _forbidden(path):
    import stat
    try:
        # This diagnostic runs before the worker starts. Do not follow a
        # submission symlink into private files or block on a named pipe.
        fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
        with os.fdopen(fd) as source:
            info = os.fstat(source.fileno())
            if not stat.S_ISREG(info.st_mode) or info.st_size > 10 * 1024 * 1024:
                return ['submission must be a regular Python file under 10 MiB']
            src = source.read().lower()
    except Exception:
        return None
    hits = [b for b in BANNED if b.lower() in src]
    return hits or None


# The version of the grading contract, written into reward.json so that a result records which verifier
# produced it.
VERIFIER_VERSION = 4

_STARTED = time.monotonic()


def _progress(stage, shape=None):
    """Retain the last operation even if the grading stage is killed during a stuck CUDA/JIT call."""
    os.makedirs(REWARD_DIR, exist_ok=True)
    record = {"stage": stage, "shape": shape,
              "elapsed_sec": round(time.monotonic() - _STARTED, 2),
              "verifier_version": VERIFIER_VERSION}
    path = os.path.join(REWARD_DIR, "progress.json")
    with open(path + ".tmp", "w") as f:
        json.dump(record, f, indent=2)
    os.replace(path + ".tmp", path)
    print("[verifier]", json.dumps(record), flush=True)


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


# Runtime guard: the source scan (above) catches the obvious cheats; this catches OBFUSCATED ones (dynamic getattr,
# etc.) by observing the actual aten ops dispatched while the agent's kernel runs. It only FLAGS the banned
# attention ops (SDPA / flash / mem-efficient / cuDNN attention) and passes everything else through, so it never
# interferes with a legitimate Triton/CuTe/CUDA kernel. Used for a one-shot DETECTION pass on the SUBMISSION at a
# tiny fixed shape, never during the timed benchmark (so it adds no timing overhead).
#
# THE ORACLE IS NOT INSIDE THIS MODE, and since one of its two routes is torch's cuDNN attention backward,
# that matters: the mode is entered around `fn(...)` alone, and the oracle is not even loaded until the perf loop
# below. The ban is on the submission calling a prebuilt attention kernel, not on the grader measuring against
# one -- the same posture as every sealed oracle library in this suite.
# PyTorch's flash/efficient/cuDNN SDPA backends ALL lower to an aten op containing "scaled_dot_product"; keying on
# that (plus cuDNN) catches every backend without false-flagging a legit hand-written flash kernel/op.
_BANNED_OPS = ("scaled_dot_product", "cudnn_attention")


class _DetectAttnOps(TorchDispatchMode):
    triggered = None

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        name = str(func).lower()
        if any(b in name for b in _BANNED_OPS):
            _DetectAttnOps.triggered = str(func)
        return func(*args, **(kwargs or {}))


def _torch_fwd(q, k, v, scale, causal):        # (B,S,H,D) -> o (B,S,H,D) bf16, lse (B,H,S) fp32
    qh, kh, vh = (t.transpose(1, 2).float() for t in (q, k, v))
    s = (qh @ kh.transpose(-2, -1)) * scale
    if causal:
        Sq, Sk = s.shape[-2], s.shape[-1]
        s = s.masked_fill(torch.triu(torch.ones(Sq, Sk, device=q.device, dtype=torch.bool), 1), float("-inf"))
    o = (s.softmax(-1) @ vh).transpose(1, 2).to(torch.bfloat16)
    lse = torch.logsumexp(s, dim=-1)
    return o, lse


def _gt_grads(q, k, v, do, scale, causal):     # independent fp32 autograd ground truth
    q2, k2, v2 = (t.detach().float().requires_grad_() for t in (q, k, v))
    qh, kh, vh = (t.transpose(1, 2) for t in (q2, k2, v2))
    s = (qh @ kh.transpose(-2, -1)) * scale
    if causal:
        Sq, Sk = s.shape[-2], s.shape[-1]
        s = s.masked_fill(torch.triu(torch.ones(Sq, Sk, device=q.device, dtype=torch.bool), 1), float("-inf"))
    o = (s.softmax(-1) @ vh).transpose(1, 2)
    return torch.autograd.grad(o, (q2, k2, v2), do.float())


def _load():
    from kernel_client import RemoteKernel
    return RemoteKernel(MODULE_PATH, 'attn_backward', 'flash-attn-backward')


def _mk(B, S, seed):
    g = torch.Generator(device="cuda").manual_seed(seed)
    t = lambda: torch.randn(B, S, H, D, device="cuda", dtype=torch.bfloat16, generator=g)
    return t(), t(), t(), t()   # q, k, v, do  (B,S,H,D)


def _relerr(a, b):
    a, b = a.float(), b.float()
    return float((a - b).norm() / (b.norm() + 1e-12))


def _worst(grads, gt):
    return max(_relerr(g, r) for g, r in zip(grads, gt))


def _bench(fn, target_s=0.05, batches=6, warm_s=0.2):
    """Back-to-back launches, one sync per batch, MINIMUM over batches. Not per-launch CUDA events.

    A sync after every launch times ONE COLD, ISOLATED launch: launch latency in, warm L2 out. For a
    block-scaled FP4 GEMM that reads 0.424 ms where 200 back-to-back launches average 0.224 ms -- an 89% error on a number the reward is divided by, and it does NOT cancel between the
    agent and the oracle, because it scales with each kernel's own launch cost. Steady-state throughput
    is both what the kernel does inside a model and what the baseline it is compared against is quoted at.

    Minimum over batches rather than mean, for the reason the per-launch version took a min: on a shared
    GPU contention slows some batches and never the fastest one. The batch size adapts so each batch is
    about `target_s`, which stops a slow submission from turning the perf loop into a timeout and a fast
    one from being measured over three launches.
    """
    t0 = time.perf_counter()
    n = 0
    while n < 5 or (time.perf_counter() - t0) < warm_s:   # JIT is primed by the caller; this ramps clocks
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


def _fa2_bwd_fn(q, k, v, do, scale, causal):
    # Build the FA-2 forward graph ONCE (SDPA FLASH backend), then return a nullary that times the BACKWARD only.
    # SDPA is the perf yardstick — it is allowed HERE (grader), only the SUBMISSION is banned from calling it.
    qb, kb, vb = (t.transpose(1, 2).contiguous().detach().requires_grad_() for t in (q, k, v))   # (B,H,S,D)
    dob = do.transpose(1, 2).contiguous()
    with sdpa_kernel([SDPBackend.FLASH_ATTENTION]):
        out = F.scaled_dot_product_attention(qb, kb, vb, is_causal=causal, scale=scale)
    return lambda: torch.autograd.grad(out, (qb, kb, vb), dob, retain_graph=True)



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
    faulthandler.enable()
    faulthandler.dump_traceback_later(300, repeat=True)
    _progress("load_submission")
    details = {"kernel_execution": "restricted-worker-v1", "benchmark_method": "isolated-batch-wall-v1"}
    correct_gate = False
    cached_output = False
    agent_geo, oracle_geo = 0.0, 1.0
    corr_ok = False
    try:
        # ---- anti-cheat #1: static source scan for banned attention APIs ----
        bad = _forbidden(MODULE_PATH)
        if bad:
            details["error"] = f"forbidden API in submission (write the kernel yourself): {bad}"
            raise SystemExit
        fn = _load()

        # ---- anti-cheat #2: runtime guard — flag any banned attention op the submission dispatches (obfuscation) --
        _DetectAttnOps.triggered = None
        _progress("submission_guard", (1, 512, True))
        gq, gk, gv, gdo = _mk(1, 512, seed=5)
        go, glse = _torch_fwd(gq, gk, gv, SCALE, True)
        try:
            with _DetectAttnOps():
                fn(gq, gk, gv, go, glse, gdo, SCALE, True)
        except Exception:
            pass  # any real error resurfaces in the correctness pass below
        if _DetectAttnOps.triggered:
            details["error"] = f"banned attention op used at runtime (write the kernel yourself): {_DetectAttnOps.triggered}"
            raise SystemExit

        # ---- correctness breadth (small shapes) vs fp32 autograd ground truth ----
        corr_ok, worst, msg = True, 0.0, ""
        for i, (B, S, causal) in enumerate(CORRECT_SHAPES):
            shape = (B, S, causal)
            _progress("correctness_reference", shape)
            q, k, v, do = _mk(B, S, seed=10 + i)
            o, lse = _torch_fwd(q, k, v, SCALE, causal)
            gt = _gt_grads(q, k, v, do, SCALE, causal)
            _progress("correctness_submission", shape)
            out = fn(q, k, v, o, lse, do, SCALE, causal)
            if i == 0:          # once per grade: is the submission computing, or replaying?
                _moved = _must_recompute(fn, (q, k, v, o, lse, do, SCALE, causal), lambda: do.mul_(-1.0))
                details["recompute_movement"] = round(_moved, 6) if _moved != float('inf') else -1.0
                if not (_moved > RECOMPUTE_MIN):
                    cached_output = True

            if out is None or len(out) != 3 or any(x is None or tuple(x.shape) != (B, S, H, D) for x in out):
                corr_ok, msg = False, f"cfg{i} (B={B},S={S},causal={causal}): bad/None output shapes"
                break
            re = _worst(out, gt)
            worst = max(worst, re)
            if re > PERF_TOL:
                corr_ok, msg = False, f"cfg{i} (B={B},S={S},causal={causal}): relerr {re:.3e}"
                break
        details.update(correct_worst_relerr=round(worst, 6), correct_msg=msg or "all within tol")

        # ---- per-shape perf: agent + oracle vs FA-2, correctness-checked at each perf shape ----
        oracle_bwd = None
        agent_sp, oracle_sp, per_shape, perf_ok = [], [], [], corr_ok
        for i, (B, S, causal) in enumerate(GRADER_SHAPES):
            shape = (B, S, causal)
            _progress("performance_reference", shape)
            q, k, v, do = _mk(B, S, seed=100 + i)
            o, lse = _torch_fwd(q, k, v, SCALE, causal)
            gt = _gt_grads(q, k, v, do, SCALE, causal)
            _progress("performance_submission_correctness", shape)
            out = fn(q, k, v, o, lse, do, SCALE, causal)
            re = _worst(out, gt) if (out is not None and len(out) == 3 and all(
                x is not None and tuple(x.shape) == (B, S, H, D) for x in out)) else 1.0
            if re > PERF_TOL:
                perf_ok = False
            _progress("baseline_timing", shape)
            fa2_t = _bench(_fa2_bwd_fn(q, k, v, do, SCALE, causal))
            _progress("submission_timing", shape)
            agt_t = fn.benchmark(q, k, v, o, lse, do, SCALE, causal)
            details.setdefault("isolated_batches", []).append(dict(fn.last_benchmark))
            a_sp = fa2_t / agt_t if agt_t > 0 else 0.0
            agent_sp.append(a_sp)
            if oracle_bwd is None:                            # lazy-load + run the sealed oracle LIVE
                _progress("oracle_load", shape)
                sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # tests/ — for oracle_bench (sealed)
                try:
                    from oracle_bench import oracle_backward as _get_oracle
                    oracle_bwd = _grader_only(_get_oracle())
                except Exception as _e:   # GRADER-side failure -- see GraderBroken
                    raise GraderBroken("sealed oracle unavailable: "
                                       + type(_e).__name__ + ": " + str(_e)[:200]) from _e
            # The oracle, on the inputs this cell's reward is computed from. A ceiling that is
            # wrong here makes every number on this task meaningless, so it voids rather than scores.
            _progress("oracle_selection_and_correctness", shape)
            _oracle_must_agree(oracle_bwd(q, k, v, o, lse, do, SCALE, causal), gt, (B, S, causal), PERF_TOL)
            _progress("oracle_timing", shape)
            o_t = _bench(lambda: oracle_bwd(q, k, v, o, lse, do, SCALE, causal))
            o_sp = fa2_t / o_t if o_t > 0 else 0.0
            oracle_sp.append(o_sp)
            per_shape.append({"shape": [B, S, causal], "agent_x": round(a_sp, 3),
                              "oracle_x": round(o_sp, 3), "relerr": round(re, 5)})
        agent_geo, oracle_geo = _geomean(agent_sp), _geomean(oracle_sp)
        details.update(per_shape=per_shape, agent_geomean=round(agent_geo, 4),
                       oracle_geomean=round(oracle_geo, 4), perf_size_ok=perf_ok)
        # WHICH oracle produced this ceiling, and what the alternatives cost. The oracle is chosen
        # live and per shape, so "the oracle" is not one name -- and a reward is unreadable without
        # knowing which route was picked where.
        try:
            from oracle_bench import ORACLE_BACKEND, SELECTION_LOG
            details.update(oracle_backend=ORACLE_BACKEND, oracle_selection=list(SELECTION_LOG))
        except Exception:                                    # noqa: BLE001
            pass
        correct_gate = corr_ok and perf_ok
    except SystemExit:
        pass
    except Exception as e:
        details["error"] = f"{e.__class__.__name__}: {e}"[:220]
        details["trace"] = traceback.format_exc()[-800:]

    # ---- reward: correctness-gated log-speedup toward the FA-4 oracle (SOTA), capped at 1.0 ----
    #   reward = correct * clamp( ln(agent_geomean) / ln(oracle_geomean), 0, 1 )  (0 at FA-2 parity, 1.0 at the oracle)
    reward = 0.0
    if correct_gate and agent_geo > 1.0 and oracle_geo > 1.0:
        reward = min(1.0, math.log(agent_geo) / math.log(oracle_geo))
    # A submission that reached for the grader's own oracle library did not write a kernel. Zeroed,
    # and SAID SO in both files, so it is never confused with a correctness failure or a slow kernel.
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
    os.makedirs(REWARD_DIR, exist_ok=True)
    result = {"reward": round(reward, 4), "correct": 1.0 if correct_gate else 0.0,
              "used_oracle_library": 1.0 if _SEAL.seen else 0.0,
              "cached_output": 1.0 if cached_output else 0.0,
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
          "| agent_geo:", round(agent_geo, 3), "| oracle_geo:", round(oracle_geo, 3))
    _progress("complete")
    faulthandler.cancel_dump_traceback_later()


try:
    main()
except BaseException as _fatal:          # GraderBroken / OracleUnavailable / anything else terminal
    _void(_fatal)
    raise
