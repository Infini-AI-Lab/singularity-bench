"""Self-assessment tool — your GEOMEAN speedup over a FlashAttention-2 backward across a RANGE of sizes.

    python /app/measure.py            # full sizes
    python /app/measure.py --quick    # smaller sizes for fast iteration

You are graded on the **geomean speedup across many shapes** (varied batch, seqlen, causal/non-causal) — NOT one
fixed size. The sizes below are a DIFFERENT sample from the same regime the grader uses, so optimize for
GENERALITY: a kernel hyper-tuned to a single shape will not score well. This prints your per-shape and geomean
speedup + correctness. (It does NOT print your grade.)

The baseline is a FlashAttention-2 backward (PyTorch's FLASH SDPA backend). It is here ONLY as your speed
yardstick — you MAY NOT call scaled_dot_product_attention / SDPA / flash_attn in your submission (the grader bans
it, incl. the raw aten ops); you must write the kernel yourself (see reference.py).
"""
import argparse
import math
import sys

import time

import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel

sys.path.insert(0, "/app")
from attn_backward import attn_backward  # your implementation

H, D = 16, 128
SCALE = 1.0 / math.sqrt(D)

# (B, S, causal) — a different sample of the same regime as the grader (varied batch/seqlen/causal, incl. a big S)
FULL = [(1, 8192, True), (2, 4096, True), (4, 2048, False), (1, 16384, False)]
QUICK = [(1, 2048, True), (2, 1024, True), (2, 2048, False), (1, 4096, True)]


def _mk(B, S, seed=777):
    g = torch.Generator(device="cuda").manual_seed(seed)
    t = lambda: torch.randn(B, S, H, D, device="cuda", dtype=torch.bfloat16, generator=g)
    return t(), t(), t(), t()          # q, k, v, do  (B,S,H,D)


def _torch_fwd(q, k, v, causal):        # o (B,S,H,D) bf16, lse (B,H,S) fp32
    qh, kh, vh = (x.transpose(1, 2).float() for x in (q, k, v))
    s = (qh @ kh.transpose(-2, -1)) * SCALE
    if causal:
        s = s.masked_fill(torch.triu(torch.ones(s.shape[-2], s.shape[-1], device="cuda", dtype=torch.bool), 1), float("-inf"))
    o = (s.softmax(-1) @ vh).transpose(1, 2).to(torch.bfloat16)
    return o, torch.logsumexp(s, dim=-1)


def _gt(q, k, v, do, causal):           # fp32 autograd ground truth
    q2, k2, v2 = (x.detach().float().requires_grad_() for x in (q, k, v))
    qh, kh, vh = (x.transpose(1, 2) for x in (q2, k2, v2))
    s = (qh @ kh.transpose(-2, -1)) * SCALE
    if causal:
        s = s.masked_fill(torch.triu(torch.ones(s.shape[-2], s.shape[-1], device="cuda", dtype=torch.bool), 1), float("-inf"))
    o = (s.softmax(-1) @ vh).transpose(1, 2)
    return torch.autograd.grad(o, (q2, k2, v2), do.float())


def _relerr(a, b):
    return float((a.float() - b.float()).norm() / (b.float().norm() + 1e-12))


def _bench(fn, target_s=0.05, batches=6, warm_s=0.2):
    """Back-to-back launches, one sync per batch, minimum over batches — the same method the grader uses.

    Not a sync after every launch: that times one cold, isolated launch (launch latency in, warm L2 out)
    and can misread a short kernel by most of its runtime. Steady-state throughput is what the kernel does
    inside a model, and it is what the baseline it is compared against is quoted at too.
    """
    t0 = time.perf_counter()
    n = 0
    while n < 5 or (time.perf_counter() - t0) < warm_s:   # call fn once before this to trigger any JIT
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
    return best


def _fa2_bwd_fn(q, k, v, do, causal):   # the baseline: FA-2 backward (SDPA FLASH), backward-only
    qb, kb, vb = (x.transpose(1, 2).contiguous().detach().requires_grad_() for x in (q, k, v))
    dob = do.transpose(1, 2).contiguous()
    with sdpa_kernel([SDPBackend.FLASH_ATTENTION]):
        out = F.scaled_dot_product_attention(qb, kb, vb, is_causal=causal, scale=SCALE)
    return lambda: torch.autograd.grad(out, (qb, kb, vb), dob, retain_graph=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="smaller sizes for fast iteration")
    args = ap.parse_args()
    shapes = QUICK if args.quick else FULL
    print(f"geomean over {len(shapes)} shapes ({'quick' if args.quick else 'full'}); "
          f"the grader uses DIFFERENT sizes in the same regime — optimize for generality")
    sps, allok = [], True
    for (B, S, causal) in shapes:
        q, k, v, do = _mk(B, S)
        o, lse = _torch_fwd(q, k, v, causal)
        gt = _gt(q, k, v, do, causal)
        out = attn_backward(q, k, v, o, lse, do, SCALE, causal)
        re = max(_relerr(a, b) for a, b in zip(out, gt)) if (out is not None and len(out) == 3) else 1.0
        ok = re <= 2e-2
        allok = allok and ok
        ft = _bench(_fa2_bwd_fn(q, k, v, do, causal))
        at = _bench(lambda: attn_backward(q, k, v, o, lse, do, SCALE, causal))
        sp = ft / at if at > 0 else 0.0
        sps.append(sp)
        print(f"  B{B} S{S} causal={int(causal)}: {sp:6.2f}x   relerr {re:.1e}  {'ok' if ok else 'FAIL'}")
    geo = math.exp(sum(math.log(max(s, 1e-9)) for s in sps) / len(sps)) if sps else 0.0
    print(f"  => GEOMEAN {geo:.2f}x over the FA-2 backward   ({'all correct' if allok else 'SOME WRONG — fix first'})")
    print("     graded continuously on this geomean vs a state-of-the-art kernel (FA-4, CuTe DSL) — higher is better.")


if __name__ == "__main__":
    main()
