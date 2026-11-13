"""Self-assessment tool — your GEOMEAN speedup over the eager reference across a RANGE of (B, L) sizes.

    python /app/measure.py            # full sizes
    python /app/measure.py --quick    # smaller sizes for fast iteration

You are graded on the **geomean speedup across many (B, L) shapes** (short & deep KV caches, small & large batch,
including a big case) — NOT one fixed size. The sizes below are a DIFFERENT sample from the same regime the grader
uses, so optimize for GENERALITY: a kernel hyper-tuned to a single shape will not score well. This prints your
per-shape and geomean speedup + correctness. (It does NOT print your grade.) H=128, head_dim=576, value dim=512
are fixed by the contract; only B and L vary (L a multiple of 64). The baseline is the eager torch reference
(/app/reference.py), which is also the correctness spec.
"""
import argparse
import math
import sys

import time

import torch

sys.path.insert(0, "/app")
from reference import mla_decode as ref_decode     # eager baseline (also the correctness spec)
from mla_decode import mla_decode as agent_decode   # your implementation

H = 128
DV = 512
SCALE = 1.0 / math.sqrt(192)

# (B, L) — a different sample of the same regime as the grader (varied batch, short..deep cache, incl. a big case)
FULL = [(16, 4096), (48, 8192), (64, 12288), (128, 8192)]
QUICK = [(8, 2048), (16, 4096), (32, 4096), (64, 2048)]


def _mk(B, L, seed=777):
    g = torch.Generator(device="cuda").manual_seed(seed)
    qn = torch.randn(B, H, 512, device="cuda", dtype=torch.bfloat16, generator=g)
    qp = torch.randn(B, H, 64, device="cuda", dtype=torch.bfloat16, generator=g)
    ck = torch.randn(B, L, 512, device="cuda", dtype=torch.bfloat16, generator=g) / math.sqrt(512)
    kp = torch.randn(B, L, 64, device="cuda", dtype=torch.bfloat16, generator=g) / math.sqrt(64)
    q = torch.cat([qn, qp], dim=-1).contiguous()      # (B, H, 576)
    kv = torch.cat([ck, kp], dim=-1).contiguous()     # (B, L, 576)
    return q, kv


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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="smaller sizes for fast iteration")
    args = ap.parse_args()
    shapes = QUICK if args.quick else FULL
    print(f"geomean over {len(shapes)} shapes ({'quick' if args.quick else 'full'}); "
          f"the grader uses DIFFERENT (B, L) in the same regime — optimize for generality")
    sps, allok = [], True
    for (B, L) in shapes:
        q, kv = _mk(B, L)
        ref = ref_decode(q, kv, SCALE)
        out = agent_decode(q, kv, SCALE)
        re = _relerr(out, ref) if (out is not None and tuple(out.shape) == (B, H, DV)) else 1.0
        ok = re <= 2e-2
        allok = allok and ok
        et = _bench(lambda: ref_decode(q, kv, SCALE))
        at = _bench(lambda: agent_decode(q, kv, SCALE))
        sp = et / at if at > 0 else 0.0
        sps.append(sp)
        print(f"  B{B:<4d} L{L:<6d}: {sp:6.2f}x   relerr {re:.1e}  {'ok' if ok else 'FAIL'}")
    geo = math.exp(sum(math.log(max(s, 1e-9)) for s in sps) / len(sps)) if sps else 0.0
    print(f"  => GEOMEAN {geo:.2f}x over the reference   ({'all correct' if allok else 'SOME WRONG — fix first'})")
    print("     graded continuously on this geomean vs a state-of-the-art kernel — higher is better.")


if __name__ == "__main__":
    main()
