"""Self-assessment tool — your GEOMEAN speedup over a bf16 cuBLAS matmul across a RANGE of (M,N,K) sizes.

    python /app/measure.py            # full sizes
    python /app/measure.py --quick    # smaller sizes for fast iteration

`--quick` uses the SMALLEST shapes in the graded regime, where a block-scaled FP4 GEMM has the least
headroom over bf16 (there is less work to amortise the MMA setup over) — expect lower speedups there than
the grader will see, not higher. Below this regime the effect is dramatic: at 4096^3 the best FP4 kernel
available on a B200 is 0.68x bf16, i.e. slower than the baseline. Nothing that small is graded.

You are graded on the **geomean speedup across many shapes** (large square AND non-square M!=N!=K, K varied) —
NOT one fixed size. The sizes below are a DIFFERENT sample from the same regime the grader uses, so optimize
for GENERALITY: a kernel hyper-tuned to a single shape will not score well. This prints your per-shape and
geomean speedup over a bf16 cuBLAS matmul + correctness vs the fp32 dequant reference. (It does NOT print
your grade.)

The bf16 matmul is only the speed yardstick; your SUBMISSION must be the NVFP4 GEMM (a bf16 matmul computes
a different number and fails correctness).

This script installs the SAME ban on torch's prebuilt block-scaled GEMM that the grader installs, so you
find out here rather than at grade time. See instruction.md.
"""
import argparse
import math
import sys
import time

import torch

sys.path.insert(0, "/app")
from reference import E2M1_VALUES, nvfp4_gemm as ref_gemm   # fp32 reference (also the correctness spec)

# (M, N, K), all multiples of 128 — a DIFFERENT sample of the same regime as the grader (big square + non-square)
FULL = [(8192, 8192, 6144), (6144, 6144, 6144), (12288, 4096, 8192), (4096, 12288, 4096)]
QUICK = [(4096, 8192, 4096), (8192, 4096, 4096), (4096, 4096, 8192), (6144, 6144, 4096)]


def _ban_prebuilt_gemm():
    """Close `aten::_scaled_mm{,_v2}` — cuBLAS's prebuilt block-scaled GEMM — exactly as the grader does.

    It is not yours to call on this task: it IS the ceiling you are measured against, so a submission that
    reaches it scores 0 no matter how fast. The grader closes it at the dispatcher, which catches every
    route in (`torch.nn.functional.scaled_mm`, `torch._scaled_mm_v2`, `torch.ops.aten._scaled_mm_v2`).
    Installed here too so you hit the wall while iterating instead of after four hours.
    """
    lib = torch.library.Library("aten", "IMPL")

    def _banned(*a, **k):
        raise RuntimeError(
            "torch's prebuilt scaled-mm reaches cuBLAS's block-scaled GEMM, which is this task's ceiling: "
            "calling it scores 0. Write the kernel — Triton (tl.dot_scaled speaks NVFP4 on Blackwell), "
            "CUTLASS / the CuTe DSL and nvcc are all available. See instruction.md.")

    for op in ("_scaled_mm", "_scaled_mm_v2"):
        try:
            lib.impl(op, _banned, "CUDA")
        except Exception:                                            # noqa: BLE001
            pass
    return lib


_BAN = _ban_prebuilt_gemm()                       # module-level: GC of the Library would deregister it
from nvfp4_gemm import nvfp4_gemm as agent_gemm    # noqa: E402  (imported after the ban is armed)


def _quantize(x):
    """bf16 (R, K) -> packed e2m1 (R, K//2), row-major 1x16 e4m3 scales (R, K//16), global scale."""
    R, K = x.shape
    lut = torch.tensor(E2M1_VALUES, device=x.device, dtype=torch.float32)
    gsf = (448.0 * 6.0) / x.float().abs().amax()
    blk = x.float().view(R, K // 16, 16) * gsf
    sf = (blk.abs().amax(-1) / 6.0).to(torch.float8_e4m3fn)
    q = blk / sf.float().clamp(min=1e-30).unsqueeze(-1)
    idx = (q.unsqueeze(-1) - lut.view(1, 1, 1, 16)).abs().argmin(-1).to(torch.uint8).view(R, K)
    return (idx[:, 0::2] | (idx[:, 1::2] << 4)).contiguous(), sf, gsf


def _relerr(a, b):
    return float((a.float() - b.float()).norm() / (b.float().norm() + 1e-12))


def _bench(fn, target_s=0.05, batches=6, warm_s=0.2):
    """Back-to-back launches, one sync per batch, minimum over batches — the same method the grader uses.

    Not a sync after every launch: that times one cold, isolated launch (launch latency in, warm L2 out)
    and on these kernels misreads the time by up to 89%. Steady-state throughput is what the kernel does
    inside a model, and it is what the bf16 baseline is quoted at too.
    """
    t0 = time.perf_counter()
    n = 0
    while n < 5 or (time.perf_counter() - t0) < warm_s:     # call fn once before this to trigger any JIT
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


def _mk(M, N, K, seed=777):
    g = torch.Generator(device="cuda").manual_seed(seed)
    a = torch.randn(M, K, device="cuda", dtype=torch.bfloat16, generator=g)
    b = torch.randn(N, K, device="cuda", dtype=torch.bfloat16, generator=g)
    Aq, As, ga = _quantize(a)
    Bq, Bs, gb = _quantize(b)
    alpha = torch.tensor(1.0 / float(ga * gb), device="cuda", dtype=torch.float32)
    return a, b, (Aq, As, Bq, Bs, alpha)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="smaller sizes for fast iteration")
    args = ap.parse_args()
    shapes = QUICK if args.quick else FULL
    print(f"geomean over {len(shapes)} shapes ({'quick' if args.quick else 'full'}); the grader uses DIFFERENT "
          f"sizes in the same regime — optimize for generality")
    sps, allok = [], True
    for (M, N, K) in shapes:
        a, b, inputs = _mk(M, N, K)
        ref = ref_gemm(*inputs)
        out = agent_gemm(*inputs)
        re = _relerr(out, ref) if (out is not None and tuple(out.shape) == (M, N)) else 1.0
        ok = re <= 1e-2
        allok = allok and ok
        bf16_t = _bench(lambda: a @ b.T)
        agt_t = _bench(lambda: agent_gemm(*inputs))
        sp = bf16_t / agt_t if agt_t > 0 else 0.0
        sps.append(sp)
        print(f"  M{M} N{N} K{K}: {sp:6.2f}x over bf16   relerr {re:.1e}  {'ok' if ok else 'FAIL'}")
        del a, b, inputs, ref, out
        torch.cuda.empty_cache()
    geo = math.exp(sum(math.log(max(s, 1e-9)) for s in sps) / len(sps)) if sps else 0.0
    print(f"  => GEOMEAN {geo:.2f}x over bf16 cuBLAS   ({'all correct' if allok else 'SOME WRONG — fix first'})")
    print("     graded continuously on this geomean vs a state-of-the-art FP4 kernel — higher is better.")


if __name__ == "__main__":
    main()
