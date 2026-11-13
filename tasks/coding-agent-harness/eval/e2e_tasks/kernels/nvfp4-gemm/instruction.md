# Write a fast NVFP4 block-scaled GEMM (beat bf16)

You are given a **correct but slow** reference at **`/app/reference.py`** — `nvfp4_gemm(A_fp4, A_scales,
B_fp4, B_scales, alpha)`, which computes an NVFP4 GEMM by unpacking the 4-bit elements to fp32 and doing a
normal matmul (so it does **not** touch the FP4 tensor cores and is several times *slower* than bf16).
**Read it** — it defines the exact computation and the exact data layout.

Write **`/app/nvfp4_gemm.py`** with the **same signature and same numerical result**, but **fast** — a kernel
that actually uses the GPU's FP4 tensor cores:

```python
def nvfp4_gemm(A_fp4, A_scales, B_fp4, B_scales, alpha):
    ...
    return C   # (M, N) bfloat16
```

## Inputs (CUDA tensors; M, N, K multiples of 128)
| name | shape / dtype | meaning |
|------|---------------|---------|
| `A_fp4` | `(M, K//2)` `uint8` | two **e2m1** values per byte: element `k` is the **low** nibble when `k` is even, the **high** nibble when `k` is odd |
| `A_scales` | `(M, K//16)` `float8_e4m3fn` | one scale per run of 16 consecutive `k`: `A[m,k] = e2m1(...) * float(A_scales[m, k//16])` |
| `B_fp4` | `(N, K//2)` `uint8` | same packing |
| `B_scales` | `(N, K//16)` `float8_e4m3fn` | same 1×16 blocking — `B` is scaled per row-block, like `A` |
| `alpha` | `()` `float32` | one global scale applied to the whole product |

`e2m1` is the 4-bit float — 1 sign bit, 2 exponent bits, 1 mantissa bit, so exactly 16 representable
values; `reference.py` lists all of them. Output `C = alpha * (A @ B.T)`, shape `(M, N)`, `bfloat16`.

**The scales you are given are row-major.** The FP4 MMA instructions do not read them in that order —
getting the block scales into the layout your matmul wants is part of writing the kernel, and it is not
free. Nothing stops you from folding that rearrangement into your own kernel.

## How success is decided (runs on GPU)
The grader generates randomized inputs, computes an independent fp32 dequant reference, and scores you on:

1. **Correctness (required gate)** — your `C` must match the reference within relative error **1e-2**
   (FP4-appropriate) at **every graded shape**. A wrong kernel scores **0**, no matter how fast.
2. **Speed (your score)** — you are graded on the **geometric-mean speedup over a bfloat16 cuBLAS matmul**
   (`a @ b.T` of the same size) across a **RANGE of `(M, N, K)` shapes** — a large square, a medium square,
   and **non-square cases** (`M != N != K`; both tall and wide), all multiples of 128 with dimensions
   roughly in the **4096–16384** range. **The exact grader sizes are not disclosed, and it is not one fixed
   size** — so write a **general** kernel: hyper-tuning a single shape (e.g. `8192³`) will not score. The
   score climbs from 0 (no faster than bf16 cuBLAS) toward 1.0 as your geomean approaches a
   **state-of-the-art block-scaled FP4 kernel**. Run **`python /app/measure.py`** to see your per-shape and
   geomean speedup (it samples different sizes in the same regime).

For calibration on a Blackwell part, measured over the graded shapes: **unpacking to bf16 and calling
cuBLAS is ~0.2× — that scores 0.** A first-cut Triton `tl.dot_scaled` kernel is ~1.1×; tuning its tiles
gets to ~1.5×. The state-of-the-art ceiling is ~2.8×. There is a lot of room between those, and partial
progress scores partially.

## What's available
- `torch` (2.11), **`triton` — its `tl.dot_scaled` speaks NVFP4 natively (`"e2m1"` elements with `e4m3`
  1×16 scales) and lowers to Blackwell's block-scaled MMA**, **`nvidia-cutlass-dsl` (CuTe DSL)**, `nvcc`
  via `torch.utils.cpp_extension`; a Blackwell GPU (`sm_100`+ — FP4 tensor cores do not exist before it).
  Read `torch.cuda.get_device_capability()` at runtime.
- **The grader's own oracle is off limits** — and on this task that is two different things, for one
  reason: each of them *is* the ceiling you are scored against, reachable in a single call.
  - `deep_gemm` and `flashinfer` — the grader's own oracle libraries. They are not importable while your
    module is loaded and run: the attempt raises `ModuleNotFoundError`, is recorded, and scores **0**
    however fast the result is.
  - **`torch._scaled_mm` / `torch._scaled_mm_v2` / `torch.nn.functional.scaled_mm`** — these reach cuBLAS's
    prebuilt block-scaled FP4 kernel, which is the same class of kernel the oracle is. They are disabled at
    the dispatcher while your module runs (every route into `aten::_scaled_mm{,_v2}` raises), the attempt
    is recorded, and it scores **0**. `/app/measure.py` installs the same ban, so you will hit it while
    iterating rather than at grade time.

  Everything else the image provides is yours, and using it is the task: ordinary `torch` ops, Triton,
  CUTLASS / the CuTe DSL, and hand-written CUDA all remain fully available. `a @ b.T` still works — it is
  the yardstick, not a solution (it computes a different number and fails correctness).
- **No internet access.** `pip install` and cloning CUTLASS/TensorRT-LLM are blocked — write the kernel
  yourself. The *computation* is given in `reference.py`; the *fast FP4 kernel* is the task.
