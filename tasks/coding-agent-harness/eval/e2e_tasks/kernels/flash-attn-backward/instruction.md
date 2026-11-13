# Write a fast FlashAttention backward (beat FlashAttention-2)

You are given a **correct but unfused** reference at **`/app/reference.py`** — `attn_backward(q, k, v, o, lse,
do, scale, causal)`, the scaled-dot-product attention backward, written with explicit matmuls (it materializes
the full `[B, H, Sq, Sk]` score/probability matrices, so it is slow). **Read it** — it defines the exact
computation and the tensor layout.

Write **`/app/attn_backward.py`** with the **same signature and same numerical result**, but **fast** — a fused
flash-attention backward kernel:

```python
def attn_backward(q, k, v, o, lse, do, scale, causal):
    ...
    return dq, dk, dv   # each (B, S, H, D) bfloat16
```

## The computation (FlashAttention backward)
Given the forward `scores = (Q Kᵀ)*scale (+causal)`, `P = softmax(scores) = exp(scores − lse)`, `O = P V`, and
upstream `dO`, recompute `P` from `lse` (no saved `P`) and produce:
`dV = Pᵀ dO`, `dP = dO Vᵀ`, `Δ = rowsum(dO∘O)`, `dS = P∘(dP−Δ)`, `dQ = (dS K)*scale`, `dK = (dSᵀ Q)*scale`.

## Inputs (CUDA tensors, bf16; FlashAttention layout, H=16 heads, head_dim D=128; S varies)
| name | shape / dtype | meaning |
|------|---------------|---------|
| `q`, `k`, `v` | `(B, S, H, D)` `bf16` | forward inputs |
| `o` | `(B, S, H, D)` `bf16` | forward output `O` |
| `lse` | `(B, H, S)` `float32` | forward log-sum-exp (natural log) |
| `do` | `(B, S, H, D)` `bf16` | upstream gradient `dO` |
| `scale` | `float` | softmax scale (`1/sqrt(D)`) |
| `causal` | `bool` | causal masking (bottom-right aligned for `Sq==Sk`) |

## How success is decided (runs on GPU)
The grader scores you on:

1. **Correctness (required gate)** — `dQ/dK/dV` must match an independent **fp32 autograd** ground truth within
   relative error **2e-2** at every graded shape. A wrong kernel scores **0**, no matter how fast.
2. **Speed (your score)** — you are graded on the **geometric-mean speedup over a FlashAttention-2 backward
   (PyTorch's FLASH SDPA backend) across a RANGE of shapes** — varied batch `1–4`, seqlen `~2k–16k`, and both
   **causal and non-causal** (H=16, D=128 fixed), including a big-S case and a non-causal case — **not one fixed
   size, and the exact grader shapes are not disclosed.** Write a **general** kernel: hyper-tuning a single
   `(B, S, causal)` will not score. The score climbs from 0 (no faster than FA-2) toward 1.0 as your geomean
   approaches a **state-of-the-art kernel** (the FA-4 CuTe-DSL fused backward), re-benchmarked live on this GPU each
   grade. The timed output is also correctness-checked. Run **`python /app/measure.py`** to see your per-shape +
   geomean speedup (it samples different sizes in the same regime).

**This requires a real fused kernel with architecture-specific optimizations.** FA-2's backward is already a strong,
fused baseline; beating it by 1.5× needs FA-3/FA-4-class engineering — **warp specialization**, **`wgmma`**, **TMA**,
and a well-pipelined `dS = P∘(dP−Δ)` recompute. `torch.compile`/eager reach ~0.1×; a straightforward Triton
backward does not beat FA-2.

**Banned (checked): `scaled_dot_product_attention`, `flash_attn`, cuDNN/xformers attention, and any SDPA
backend.** You must *write* the kernel — borrowing a prebuilt attention backward scores 0.

## What's available
- `torch` (2.11, CUDA 13), `triton`, the **CuTe DSL** (`cutlass` / `quack` are installed — general CuTe
  primitives, no attention kernel), `nvcc` (via `torch.utils.cpp_extension`); a CUDA GPU. The graded
  machine may be sm_90 (Hopper) or sm_10x (Blackwell) — read `torch.cuda.get_device_capability()`
  at runtime; the fast path differs between them.
- **The grader's own oracle is off limits.** `flash_attn` and `flashinfer` are the state-of-the-art kernels your score is
  measured *against*, so a submission that calls one has not done the task. They are not importable
  while your module is loaded and run: the attempt raises `ModuleNotFoundError`, is recorded, and
  scores **0** however fast the result is. Everything else the image provides is yours to use.
- **No internet access.** `pip install flash-attn` / cloning FlashAttention/CUTLASS are blocked — the FA-4
  backward implementation is sealed; write your own (the *computation* is in `reference.py`).
