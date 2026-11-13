# Write a fast MLA-decode kernel (beat the torch reference)

You are given a **correct but unfused** reference at **`/app/reference.py`** — `mla_decode(q, kv, scale)`, the
Multi-head Latent Attention (MLA) decode step (DeepSeek-V2/V3), computed as two batched cuBLAS matmuls + a
softmax. It materializes the full `[B, H, L]` score matrix and reads the KV latent twice. **Read it** — it
defines the exact computation and the tensor layout.

Write **`/app/mla_decode.py`** with the **same signature and same numerical result**, but **fast** — a fused
flash-decode kernel that never materializes the score matrix:

```python
def mla_decode(q, kv, scale):
    ...
    return out   # (B, H, 512) bfloat16
```

## The computation (absorbed MLA decode — one decode token per sequence)
At decode, MLA caches a single **576-dim latent per position, shared across all 128 query heads** (i.e. MQA with
a 576-dim "head"). The latent's first **512** dims are the low-rank KV (`kv_lora_rank`) and serve as **both the
key-content and the value**; the last **64** dims are the decoupled-RoPE key that only affects the score.

## Inputs (CUDA tensors, bfloat16; H=128, head_dim=576, value_dim=512; L a multiple of 64)
| name | shape / dtype | meaning |
|------|---------------|---------|
| `q`  | `(B, H, 576)` `bfloat16` | query for the decode token, per head — `[content(512) \| rope(64)]` |
| `kv` | `(B, L, 576)` `bfloat16`, contiguous | latent KV cache — `[c_kv(512) \| k_rope(64)]` |
| `scale` | `float` | softmax scale (grader uses `1/sqrt(192)`) |

```
scores[b,h,l] = (q[b,h] · kv[b,l]) * scale        # full 576-dim dot  -> (B, H, L)
attn          = softmax(scores, dim=l)            # over the L cached positions
out[b,h]      = sum_l attn[b,h,l] * kv[b,l,:512]  # value = first 512 dims -> (B, H, 512)
```

## How success is decided (runs on GPU)
The grader generates randomized inputs, computes an independent fp32 reference, and scores you on:

1. **Correctness (required gate)** — your `out` must match the reference within relative error **2e-2** across
   several `(B, L)` sizes. A wrong kernel scores **0**, no matter how fast.
2. **Speed (your score)** — you are graded on the **geometric-mean speedup over the eager reference across a
   RANGE of `(B, L)` sizes** (batch `B` ~8–128, cache length `L` ~2k–32k, `L` a multiple of 64, including a big
   large-`B`×large-`L` case) — **not one fixed size, and the exact grader sizes are not disclosed.** Write a
   **general** kernel: hyper-tuning a single shape (block size, split-KV count, batch) will not score. The score
   climbs from 0 (no faster than the reference) toward 1.0 as your geomean approaches a **optimized
   kernel** (a FlashMLA-class fused flash-decode, **re-benchmarked live** so the ceiling reflects this GPU). The
   timed output is also correctness-checked at every shape (no shape-conditional shortcuts). Run **`python
   /app/measure.py`** to see your per-shape + geomean speedup (it samples different sizes in the same regime).

**This requires a real fused kernel.** `torch.compile` on the reference only reaches ~2.2× (it fuses the
softmax) — real-kernel territory is well beyond that. There is **no `scaled_dot_product_attention` shortcut**: head_dim 576
falls off SDPA's fast paths (it OOMs / runs an order of magnitude slower). The hard part is MLA's **512-wide
per-head accumulator shared across 128 heads** — a naive Triton flash-decode is occupancy-limited and lands
*below* `torch.compile`. Expect to need a carefully-tuned flash-decode (Triton split-KV, or **CUDA / the CuTe
DSL**, i.e. a FlashMLA-style design: online softmax, no score materialization, KV streamed once).

## What's available
- `torch` (2.11), `triton`, **`nvidia-cutlass-dsl` (CuTe DSL)**, `nvcc` (via
  `torch.utils.cpp_extension`); a CUDA GPU. The graded machine may be sm_90 (Hopper) or sm_10x
  (Blackwell) — read `torch.cuda.get_device_capability()` at runtime; the fast path differs.
- **The grader's own oracle is off limits.** `flash_mla`, `flash_attn`, and `flashinfer` are the optimized kernels your score is
  measured *against*, so a submission that calls one has not done the task. They are not importable
  while your module is loaded and run: the attempt raises `ModuleNotFoundError`, is recorded, and
  scores **0** however fast the result is. Everything else the image provides is yours to use.
- **No internet access.** `pip install flash_mla` / cloning FlashMLA/FlashInfer/CUTLASS are blocked — write the
  kernel yourself (the *computation* is given in `reference.py`; the *fast fused kernel* is the task).
