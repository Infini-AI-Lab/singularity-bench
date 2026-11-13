"""Reference: Multi-head Latent Attention (MLA) DECODE, in the absorbed latent space (DeepSeek-V2/V3 style).

This is a CORRECT, vectorized-torch implementation (two batched cuBLAS matmuls + a softmax). It is the realistic
baseline a competent engineer writes — it is NOT slow-on-purpose. But it materializes the full [B, H, L] score
matrix and reads the KV latent twice, so it leaves a lot on the table versus a fused flash-decode kernel. It is
GIVEN to you as the specification of the exact computation + the tensor layout. Your job (in /app/mla_decode.py)
is to implement `mla_decode` with the SAME signature and SAME numerical result, but FAST — see instruction.md.

Setting: autoregressive DECODE (one new query token per sequence). The KV cache is stored compressed: instead of
per-head K and V, each position keeps a single 576-dim latent shared across all query heads. By weight absorption
the attention runs entirely in this latent space, so at decode MLA is effectively MQA with a 576-dim "head":

  head_dim 576 = 512 (low-rank latent `c_kv`, DeepSeek kv_lora_rank)  ++  64 (decoupled-RoPE key, qk_rope_head_dim)

The latent's first 512 dims serve as BOTH the key-content AND the value; the last 64 dims are the RoPE key that
only contributes to the score. Everything is already projected/absorbed into this space — you do NOT do any RoPE
or projection here; you compute the attention.

Tensor layout (the grader always uses H=128 query heads, head_dim=576, value dim=512; B and L vary, L a multiple
of 64; CUDA, bfloat16):
  q  : (B, H, 576)   the query for the single decode token, per head  ([content(512) | rope(64)])
  kv : (B, L, 576)   the compressed latent KV cache, contiguous       ([c_kv(512)    | k_rope(64)])
  scale : float      softmax scale (grader uses 1/sqrt(qk_nope_head_dim + qk_rope_head_dim) = 1/sqrt(192))
Returns:
  out : (B, H, 512)  bfloat16   (attention output in the latent/value space)

Computation:
  scores[b,h,l] = (q[b,h] . kv[b,l]) * scale          # full 576-dim dot -> (B, H, L)
  attn          = softmax(scores, dim=l)              # over the L cached positions
  out[b,h]      = sum_l attn[b,h,l] * kv[b,l, :512]   # value = the first 512 dims of the latent -> (B, H, 512)
"""
import torch

DV = 512   # value / latent-content dim (the output width); the score dot is over the full 576


def mla_decode(q, kv, scale):
    scores = torch.einsum("bhd,bld->bhl", q, kv)               # (B, H, L), full 576-dim dot
    attn = (scores.float() * scale).softmax(dim=-1).to(q.dtype)  # softmax in fp32, back to bf16
    return torch.einsum("bhl,bld->bhd", attn, kv[..., :DV])     # weighted sum of the latent-content -> (B,H,512)
