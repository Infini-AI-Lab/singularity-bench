"""Reference: FlashAttention BACKWARD, written as an explicit (unfused) torch backward.

This is CORRECT and shows the exact math, but it materializes the full [B, H, Sq, Sk] score / probability
matrices, so it is far slower than a fused flash kernel. It is GIVEN to you as the specification of the exact
computation + the tensor layout. Your job (in /app/attn_backward.py) is to implement `attn_backward` with the
SAME signature and SAME numerical result, but FAST — see instruction.md.

Setting: the standard scaled-dot-product attention backward. Given the forward
    scores = (Q Kᵀ) * scale         (+ causal mask)
    P      = softmax(scores)         = exp(scores - lse)      # lse = logsumexp(scores, dim=key)
    O      = P V
and the upstream gradient dO, compute dQ, dK, dV via the flash recompute (no saved P):
    dV = Pᵀ dO
    dP = dO Vᵀ
    D  = rowsum(dO ∘ O)              # per query row
    dS = P ∘ (dP - D)               # gradient wrt scores (pre-scale)
    dQ = (dS K) * scale ,  dK = (dSᵀ Q) * scale

Tensor layout (FlashAttention-native; the grader uses H=16 heads, head_dim D=128; B and S vary; CUDA, bf16):
  q, k, v : (B, S, H, D)   bfloat16    (batch, seqlen, heads, head_dim)
  o       : (B, S, H, D)   bfloat16    forward output   O
  lse     : (B, H, S)      float32     forward log-sum-exp (natural log)
  do      : (B, S, H, D)   bfloat16    upstream grad dO
  scale   : float          softmax scale (grader uses 1/sqrt(D))
  causal  : bool
Returns:
  dq, dk, dv : (B, S, H, D)  bfloat16
"""
import torch


def attn_backward(q, k, v, o, lse, do, scale, causal):
    qh, kh, vh, oh, doh = (t.transpose(1, 2).float() for t in (q, k, v, o, do))   # (B,H,S,D)
    S = (qh @ kh.transpose(-2, -1)) * scale                                        # (B,H,Sq,Sk)
    if causal:
        Sq, Sk = S.shape[-2], S.shape[-1]
        m = torch.triu(torch.ones(Sq, Sk, device=q.device, dtype=torch.bool), 1)
        S = S.masked_fill(m, float("-inf"))
    P = torch.exp(S - lse.unsqueeze(-1))                                           # (B,H,Sq,Sk)
    dV = P.transpose(-2, -1) @ doh                                                 # (B,H,Sk,D)
    dP = doh @ vh.transpose(-2, -1)                                                # (B,H,Sq,Sk)
    Delta = (doh * oh).sum(-1, keepdim=True)                                       # (B,H,Sq,1)
    dS = P * (dP - Delta)                                                          # (B,H,Sq,Sk)
    dQ = (dS @ kh) * scale                                                         # (B,H,Sq,D)
    dK = (dS.transpose(-2, -1) @ qh) * scale                                       # (B,H,Sk,D)
    to = lambda x: x.transpose(1, 2).to(torch.bfloat16)                           # back to (B,S,H,D)
    return to(dQ), to(dK), to(dV)
