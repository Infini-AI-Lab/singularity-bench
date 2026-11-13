"""FlashAttention backward — YOUR fast implementation goes here.

A correct but unfused torch reference is given in /app/reference.py. Replace this stub with an `attn_backward`
(same signature/behavior) that fuses the backward into a flash kernel and beats FlashAttention-2 by the required
margin — see instruction.md. The untouched stub raises, so the baseline scores 0.0.
"""


def attn_backward(q, k, v, o, lse, do, scale, causal):
    """Fast FA backward. Same contract as /app/reference.py:attn_backward, but >= the required speedup over FA-2."""
    raise NotImplementedError("implement a fast fused attention backward (see /app/reference.py and instruction.md)")
