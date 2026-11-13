"""MLA decode — YOUR fast implementation goes here.

A correct but unfused torch reference is given in /app/reference.py. Replace this stub with an `mla_decode`
(same signature/behavior) that fuses the attention into a flash-decode kernel and beats the reference by the
required margin — see instruction.md. The untouched stub raises, so the baseline scores 0.0.
"""


def mla_decode(q, kv, scale):
    """Fast MLA decode. Same contract as /app/reference.py:mla_decode, but >= the required speedup over it."""
    raise NotImplementedError("implement a fast fused MLA decode kernel (see /app/reference.py and instruction.md)")
