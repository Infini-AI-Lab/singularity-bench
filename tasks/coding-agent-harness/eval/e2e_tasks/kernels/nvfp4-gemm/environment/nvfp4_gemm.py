"""NVFP4 block-scaled GEMM -- YOUR fast implementation goes here.

A correct but slow reference is given in /app/reference.py. Replace this stub with an `nvfp4_gemm` (same
signature and same numerical result) that drives the FP4 tensor cores and beats a bfloat16 cuBLAS matmul
of the same size -- see instruction.md. The untouched stub raises, so the baseline scores 0.0.
"""


def nvfp4_gemm(A_fp4, A_scales, B_fp4, B_scales, alpha):
    """Fast NVFP4 GEMM. Same contract as /app/reference.py:nvfp4_gemm, but faster than a bf16 GEMM."""
    raise NotImplementedError("implement a fast NVFP4 block-scaled GEMM (see /app/reference.py and instruction.md)")
