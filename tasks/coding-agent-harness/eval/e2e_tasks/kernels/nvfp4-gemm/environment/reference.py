"""Reference: NVFP4 block-scaled GEMM, computed by unpacking to fp32 and doing a plain matmul.

This is CORRECT but does NOT touch the FP4 tensor cores (it unpacks, upcasts and runs an ordinary matmul),
so it is far SLOWER than a bfloat16 GEMM. It is GIVEN to you as the specification of the exact computation
and of the NVFP4 data layout. Your job (in /app/nvfp4_gemm.py) is to implement `nvfp4_gemm` with the SAME
signature and the SAME numerical result, but FAST -- see instruction.md.

NVFP4 layout (NVIDIA's Blackwell 4-bit format, as used by TensorRT-LLM / vLLM / SGLang):

  A_fp4     (M, K//2)   uint8            two e2m1 values per byte: element k lives in the LOW nibble when k
                                         is even and the HIGH nibble when k is odd
  A_scales  (M, K//16)  float8_e4m3fn    one scale per run of 16 consecutive k
  B_fp4     (N, K//2)   uint8            same packing
  B_scales  (N, K//16)  float8_e4m3fn    same blocking -- note B is scaled per 1x16 row-block, like A
  alpha     ()          float32          one global scale for the whole product

  A[m, k] = e2m1(A_fp4[m, k//2] nibble k%2) * float(A_scales[m, k//16])
  B[n, k] = e2m1(B_fp4[n, k//2] nibble k%2) * float(B_scales[n, k//16])
  C       = alpha * (A @ B.T)            (M, N) bfloat16

e2m1 is the 4-bit float: 1 sign bit, 2 exponent bits, 1 mantissa bit, and therefore exactly 16 values --
the table below IS the format. M, N and K are multiples of 128, so nothing is padded.

The scale layout above is the plain, row-major one. The tensor cores do not read it in that order; getting
the scales into the layout your MMA wants is part of the kernel.
"""
import torch

# every value an e2m1 nibble can take, indexed by the nibble itself (bit 3 = sign, bits 2-1 = exponent,
# bit 0 = mantissa). This table is the complete definition of the element format.
E2M1_VALUES = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
               -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0)


def unpack_nvfp4(packed, scales):
    """(R, K//2) uint8 + (R, K//16) float8_e4m3fn -> (R, K) float32. The dequantization spec."""
    R, half = packed.shape
    K = half * 2
    lut = torch.tensor(E2M1_VALUES, device=packed.device, dtype=torch.float32)
    nibbles = torch.stack([packed & 0xF, packed >> 4], dim=-1).view(R, K)   # low nibble first = even k
    values = lut[nibbles.long()]
    return (values.view(R, K // 16, 16) * scales.float().view(R, K // 16, 1)).view(R, K)


def nvfp4_gemm(A_fp4, A_scales, B_fp4, B_scales, alpha):
    A = unpack_nvfp4(A_fp4, A_scales)
    B = unpack_nvfp4(B_fp4, B_scales)
    return (float(alpha) * (A @ B.T)).to(torch.bfloat16)
