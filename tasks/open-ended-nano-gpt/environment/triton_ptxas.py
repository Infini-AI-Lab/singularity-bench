#!/usr/bin/env python3
"""Make Triton compile for the GB300 GPUs, at image-build time, and prove it.

    python triton_ptxas.py [PTXAS]      (default: the image's /usr/local/cuda/bin/ptxas)

torch 2.9.1 uses Triton 3.5.1, whose compiler knows GB300 (sm_103a) but which assembles with its own bundled ptxas
from CUDA 12.8, whatever CUDA PyTorch was built for; CUDA 12.9 is the first toolkit that targets sm_103. This links
Triton's bundled assembler to the image's CUDA 12.9 ptxas (a link, not an environment variable, so it holds in
every process, including evaluation workers that start from a clean environment). It then compiles a reduction and
a matrix multiply for GB300 ahead of time, which needs no GPU, so a toolchain that cannot target GB300 fails the
build.
"""

import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

# Triton caches compiled kernels under its home; keep this build's out of the image.
SCRATCH = tempfile.mkdtemp(prefix='triton-build-check-')
os.environ['TRITON_HOME'] = os.environ['TRITON_CACHE_DIR'] = SCRATCH

import triton  # noqa: E402
import triton.language as tl  # noqa: E402
from triton.backends.compiler import GPUTarget  # noqa: E402
from triton.compiler import ASTSource  # noqa: E402

GB300 = GPUTarget('cuda', 103, 32)


@triton.jit
def row_sum(x, out, N: tl.constexpr):
    row = tl.program_id(0)
    tl.store(out + row, tl.sum(tl.load(x + row * N + tl.arange(0, N)), axis=0))


@triton.jit
def tile_dot(a, b, c, M: tl.constexpr):
    r = tl.arange(0, M)
    x = tl.load(a + r[:, None] * M + r[None, :])
    y = tl.load(b + r[:, None] * M + r[None, :])
    tl.store(c + r[:, None] * M + r[None, :], tl.dot(x, y).to(tl.float16))


KERNELS = ((row_sum, {'x': '*fp32', 'out': '*fp32', 'N': 'constexpr'}, {'N': 1024}),
           (tile_dot, {'a': '*fp16', 'b': '*fp16', 'c': '*fp16', 'M': 'constexpr'}, {'M': 64}))


def link_assembler(ptxas):
    version = subprocess.run([str(ptxas), '--version'], capture_output=True, text=True, check=True).stdout
    if 'release 12.9' not in version:
        raise RuntimeError(f'expected the CUDA 12.9 ptxas, found: {version.strip()}')
    folder = Path(triton.__file__).parent / 'backends' / 'nvidia' / 'bin'
    bundled = [folder / name for name in ('ptxas', 'ptxas-blackwell') if (folder / name).is_file()]
    if not bundled:
        raise RuntimeError(f'Triton has no bundled ptxas in {folder}')
    for path in bundled:
        path.unlink()
        path.symlink_to(ptxas)
    return ptxas, bundled


def main():
    try:
        ptxas, bundled = link_assembler(Path(sys.argv[1] if len(sys.argv) > 1 else '/usr/local/cuda/bin/ptxas'))
        for kernel, signature, constexprs in KERNELS:
            compiled = triton.compile(ASTSource(fn=kernel, signature=signature, constexprs=constexprs), target=GB300)
            if '.target sm_103a' not in compiled.asm['ptx'] or not compiled.asm['cubin']:
                raise RuntimeError(f'{kernel.__name__} was not compiled for GB300')
    finally:
        shutil.rmtree(SCRATCH, ignore_errors=True)
    print(f'{", ".join(path.name for path in bundled)} -> {ptxas}: Triton {triton.__version__} compiles a reduction '
          'and a matrix multiply for sm_103a')


if __name__ == '__main__':
    main()
