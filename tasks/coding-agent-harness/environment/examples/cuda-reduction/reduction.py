from pathlib import Path
import hashlib

from torch.utils.cpp_extension import load

_extension = None


def row_sum(x):
    global _extension
    if _extension is None:
        root = Path(__file__).resolve().parent
        name = 'example_row_reduction_' + hashlib.sha256(str(root).encode()).hexdigest()[:12]
        _extension = load(name=name,
                          sources=[str(root / 'binding.cpp'), str(root / 'reduction.cu')],
                          extra_cuda_cflags=['-O2'], verbose=True)
    return _extension.row_sum(x)
