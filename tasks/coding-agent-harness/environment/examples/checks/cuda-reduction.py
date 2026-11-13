import sys

import torch

from common import gpu_call, load, milliseconds, require_cuda

require_cuda()
operation = load(sys.argv[1], 'reduction.py').row_sum
torch.manual_seed(42)
for rows, cols in ((7, 256), (3, 1), (5, 255), (9, 257), (17, 1025), (4, 4099)):
    x = torch.randn(rows, cols, device='cuda')
    before = x.clone()
    out = operation(x)
    assert out.device == x.device and out.dtype == x.dtype and out.shape == (rows,)
    torch.testing.assert_close(out, x.sum(dim=1), rtol=2e-5, atol=2e-4)
    torch.testing.assert_close(x, before, rtol=0, atol=0)
    print(f'PASS {rows}x{cols}', flush=True)
gpu_call(lambda: operation(x))
print(f'Informational timing: {milliseconds(lambda: operation(x)):.4f} ms (no threshold)')
