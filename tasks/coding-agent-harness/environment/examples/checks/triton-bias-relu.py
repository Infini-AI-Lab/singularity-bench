import sys

import torch

from common import gpu_call, load, milliseconds, require_cuda

require_cuda()
operation = load(sys.argv[1], 'kernel.py').bias_relu
torch.manual_seed(41)
for dtype in (torch.float16, torch.float32):
    for rows, cols in ((1, 1), (3, 17), (7, 255), (5, 256), (9, 257), (33, 1025), (128, 4096)):
        x = torch.randn(rows, cols, device='cuda', dtype=dtype)
        bias = torch.randn(cols, device='cuda', dtype=dtype)
        before_x, before_bias = x.clone(), bias.clone()
        expected = torch.relu(x + bias)
        out = operation(x, bias)
        assert out.device == x.device and out.dtype == dtype and out.shape == x.shape
        torch.testing.assert_close(out, expected)
        torch.testing.assert_close(x, before_x, rtol=0, atol=0)
        torch.testing.assert_close(bias, before_bias, rtol=0, atol=0)
        print(f'PASS {dtype} {rows}x{cols}', flush=True)
gpu_call(lambda: operation(x, bias))
print(f'Informational timing: {milliseconds(lambda: operation(x, bias)):.4f} ms (no threshold)')
