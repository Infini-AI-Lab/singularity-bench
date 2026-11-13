import platform
import shutil
import subprocess
import sys

import torch
import triton

from common import gpu_call, require_cuda

require_cuda()
print(f'Python {platform.python_version()} ({platform.machine()}); Triton {triton.__version__}')
assert sys.version_info[:2] == (3, 11), 'Harness portability requires Python 3.11.'
for tool in ('nvcc', 'c++', 'ninja'):
    assert shutil.which(tool), f'Missing build tool: {tool}'
    subprocess.run([tool, '--version'], check=True)
x = torch.ones(16, device='cuda')
torch.testing.assert_close(gpu_call(lambda: x + x), torch.full_like(x, 2))
print('GPU/runtime diagnostic passed. Test through your harness to check its tool execution too.')
