"""Protected four-GPU computation fixture; no model, dataset or provider API."""
import ctypes
import json
import os
from pathlib import Path
import signal

if ctypes.CDLL('/opt/stdio_capture.so', use_errno=True).breakwater_capture_start():
    raise OSError(ctypes.get_errno(), 'capture startup')
os.kill(os.getpid(), signal.SIGSTOP)
os.setgroups([])
os.setresgid(1101, 1101, 1101)
os.setresuid(1101, 1101, 1101)
os.environ['HOME'] = '/tmp'
import torch

assert torch.cuda.device_count() == 4
names = [torch.cuda.get_device_name(index) for index in range(4)]
assert all('B300' in name for name in names), names
assert {'sm_100', 'sm_103'} & set(torch._C._cuda_getArchFlags().split())
for index in range(4):
    values = torch.ones((128, 128), device=f'cuda:{index}', dtype=torch.bfloat16)
    product = values @ values
    torch.cuda.synchronize(index)
    assert torch.all(product == 128).item()
Path('/tmp/worker-stage-marker').write_text('This file must not cross into another container.\n')
print(json.dumps({'status': 'GPU_COMPUTATION_OK', 'gpus': names, 'torch': torch.__version__,
                  'cuda': torch.version.cuda}), flush=True)
