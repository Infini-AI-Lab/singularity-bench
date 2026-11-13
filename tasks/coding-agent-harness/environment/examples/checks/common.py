import importlib.util
from pathlib import Path
import sys

import torch


def load(workspace, filename):
    workspace = Path(workspace).resolve()
    sys.path.insert(0, str(workspace))
    spec = importlib.util.spec_from_file_location('example_solution', workspace / filename)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def require_cuda():
    assert torch.cuda.is_available(), 'No usable GPU; inspect allocation, device permissions, and CUDA libraries.'
    print(f'GPU: {torch.cuda.get_device_name()}; torch={torch.__version__}', flush=True)


def gpu_call(fn):
    """Observe the tested operation itself, excluding setup/reference GPU operations."""
    torch.cuda.synchronize()
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                           torch.profiler.ProfilerActivity.CUDA]) as profile:
        result = fn()
        torch.cuda.synchronize()
    kernels = [event.name for event in profile.events()
               if event.device_type == torch.autograd.DeviceType.CUDA
               and not event.name.startswith(('Memcpy', 'Memset', '[memory]'))]
    assert kernels, 'No GPU kernel execution observed in the tested operation.'
    print(f'Observed {len(kernels)} GPU kernel(s): {kernels[0]}', flush=True)
    return result


def milliseconds(fn):
    for _ in range(5):
        fn()
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(20):
        fn()
    end.record()
    end.synchronize()
    return start.elapsed_time(end) / 20
