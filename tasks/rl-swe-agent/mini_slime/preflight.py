"""Four-rank BF16/SDPA/NCCL allocation smoke test; no policy model allocation."""
import argparse
import json
import os
from pathlib import Path


def host_memory():
    """Report host and cgroup budgets; a host total alone can hide container limits."""
    result = {}
    path = Path('/proc/meminfo')
    if path.exists():
        fields = {line.split(':')[0]: int(line.split()[1]) * 1024 for line in path.read_text().splitlines()}
        result.update(total_bytes=fields['MemTotal'], available_bytes=fields['MemAvailable'])
    for label, path in [('cgroup_limit_bytes', '/sys/fs/cgroup/memory.max'),
                        ('cgroup_current_bytes', '/sys/fs/cgroup/memory.current'),
                        ('cgroup_v1_limit_bytes', '/sys/fs/cgroup/memory/memory.limit_in_bytes')]:
        p = Path(path)
        if p.exists():
            raw = p.read_text().strip()
            result[label] = int(raw) if raw.isdigit() else raw
    return result


def main():
    from mini_slime.config import validate_gpu_names
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--profile', choices=['l40-debug', 'h200-debug'], default='l40-debug')
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    import torch
    import torch.distributed as dist
    import torch.nn.functional as F
    from datetime import timedelta
    if int(os.environ.get('WORLD_SIZE', '1')) != 4:
        raise RuntimeError('Launch with torchrun --standalone --nproc-per-node=4')
    names = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
    validate_gpu_names(a.profile, names)
    rank = int(os.environ['RANK'])
    torch.cuda.set_device(int(os.environ['LOCAL_RANK']))
    if not torch.cuda.is_bf16_supported():
        raise RuntimeError('BF16 unsupported by this device/runtime')
    dist.init_process_group('nccl', timeout=timedelta(minutes=5))
    try:
        total = torch.tensor(float(rank + 1), device='cuda')
        dist.all_reduce(total)
        if total.item() != 10:
            raise RuntimeError('NCCL all-reduce produced the wrong sum')
        torch.manual_seed(42 + rank)
        q = torch.randn(1, 4, 64, 64, device='cuda', dtype=torch.bfloat16, requires_grad=True)
        out = F.scaled_dot_product_attention(q, q, q, is_causal=True)
        out.float().square().mean().backward()
        if not torch.isfinite(out).all() or not torch.isfinite(q.grad).all() or not q.grad.abs().sum():
            raise RuntimeError('BF16 attention forward/backward failed')
        rows = [None] * 4
        dist.all_gather_object(rows, {'rank': rank, 'name': names[rank],
            'capability': torch.cuda.get_device_capability(),
            'device_bytes': torch.cuda.get_device_properties(torch.cuda.current_device()).total_memory,
            'nccl_sum': total.item(), 'gradient_norm': float(q.grad.float().norm())})
        if rank == 0:
            a.output.parent.mkdir(parents=True, exist_ok=True)
            with a.output.open('x') as f:
                json.dump({'profile': a.profile, 'torch': torch.__version__, 'cuda': torch.version.cuda,
                           'ranks': rows, 'host_memory': host_memory(),
                           'scope': 'Allocation, small BF16 attention and NCCL only; not GLM or Harbor'}, f, indent=2)
                f.write('\n')
        dist.barrier()
    finally:
        dist.destroy_process_group()


if __name__ == '__main__':
    main()
