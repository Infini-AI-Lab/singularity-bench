"""Check the native ARM64 build, dependencies, BF16 attention and NCCL on the four allocated GB300 GPUs."""
import argparse
import importlib
import importlib.metadata
import json
import os
import platform
import socket
from datetime import timedelta

GPUS = 4


def gpu_worker(rank, port, count):
    os.environ.update(NVTE_FLASH_ATTN='0', NVTE_FUSED_ATTN='1', NVTE_UNFUSED_ATTN='0')
    import torch
    import torch.distributed as dist
    import transformer_engine.pytorch as te
    torch.cuda.set_device(rank)
    assert torch.cuda.get_device_capability(rank) == (10, 3), 'Expected a GB300 GPU (compute capability 10.3)'
    x = torch.randn(16, 256, device=f'cuda:{rank}', dtype=torch.bfloat16, requires_grad=True)
    layer = te.Linear(256, 256, params_dtype=torch.bfloat16, device=f'cuda:{rank}')
    layer(x).float().square().mean().backward()
    assert torch.isfinite(x.grad).all()
    attention = te.DotProductAttention(num_attention_heads=32, kv_channels=128,
        num_gqa_groups=8, attention_dropout=0.0, qkv_format='bshd', attn_mask_type='causal')
    q = torch.randn(1, 16, 32, 128, device=f'cuda:{rank}', dtype=torch.bfloat16, requires_grad=True)
    k = torch.randn(1, 16, 8, 128, device=f'cuda:{rank}', dtype=torch.bfloat16, requires_grad=True)
    v = torch.randn_like(k, requires_grad=True)
    attention(q, k, v).float().square().mean().backward()
    assert all(torch.isfinite(t.grad).all() for t in (q, k, v))
    dist.init_process_group('nccl', init_method=f'tcp://127.0.0.1:{port}',
                            rank=rank, world_size=count, timeout=timedelta(seconds=120))
    try:
        total = torch.tensor(float(rank), device=f'cuda:{rank}')
        dist.all_reduce(total)
        assert total.item() == count * (count - 1) / 2
        torch.cuda.synchronize()
    finally:
        dist.destroy_process_group()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--image-only', action='store_true', help='check the build and imports without GPUs')
    args = parser.parse_args()
    assert platform.machine() in ('aarch64', 'arm64'), 'Build and run the native ARM64 image'
    for module in ('torch', 'transformer_engine.pytorch', 'apex', 'ray', 'sglang',
                   'sglang_router', 'torch_memory_saver', 'megatron.core', 'pyarrow',
                   'slime.backends.megatron_utils.actor', 'megatron.training',
                   'datasets', 'accelerate', 'math_verify', 'tiktoken', 'sentencepiece',
                   'megatron.core.datasets.helpers_cpp'):
        importlib.import_module(module)
    import torch
    assert int(torch.version.cuda.split('.')[0]) >= 13, 'CUDA 13 is required by this recipe'
    print(json.dumps({p: importlib.metadata.version(p) for p in
                      ('torch', 'transformer-engine', 'ray', 'sglang', 'transformers')}, indent=2))
    if not args.image_only:
        count = GPUS
        assert torch.cuda.device_count() == count, f'Expose exactly {count} allocated GPUs'
        print(json.dumps([{'gpu': i, 'name': torch.cuda.get_device_name(i),
                           'memory_bytes': torch.cuda.get_device_properties(i).total_memory}
                          for i in range(count)]))
        with socket.socket() as sock:
            sock.bind(('127.0.0.1', 0))
            port = sock.getsockname()[1]
        torch.multiprocessing.spawn(gpu_worker, args=(port, count), nprocs=count, join=True)
        print(f'PASS: BF16 forward/backward on all {count} GB300 GPUs and NCCL all-reduce')


if __name__ == '__main__':
    main()
