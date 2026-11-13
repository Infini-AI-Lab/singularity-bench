"""Fixed downstream cell contracts. No task, module path or entry point comes from submitted code.

Kernel cells share one raw protocol: the task's own sealed verifier (``tests/verify_env.py``)
runs in the trusted root process with every reference, oracle, comparison and timing
there, while the submitted kernel runs in a separate worker process. PR cells use
public/private splits: a fixed public runner invokes the submitted vLLM API as the worker,
and private expectations are compared by the data-only scorer.

The grid ``compact5-v1`` covers exactly these five tasks: flash-attn-backward, nvfp4-gemm,
mla-decode, pr-vllm-35022 and pr-vllm-33698. Any other name is refused.
"""
KERNELS = {
    'flash-attn-backward': {'module': '/app/attn_backward.py', 'entry': 'attn_backward'},
    'nvfp4-gemm': {'module': '/app/nvfp4_gemm.py', 'entry': 'nvfp4_gemm'},
    'mla-decode': {'module': '/app/mla_decode.py', 'entry': 'mla_decode'},
}
PRS = {
    'pr-vllm-33698': {'adapter': 'pr33698', 'gpu': False},
    'pr-vllm-35022': {'adapter': 'pr35022', 'gpu': True},
}
GROUPS = {name: 'kernels' for name in KERNELS} | {name: 'pr-reproduction' for name in PRS}
WORKDIRS = {name: '/app' for name in KERNELS} | {name: '/app/vllm' for name in PRS}
TASKS = frozenset(GROUPS)


def require_task(name):
    if name not in TASKS:
        raise ValueError(f'No protected raw protocol for downstream task {name!r}; it is refused, not graded '
                         'through an unprotected path')
    return name


def task_path(name):
    """Relative to tasks/coding-agent-harness/eval/e2e_tasks."""
    return f'{GROUPS[require_task(name)]}/{name}'


def protocol(name):
    return 'breakwater-harness-cell/' + require_task(name) + ':raw-v1'
