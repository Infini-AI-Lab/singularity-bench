"""Resolve pinned checkpoint defaults with the task's installed serving runtime.

Run inside the pinned fleet image, mounting an existing HF config cache read-only.
No weights or network access are needed. Only ERNIE's hash-reviewed, pinned
configuration class executes; model and tokenizer code never execute. The output
is an input to build_architecture.py, not a runtime dependency of world generation.
"""
import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'_kit'))
import runtime_compat


def main():
    from transformers import AutoConfig
    from vllm.transformers_utils.config import _CONFIG_REGISTRY
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--catalogue', type=Path, required=True)
    p.add_argument('--cache', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    versions = {k: importlib.metadata.version(k) for k in ('vllm', 'transformers')}
    if versions != {'vllm': '0.28.0', 'transformers': '5.15.1'}:
        raise ValueError(f'use the pinned fleet image, got {versions}')
    records = {}
    for m in json.loads(a.catalogue.read_text())['models']:
        path = a.cache/('models--'+m['checkpoint'].replace('/', '--'))/'snapshots'/m['checkpoint_revision']
        raw = (path/'config.json').read_bytes()
        kind = json.loads(raw)['model_type']
        # Match the serving runtime, including the single reviewed legacy config.
        if kind in _CONFIG_REGISTRY:
            config = _CONFIG_REGISTRY[kind].from_pretrained(path, local_files_only=True)
            resolved, resolver = config.to_dict(), 'vllm-builtin'
        else:
            custom = runtime_compat.needs_config_code(m['checkpoint'],path)
            config = AutoConfig.from_pretrained(path, local_files_only=True, trust_remote_code=custom)
            resolved, resolver = config.to_dict(), ('reviewed-pinned-config' if custom else 'transformers-builtin')
        records[m['id']] = {'config_sha256': hashlib.sha256(raw).hexdigest(),
                            'resolver': resolver, 'config': resolved}
    # The legacy ERNIE class overrides to_dict without transformers' dtype
    # normalization. Normalize only torch dtypes; unexpected objects still fail.
    def encode(value):
        import torch
        if isinstance(value, torch.dtype):
            return str(value).removeprefix('torch.')
        raise TypeError(f'unsupported configuration value: {type(value).__name__}')
    a.out.write_text(json.dumps({'schema_version': 1, 'versions': versions,
        'resolver_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'models': records}, indent=1, sort_keys=True, default=encode)+'\n')
    print(f'resolved {len(records)} pinned configs')


if __name__ == '__main__':
    main()
