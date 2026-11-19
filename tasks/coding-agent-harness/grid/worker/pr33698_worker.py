"""Invoke submitted PP-cache API using public config only; no expected answers."""
import contextlib
import json
from pathlib import Path
import sys


def run_cases(input_path, output_path):
    with contextlib.redirect_stdout(sys.stderr):
        import torch
        from vllm.config import ModelConfig, VllmConfig
        from vllm.v1.core.kv_cache_utils import get_kv_cache_configs
        from vllm.v1.kv_cache_interface import FullAttentionSpec
    cases = json.loads(Path(input_path).read_text())
    with Path(output_path).open('w') as output:
        for case in cases:
            row = {'example_id': case['example_id']}
            try:
                with contextlib.redirect_stdout(sys.stderr):
                    spec_args = {**case['spec'], 'dtype': getattr(torch, case['spec']['dtype'])}
                    spec = FullAttentionSpec(**spec_args)
                    configs = get_kv_cache_configs(VllmConfig(model_config=ModelConfig(max_model_len=case['max_model_len'])),
                        [{name: spec} for name in case['layer_names']], case['available_memory'])
                    returned = []
                    for config in configs:
                        groups = []
                        for group in config.kv_cache_groups:
                            info = {key: getattr(group.kv_cache_spec, key) for key in case['spec']}
                            info['dtype'] = str(info['dtype']).removeprefix('torch.')
                            groups.append({'layer_names': list(group.layer_names), 'spec': info,
                                           'spec_type': type(group.kv_cache_spec).__name__})
                        returned.append({'num_blocks': config.num_blocks,
                            'tensors': [{'size': tensor.size, 'shared_by': list(tensor.shared_by)}
                                        for tensor in config.kv_cache_tensors], 'groups': groups})
                row.update(status='returned', configs=returned)
            except BaseException as error:
                row.update(status='raised', error_type=type(error).__name__, error=str(error))
            output.write(json.dumps(row, allow_nan=False) + '\n')
            output.flush()


if __name__ == '__main__':
    run_cases(sys.argv[1], sys.argv[2])
