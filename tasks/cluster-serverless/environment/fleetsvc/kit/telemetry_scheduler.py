"""Common trusted instrumentation on ai-dynamo's instrumented vLLM scheduler; scheduling stays upstream.

Installed in workers for every policy. The only subclass addition is publishing
the engine's resolved cache geometry and scheduler limits to its local wrapper.
No requests or performance measurements are synthesized here.
"""
import json
import os
from pathlib import Path

from dynamo.vllm.instrumented_scheduler import InstrumentedScheduler
from dynamo.vllm.cache_info import select_main_attention_block_size
from vllm.v1.kv_cache_interface import get_kv_cache_spec_kind


class TelemetryScheduler(InstrumentedScheduler):
    def __init__(self, vllm_config, kv_cache_config, structured_output_manager,
                 block_size, hash_block_size=None, **kwargs):
        super().__init__(vllm_config, kv_cache_config, structured_output_manager,
                         block_size, hash_block_size, **kwargs)
        groups = [{'group_idx': i, 'kind': get_kv_cache_spec_kind(g.kv_cache_spec).value,
                   'block_size': g.kv_cache_spec.block_size,
                   'sliding_window': getattr(g.kv_cache_spec, 'sliding_window', None)}
                  for i, g in enumerate(kv_cache_config.kv_cache_groups)]
        event_block = select_main_attention_block_size(groups, block_size)
        # Same capacity convention as upstream WorkerInfo.max_kv_tokens. Group
        # metadata is also public so a policy need not assume uniform attention.
        blocks = int(kv_cache_config.num_blocks)
        sched = vllm_config.scheduler_config
        parallel = vllm_config.parallel_config
        value = {'schema_version': 1, 'backend': 'real', 'worker_id': os.environ['DYN_FPM_WORKER_ID'],
                 'block_size': event_block, 'total_kv_blocks': blocks, 'cache_groups': groups,
                 'max_kv_tokens': blocks * event_block, 'max_num_seqs': sched.max_num_seqs,
                 'max_num_batched_tokens': sched.max_num_batched_tokens,
                 'context_length': vllm_config.model_config.max_model_len,
                 'num_gpu': parallel.tensor_parallel_size * parallel.pipeline_parallel_size,
                 'prefix_caching': vllm_config.cache_config.enable_prefix_caching}
        path = Path(os.environ['CLUSTER_CAPABILITIES_FILE'])
        staged = path.with_suffix('.new')
        staged.write_text(json.dumps(value))
        staged.replace(path)
