"""Block accounting for the pinned vLLM 0.28 hybrid cache manager.

The formulas follow v1/kv_cache_interface.py and v1/core/kv_cache_utils.py in
the pinned fleet image, and model_executor/layers/mamba/mamba_utils.py. They
describe BF16, no speculative decoding, TP/PP, and the allowed engine options.
The simulator reserves a full request's *full-attention* output up front; vLLM
can instead preempt/recompute. Window and recurrent memory are bounded, not
misrepresented as full attention. Kernel selection remains hardware dependent.
"""
from collections import defaultdict
from dataclasses import dataclass
import math

import engine_args


def cdiv(n, d):
    return (int(n) + int(d) - 1) // int(d)


@dataclass(frozen=True)
class Group:
    kind: str
    block_size: int
    count: int
    window: int = 0


@dataclass(frozen=True)
class Layout:
    groups: tuple[Group, ...]
    bytes_per_block: int
    max_model_len: int
    in_flight_tokens: int
    prefix_caching: bool
    padding_layers: int

    def blocks(self, length, *, decoding=False):
        """Conservative admission reservation, in shared-pool block IDs."""
        length = max(1, int(length))
        in_flight = 2 if decoding else self.in_flight_tokens
        result = 0
        for g in self.groups:
            if g.kind == 'linear_attention':
                count = 2 if self.prefix_caching else 1
            elif g.kind == 'sliding_attention':
                count = min(cdiv(length, g.block_size),
                            cdiv(g.window - 1 + in_flight, g.block_size) + 1)
            elif g.kind == 'chunked_attention':
                count = cdiv(min(length, g.window + in_flight), g.block_size)
            else:
                count = cdiv(length, g.block_size)
            result += count * g.count
        return result

    def startup_blocks(self):
        # vLLM reports the worst alignment even for a short sliding window.
        return sum(g.count * (
            (2 if self.prefix_caching else 1) if g.kind == 'linear_attention' else
            cdiv(min(g.window - 1 + self.in_flight_tokens, self.max_model_len), g.block_size) + 1
            if g.kind == 'sliding_attention' else
            cdiv(min(g.window + self.in_flight_tokens, self.max_model_len), g.block_size)
            if g.kind == 'chunked_attention' else cdiv(self.max_model_len, g.block_size)
        ) for g in self.groups)

    def capacity_tokens(self, available_bytes):
        return int((available_bytes // self.bytes_per_block) / self.startup_blocks() * self.max_model_len)

    @property
    def simple_prefix_cache(self):
        # A hash alone cannot represent the sliding tail or a recurrent state
        # snapshot. Until their eviction is simulated, only ordinary full KV
        # receives prefix-reuse credit (conservative, explicitly reported).
        return len(self.groups) == 1 and self.groups[0].kind == 'full_attention'


def recurrent_bytes(a, tp):
    # vLLM's Qwen3.5 model hook reads mamba_ssm_dtype; Nemotron uses
    # mamba_ssm_cache_dtype. Neither changes the BF16 activation/KV dtype.
    state_bytes = 4 if a.get('mamba_ssm_cache_dtype', a.get('mamba_ssm_dtype')) == 'float32' else 2
    if 'linear_num_value_heads' in a:
        keys, values = a['linear_num_key_heads'], a['linear_num_value_heads']
        kd, vd = a['linear_key_head_dim'], a['linear_value_head_dim']
        conv = cdiv(2*keys*kd + values*vd, tp) * (a['linear_conv_kernel_dim']-1)
        state = cdiv(values, tp)*kd*vd
    else:
        heads, dim, size = a['mamba_num_heads'], a['mamba_head_dim'], a['ssm_state_size']
        groups = cdiv(a['n_groups'], tp)*tp
        conv = cdiv(heads*dim + 2*groups*size, tp)*(a['conv_kernel']-1)
        state = cdiv(heads, tp)*dim*size
    return conv*2 + state*state_bytes


def build(a, *, tp=1, pp=1, max_model_len=32768, extra_args=()):
    """Build global cache groups; use the limiting PP rank's physical pool."""
    if not a or not a.get('num_hidden_layers'):
        return None  # entries without architecture metadata (small test fixtures); catalogue models have it
    layers = a['num_hidden_layers']
    types = a.get('layer_types') or ['full_attention']*layers
    if len(types) != layers:
        raise ValueError('architecture layer count and layer_types disagree')
    prefix = not engine_args.has(extra_args, '--no-enable-prefix-caching')
    base_block = engine_args.int_of(extra_args, '--block-size', 16)
    shared = int(a.get('num_kv_shared_layers') or 0)
    # TP-local recurrent state promotes the attention block size; it does not
    # acquire a separate unbounded per-token KV allocation.
    state = recurrent_bytes(a, tp) if 'linear_attention' in types else 0
    specs = []
    for i, kind in enumerate(types[:layers-shared]):
        if kind in ('moe', 'mlp'):
            continue
        if kind == 'deepseek_sparse_attention':
            kind = 'full_attention'
        if kind not in ('full_attention', 'sliding_attention', 'chunked_attention', 'linear_attention'):
            raise ValueError(f'unsupported cache type {kind!r}')
        window = int(a.get('sliding_window') or 0) if kind == 'sliding_attention' else int(a.get('attention_chunk_size') or 0)
        if kind == 'linear_attention':
            specs.append([i, kind, base_block, state, window]); continue
        override = a.get('per_layer_config', {}).get(str(i).zfill(2), {})
        head = override.get('head_dim', a['head_dim'])
        heads = override.get('num_key_value_heads', a['num_key_value_heads'])
        per_token = ((a['kv_lora_rank'] + a.get('qk_rope_head_dim', 64))*2 if a.get('kv_lora_rank')
                     else max(1, cdiv(heads, tp))*head*4)
        block = max(base_block, cdiv(state, per_token*base_block)*base_block)
        specs.append([i, kind, block, per_token*block, window])
    if not specs:
        raise ValueError('architecture has no supported cache layers')
    page = max(s[3] for s in specs)
    buckets = defaultdict(list)
    for i, kind, block, raw_page, window in specs:
        if kind != 'linear_attention' and page % raw_page == 0:
            block *= page // raw_page
        buckets[kind, block, window].append(i)
    counts = [len(v) for v in buckets.values()]
    size = max(counts) if max(counts) < min(counts)*1.5 else min(counts)
    groups, rank_size, padding = [], 0, 0
    for (kind, block, window), indices in buckets.items():
        count = cdiv(len(indices), size)
        groups.append(Group(kind, block, count, window))
        padding += count*size-len(indices)
        for g in range(count):
            group = indices[g::count]
            for rank in range(pp):
                lo, hi = layers*rank//pp, layers*(rank+1)//pp
                rank_size = max(rank_size, sum(lo <= i < hi for i in group))
    batched = engine_args.int_of(extra_args, '--max-num-batched-tokens', 4096)
    if engine_args.has(extra_args, '--no-enable-chunked-prefill'):
        batched = max(batched, max_model_len)
    concurrent_batches = 2 if pp == 1 else pp
    return Layout(tuple(groups), page*rank_size, max_model_len,
                  batched*concurrent_batches, prefix, padding)
