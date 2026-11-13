"""Freeze simulator architecture metadata from pinned public checkpoint configs.

Reads an existing Hugging Face cache; downloads no weights. The resulting small
JSON ships with the task, and records immutable source URLs and content hashes.
"""
import argparse
import hashlib
import json
from pathlib import Path


def extract(model, config_bytes, resolved=None):
    config = json.loads(config_bytes) if resolved is None else resolved['config']
    if resolved and resolved['config_sha256'] != hashlib.sha256(config_bytes).hexdigest():
        raise ValueError('resolved defaults must match the pinned config bytes')
    text = config.get('text_config', config)
    counts = [text.get(k) for k in ('num_experts', 'num_local_experts', 'n_routed_experts')]
    moe = any(isinstance(n, int) and n > 1 for n in counts)
    # ERNIE describes per-modality experts in lists rather than a scalar count.
    if text.get('model_type') == 'ernie4_5_moe_vl':
        moe = True
    keep = ('model_type', 'hidden_size', 'num_hidden_layers', 'num_attention_heads',
            'num_key_value_heads', 'head_dim', 'vocab_size', 'num_experts', 'num_local_experts',
            'n_routed_experts', 'num_experts_per_tok', 'num_experts_per_token', 'top_k_experts')
    keep += ('kv_lora_rank', 'qk_rope_head_dim', 'index_head_dim', 'index_n_heads',
             'layer_types', 'layers_block_type', 'sliding_window', 'attention_chunk_size',
             'per_layer_config', 'num_kv_shared_layers', 'attention_k_eq_v',
             'linear_conv_kernel_dim', 'linear_key_head_dim', 'linear_value_head_dim',
             'linear_num_key_heads', 'linear_num_value_heads', 'mamba_head_dim',
             'mamba_num_heads', 'ssm_state_size', 'conv_kernel', 'n_groups',
             'mamba_ssm_cache_dtype', 'mamba_ssm_dtype')
    result = {'checkpoint': model['checkpoint'], 'checkpoint_revision': model['checkpoint_revision'],
              'config_sha256': hashlib.sha256(config_bytes).hexdigest(), 'is_moe': moe,
              'architectures': config.get('architectures', []),
              'source': f"https://huggingface.co/{model['checkpoint']}/resolve/{model['checkpoint_revision']}/config.json",
              'text_config': {k: text[k] for k in keep if k in text}}
    if not result['text_config'].get('head_dim') and text.get('hidden_size') and text.get('num_attention_heads'):
        result['text_config']['head_dim'] = text['hidden_size']//text['num_attention_heads']
    arch = result['text_config']
    if 'layers_block_type' in arch:
        arch['layer_types'] = arch.pop('layers_block_type')
        arch['num_hidden_layers'] = len(arch['layer_types'])
    if not arch.get('num_key_value_heads'):
        arch['num_key_value_heads'] = arch['num_attention_heads']
    if not arch.get('layer_types'):
        arch['layer_types'] = [('sliding_attention' if arch.get('sliding_window') else 'full_attention')] * arch['num_hidden_layers']
    if resolved:
        result['defaults_resolver'] = resolved['resolver']
    if model['id'] == 'mistral-small-2603':
        result['active_params_b'] = 6.5
        result['active_params_source'] = 'https://docs.mistral.ai/models/mistral-small-4-0-26-03'
    return result


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--catalogue', type=Path, required=True)
    p.add_argument('--cache', type=Path, default=Path.home()/'.cache/huggingface/hub')
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--resolved', type=Path, required=True,
                   help='resolve_architecture.py output from the pinned fleet image')
    a = p.parse_args()
    resolved = json.loads(a.resolved.read_text())
    models = {}
    for model in json.loads(a.catalogue.read_text())['models']:
        path = a.cache/('models--'+model['checkpoint'].replace('/', '--'))/'snapshots'/model['checkpoint_revision']/'config.json'
        models[model['id']] = extract(model, path.read_bytes(), resolved['models'][model['id']])
    a.out.write_text(json.dumps({'schema_version': 2, 'models': models,
        'defaults': {k: resolved[k] for k in ('versions', 'resolver_sha256')}}, indent=1, sort_keys=True)+'\n')
    print(f'froze {len(models)} pinned architecture records')
