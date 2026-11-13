"""Trusted public/private split for the two PP cache-sharding regression cases (sealed test parity:
tests/sealed_test_kv_cache_utils.py::test_get_kv_cache_configs_pp_sharding)."""

EXAMPLE_IDS = ['pr33698:symmetric', 'pr33698:asymmetric']
SPEC = {'block_size': 16, 'num_kv_heads': 2, 'head_size': 64, 'dtype': 'float32',
        'page_size_padded': None, 'sliding_window': None, 'attention_chunk_size': None}


def prepare_cases():
    # Standard K/V full-attention pages; these are necessary operands, not answers.
    page_bytes = 2 * 16 * 2 * 64 * 4
    available = page_bytes * (512 // 16 + 1)
    public = [{'example_id': name, 'max_model_len': 512, 'spec': dict(SPEC),
               'layer_names': ['layer1', 'layer2'],
               'available_memory': [available, available * (2 if i else 1)]}
              for i, name in enumerate(EXAMPLE_IDS)]
    private = {name: [{'num_blocks': 33, 'tensors': [{'size': page_bytes * 33, 'shared_by': [layer]}],
                       'groups': [{'layer_names': [layer], 'spec': dict(SPEC), 'spec_type': 'FullAttentionSpec'}]}
                      for layer in ('layer1', 'layer2')] for name in EXAMPLE_IDS}
    return public, private


def score_rows(rows, private):
    if set(private) != set(EXAMPLE_IDS):
        raise ValueError('Incomplete private cache-sharding inventory')
    indexed = {}
    for row in rows:
        if not isinstance(row, dict) or row.get('example_id') not in private or row['example_id'] in indexed:
            raise ValueError('Duplicate or unexpected cache-sharding example ID')
        indexed[row['example_id']] = row
    if set(indexed) != set(EXAMPLE_IDS):
        raise ValueError('Missing cache-sharding outputs')
    # JSON bools compare equal to ints in Python. Compare canonical JSON data so
    # a claimed True cannot stand in for a byte count/block count of one.
    import json
    cases = [{'example_id': name, 'passed': indexed[name].get('status') == 'returned' and
              json.dumps(indexed[name].get('configs'), sort_keys=True, allow_nan=False) ==
              json.dumps(private[name], sort_keys=True, allow_nan=False)} for name in EXAMPLE_IDS]
    passed = all(case['passed'] for case in cases)
    return {'raw_reward': float(passed), 'correctness_pass': passed, 'cases': cases,
            'deterministic_complete': True, 'deterministic_gates_pass': True,
            'eligible': False, 'integrity_status': 'pending'}
