"""Trusted structured-beam-output scorer; submitted vLLM executes only in main."""
import json

EXAMPLE_IDS = ['pr35022:tinyllama:half:beam4']
SCHEMA = {'type': 'object', 'properties': {'name': {'type': 'string'}, 'age': {'type': 'integer'}},
          'required': ['name', 'age'], 'additionalProperties': False}


def prepare_cases():
    import copy
    return [{'example_id': EXAMPLE_IDS[0], 'model': 'TinyLlama/TinyLlama-1.1B-Chat-v1.0',
             'dtype': 'half', 'beam_width': 4, 'max_tokens': 64, 'max_model_len': 512,
             'prompt': 'Generate a JSON object for a person with name and age:',
             'schema': copy.deepcopy(SCHEMA)}], {EXAMPLE_IDS[0]: copy.deepcopy(SCHEMA)}


def score_rows(rows, private):
    import jsonschema
    if set(private) != set(EXAMPLE_IDS) or len(rows) != 1 or rows[0].get('example_id') != EXAMPLE_IDS[0]:
        raise ValueError('Missing, duplicate or unexpected structured beam example ID')
    row = rows[0]
    reason = None
    outputs = row.get('outputs')
    if row.get('status') != 'returned':
        reason = 'execution_failed'
    elif not isinstance(outputs, list) or len(outputs) != 1:
        reason = 'prompt_output_count_mismatch'
    else:
        sequences = outputs[0].get('sequences') if isinstance(outputs[0], dict) else None
        if not isinstance(sequences, list) or not sequences:
            reason = 'no_sequences'
        else:
            for sequence in sequences:
                text = sequence.get('text') if isinstance(sequence, dict) else None
                tokens = sequence.get('token_ids') if isinstance(sequence, dict) else None
                if not isinstance(tokens, list) or not tokens or any(type(t) is not int or t < 0 for t in tokens):
                    reason = 'missing_or_invalid_token_ids'
                    break
                if not isinstance(text, str) or '{' not in text:
                    reason = 'missing_json_text'
                    break
                generated = text[text.find('{'):].replace('</s>', '').strip()
                try:
                    parsed = json.loads(generated)
                    jsonschema.validate(parsed, private[EXAMPLE_IDS[0]])
                except (ValueError, jsonschema.ValidationError):
                    reason = 'invalid_structured_output'
                    break
    return {'raw_reward': float(reason is None), 'correctness_pass': reason is None,
            'cases': [{'example_id': EXAMPLE_IDS[0], 'passed': reason is None, 'reason': reason}],
            'deterministic_complete': True, 'deterministic_gates_pass': True,
            'eligible': False, 'integrity_status': 'pending'}
