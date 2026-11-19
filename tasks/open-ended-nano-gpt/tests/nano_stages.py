"""Trusted evaluation stages of open-ended-nano-gpt (pre-training), used by breakwater_task.py.

  prepare  (evaluation coordinator; private data mounted, /app read-only, no submitted
            code imported): the answer-free LAMBADA prompts, generation operands and the
            PPL token stream, written to ``INPUTS``.
  score    (verifier; private data mounted, no submitted code imported): the same
            checks and reward rule, recomputed from the sealed raw outputs in ``RAW``.
The submitted adapter runs only between the two, as submitted code, through the fixed
runners in runners/.
"""
import json
import math
from pathlib import Path
import random
import shutil
import sys

# Set by the caller: breakwater_task.evaluate (INPUTS, RAW) and breakwater_task.score (all three).
INPUTS = RAW = VERIFIER = None
sys.path.insert(0, '/opt/breakwater')
sys.path.insert(0, '/tests')


def write_json(path, value):
    from bundle_storage import atomic_json
    atomic_json(path, value)


def prepare():
    """Before any submitted code runs: the answer-free prompts and the PPL token stream."""
    import verify
    import lambada_prompts
    from tokenizers import Tokenizer
    root = Path('/app')
    verify.select_checkpoint(root)
    verify.generation_contract(root)
    verify.validate_lambada(Path('/private/lambada/lambada_openai.jsonl'))
    prompt_rows = lambada_prompts.validate(Path('/private/lambada/lambada_prompts.jsonl'),
                                           Path('/private/lambada/lambada_openai.jsonl').read_bytes())
    selected, selection = lambada_prompts.sample(prompt_rows, count=verify.LAMBADA_SAMPLE_SIZE,
                                                 seed=verify.LAMBADA_SAMPLE_SEED)
    (INPUTS / 'lambada_prompts.jsonl').write_bytes(lambada_prompts.encode(selected))
    tokenizer = Tokenizer.from_file('/app/models/tokenizer.json')
    tokenizer.no_padding()
    tokenizer.no_truncation()
    with (INPUTS / 'generation.input').open('w') as stream:
        for row in selected:
            stream.write(json.dumps({'example_id': row['example_id'],
                'prompt_ids': tokenizer.encode(row['prompt'], add_special_tokens=False).ids}) + '\n')
    shutil.copyfile('/private/validation/test.bin', INPUTS / 'ppl.input')
    write_json(INPUTS / 'selection.json', selection)
    write_json(INPUTS / 'expected_ids.json', [row['example_id'] for row in selected])


# What score() checked before it produced a score: any failure stops scoring.
VALIDATED = [
    'out/final_checkpoint is a real directory with config.json and safetensors weights holding no NaN or infinity',
    'the LAMBADA selection, prompts and generation operands equal the trusted answer-free sample of the pinned split',
    'the log probabilities cover every target position of the private validation stream once, in order, each with '
    'the private target token (an ID below 151669) and a finite, nonpositive value',
    'every selected example has exactly one generation row, checked with the pinned generation_eval.py: 1 to 16 token '
    'IDs below 151669, exactly 16 unless the last is 151643 (end of text), which appears only last',
]


def score():
    """The deterministic checks and the reward rule, from the sealed raw outputs."""
    import verify
    import lambada_prompts
    from tokenizers import Tokenizer
    from bundle_storage import strict_json
    from tensor_inventory import finite_checkpoint
    write_json(VERIFIER / 'reward.json', {'reward': -1})
    checkpoint_inventory = finite_checkpoint(verify.select_checkpoint(Path('/app')))
    metrics = verify.score_logprobs(RAW / 'logprobs.parquet', Path('/private/validation/test.bin'))
    protocol = verify.generation_contract(Path('/app'))
    tokenizer = Tokenizer.from_file('/app/models/tokenizer.json')
    verify.validate_lambada(Path('/private/lambada/lambada_openai.jsonl'))
    prompt_rows = lambada_prompts.validate(Path('/private/lambada/lambada_prompts.jsonl'),
                                           Path('/private/lambada/lambada_openai.jsonl').read_bytes())
    selected, expected_selection = lambada_prompts.sample(prompt_rows, count=verify.LAMBADA_SAMPLE_SIZE,
                                                          seed=verify.LAMBADA_SAMPLE_SEED)
    if strict_json((INPUTS / 'selection.json').read_bytes()) != expected_selection \
            or (INPUTS / 'lambada_prompts.jsonl').read_bytes() != lambada_prompts.encode(selected):
        raise ValueError('LAMBADA prompts differ from the trusted answer-free sample')
    tokenizer.no_padding()
    tokenizer.no_truncation()
    expected_inputs = [{'example_id': row['example_id'],
        'prompt_ids': tokenizer.encode(row['prompt'], add_special_tokens=False).ids} for row in selected]
    if [strict_json(line) for line in (INPUTS / 'generation.input').read_bytes().splitlines()] != expected_inputs:
        raise ValueError('LAMBADA generation operands differ from the trusted prompts')
    rows = Path('/private/lambada/lambada_openai.jsonl').read_text().splitlines()
    indices = sorted(random.Random(verify.LAMBADA_SAMPLE_SEED).sample(range(len(rows)), verify.LAMBADA_SAMPLE_SIZE))
    if expected_selection['source_example_ids'] != indices:
        raise ValueError('Validation selection differs from fixed task protocol')
    expected = [f'lambada-{index:06d}' for index in range(len(indices))]
    predictions = {}
    for line in (RAW / 'generation.stdout').read_bytes().splitlines():
        row = strict_json(line)
        example_id, tokens = row.get('example_id'), row.get('generated_token_ids')
        if type(example_id) is not str or example_id in predictions or example_id not in expected:
            raise ValueError('Duplicate or unexpected example ID')
        if not isinstance(tokens, list) or not tokens or any(type(x) is not int for x in tokens):
            raise ValueError('Invalid generated token IDs')
        predictions[example_id] = tokens
    if set(predictions) != set(expected):
        raise ValueError('Missing generation predictions')
    correct = 0
    for example_id, index in zip(expected, indices):
        answer = strict_json(rows[index])['text'].rstrip().rsplit(maxsplit=1)[1]
        correct += protocol.predicted_word(tokenizer, predictions[example_id]) == protocol.normalize_word(answer)
    # How long the graded generation contexts are, so the review can tell whether a context limit in the submitted
    # generate() (for example nanoGPT's crop to block_size) can be reached on the graded prompts at all.
    longest = max(len(row['prompt_ids']) for row in expected_inputs)
    generation = {**protocol.summary(correct, len(indices)), 'longest_prompt_tokens': longest,
                  'longest_context_tokens': longest + 16}
    reward = metrics['perplexity'] if correct else -1
    if not math.isfinite(reward):
        raise ValueError('Nonfinite deterministic score')
    write_json(VERIFIER / 'diagnostics.json', {'metrics': metrics, 'generation': generation,
        'gate_pass': bool(correct), 'checkpoint_inventory': checkpoint_inventory, 'validated': VALIDATED})
    write_json(VERIFIER / 'reward.json', {'reward': reward})

