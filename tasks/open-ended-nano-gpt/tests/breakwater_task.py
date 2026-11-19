"""open-ended-nano-gpt's trusted task code for Breakwater (breakwater/SPEC.md).

evaluate: before any submitted code runs, the answer-free LAMBADA prompts, generation operands and PPL token stream
(nano_stages.prepare); then the fixed runners in runners/ drive the submitted adapter as submitted code: the trusted
PPL supervisor (runners/nano_ppl.py) exchanges label-free tensors with it, and nano_driver.py generates.
score: the same checks and reward rule recomputed from the sealed raw outputs (nano_stages.score).
"""
from pathlib import Path
import sys

sys.path[:0] = [str(Path(__file__).resolve().parent), '/opt/breakwater-drivers']


def evaluate(run):
    import nano_stages
    nano_stages.INPUTS, nano_stages.RAW = run.inputs, run.raw
    run.inputs.mkdir()
    nano_stages.prepare()
    # Only the answer-free generation operands are visible to submitted code.
    run.publish('generation.input', (run.inputs / 'generation.input').read_bytes())
    from nano_ppl import evaluate as perplexity
    perplexity(str(run.asset('validation') / 'test.bin'), run.raw / 'logprobs.parquet')
    with (run.raw / 'generation.stdout').open('wb') as out, (run.raw / 'generation.stderr').open('wb') as err:
        run.run_submission(['python3', '/opt/breakwater-drivers/nano_driver.py', '--stage', 'generation',
                            '--input', '/eval-inputs/generation.input', '--output', '/dev/null'],
                           check=True, stdout=out, stderr=err, timeout=1800)


def score(scoring):
    import nano_stages
    from contract import read_json
    nano_stages.INPUTS, nano_stages.RAW, nano_stages.VERIFIER = scoring.inputs, scoring.raw, scoring.records
    nano_stages.score()
    diagnostics = read_json(scoring.records / 'diagnostics.json')
    # The judge checks the reported perplexity against the generation score.
    return {'reward': read_json(scoring.records / 'reward.json')['reward'],
            'details': {key: diagnostics[key] for key in ('metrics', 'generation', 'gate_pass')},
            'validated': nano_stages.VALIDATED}
