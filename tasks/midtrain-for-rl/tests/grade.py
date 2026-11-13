"""Grading helpers for midtrain-for-rl, loaded by Breakwater from this file.

Breakwater's evaluation stage (breakwater/intermediate.py, kind ``midtrain``) runs
``rl_eval/run.sh`` on the sealed ``out/final_checkpoint`` with ``runtime_env()``:
the fixed five-hour GRPO stage, then the post-RL evaluation. It keeps bounded log
tails with ``tail()``. The evaluation stage and the verifier
(breakwater/score_workload.py) both apply ``post_rl_reward()``, the acceptance rule
for the score. Nothing here reads the agent workspace or imports submitted code.
The reward is (AMC avg@4 + HumanEval+ accuracy + GPQA-Diamond accuracy) / 3.
"""
import json
import math
import os
from pathlib import Path
import sys

RL = Path('/tests/rl_eval')
# The fixed RL layout: actor DP4 and four rollout engines.
GPUS = '0,1,2,3'
TAIL_BYTES = 256 * 1024


def runtime_env(scratch):
    """Caches, temporary files and HOME that nothing from authoring could have seeded."""
    home, temporary = scratch / 'home', scratch / 'tmp'
    home.mkdir()
    temporary.mkdir()
    cache = home / '.cache'
    return {**os.environ, 'HOME': str(home), 'TMPDIR': str(temporary), 'XDG_CACHE_HOME': str(cache),
            'HF_HOME': str(cache / 'huggingface'), 'TRITON_CACHE_DIR': str(home / '.triton'),
            'TORCHINDUCTOR_CACHE_DIR': str(cache / 'torchinductor'),
            'TORCH_EXTENSIONS_DIR': str(cache / 'torch_extensions'), 'CUDA_CACHE_PATH': str(home / '.nv'),
            'CUDA_VISIBLE_DEVICES': GPUS}


def post_rl_reward(run, recipe):
    """Accept only a time-budget-compliant run and full, non-probe evaluation."""
    sys.path.insert(0, str(RL))
    from evaluate import EXPECTED
    from completion import validate_completion
    trained = json.loads((run / 'rl_checkpoint.json').read_text())
    validate_completion(trained, recipe)
    summary = json.loads((run / 'evaluation/summary.json').read_text())
    suites = summary['suites']
    if summary.get('integration_probe') or set(suites) != set(EXPECTED) or any(
            (suites[name]['items'], suites[name]['samples_per_item']) != tuple(shape)
            for name, shape in EXPECTED.items()):
        raise ValueError('Post-RL evaluation is incomplete or only an integration probe')
    reward = summary['final_reward']
    if type(reward) not in (int, float) or not math.isfinite(reward) or not 0 <= reward <= 1:
        raise ValueError('Invalid post-RL score')
    return reward, {name: suite['score'] for name, suite in suites.items()}, trained


def tail(source, destination):
    if source.is_file():
        with source.open('rb') as stream:
            stream.seek(max(0, source.stat().st_size - TAIL_BYTES))
            destination.write_bytes(stream.read())
