"""rl-swe-agent's trusted task code for Breakwater (breakwater/SPEC.md).

No evaluation runs in the evaluation stage, and no submitted code: the held-out suites run one Harbor sandbox per task,
on the host, after the integrity review (evaluation/README.md). evaluate records that; score defers the reward to the
host evaluator, which keeps it only if the integrity verdict is PASS (evaluation/score.py --integrity).
"""
from contract import atomic_json, read_json, require

HOST_EVALUATION = {'scored_by': 'tasks/rl-swe-agent/evaluation (evaluate_checkpoint.py, then score.py --integrity)',
                   'checkpoint': 'out/final_checkpoint'}


def evaluate(run):
    atomic_json(run.raw / 'host-evaluation.json', HOST_EVALUATION)


def score(scoring):
    require(read_json(scoring.raw / 'host-evaluation.json').get('checkpoint') == 'out/final_checkpoint',
            'Missing host-evaluation record')
    return {'reward': None, 'deferred': {'scored_by': 'tasks/rl-swe-agent/evaluation/score.py, after this review',
                                         'rule': 'the final reward is the host score only if this verdict is PASS'}}
