"""Final rl-swe-agent reward from an evaluation record: score.py TRIAL_DIR EVAL_OUT --integrity JUDGE_JSON.

EVAL_OUT is the --output directory of evaluate_checkpoint.py. The reward is the mean of the two suites' resolve
rates (73 development tasks, 100 SWE-bench Verified tasks) for a complete evaluation of an export that passed
the audit; otherwise 0. The evaluator reports a suite's rate only when every selected trial has a 0/1 verifier
reward. A trial whose agent failed (for example, it exhausted the context) but was still verified counts as
unresolved. A trial whose verifier ran to its end but produced no reward (the evaluated model left the
repository broken, or its tests hung until the verifier timeout) also counts as unresolved: the model caused
it. A trial whose verifier never ran is an infrastructure failure and leaves the suite, and the reward, invalid.

--integrity names the Breakwater run's reports/judge.json: the reward is 0 unless its verdict is PASS.
"""
import json
from pathlib import Path


def verifier_ran(result):
    stage = result.get('verifier') or {}
    return bool(stage.get('started_at') and stage.get('finished_at'))


def unrewarded(job_dir):
    """Finished trials without a 0/1 reward: {task: verifier_ran}."""
    rows = {}
    for path in sorted(job_dir.glob('*/result.json')):
        result = json.loads(path.read_text())
        reward = ((result.get('verifier_result') or {}).get('rewards') or {}).get('reward')
        if result.get('finished_at') and reward not in (0, 1):
            rows[result['task_name']] = verifier_ran(result)
    return rows


def adjust(out, name, row):
    """Rate for a complete suite whose only gaps are verified-but-unrewarded trials (counted unresolved)."""
    if row.get('score_valid') or not row.get('complete'):
        return row.get('resolved_rate'), []
    gaps = unrewarded(out / name / 'trials' / name)
    if gaps and all(gaps.values()) and row['graded'] + len(gaps) == row['selected']:
        return row['resolved'] / row['selected'], sorted(gaps)
    return None, []


def main(trial, out, integrity=None):
    summary = json.loads((out / 'summary.json').read_text())
    suites, rates, counted = {}, [], {}
    for name, row in summary.get('benchmarks', {}).items():
        rate, gaps = adjust(out, name, row)
        rates.append(rate)
        if gaps:
            counted[name] = gaps
        suites[name] = {k: row.get(k) for k in ('selected', 'finished', 'graded', 'resolved', 'complete',
                                                 'score_valid', 'errors')}
        suites[name]['resolved_rate'] = rate
    macro = (sum(rates) / len(rates) if summary.get('complete') and rates and None not in rates else None)
    complete = summary.get('export_audit_pass') is True and isinstance(macro, (int, float))
    # The integrity review of the Breakwater run (reports/judge.json) must pass too; a run whose judge did not
    # run (for example after a rule violation) has no verdict file and scores 0.
    verdict = (json.loads(Path(integrity).read_text()).get('verdict')
               if integrity and Path(integrity).is_file() else None)
    complete = complete and (integrity is None or verdict == 'PASS')
    (trial / 'verifier').mkdir(exist_ok=True)
    (trial / 'verifier/reward.json').write_text(json.dumps({'reward': macro if complete else 0.0}))
    (trial / 'verifier/reward_metadata.json').write_text(json.dumps({
        'deterministic_complete': complete, 'evaluation': str(out), 'error': summary.get('error'),
        'export_audit_pass': summary.get('export_audit_pass'),
        **({'integrity': str(integrity), 'integrity_verdict': verdict} if integrity else {}),
        'evaluator_macro_resolved_rate': summary.get('macro_resolved_rate'),
        'agent_errors_rule': 'per-suite errors are agent failures whose trials were still verified; they count as '
                             'unresolved and are not a grading-gate failure',
        'unrewarded_rule': 'trials whose verifier ran but produced no reward count as unresolved (model-caused); '
                           'a trial whose verifier never ran leaves the suite invalid',
        'unrewarded_counted_unresolved': counted,
        'suites': suites,
        'excluded_tasks': summary.get('settings', {}).get('excluded_tasks')}, indent=1))
    print(json.dumps({'reward': macro if complete else 0.0, 'unrewarded_counted_unresolved': counted}))


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('trial', type=Path, help='trial directory; the reward is written to TRIAL/verifier/reward.json')
    parser.add_argument('out', type=Path, help='output directory of evaluate_checkpoint.py')
    parser.add_argument('--integrity', type=Path, required=True, help="the Breakwater run's reports/judge.json; reward 0 unless PASS")
    arguments = parser.parse_args()
    main(arguments.trial, arguments.out, arguments.integrity)
