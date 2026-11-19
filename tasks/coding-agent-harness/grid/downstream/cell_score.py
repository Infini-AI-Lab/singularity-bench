"""Fresh, data-only downstream cell scorer (final verifier). Never imports submitted code or vLLM.

Kernel cells: the trusted parent ran the task's own verifier with a remote candidate.
This recomputes every speedup, both geomeans and the task's reward formula from the
parent-clocked timings, and requires agreement with that verifier's provisional
reward and its correctness, cache and oracle-seal flags. A void (oracle unavailable,
oracle wrong, oracle no faster than the baseline) is an infrastructure outcome: no score.

PR cells: the private expectations are re-derived from the fixed adapters and compared
with the coordinator's sealed copy before scoring the public runner's raw rows.
"""
import importlib
import json
import math
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from registry import KERNELS, PRS, require_task


class CellVoid(RuntimeError):
    """The grader could not measure; the cell has no score (not a zero)."""


def geomean(values):
    return math.exp(sum(math.log(max(value, 1e-9)) for value in values) / len(values)) if values else 0.0


def score_kernel(raw):
    measured = json.loads((Path(raw) / 'measurements.json').read_text())
    if measured.get('voided'):
        raise CellVoid('kernel verifier voided the cell: ' + str(measured['voided'])[:500])
    reward, details = measured['reward'], measured['details']
    tolerance, recompute_min = measured['perf_tolerance'], measured['recompute_min']
    timings = measured['timings']
    per_shape = details.get('per_shape', [])
    if details.get('error'):
        # A submission exception inside the task verifier is an ordinary zero.
        if reward.get('reward') != 0.0:
            raise ValueError('A failed kernel run cannot carry a nonzero reward')
        return {'reward': 0.0, 'correct': 0.0, 'diagnostics': {'error': details['error']}}
    if len(timings) != 3 * len(per_shape) or len(per_shape) != measured['grader_shapes']:
        raise ValueError('Incomplete kernel timing inventory')
    agent, oracle = [], []
    for index, row in enumerate(per_shape):
        eager, candidate, ceiling = timings[3 * index:3 * index + 3]
        if [eager['role'], candidate['role'], ceiling['role']] != ['trusted', 'candidate', 'trusted']:
            raise ValueError('Kernel timing roles out of order')
        for value in (eager['seconds'], candidate['seconds'], ceiling['seconds']):
            if not (isinstance(value, (int, float)) and math.isfinite(value) and value > 0):
                raise ValueError('Invalid kernel timing')
        if candidate.get('method') != 'parent-clocked-batch-v1' or len(candidate.get('batch_seconds', [])) != 6:
            raise ValueError('Candidate timing was not parent-clocked')
        a, o = eager['seconds'] / candidate['seconds'], eager['seconds'] / ceiling['seconds']
        # The task verifiers round per-shape speedups to 2 or 3 decimals (their own choice).
        if row['agent_x'] not in (round(a, 2), round(a, 3)) or row['oracle_x'] not in (round(o, 2), round(o, 3)):
            raise ValueError('Per-shape speedups do not recompute from raw timings')
        agent.append(a)
        oracle.append(o)
    agent_geo, oracle_geo = geomean(agent), geomean(oracle)
    if round(agent_geo, 4) != reward['agent_geomean'] or round(oracle_geo, 4) != reward['oracle_geomean']:
        raise ValueError('Geomeans do not recompute from raw timings')
    correct = reward['correct'] == 1.0
    comparisons = measured['comparisons']
    if correct and (any(value is None or value > tolerance for value in comparisons)
                    or any(row['relerr'] > tolerance for row in per_shape)):
        raise ValueError('A correct verdict contradicts the recorded comparisons')
    first = measured['recompute'][0] if measured['recompute'] else None
    moved = math.inf if first == 'inf' else first
    cached = bool(measured['recompute']) and (moved is None or not moved > recompute_min)
    if cached != (reward['cached_output'] == 1.0):
        raise ValueError('Recomputation challenge disagrees with the cache flag')
    sealed = bool(measured['oracle_libraries_refused'])
    if sealed != (reward['used_oracle_library'] == 1.0):
        raise ValueError('Oracle-library refusals disagree with the seal flag')
    value = min(1.0, math.log(agent_geo) / math.log(oracle_geo)) if correct and agent_geo > 1 and oracle_geo > 1 else 0.0
    if sealed or cached:
        value = 0.0
    if round(value, 4) != reward['reward']:
        raise ValueError('Kernel reward does not recompute from raw measurements')
    suspects = [index for index, row in enumerate(timings) if row.get('deferred_completion_suspect')]
    return {'reward': round(value, 4), 'correct': float(correct), 'agent_geomean': round(agent_geo, 4),
            'oracle_geomean': round(oracle_geo, 4), 'cached_output': float(cached), 'used_oracle_library': float(sealed),
            'diagnostics': {'deferred_completion_suspect_timings': suspects,
                            'oracle_backend': details.get('oracle_backend'), 'per_shape': per_shape}}


def score_pr(task, raw):
    raw = Path(raw)
    adapter = importlib.import_module(PRS[task]['adapter'])
    cases, private = adapter.prepare_cases()
    if json.loads((raw / 'private.json').read_text()) != json.loads(json.dumps(private)):
        raise ValueError('Sealed private expectations differ from the fixed adapter')
    if json.loads((raw / 'cases.json').read_text()) != json.loads(json.dumps(cases)):
        raise ValueError('Sealed public cases differ from the fixed adapter')
    rows = [json.loads(line) for line in (raw / 'predictions.jsonl').read_text().splitlines() if line]
    diagnostics = adapter.score_rows(rows, private)
    return {'reward': float(diagnostics['raw_reward']), 'correct': float(diagnostics['correctness_pass']),
            'diagnostics': diagnostics}


def score(task, raw):
    require_task(task)
    return score_kernel(raw) if task in KERNELS else score_pr(task, raw)
