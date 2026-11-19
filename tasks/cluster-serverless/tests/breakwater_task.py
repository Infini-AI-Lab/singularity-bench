"""cluster-serverless's trusted task code for Breakwater (breakwater/SPEC.md).

evaluate: the graded control-plane cells against the operator fleet; the submitted plane runs as submitted code and the
trusted driver replays the private traces.
score: the absolute three-cell utility recomputed from root-written driver records and the private traces.
mechanical: the fleet's inference journals, which the host collected after both execution stages ended, recomputed.
review_views: each fleet journal as an index, agent-stage requests in full and graded traffic one line each.
"""
import json
import math
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from contract import digest, read_json, regular, require  # noqa: E402


def evaluate(run):
    from cluster_cells import run_cells
    run_cells(run.raw / 'cluster', run.contract.get('options') or {}, smoke=run.smoke)


def score(scoring):
    roots = [Path(scoring.contract['workdir'])] + ([scoring.home] if scoring.home else [])
    return {'reward': cluster_score(scoring.raw, tests=str(scoring.tests), config=scoring.contract.get('options') or {},
                                    roots=roots, smoke=scoring.smoke)}


def mechanical(scoring):
    evidence = scoring.evidence / 'task'  # what tests/breakwater_host.py collected
    result = {}
    from cluster_fleet import check_journals
    recorded = read_json(regular(evidence, 'fleet-inference.json'))
    manifests = read_json(regular(evidence, 'fleet-manifests.json'))
    folder = Path(evidence) / 'fleet-inference'
    require(folder.is_dir() and not folder.is_symlink(), 'Fleet inference evidence is missing')
    recomputed = check_journals(folder, manifests.get('run_id'))
    require(recomputed['journals'] == recorded['journals'] and recomputed['incomplete'] == recorded['incomplete'],
            'Fleet inference evidence differs from its collection report')
    require(sorted(recomputed['journals']) == sorted(recorded['collected']), 'Uncollected or extra fleet journals')
    result['fleet_inference'] = {key: recomputed[key] for key in ('requests', 'incomplete', 'run_id')}
    result['fleet_inference']['journals'] = len(recomputed['journals'])
    return result


def review_views(scoring):
    return [('external/task/fleet-inference/*.jsonl', fleet_journal_view)]


def fleet_journal_view(path, scoring):
    """A fleet journal for the judge, with each request's stage from the windows the host recorded."""
    recorded = Path(scoring.evidence) / 'task' / 'fleet-inference.json'
    windows = read_json(recorded).get('windows') if recorded.is_file() else None
    return index_fleet_journal(path, windows)


def index_fleet_journal(path, windows):
    """Cluster fleet inference: agent-stage requests in full; graded traffic as one line each."""
    from cluster_fleet import phase_of
    lines = [f'FLEET INFERENCE JOURNAL {Path(path).name} sha256 {digest(path)} (worker-recorded before inference '
             'and before forwarding; prompts are token IDs; no tokenizer is available to the verifier)']
    texts = {}
    with Path(path).open('rb') as stream:
        for raw in stream:
            row = json.loads(raw)
            event, phase = row.get('event'), phase_of(int(row.get('time_ns', 0)), windows)
            if event == 'worker_started':
                lines.append('WORKER ' + json.dumps({k: row.get(k) for k in ('run_id', 'node', 'worker', 'model',
                                                                             'model_info', 'utc')}, sort_keys=True))
            elif event == 'request':
                request = row.get('request') or {}
                prompt = request.get('prompt') if isinstance(request, dict) else None
                summary = {'seq': row.get('seq'), 'phase': phase, 'audit_id': row.get('audit_id'),
                           'model': request.get('model') if isinstance(request, dict) else None,
                           'prompt_tokens': len(prompt) if isinstance(prompt, list) else None,
                           'max_tokens': request.get('max_tokens') if isinstance(request, dict) else None,
                           'routing': request.get('routing') if isinstance(request, dict) else None}
                lines.append('REQUEST ' + json.dumps(summary, sort_keys=True, default=str))
                if phase == 'agent':
                    lines.append('AGENT-STAGE REQUEST BODY ' + json.dumps(request, sort_keys=True, default=str))
            elif event == 'response':
                frame = row.get('frame') or {}
                for choice in frame.get('choices', []) if isinstance(frame, dict) else []:
                    if isinstance(choice, dict) and isinstance(choice.get('text'), str):
                        texts.setdefault(row.get('audit_id'), []).append(choice['text'])
            elif event == 'request_finished':
                text = ''.join(texts.pop(row.get('audit_id'), []))
                lines.append('FINISHED ' + json.dumps({k: row.get(k) for k in ('seq', 'audit_id', 'outcome')},
                                                      sort_keys=True, default=str)
                             + (f' phase={phase} response_text={json.dumps(text)}' if phase == 'agent'
                                else f' phase={phase} response_chars={len(text)}'))
            elif event in ('upstream_error', 'upstream_status', 'worker_stopped'):
                lines.append(event.upper() + ' ' + json.dumps({k: v for k, v in row.items()
                                                               if k not in ('bytes_base64',)}, sort_keys=True, default=str))
    for audit_id, parts in texts.items():
        lines.append(f'UNFINISHED {audit_id} response_text={json.dumps("".join(parts))}')
    return '\n'.join(lines) + '\n'


def cluster_score(raw, tests='/tests', config=None, roots=None, smoke=False):
    """Absolute three-cell utility recomputed from root-written driver records and private traces (a smoke run: the
    one graded cell's, which is not a benchmark score).

    With the sealed submission ``roots``, the entry and byte limits are rechecked: an
    oversized submission must have been graded as invalid.
    """
    # The task kit's modules are imported fresh from the trusted tests tree.
    for name in ('score', 'reference', 'traceunits'):
        sys.modules.pop(name, None)
    sys.path[:0] = [str(Path(tests) / 'kit')]
    import reference
    import score as scorer
    measurement = read_json(raw / 'cluster/measurement.json')
    suite = read_json(Path(tests) / 'suite-graded.json')
    if config is None:
        config = read_json('/trusted-task/contract.json').get('options') or {}
    qualified = set(suite['qualification'].values()) == {'target-qualified'}
    require(measurement['development_only'] == (not qualified), 'Qualification mode differs from the graded suite')
    if roots is not None:
        from cluster_cells import submission_problem
        problem = submission_problem(roots)
        require(problem is None or measurement['invalid'] is not None, 'A submission over its limits was graded: ' + str(problem))
    require(qualified or config.get('allow_unqualified'), 'Unqualified suite graded without a development declaration')
    require(qualified == (measurement['measurement_binding'] is not None), 'Qualified grading lacks its measurement binding')
    require(measurement['score_version'] == scorer.SCORE_VERSION and measurement['model_ids'] == suite['model_ids'],
            'Scoring version or catalogue coverage changed')
    import hashlib
    traces = {}
    for case in suite['cases']:
        path = Path(tests) / f"trace-graded-{case['name']}.jsonl"
        require(hashlib.sha256(path.read_bytes()).hexdigest() == case['trace_sha256']
                == measurement['trace_sha256'][case['name']], 'Graded trace fingerprint mismatch')
        traces[case['name']] = [json.loads(line) for line in path.read_text().splitlines()]
    repeats = measurement['repeats']
    from cluster_cells import SMOKE_CELL
    require(bool(measurement.get('smoke')) == smoke, 'The graded cells differ from the sealed run (smoke or not)')
    labels = [f'{name}-{repeat}' for name in ((SMOKE_CELL,) if smoke else scorer.CELLS) for repeat in range(repeats)]
    require(measurement['labels'] == labels and [cell['label'] for cell in measurement['cells']] == labels,
            'Incomplete or reordered cell inventory')
    passes = []
    for cell in measurement['cells']:
        trace = traces[cell['trace']]
        if measurement['invalid'] is not None:
            passes.append(scorer.summarize([], trace))
            continue
        records = []
        if cell.get('records'):
            lines = (raw / 'cluster' / cell['records']).read_text().splitlines(keepends=True)
            for index, line in enumerate(lines):
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    require(index == len(lines) - 1 and not line.endswith('\n'), 'Corrupt driver record')
        stats = read_json(raw / 'cluster' / cell['stats']) if cell.get('stats') else None
        summary = scorer.summarize(records, trace, stats)
        if cell.get('records'):
            require(math.isclose(summary['utility'], cell['provisional_utility'], abs_tol=1e-12)
                    and scorer.gates_pass(summary) == cell['provisional_gates_pass'],
                    'Cell utility does not recompute from its raw records')
        passes.append(summary)
    if smoke:
        # One cell, scored as the aggregate scores each: its utility if its gates pass, nothing if tampered.
        summary = passes[0]
        if measurement['invalid'] is not None or summary['gates']['tampered'] or not scorer.gates_pass(summary):
            return 0.0
        require(scorer.finite(summary['utility']) and 0 <= summary['utility'] <= 1, 'Invalid cell utility')
        return round(summary['utility'], 6)
    final, rewards, worst = reference.aggregate(passes, labels, suite['model_ids'],
                                                development_only=measurement['development_only'])
    if measurement['invalid'] is not None:
        return 0.0
    value = final['reward']
    require(type(value) in (int, float) and math.isfinite(value) and 0 <= value <= 1, 'Invalid cluster reward')
    return round(value, 6)
