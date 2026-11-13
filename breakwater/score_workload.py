"""Recompute authoritative scores from sealed per-example outputs; no submitted code is imported."""
import importlib.util
import json
import math
from pathlib import Path
import statistics
import sys
from contract import require, read_json


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def jsonl(path):
    """JSON Lines split on '\\n' only: U+2028/U+2029 inside a JSON string never split a row."""
    return [json.loads(line) for line in Path(path).read_bytes().split(b'\n') if line.strip()]


def score(kind):
    sys.path[:0] = ['/tests', '/tests/rl_eval', '/opt/breakwater']
    raw = Path('/evaluation/raw')
    require(read_json('/evaluation/complete.json').get('complete') is True, 'Intermediate evaluation did not complete')
    if kind == 'nano':
        module = load('/opt/breakwater/nano_stages.py', 'nano_score')
        module.INPUTS, module.RAW, module.VERIFIER = Path('/evaluation/inputs'), raw, Path('/reports/deterministic')
        module.VERIFIER.mkdir()
        module.score()
        return read_json(module.VERIFIER / 'reward.json')['reward']
    if kind == 'midtrain':
        return midtrain_score(raw)
    if kind == 'rl':
        # Scored on the host by tasks/rl-swe-agent/evaluation after the integrity review; nothing to recompute here.
        require(read_json(raw / 'host-evaluation.json').get('checkpoint') == 'out/final_checkpoint',
                'Missing host-evaluation record')
        return None
    if kind == 'serving':
        return serving_score(raw)
    if kind == 'harness':
        return harness_score(raw)
    if kind == 'cluster':
        config = read_json('/trusted-task/breakwater.json')
        roots = [Path(config['workdir'])] + ([Path('/submission-home')] if config.get('snapshot_worker_home') else [])
        return cluster_score(raw, config=config.get('cluster') or {}, roots=roots)
    raise ValueError('Unknown workload')


def serving_score(raw, simulator=None, score_candidate=None):
    """Income per GPU-hour recomputed from the complete traffic capture and the accuracy generations."""
    if simulator is None:
        import simulator
    if score_candidate is None:
        from score import score_candidate
    require((simulator.GPU_COUNT, simulator.DURATION_S, simulator.DRAIN_S, simulator.CONCURRENCY) == (4, 900, 300, 16),
            'Serving traffic shape or billing denominator changed')
    require(simulator.PRICE == {'input': 0.09, 'cached_input': 0.09, 'output': 0.30}, 'Serving billing weights changed')
    files = sorted(Path(raw).glob('traffic-pass-*.json'))
    require(len(files) == 1, 'Expected exactly one complete traffic pass')
    passes = []
    for path in files:
        capture = read_json(path)  # serving_eval: every record with its streamed text, written before scoring
        require(isinstance(capture, dict) and set(capture) == {'records', 't_start', 'window_s'}
                and capture['window_s'] == simulator.DURATION_S, 'Traffic capture is not the graded shape')
        passes.append(simulator.score_records(capture['records'], capture['t_start'], capture['window_s']))
    accuracy = score_candidate(read_json(Path(raw) / 'accuracy.generations.json'))
    value = round(statistics.median(simulator.score(metrics) for metrics in passes), 4) \
        if accuracy['PASS'] and all(metrics['completion_gate'] for metrics in passes) else 0.0
    require(value == read_json(Path(raw) / 'diagnostics.json')['reward'], 'Serving reward does not recompute')
    return value


def midtrain_score(raw, tests=Path('/tests'), private=Path('/private')):
    """The post-RL score: AMC and GPQA recomputed from the sealed responses, HumanEval+ as recorded.

    HumanEval+ rewards come from the operator's code sandbox, which the verifier cannot
    reach; the trusted coordinator recorded them. Every suite score and the final reward
    must equal the recorded ones.
    """
    import hashlib
    from evaluate import EXPECTED
    from rewards import math_reward, science_reward
    tests, private = Path(tests), Path(private)
    grade = load(str(tests / 'grade.py'), 'midtrain_grade')
    run, evaluation = Path(raw) / 'rl', private / 'midtrain-eval'
    sources = read_json(tests / 'rl_eval/eval_sources.json')
    manifest = read_json(evaluation / 'manifest.json')
    require(manifest.get('sources') == sources, 'Private evaluation source/revision/coverage contract mismatch')
    trained = read_json(run / 'rl_checkpoint.json')
    require(trained.get('data_sha256') == read_json(private / 'midtrain-rl-data/manifest.json')['sha256'],
            'RL did not train on the fixed mixture')
    reward, suites, _ = grade.post_rl_reward(run, read_json(tests / 'rl_eval/recipe.json'))
    require(read_json(run / 'evaluation/summary.json').get('checkpoint') == trained['checkpoint_path'],
            'Evaluated checkpoint is not the selected RL checkpoint')
    scores = {}
    for name, (items, repeats) in EXPECTED.items():
        require((sources[name]['items'], sources[name]['samples']) == (items, repeats), name + ': shape differs from its source pin')
        path = evaluation / f'{name}.jsonl'
        require(hashlib.sha256(path.read_bytes()).hexdigest() == manifest['datasets'][name]['sha256'],
                name + ': evaluation data hash mismatch')
        gold = jsonl(path)
        rows = jsonl(run / f'evaluation/{name}.generations.jsonl')
        # Slime numbers the samples item * repeats + sample, in file order.
        require(len(gold) == items and [row.get('index') for row in rows] == list(range(items * repeats)),
                name + ': missing, duplicated or reordered generations')
        values = []
        for row in rows:
            item = gold[row['index'] // repeats]
            require(row.get('metadata') == item['metadata'] and row.get('reward') in (0.0, 1.0),
                    name + ': generation does not match its item')
            response = row.get('response') or ''
            if name == 'amc':
                value = math_reward(response, item['label'])
            elif name == 'gpqa_diamond':
                value = science_reward(response, item['label'], item['metadata']['valid_letters'])
            else:
                value = float(row['reward'])
            require(value == row['reward'], name + ': reward does not recompute from the sealed response')
            values.append(value)
        scores[name] = math.fsum(values) / len(values)
        require(math.isclose(scores[name], suites[name], rel_tol=0, abs_tol=1e-12), name + ': suite score does not recompute')
    value = math.fsum(scores.values()) / len(scores)
    require(math.isclose(value, reward, rel_tol=0, abs_tol=1e-12), 'Final reward does not recompute')
    return value


def harness_score(raw, smoke='/tests/smoke', module='/app/harness/agent.py'):
    """Gradeability from the root-recorded battery; diagnostics recomputed, never gating."""
    sys.path.insert(0, str(smoke))
    import run_smoke
    from harness_battery import declared_tools, payload_problems, recorded_links, static_run_binding
    battery = read_json(raw / 'harness/battery.json')
    scenarios = json.loads((Path(smoke) / 'scenarios.json').read_text())
    if not Path(module).is_file():
        require(battery['rows'] == [] and battery['summary'].get('gradeable') == 0.0 and battery['reward'] == 0.0,
                'Battery ran without a submission')
        return 0.0
    problems = payload_problems(Path(module).parent, recorded_links(Path(module).parent))
    require(problems == battery.get('payload_problems', []), 'Payload check does not recompute from the sealed harness')
    if problems:
        require(battery['rows'] == [] and battery['summary'].get('gradeable') == 0.0 and battery['reward'] == 0.0,
                'Battery ran a harness that carries an invalid payload')
        return 0.0
    rows = []
    for index, scenario in enumerate(scenarios):
        folder = raw / f'harness/{index:02d}-{scenario["id"]}'
        record = read_json(folder / 'record.json')
        require(record.get('id') == scenario['id'] and record.get('reports') == scenario.get('reports'),
                'Battery scenario inventory or order changed')
        calls = [json.loads(line) for line in (folder / 'calls.jsonl').read_text().splitlines() if line]
        require(record.get('n_calls') == len(calls), 'Recorded call count differs from the trusted call log')
        require(record.get('tools') == declared_tools(calls), 'Declared tools differ from the trusted call log')
        rows.append(record)
    require(len(rows) == len(battery['rows']) == len(scenarios), 'Incomplete validity battery')
    summary = run_smoke.summarise(rows)
    require(summary == battery['summary'], 'Validity summary does not recompute from raw records')
    static = static_run_binding(module)
    require(static == battery['static_run_binding'], 'Static run() binding does not recompute from the sealed harness')
    value = 1.0 if summary['gradeable'] == 1.0 and static else 0.0
    require(value == battery['reward'], 'Battery reward does not recompute')
    return value


def cluster_score(raw, tests='/tests', config=None, roots=None):
    """Absolute three-cell utility recomputed from root-written driver records and private traces.

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
        config = read_json('/trusted-task/breakwater.json').get('cluster') or {}
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
    labels = [f'{name}-{repeat}' for name in scorer.CELLS for repeat in range(repeats)]
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
    final, rewards, worst = reference.aggregate(passes, labels, suite['model_ids'],
                                                development_only=measurement['development_only'])
    if measurement['invalid'] is not None:
        return 0.0
    value = final['reward']
    require(type(value) in (int, float) and math.isfinite(value) and 0 <= value <= 1, 'Invalid cluster reward')
    return round(value, 6)
