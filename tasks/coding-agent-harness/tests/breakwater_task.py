"""coding-agent-harness's trusted task code for Breakwater (breakwater/SPEC.md).

evaluate: the validity battery: a scripted model in the coordinator drives the sealed harness, which runs as submitted
code, through every smoke scenario.
score: gradeability recomputed from the root-recorded battery; diagnostics recomputed, never gating. The harness's
downstream score comes from the task's grid (grid/grid.py) after this run passes.
"""
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from contract import read_json, require  # noqa: E402


def evaluate(run):
    from harness_battery import run_battery
    rows, summary = run_battery(run.submission / 'harness/agent.py', Path('/private/tests/smoke'), run.raw / 'harness')
    require(isinstance(summary, dict) and 'gradeable' in summary, 'Validity battery did not complete')


def score(scoring):
    return {'reward': harness_score(scoring.raw, smoke=str(scoring.tests / 'smoke'),
                                    module=str(scoring.submission / 'harness/agent.py'))}


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
