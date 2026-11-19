"""serve-glm53-flash's trusted task code for Breakwater (breakwater/SPEC.md).

evaluate: the task's grader (serving_eval.py) against a fresh, writable, worker-owned copy of the sealed submission at
/app; the submitted server (bash /app/serve.sh) runs as submitted code.
score: income per GPU-hour recomputed from the complete traffic capture and the accuracy generations.
"""
import os
from pathlib import Path
import statistics
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from contract import read_json, require  # noqa: E402


def evaluate(run):
    import serving_eval
    from workers import install_tree, kill_workers
    require(run.contract['submission_mount'] and run.submission.is_dir(),
            'Serving evaluation needs the submission apart from its writable /app copy (workspace.submission_mount)')
    install_tree([(run.submission, Path(run.contract['workdir']))], allowed=run.input_roots())
    os.environ['REWARD_DIR'] = str(run.raw)
    serving_eval.OUTDIR = str(run.raw)
    serving_eval.AGENT_CMD = 'bash /app/serve.sh'  # measure_income() starts it as submitted code
    serving_eval.MEASURE_STOCK = False
    try:
        serving_eval.run_grader(None, 774185263)  # no starter-server measurement
    finally:
        # Detached server children; this also unblocks accuracy threads still reading at exit.
        kill_workers(grace=10)
    report = read_json(run.raw / 'diagnostics.json')
    require(not report.get('grader_error'), 'Serving evaluation failed')


def score(scoring):
    return {'reward': serving_score(scoring.raw)}


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
