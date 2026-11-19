"""midtrain-for-rl's trusted task code for Breakwater (breakwater/SPEC.md).

evaluate: the grading command (tests/grade.py): the fixed five-hour GRPO stage on the submitted checkpoint, then the
post-RL evaluation. No submitted code runs: the RL stage reads the checkpoint as weights. The operator's code sandbox
(RL coding rewards, HumanEval+) is the coordinator's only network peer.
score: AMC and GPQA recomputed from the sealed responses, HumanEval+ as recorded, and the post-RL reward.
"""
import math
import os
from pathlib import Path
import subprocess
import sys

sys.path[:0] = [str(Path(__file__).resolve().parent), str(Path(__file__).resolve().parent / 'rl_eval')]
from contract import read_json, require  # noqa: E402
from task_api import jsonl  # noqa: E402


def evaluate(run):
    """The grading command (tests/grade.py): the fixed five-hour GRPO stage, then post-RL evaluation.

    Only trusted code runs here: the submitted checkpoint is read as weights by the RL
    stage. The code sandbox (RL coding rewards, HumanEval+) is the only network peer.
    """
    import shutil
    import socket
    import urllib.parse
    import grade
    require(read_json('/private/midtrain-eval/manifest.json').get('sources') == read_json('/tests/rl_eval/eval_sources.json'),
            'Private evaluation source/revision/coverage contract mismatch')
    sandbox = urllib.parse.urlsplit(os.environ['MIDTRAIN_CODE_SANDBOX_URL'])
    require(sandbox.scheme in ('http', 'https') and sandbox.hostname, 'Invalid code sandbox URL')
    if os.environ.get('BREAKWATER_BACKEND') != 'apptainer':
        # Docker: the coordinator is on a bridge network; a firewall leaves it the code sandbox only. On Apptainer its
        # network namespace has loopback alone, and the code sandbox's 127.0.0.1 URL is relayed to the host service.
        address = socket.gethostbyname(sandbox.hostname)
        port = sandbox.port or (443 if sandbox.scheme == 'https' else 80)
        for rule in (['-P', 'OUTPUT', 'DROP'], ['-A', 'OUTPUT', '-o', 'lo', '-j', 'ACCEPT'],
                     ['-A', 'OUTPUT', '-m', 'conntrack', '--ctstate', 'ESTABLISHED,RELATED', '-j', 'ACCEPT'],
                     ['-A', 'OUTPUT', '-d', address, '-p', 'tcp', '--dport', str(port), '-j', 'ACCEPT']):
            subprocess.run(['iptables', *rule], check=True)
        # IPv6 fails closed, as in cluster_cells.firewall: loopback only.
        interfaces = Path('/proc/net/if_inet6')
        if interfaces.is_file() and any(line.split()[-1] != 'lo' for line in interfaces.read_text().splitlines()
                                        if line.split()):
            for rule in (['-P', 'OUTPUT', 'DROP'], ['-A', 'OUTPUT', '-o', 'lo', '-j', 'ACCEPT']):
                subprocess.run(['ip6tables', *rule], check=True)
    # A smoke run (run.smoke) trains two optimizer updates instead of the five-hour budget; nothing else changes.
    recipe = Path('/tests/rl_eval') / ('recipe-smoke.json' if run.smoke else 'recipe.json')
    scratch = Path('/tmp/midtrain-grading')  # RL checkpoints stay in this container; only receipts are retained
    scratch.mkdir(mode=0o700)
    env = grade.runtime_env(scratch)  # fresh HOME/TMPDIR/caches; CUDA_VISIBLE_DEVICES=0,1,2,3
    rl_run, log, retained = scratch / 'run', scratch / 'rl-eval.log', run.raw / 'rl'
    retained.mkdir()
    try:
        with log.open('wb') as stream:
            subprocess.run(['bash', '/tests/rl_eval/run.sh', str(run.submission / 'out/final_checkpoint'), str(rl_run),
                            '--base', str(run.asset('evaluation-base-model')), '--data', '/private/midtrain-rl-data/train.jsonl',
                            '--eval-data', '/private/midtrain-eval', '--recipe', str(recipe)],
                           cwd=scratch, env=env, stdin=subprocess.DEVNULL, stdout=stream, stderr=subprocess.STDOUT,
                           check=True)
    finally:
        grade.tail(log, retained / 'rl-eval.log.tail')
        grade.tail(rl_run / 'train.log', retained / 'train.log.tail')
        for name in ('run.json', 'command.json', 'rl_checkpoint.json'):
            if (rl_run / name).is_file():
                shutil.copyfile(rl_run / name, retained / name)
        if (rl_run / 'evaluation').is_dir():
            shutil.copytree(rl_run / 'evaluation', retained / 'evaluation')
    grade.post_rl_reward(retained, read_json(recipe))


def score(scoring):
    return {'reward': midtrain_score(scoring.raw, tests=scoring.tests, private=Path('/private'),
                                     recipe='recipe-smoke.json' if scoring.smoke else 'recipe.json')}


def midtrain_score(raw, tests=Path('/tests'), private=Path('/private'), recipe='recipe.json'):
    """The post-RL score: AMC and GPQA recomputed from the sealed responses, HumanEval+ as recorded.

    HumanEval+ rewards come from the operator's code sandbox, which the verifier cannot
    reach; the trusted coordinator recorded them. Every suite score and the final reward
    must equal the recorded ones.
    """
    import hashlib
    from evaluate import EXPECTED
    from rewards import math_reward, science_reward
    tests, private = Path(tests), Path(private)
    import grade
    run, evaluation = Path(raw) / 'rl', private / 'midtrain-eval'
    sources = read_json(tests / 'rl_eval/eval_sources.json')
    manifest = read_json(evaluation / 'manifest.json')
    require(manifest.get('sources') == sources, 'Private evaluation source/revision/coverage contract mismatch')
    trained = read_json(run / 'rl_checkpoint.json')
    require(trained.get('data_sha256') == read_json(private / 'midtrain-rl-data/manifest.json')['sha256'],
            'RL did not train on the fixed mixture')
    reward, suites, _ = grade.post_rl_reward(run, read_json(tests / 'rl_eval' / recipe))
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
