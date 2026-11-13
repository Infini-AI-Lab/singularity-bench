"""Trusted evaluation coordinator in a fresh container; never receives external logs or judge keys."""
import importlib.util
import os
from pathlib import Path
import stat
import subprocess
import sys

sys.path[:0] = ['/opt/breakwater', '/tests', '/tests/rl_eval', '/opt/breakwater-drivers']
from contract import atomic_json, read_json, require

RAW = Path('/evaluation/raw')
SUBMISSION = Path('/submission')
AUTHOR_GID, SHARED_GID = 1100, 1200


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def input_roots():
    """Read-only task inputs a submitted symbolic link may point into (never private evaluation data)."""
    rows = read_json('/trusted-task/bindings.example.json')
    return [Path(row['target']) for row in rows
            if 'intermediate' in row['stages'] and not row['target'].startswith('/private/')]


def nano():
    # Reuse the audited Nano score protocol and the label-free numeric tensor wire.
    module = load('/opt/breakwater/nano_stages.py', 'nano_stages')
    module.INPUTS, module.RAW = Path('/evaluation/inputs'), RAW
    module.INPUTS.mkdir()
    module.prepare()
    # Only answer-free generation operands are copied into a root-owned, worker-readable directory.
    public = Path('/eval-inputs')
    public.mkdir(mode=0o755)
    public.chmod(0o755)
    for name in ('generation.input',):
        (public / name).write_bytes((module.INPUTS / name).read_bytes())
        (public / name).chmod(0o444)
    from nano_ppl import evaluate
    evaluate('/validation/test.bin', RAW / 'logprobs.parquet')
    with (RAW / 'generation.stdout').open('wb') as out, (RAW / 'generation.stderr').open('wb') as err:
        subprocess.run(['python3', '-I', '-B', '/opt/breakwater-runtime/run_worker.py', 'python3',
                        '/opt/breakwater-drivers/nano_driver.py', '--stage', 'generation',
                        '--input', '/eval-inputs/generation.input', '--output', '/dev/null'],
                       check=True, stdout=out, stderr=err, timeout=1800)


def midtrain():
    """The grading command (tests/grade.py): the fixed five-hour GRPO stage, then post-RL evaluation.

    Only trusted code runs here: the submitted checkpoint is read as weights by the RL
    stage. The code sandbox (RL coding rewards, HumanEval+) is the only network peer.
    """
    import shutil
    import socket
    import urllib.parse
    grade = load('/tests/grade.py', 'midtrain_grade')
    require(read_json('/private/midtrain-eval/manifest.json').get('sources') == read_json('/tests/rl_eval/eval_sources.json'),
            'Private evaluation source/revision/coverage contract mismatch')
    sandbox = urllib.parse.urlsplit(os.environ['MIDTRAIN_CODE_SANDBOX_URL'])
    require(sandbox.scheme in ('http', 'https') and sandbox.hostname, 'Invalid code sandbox URL')
    address = socket.gethostbyname(sandbox.hostname)
    port = sandbox.port or (443 if sandbox.scheme == 'https' else 80)
    for rule in (['-P', 'OUTPUT', 'DROP'], ['-A', 'OUTPUT', '-o', 'lo', '-j', 'ACCEPT'],
                 ['-A', 'OUTPUT', '-m', 'conntrack', '--ctstate', 'ESTABLISHED,RELATED', '-j', 'ACCEPT'],
                 ['-A', 'OUTPUT', '-d', address, '-p', 'tcp', '--dport', str(port), '-j', 'ACCEPT']):
        subprocess.run(['iptables', *rule], check=True)
    # IPv6 fails closed, as in cluster_cells.firewall: loopback only.
    interfaces = Path('/proc/net/if_inet6')
    if interfaces.is_file() and any(line.split()[-1] != 'lo' for line in interfaces.read_text().splitlines() if line.split()):
        for rule in (['-P', 'OUTPUT', 'DROP'], ['-A', 'OUTPUT', '-o', 'lo', '-j', 'ACCEPT']):
            subprocess.run(['ip6tables', *rule], check=True)
    scratch = Path('/tmp/midtrain-grading')  # RL checkpoints stay in this container; only receipts are retained
    scratch.mkdir(mode=0o700)
    env = grade.runtime_env(scratch)  # fresh HOME/TMPDIR/caches; CUDA_VISIBLE_DEVICES=0,1,2,3
    run, log, retained = scratch / 'run', scratch / 'rl-eval.log', RAW / 'rl'
    retained.mkdir()
    try:
        with log.open('wb') as stream:
            subprocess.run(['bash', '/tests/rl_eval/run.sh', '/app/out/final_checkpoint', str(run),
                            '--base', '/opt/base-model', '--data', '/private/midtrain-rl-data/train.jsonl',
                            '--eval-data', '/private/midtrain-eval'],
                           cwd=scratch, env=env, stdin=subprocess.DEVNULL, stdout=stream, stderr=subprocess.STDOUT,
                           check=True)
    finally:
        grade.tail(log, retained / 'rl-eval.log.tail')
        grade.tail(run / 'train.log', retained / 'train.log.tail')
        for name in ('run.json', 'command.json', 'rl_checkpoint.json'):
            if (run / name).is_file():
                shutil.copyfile(run / name, retained / name)
        if (run / 'evaluation').is_dir():
            shutil.copytree(run / 'evaluation', retained / 'evaluation')
    grade.post_rl_reward(retained, read_json('/tests/rl_eval/recipe.json'))


HOST_EVALUATION = {'scored_by': 'tasks/rl-swe-agent/evaluation (evaluate_checkpoint.py, then score.py --integrity)',
                   'checkpoint': 'out/final_checkpoint'}


def rl():
    """rl-swe-agent: no evaluation in this container, and no submitted code runs here.

    Its held-out suites run one Harbor sandbox per task, which this container cannot start.
    The trusted host evaluator scores the sealed checkpoint (tasks/rl-swe-agent/evaluation/README.md),
    and score.py combines that score with the integrity verdict of this run.
    """
    atomic_json(RAW / 'host-evaluation.json', HOST_EVALUATION)


def serving():
    """The task's grader against a fresh, writable, worker-owned copy of the sealed /app."""
    import serving_eval
    from workers import install_tree, kill_workers
    require(SUBMISSION.is_dir(), 'Serving evaluation needs submission_mount /submission for its writable /app copy')
    install_tree([(SUBMISSION, Path('/app'))], allowed=input_roots())
    os.environ['REWARD_DIR'] = str(RAW)
    serving_eval.OUTDIR = str(RAW)
    serving_eval.AGENT_CMD = 'bash /app/serve.sh'  # measure_income() starts it through run_worker.py
    serving_eval.MEASURE_STOCK = False
    try:
        serving_eval.run_grader(None, 774185263)  # no starter-server measurement
    finally:
        # Detached server children; this also unblocks accuracy threads still reading at exit.
        kill_workers(grace=10)
    report = read_json(RAW / 'diagnostics.json')
    require(not report.get('grader_error'), 'Serving evaluation failed')


def harness():
    """Validity battery: scripted model in this root process, sealed harness as the worker."""
    from harness_battery import run_battery
    rows, summary = run_battery(Path('/app/harness/agent.py'), Path('/private/tests/smoke'), RAW / 'harness')
    require(isinstance(summary, dict) and 'gradeable' in summary, 'Validity battery did not complete')


def cluster():
    """Graded control-plane cells against the operator fleet; plane as worker, driver as root."""
    from cluster_cells import run_cells
    config = read_json('/trusted-task/breakwater.json')
    run_cells(RAW / 'cluster', config.get('cluster') or {})


# Each kind's worker environment mirrors its authoring worker (author/task_config.py
# plus the image ENV). HARNESS_LLM_BASE_URL is published (as during authoring) but unreachable:
# the battery runs with no network and the harness must survive a failed utility call.
WORKER_ENV = {
    'nano': {'PATH': '/opt/venv/bin:/usr/local/cuda/bin:/usr/local/bin:/usr/bin:/bin', 'VIRTUAL_ENV': '/opt/venv',
             'HOME': '/tmp/worker', 'TMPDIR': '/tmp/worker', 'BREAKWATER_WORKDIR': '/app', 'LANG': 'C.UTF-8',
             'PYTHONUNBUFFERED': '1', 'PYTHONDONTWRITEBYTECODE': '1',
             'PYTHONPATH': '/opt/breakwater-drivers:/opt/megatron-lm',
             'HF_HUB_OFFLINE': '1', 'TRANSFORMERS_OFFLINE': '1', 'TOKENIZERS_PARALLELISM': 'false',
             'TORCH_CUDA_ARCH_LIST': '10.0+PTX'},
    'serving': {'PATH': '/usr/local/nvidia/bin:/usr/local/cuda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin',
                'HOME': '/tmp/worker', 'TMPDIR': '/tmp/worker', 'BREAKWATER_WORKDIR': '/app', 'LANG': 'C.UTF-8',
                'PYTHONUNBUFFERED': '1', 'PYTHONDONTWRITEBYTECODE': '1', 'HF_HUB_OFFLINE': '1',
                'TORCH_CUDA_ARCH_LIST': '10.0 10.3', 'HF_HOME': '/home/worker/.cache/huggingface'},
    'harness': {'PATH': '/usr/local/cuda/bin:/usr/local/bin:/usr/bin:/bin', 'HOME': '/tmp/worker',
                'TMPDIR': '/tmp/worker', 'BREAKWATER_WORKDIR': '/app', 'LANG': 'C.UTF-8', 'PYTHONUNBUFFERED': '1',
                'PYTHONDONTWRITEBYTECODE': '1', 'CUDA_HOME': '/usr/local/cuda',
                'HARNESS_LLM_BASE_URL': 'http://gateway:8000/v1', 'HARNESS_CHAT_MODEL': 'chat',
                'HARNESS_EMBED_MODEL': 'embed'},
    # The plane's HOME is the fresh /home/worker each cell installs (cluster_cells.install_app).
    'cluster': {'PATH': '/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin', 'HOME': '/home/worker',
                'TMPDIR': '/tmp/worker', 'BREAKWATER_WORKDIR': '/app', 'LANG': 'C.UTF-8', 'PYTHONUNBUFFERED': '1',
                'PYTHONDONTWRITEBYTECODE': '1'},
}
# Kinds that start no submitted worker (midtrain) or none here (rl).
GENERIC_ENV = {'PATH': '/opt/venv/bin:/usr/local/cuda/bin:/usr/local/bin:/usr/bin:/bin', 'HOME': '/tmp/worker',
               'TMPDIR': '/tmp/worker', 'PYTHONPATH': '/opt/breakwater-drivers', 'HF_HUB_OFFLINE': '1',
               'TRANSFORMERS_OFFLINE': '1', 'OMP_NUM_THREADS': '8', 'MKL_NUM_THREADS': '8',
               'PYTHONDONTWRITEBYTECODE': '1'}


def worker_groups():
    """The workspace group plus the device-node groups, exactly as authoring grants them."""
    groups = {SHARED_GID}
    for pattern in ('nvidia*', 'nvidia-caps/*', 'dri/*', 'infiniband/*'):
        for device in Path('/dev').glob(pattern):
            info = device.stat()
            if stat.S_ISCHR(info.st_mode) and info.st_gid not in (0, AUTHOR_GID):
                groups.add(info.st_gid)
    return sorted(groups)


def main():
    require(os.geteuid() == 0 and os.environ.get('BREAKWATER_STAGE') == 'intermediate', 'Trusted evaluation coordinator only')
    config = read_json('/trusted-task/breakwater.json')
    RAW.mkdir()
    # A private parent directory, not file permissions supplied by the agent, protects
    # trusted tests, labels, raw outputs and the coordinator from submitted processes.
    Path('/private').chmod(0o700)
    Path('/evaluation').chmod(0o700)
    for alias, target in {'/tests': '/private/tests', '/lambada': '/private/lambada',
                          '/validation': '/private/validation', '/opt/base-model': '/private/base-model'}.items():
        path = Path(alias)
        if path.is_symlink():
            path.unlink()
        if path.is_dir() and not any(path.iterdir()):
            path.rmdir()
        require(not path.exists(), 'Evaluation image has an unexpected private-data path: ' + alias)
        path.symlink_to(target, target_is_directory=True)
    runtime = Path('/run/breakwater')
    runtime.mkdir(exist_ok=True, parents=True)
    env = dict(WORKER_ENV.get(config['kind'], {**GENERIC_ENV, 'BREAKWATER_WORKDIR': config['workdir']}))
    if os.environ.get('LD_LIBRARY_PATH'):
        env['LD_LIBRARY_PATH'] = os.environ['LD_LIBRARY_PATH']
    (runtime / 'worker.env').write_bytes(b'\0'.join((key + '=' + value).encode() for key, value in env.items()))
    (runtime / 'worker-groups').write_text(' '.join(map(str, worker_groups())) + '\n')
    (runtime / 'finished').write_text('Fresh intermediate container; no author processes exist here.\n')
    # The coordinator already runs under the observer's `strace -f` (observed.py barrier), so
    # every worker it starts is traced from fork. A nested run_worker capture barrier would
    # SIGSTOP an already-traced process that the observer never attaches to (it only attaches
    # to untraced stopped processes) and would hang until the evaluation deadline. Workers here
    # are therefore observed by inheritance, and run_worker drops privileges without a barrier.
    Path('/tmp/worker').mkdir(mode=0o700, exist_ok=True)
    os.chown('/tmp/worker', 1101, 1200)
    {'nano': nano, 'midtrain': midtrain, 'rl': rl, 'serving': serving,
     'harness': harness, 'cluster': cluster}[config['kind']]()
    atomic_json(Path('/evaluation/complete.json'), {'complete': True, 'kind': config['kind']})
    # Publish readable, immutable raw files only after the entire evaluation has ended.
    for directory, dirs, files in os.walk('/evaluation'):
        for name in files:
            (Path(directory) / name).chmod(0o444)
        Path(directory).chmod(0o555)


if __name__ == '__main__':
    main()
