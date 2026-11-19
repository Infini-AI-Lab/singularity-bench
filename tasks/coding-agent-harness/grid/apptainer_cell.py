#!/usr/bin/env python3
"""One coding-agent-harness downstream grid cell on Breakwater's Apptainer backend (the counterpart of cell.py).

    python3 tasks/coding-agent-harness/grid/apptainer_cell.py --task TASK --driver DRIVER --harness-run ROOT_RUN \
      --harness-authority SHA --image CELL.sif --judge-dir JUDGE_DIR --run-dir RUN_DIR \
      [--driver-key-env FIREWORKS_API_KEY] [--judge-key-env FIREWORKS_API_KEY] [--gpus 0] [--wall-sec N] [--scratch DIR]

CELL.sif is the downstream task's image with the cell overlay (images/apptainer.def: installed agent CLIs and
JavaScript runtimes purged). The same three stages and evidence as the Docker cell, with the submitted harness as the
agent, which only a root harness run that passed its own verification may be:

1. Authoring: a command sandbox from CELL.sif on a writable copy of the task's workspace. The trusted cell driver
   (cell_drive.py) runs the sealed harness there as one traced broker command, started by a minimal trusted CLI in the
   author sandbox. Its only network is the driver gateway (dev_gateway.py, driver role, holding the driver key on the
   host), relayed in as driver-gateway:8003; a one-token probe through it first voids a cell whose endpoint refuses.
   The image's oracle libraries are hidden. Only the workspace change set crosses: the files the harness added or
   changed in the declared workspace, and deletions.
2. Evaluation: the task's pristine workspace receives that change set and the task's own sealed checks run in the
   coordinator (cell_intermediate.py); submitted kernels and vLLM code run in the submission sandbox through fixed
   public runners, with the oracle libraries hidden from them.
3. Verification: Breakwater's verifier, with the cell's verification task (verify/tests: the data-only scorer, the
   overlay checks and the cell rubric) and the cell record sealed as the run's task metadata.

A cell that ends without a score is void (reports/void.json), never zero.
"""
from __future__ import annotations

import argparse
import hashlib
import http.client
import json
import os
from pathlib import Path, PurePosixPath
import secrets
import shutil
import socket
import stat
import sys
import tempfile
import time
import tomllib
import traceback

HERE = Path(__file__).resolve().parent
FRAMEWORK = HERE.parents[2] / 'breakwater'  # the Breakwater framework this grid runs on
sys.path[:0] = [str(FRAMEWORK / 'apptainer'), str(FRAMEWORK), str(HERE / 'downstream'), str(HERE)]
import run as launcher  # noqa: E402  (breakwater/apptainer/run.py)
from agent_stage import AuthoringStage, image_environment  # noqa: E402
from contract import (CREDENTIAL_ENV, ContractError, RulesViolation, atomic_json, digest, read_json, require,  # noqa: E402
                      seal, verify_seal, write_canary)
from evaluation_stage import EvaluationStage  # noqa: E402
from registry import WORKDIRS, require_task, task_path  # noqa: E402
import spec  # noqa: E402
from verification import RECORDING, materialize  # noqa: E402

TASK = HERE.parent
IGNORED = ('__pycache__', '.pytest_cache')
# The grader's own oracle libraries: root-only in the Docker cell image, hidden from the harness and submitted code.
ORACLE_PATHS = ('/opt/oracle-libs', '/opt/flashinfer-cubins')
# What breakwater/author/purge.py removes; none of it may be reachable in a cell.
PURGED = ('claude', 'codex', 'kimi', 'kimi-cli', 'grok', 'mini', 'mini-extra', 'mini-swe-agent', 'aider', 'opencode',
          'gemini', 'qwen', 'cursor-agent', 'openhands', 'sweagent', 'node', 'nodejs', 'npm', 'npx', 'corepack', 'bun',
          'deno')
# The trusted CLI in the author sandbox: it starts the cell driver as one broker command and passes its streams on.
CELL_CLI = r'''
import subprocess, sys
workdir = sys.argv[1]
command = ('mkdir -p /tmp/worker && exec python3 -I -B /opt/breakwater-cell/cell_drive.py /harness '
           '/run/breakwater-cell-task/task.txt ' + workdir)
sys.exit(subprocess.call(['/opt/breakwater-runtime/breakwater-exec', command]))
'''


class CellVoid(RuntimeError):
    """The cell ends without a score (not the harness's doing)."""


def root_submission(run_dir, authority_sha):
    """Accept only a root harness run that passed its own protected verification."""
    run_dir = Path(run_dir).resolve()
    launch = verify_seal(run_dir / 'authority', authority_sha)
    identity = launch['identity']
    require(identity.get('task') == spec.load(TASK)['task'], 'Not a root run of this harness task')
    reports = run_dir / 'reports'
    require(read_json(reports / 'reward.json').get('reward') == 1.0 and not read_json(reports / 'reward.json').get('smoke'),
            'Root run is not gradeable')
    require(not (reports / 'rule-violation.json').exists(), 'Root run violated the rules')
    require(read_json(reports / 'judge.json').get('verdict') == 'PASS', 'Root integrity verdict is not PASS')
    manifest = verify_seal(run_dir / 'submission', launch['submission_sha256'], identity)
    harness = {name: value for name, value in manifest['files'].items() if name.startswith('harness/')}
    require('harness/agent.py' in harness, 'Sealed root submission has no harness/agent.py')
    return launch, harness, run_dir / 'submission' / 'harness'


def inventory(root):
    """A workspace's regular files (sha256, bytes, executable) and symbolic links, as the Docker cell records them."""
    files, links = {}, {}
    root = Path(root)
    for directory, dirs, names in os.walk(root):
        for name in sorted([*dirs, *names]):
            path = Path(directory) / name
            relative = path.relative_to(root).as_posix()
            info = path.lstat()
            if stat.S_ISLNK(info.st_mode):
                links[relative] = os.readlink(path)
                if name in dirs:
                    dirs.remove(name)
            elif stat.S_ISREG(info.st_mode):
                files[relative] = {'sha256': launcher.file_sha256(path), 'bytes': info.st_size,
                                   'exec': bool(info.st_mode & 0o111)}
    return {'files': files, 'symlinks': links}


def kept(name):
    return not any(part in IGNORED for part in PurePosixPath(name).parts) and not name.endswith(('.pyc', '.pyo'))


def overlay(workspace, baseline, submission, external):
    """The change set that crosses to evaluation (submission/changes.json and submission/files), as in Docker."""
    final = inventory(workspace)
    before, after = baseline['files'], final['files']
    changes = {'workdir': None,
               'added': sorted(name for name in after if name not in before and kept(name)),
               'modified': sorted(name for name in after if name in before and before[name]['sha256'] != after[name]['sha256']
                                  and kept(name)),
               'deleted': sorted(name for name in before if name not in after),
               'executable': sorted(name for name in after if after[name]['exec'] and kept(name)
                                    and (name not in before or before[name]['sha256'] != after[name]['sha256']))}
    (submission / 'files').mkdir(parents=True, mode=0o700)
    for name in changes['added'] + changes['modified']:
        target = submission / 'files' / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(Path(workspace) / name, target)
    atomic_json(external / 'task' / 'cell-workspace-final.json', {
        'files_sha256': hashlib.sha256(json.dumps(after, sort_keys=True).encode()).hexdigest(), 'files': after,
        'ignored_cache_changes': sorted(name for name in after if not kept(name) and before.get(name) != after[name]),
        'symlinks_added_or_changed': {name: target for name, target in final['symlinks'].items()
                                      if baseline['symlinks'].get(name) != target},
        'symlinks_removed': sorted(set(baseline['symlinks']) - set(final['symlinks']))})
    return changes


def unix_post(socket_path, path, payload, timeout=180):
    """POST to a host service on its unix socket (the driver gateway); the HTTP status, 0 on a transport failure."""
    class Connection(http.client.HTTPConnection):
        def connect(self):
            self.sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            self.sock.settimeout(timeout)
            self.sock.connect(str(socket_path))
    connection = Connection('driver-gateway', timeout=timeout)
    try:
        connection.request('POST', path, body=json.dumps(payload), headers={'Content-Type': 'application/json'})
        return connection.getresponse().status
    except OSError:
        return 0
    finally:
        connection.close()


def inspect_image(image):
    """The cell image must carry the overlay: no agent CLI or JavaScript runtime the harness could hand the task to.
    Returns whether it has the overlay's moved Hugging Face cache."""
    from sandbox import Sandbox
    box = Sandbox(image, role='probe')
    try:
        code, out, _ = box.run(['sh', '-c', 'for c in ' + ' '.join(PURGED) + '; do command -v "$c"; done; '
                                'if [ -d /opt/breakwater-hf-cache ]; then echo HF-CACHE; fi; true'], timeout=300)
    finally:
        box.stop()
    found = [line for line in out.splitlines() if line.strip() and line.strip() != 'HF-CACHE']
    require(code == 0 and not found, 'The cell image still provides agent CLIs or JavaScript runtimes ('
            + ', '.join(found) + '): build it with tasks/coding-agent-harness/grid/images/apptainer.def')
    return 'HF-CACHE' in out.split()


def cell_environment(image_env, workdir, gpus, hf_cache, authoring):
    """The harness's environment (authoring, as cell_setup.py writes it) or the submitted code's (evaluation, as
    cell_intermediate.worker_environment): no key, the driver gateway only."""
    drop = (('BREAKWATER_STAGE',) if authoring else ('BREAKWATER_',)) + ('FIREWORKS', 'OPENAI', 'ANTHROPIC', 'UPSTREAM',
                                                                         'HARBOR_', 'GATEWAY_')
    env = {key: value for key, value in image_env.items() if not key.startswith(drop)}
    env.update(HOME='/tmp/worker', TMPDIR='/tmp/worker', BREAKWATER_WORKDIR=workdir, PYTHONDONTWRITEBYTECODE='1',
               PYTHONUNBUFFERED='1', TRITON_CACHE_DIR='/tmp/worker/.triton',
               TORCH_EXTENSIONS_DIR='/tmp/worker/torch_extensions', HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1')
    if authoring:
        env['HARNESS_RECORD'] = '/run/breakwater-cell/record/eval_record.json'
    if hf_cache:
        env['HF_HOME'] = '/opt/breakwater-hf-cache'
    env['PATH'] = ':'.join(part for part in env.get('PATH', '/usr/local/bin:/usr/bin:/bin').split(':')
                           if part and not part.startswith(('/root', '/home/')))
    if gpus:
        env['CUDA_VISIBLE_DEVICES'] = gpus
    return env


def main():
    static = read_json(HERE / 'cell-pipeline.json')
    drivers = read_json(TASK / 'eval' / 'config.json')
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--task', required=True)
    parser.add_argument('--driver', required=True, choices=sorted(drivers['drivers']))
    parser.add_argument('--harness-run', type=Path, required=True)
    parser.add_argument('--harness-authority', required=True, help='authority_sha256 printed by the root run')
    parser.add_argument('--image', type=Path, required=True, help='the cell image: the task image with the cell overlay')
    parser.add_argument('--judge-dir', type=Path, required=True)
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--driver-key-env', default='FIREWORKS_API_KEY')
    parser.add_argument('--judge-key-env', default='FIREWORKS_API_KEY')
    parser.add_argument('--gpus', help="CUDA_VISIBLE_DEVICES for the cell (the grid gives each cell its own); by default "
                        "the launcher's own, as Slurm sets it for a job's GPUs")
    parser.add_argument('--wall-sec', type=float, help='cost control; defaults to the task agent.timeout_sec')
    parser.add_argument('--scratch', type=Path)
    args = parser.parse_args()
    if args.gpus is None:
        args.gpus = os.environ.get('CUDA_VISIBLE_DEVICES', '')
    keys = {name: os.environ[name] for name in {*CREDENTIAL_ENV, args.driver_key_env, args.judge_key_env}
            if name in os.environ}
    for name in keys:
        os.environ.pop(name)
    require(keys.get(args.driver_key_env) and keys.get(args.judge_key_env), 'Missing the driver or judge key')
    task = require_task(args.task)
    workdir = WORKDIRS[task]
    source = TASK / 'eval' / 'e2e_tasks' / task_path(task)
    task_config = tomllib.loads((source / 'task.toml').read_text())
    wall = float(args.wall_sec or task_config['agent']['timeout_sec'])
    evaluation_timeout = float(task_config['verifier']['timeout_sec'])
    driver = drivers['drivers'][args.driver]
    scratch = Path(os.path.realpath(args.scratch or launcher.SCRATCH))
    launch_root, harness_files, harness_source = root_submission(args.harness_run, args.harness_authority)
    run = Path(os.path.realpath(args.run_dir))
    overlaps = launcher.contract_tools.overlaps
    require(not run.exists() and not overlaps(run, TASK) and not overlaps(run, Path(args.harness_run).resolve()),
            'Use a fresh run directory outside the task and the root run')
    run.mkdir(parents=True, mode=0o700)
    external, authority, local, submission, evaluation, reports = (
        run / name for name in ('external', 'authority', 'agent-local', 'submission', 'evaluation', 'reports'))
    for folder in (external, authority, local, evaluation, reports):
        folder.mkdir(mode=0o700)
    identity = {'task': static['task'], 'run_id': secrets.token_hex(16)}
    stages, cli = {}, None
    live, pristine = run / 'workspace-live', run / 'workspace-pristine'
    try:
        hf_cache = inspect_image(args.image)
        # The sealed root harness, copied and re-verified byte for byte.
        harness = run / 'harness'
        for name, value in harness_files.items():
            target = run / name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(harness_source.parent / name, target)
            require(digest(target) == value['sha256'], 'Root harness changed after sealing')
        (run / 'task.txt').write_text((source / 'instruction.md').read_text())
        cell = {'task': task, 'workdir': workdir, 'driver': args.driver, 'driver_model': driver['model'],
                'embed_model': drivers['embed_model'], 'wall_sec': wall, 'evaluation_timeout_sec': evaluation_timeout,
                'root_run_id': launch_root['identity']['run_id'], 'root_authority_sha256': args.harness_authority,
                'root_submission_sha256': launch_root['submission_sha256'],
                'harness_inventory_sha256': hashlib.sha256(json.dumps(harness_files, sort_keys=True).encode()).hexdigest(),
                'harness_files': harness_files}
        private = run / 'cell-private'
        private.mkdir(mode=0o700)
        atomic_json(private / 'cell.json', {'task': task, 'workdir': workdir})
        # The cell's contract for the coordinator and the verifier: graded at its wall clock, but its trusted driver
        # must exit cleanly before that; the driver key never enters a sandbox.
        contract = run / 'contract.json'
        atomic_json(contract, {**spec.DEFAULTS, **static, 'workdir': workdir, 'submission_root': '/submission',
                               'submission_mount': '/submission', 'agent_timeout_sec': wall,
                               'evaluation_timeout_sec': evaluation_timeout, 'credential_env': [args.driver_key_env]})
        contract.chmod(0o444)
        write_canary(external)
        (external / 'task').mkdir(mode=0o700)
        # The overlay carries regular files only: no symbolic links cross to evaluation.
        atomic_json(external / 'workspace-links.json', {'symlinks': {}})
        # The task's own workspace from the image: one copy for the harness, one pristine for evaluation.
        launcher.seed({'workspace_source': 'image', 'workdir': workdir}, source, args.image, live, scratch)
        shutil.copytree(live, pristine, symlinks=True)
        baseline = inventory(live)
        atomic_json(external / 'task' / 'cell-workspace-baseline.json', baseline)
        image_sha = launcher.file_sha256(args.image)
        baseline_dir = run / 'environment-baseline'
        baseline_dir.mkdir(mode=0o700)
        # As the Docker cell: no shadow policy, so the verifier's check of the overlay must find none.
        from environment_guard import shadow_policy
        policy = shadow_policy({'packages': [], 'hooks': [], 'environments': False})
        atomic_json(baseline_dir / 'environment.json', {'version': 'breakwater-apptainer-v1', 'identity': identity,
                                                        'image_sha256': image_sha, 'specification': {'shadow': policy},
                                                        'workspace_shadows': {}})
        (baseline_dir / 'environment.json').chmod(0o444)
        shutil.copyfile(baseline_dir / 'environment.json', external / 'environment-baseline.json')
        image_env = image_environment(args.image)
        hidden = run / 'hidden'
        hidden.mkdir(mode=0o500)
        hide = [(hidden, path, 'ro') for path in ORACLE_PATHS]
        record_dir, cli = run / 'cell-record', Path(tempfile.mkdtemp(prefix='breakwater-cell-cli-'))
        record_dir.mkdir(mode=0o700)
        (cli / 'cell_cli.py').write_text(CELL_CLI)
        folder = external / 'driver-gateway'
        gateway = {'name': 'driver-gateway', 'port': 8003, 'host': 'driver-gateway',
                   'argv': [sys.executable, '-I', str(FRAMEWORK / 'dev_gateway.py')],
                   'env': {'PATH': os.environ.get('PATH', '/usr/bin:/bin'), 'UPSTREAM_API_KEY': keys[args.driver_key_env],
                           'GATEWAY_ROLE': 'driver', 'GATEWAY_SOURCE': 'driver-gateway',
                           'UPSTREAM_BASE_URL': drivers['api_base'], 'DRIVER_MODEL': driver['model'],
                           'EMBED_MODEL': drivers['embed_model'], 'GATEWAY_LOG': str(folder / 'requests.jsonl'),
                           'GATEWAY_STATE': str(folder / 'budget.json'),
                           **{name: os.environ[name] for name in launcher.PROXY_ENV if name in os.environ}}}

        def probe(sockets):
            # One token through the journaled gateway: a deterministic rejection voids the cell.
            status = unix_post(sockets['driver-gateway'], '/driver/v1/chat/completions',
                               {'model': 'driver', 'max_tokens': 1, 'messages': [{'role': 'user', 'content': 'ok'}]})
            cell['driver_probe_status'] = status
            if 400 <= status < 500 and status not in (408, 429):
                raise CellVoid(f'driver endpoint rejected the probe: {status}')

        stage = AuthoringStage(
            image=args.image, workspace=live, workdir=workdir, external=external, local=local, cli_dir=cli,
            cli_argv=['python3', '-I', '/opt/author-cli/cell_cli.py', workdir],
            cli_env={'PATH': '/usr/local/bin:/usr/bin:/bin', 'HOME': '/home/author'}, gateway=None,
            budget=wall + static['wall_grace_sec'], gpus=True, scratch=scratch, native='stdout', on_services=probe,
            command_env=cell_environment(image_env, workdir, args.gpus, hf_cache, authoring=True) | {
                'BREAKWATER_DRIVER_BASE': 'http://driver-gateway:8003', 'HARNESS_WALL_SEC': str(wall),
                'HARNESS_SESSIONS': '1'},
            services=[gateway],
            agent_binds=[(harness, '/harness', 'ro'), (run / 'task.txt', '/run/breakwater-cell-task/task.txt', 'ro'),
                         (HERE / 'cell_drive.py', '/opt/breakwater-cell/cell_drive.py', 'ro'),
                         (record_dir, '/run/breakwater-cell/record', 'rw'),
                         (baseline_dir, '/opt/environment-baseline', 'ro'), *hide])
        record = stages['agent'] = stage.run()
        timed_out = record['deadline_reached']
        require(timed_out or record['exit_code'] == 0, 'Cell driver failed; retaining evidence without a score')
        after = launcher.file_sha256(args.image)
        integrity = {'baseline_sha256': digest(external / 'environment-baseline.json'), 'image_sha256_before': image_sha,
                     'image_sha256_after': after, 'workspace_shadow_check': 'PASS',
                     'status': 'PASS' if after == image_sha and record['process_wall'] == {'author': True, 'command': True}
                     else 'FAIL'}
        atomic_json(external / 'runtime-integrity.json', integrity)
        if integrity['status'] != 'PASS':
            raise RulesViolation('Image integrity or process-wall check failed')
        changes = overlay(live, baseline, submission, external)
        changes['workdir'] = workdir
        atomic_json(submission / 'changes.json', changes)
        for path in sorted(record_dir.iterdir()):
            if path.is_file() and not path.is_symlink():
                shutil.copyfile(path, local / ('drive-' + path.name))
        submission_sha = seal(submission, identity)
        local_sha = seal(local, identity)
        # Evaluation: the pristine workspace plus the change set; the task's sealed checks; submitted code apart.
        worker_env = cell_environment(image_env, workdir, args.gpus, hf_cache, authoring=False)
        coordinator_env = {**image_env, 'HOME': '/tmp', **({'CUDA_VISIBLE_DEVICES': args.gpus} if args.gpus else {})}
        evaluation_record = stages['intermediate'] = EvaluationStage(
            image=args.image, task=source, contract=contract, submission=submission, workdir=workdir,
            evaluation=evaluation, external=external, worker_env=worker_env, coordinator_env=coordinator_env,
            timeout=evaluation_timeout, gpus=True, scratch=scratch, submission_mount='/submission',
            workdir_seed=pristine, shared=['/run/breakwater-cell'],
            private_binds=[(FRAMEWORK, '/private/breakwater', 'ro'), (HERE, '/private/breakwater-cell', 'ro'),
                           (private, '/private/cell', 'ro')],
            public_binds=[(HERE / 'worker', '/opt/breakwater-worker', 'ro'),
                          (baseline_dir, '/opt/environment-baseline', 'ro')],
            submission_binds=hide,
            coordinator_argv=['python3', '-I', '-B', '/private/breakwater-cell/cell_intermediate.py']).run()
        evaluation_record['image_sha256_after'] = launcher.file_sha256(args.image)
        require(evaluation_record['image_sha256_after'] == image_sha, 'The image changed during the evaluation stage')
        require(evaluation_record['exit_code'] == 0 and not evaluation_record['timed_out'], 'Cell evaluation failed')
        require(read_json(evaluation / 'complete.json').get('complete') is True, 'Missing cell evaluation completion')
        evaluation.chmod(0o700)
        evaluation_sha = seal(evaluation, identity)
        atomic_json(external / 'task' / 'cell.json', cell)
        external_sha = seal(external, identity, finalized=True, stage_order=['agent', 'intermediate'])
        for root in (external, local, evaluation, submission):
            launcher.read_only(root)
        code, authority_sha = launcher.verify(
            run=run, task=materialize(run, source), config=read_json(contract), contract=contract, image=args.image,
            image_env=image_env,
            bindings=[], judge_dir=args.judge_dir, judge_key=keys[args.judge_key_env], scratch=scratch,
            identity=identity, stages=stages, authority=authority, reports=reports, logs=run / 'verifier',
            seals={'external_sha256': external_sha, 'evaluation_sha256': evaluation_sha,
                   'submission_sha256': submission_sha, 'local_log_sha256': local_sha, 'image_sha256': image_sha},
            extra={'recording': RECORDING,
                   'task_metadata': {key: value for key, value in cell.items() if key != 'harness_files'}})
        print(json.dumps({'task': task, 'driver': args.driver, 'run_id': identity['run_id'], 'verifier_exit_code': code,
                          'wall_clock_exhausted': timed_out, 'reports': str(reports), 'authority_sha256': authority_sha}))
        return code
    except BaseException as error:
        print(traceback.format_exc().rstrip(), file=sys.stderr)
        violation = isinstance(error, RulesViolation)
        if isinstance(error, CellVoid):
            atomic_json(reports / 'void.json', {'status': 'VOID', 'reason': str(error)})
        if violation:
            reward = static.get('rule_violation_reward', 0.0)
            atomic_json(reports / 'reward.json', {'reward': reward})
            atomic_json(reports / 'rule-violation.json', {'reward': reward, 'reason': 'violation of rules',
                                                          'detail': str(error), 'judge_skipped': True})
        atomic_json(reports / 'pipeline-error.json', {'status': 'RULE_VIOLATION' if violation else 'VOID',
                                                      'error': f'{type(error).__name__}: {error}',
                                                      'completed_stages': list(stages)})
        print(json.dumps({'task': task, 'driver': args.driver, 'run_id': identity['run_id'],
                          'status': 'RULE_VIOLATION' if violation else 'VOID', 'error': f'{type(error).__name__}: {error}',
                          'reports': str(reports)}))
        return 2
    finally:
        # The workspace copies are not evidence (the change set is), and a vLLM checkout is large.
        for path in (live, pristine, cli):
            if path is not None:
                shutil.rmtree(path, ignore_errors=True)


if __name__ == '__main__':
    sys.exit(main())
