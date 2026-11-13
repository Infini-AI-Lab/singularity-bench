#!/usr/bin/env python3
"""Trusted host controller for ONE protected coding-agent-harness downstream grid cell.

The same three-container contract as run.py, with the submitted harness as the agent:

1. Cell authoring container: the pristine downstream task image plus a small runtime
   overlay (worker account, run_worker.py, stdio supervisor; installed agent CLIs and JS
   runtimes purged). The sealed root submission is mounted read-only at /harness and
   driven by the trusted cell_drive.py as the worker (UID 1101) behind an external
   observer. Its only network is an internal Docker network whose sole peer is the
   keyless-to-the-cell driver gateway, which journals every request/response.
2. Intermediate: a fresh container from the same image. Only the sealed workspace
   overlay crosses. The task's own sealed verifier logic runs as root; the submitted
   kernel or vLLM code runs as the worker through fixed public runners.
3. Verifier: read-only, rootless, no GPU; mechanical reconciliation, data-only scoring
   and the Codex/DeepSeek checklist over indexed evidence.

Only a root harness run that passed its own three-stage verification (gradeable reward
1.0 and a PASS judge verdict) can be graded. Cells that end without a score are voids.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import secrets
import shutil
import subprocess
import sys
import tarfile
import time
import tomllib

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent / 'downstream'))
from contract import (RulesViolation, atomic_json, digest, inspect_boundary, inventory, overlaps, read_json, require,
                      seal, verify_seal)
import run as controller
from registry import KERNELS, PRS, WORKDIRS, require_task, task_path

HERE = Path(__file__).resolve().parent
TASK = HERE.parent / 'tasks' / 'coding-agent-harness'
IGNORED = ('__pycache__', '.pytest_cache')


def tar_inventory(container, workdir, extract_to=None, baseline=None):
    """Hash a container workspace from Docker's archive stream (no target code executes).

    -> {'files': {rel: {sha256, bytes, exec}}, 'symlinks': {rel: target}}. With
    ``extract_to``, regular files that differ from ``baseline`` (and are not bytecode or
    test caches) are streamed there; nothing is buffered whole in memory.
    """
    files, links = {}, {}
    staging = None
    if extract_to is not None:
        staging = Path(extract_to).parent.parent / '.overlay-staging'
        staging.mkdir(mode=0o700)
    process = subprocess.Popen(['docker', 'cp', container + ':' + workdir.rstrip('/') + '/.', '-'],
                               stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    try:
        with tarfile.open(fileobj=process.stdout, mode='r|*') as archive:
            for member in archive:
                parts = PurePosixPath(member.name).parts
                require(not member.name.startswith('/') and '..' not in parts, 'Unsafe workspace archive path')
                relative = str(PurePosixPath(*[part for part in parts if part != '.'])) if parts else ''
                if not relative or relative == '.':
                    continue
                if member.issym() or member.islnk():
                    links[relative] = member.linkname
                    continue
                if not member.isfile():
                    continue
                stream = archive.extractfile(member)
                value = hashlib.sha256()
                temporary = staging / 'pending' if staging is not None else None
                output = temporary.open('xb') if temporary is not None else None
                try:
                    for chunk in iter(lambda: stream.read(8 * 1024**2), b''):
                        value.update(chunk)
                        if output is not None:
                            output.write(chunk)
                finally:
                    if output is not None:
                        output.close()
                record = {'sha256': value.hexdigest(), 'bytes': member.size, 'exec': bool(member.mode & 0o111)}
                files[relative] = record
                if temporary is None:
                    continue
                keep = ((baseline or {}).get(relative, {}).get('sha256') != record['sha256']
                        and not any(part in IGNORED for part in PurePosixPath(relative).parts)
                        and not relative.endswith(('.pyc', '.pyo')))
                if keep:
                    target = Path(extract_to) / relative
                    target.parent.mkdir(parents=True, exist_ok=True)
                    temporary.rename(target)
                else:
                    temporary.unlink()
        require(process.wait(timeout=60) == 0, 'Workspace archive failed')
    finally:
        if process.poll() is None:
            process.kill()
            process.wait()
        process.stdout.close()
        process.stderr.close()
        if staging is not None:
            shutil.rmtree(staging, ignore_errors=True)
    return {'files': files, 'symlinks': links}


def root_submission(run_dir, authority_sha):
    """Accept only a root harness run that passed its own protected verification."""
    run_dir = Path(run_dir).resolve()
    launch = verify_seal(run_dir / 'authority', authority_sha)
    identity = launch['identity']
    require(identity.get('task') == controller.load_config(TASK / 'breakwater.json')['task'], 'Not a root run of this harness task')
    reports = run_dir / 'reports'
    require(read_json(reports / 'reward.json') == {'reward': 1.0}, 'Root run is not gradeable')
    require(not (reports / 'rule-violation.json').exists(), 'Root run violated the rules')
    require(read_json(reports / 'judge.json').get('verdict') == 'PASS', 'Root integrity verdict is not PASS')
    manifest = verify_seal(run_dir / 'submission', launch['submission_sha256'], identity)
    harness = {name: value for name, value in manifest['files'].items() if name.startswith('harness/')}
    require('harness/agent.py' in harness, 'Sealed root submission has no harness/agent.py')
    return launch, harness, run_dir / 'submission' / 'harness'


def main():
    config = read_json(HERE / 'cell-pipeline.json')
    drivers = read_json(TASK / 'eval' / 'config.json')
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--task', required=True)
    parser.add_argument('--driver', required=True, choices=sorted(drivers['drivers']))
    parser.add_argument('--harness-run', type=Path, required=True)
    parser.add_argument('--harness-authority', required=True, help='authority_sha256 printed by the root run')
    parser.add_argument('--cell-image', required=True)
    parser.add_argument('--verifier-image', required=True)
    parser.add_argument('--observer-image', required=True)
    parser.add_argument('--gateway-image', required=True, help='breakwater-dev-gateway image (driver role)')
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--gpus', default='device=0')
    parser.add_argument('--driver-key-env', default='FIREWORKS_API_KEY')
    parser.add_argument('--wall-sec', type=float, help='Cost control; defaults to the task agent.timeout_sec')
    args = parser.parse_args()
    task = require_task(args.task)
    workdir = WORKDIRS[task]
    source = TASK / 'eval' / 'e2e_tasks' / task_path(task)
    task_config = tomllib.loads((source / 'task.toml').read_text())
    wall = float(args.wall_sec or task_config['agent']['timeout_sec'])
    evaluation_timeout = float(task_config['verifier']['timeout_sec'])
    driver = drivers['drivers'][args.driver]
    gpu_options = controller.gpu_devices(args.gpus, config['gpus'])
    require(os.environ.get(args.driver_key_env) and os.environ.get('FIREWORKS_API_KEY'), 'Missing driver/judge credentials')
    launch_root, harness_files, harness_source = root_submission(args.harness_run, args.harness_authority)
    run = args.run_dir.resolve()
    require(not run.exists() and not overlaps(run, TASK) and not overlaps(run, args.harness_run), 'Use a fresh run directory')
    run.mkdir(parents=True, mode=0o700)
    external, authority, local, submission, evaluation, reports = [run / key for key in
        ('external', 'authority', 'agent-local', 'submission', 'evaluation', 'reports')]
    local_live, baseline_dir, cell_private = run / 'local-live', run / 'environment-baseline', run / 'cell-private'
    for folder in (external, authority, local_live, evaluation, reports, baseline_dir, cell_private):
        folder.mkdir(mode=0o700)
    for folder in (local_live, evaluation, reports):
        folder.chmod(0o777)
    # The sealed root harness, copied and re-verified byte for byte.
    harness = run / 'harness'
    for name, value in harness_files.items():
        target = run / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(harness_source.parent / name, target)
        require(digest(target) == value['sha256'], 'Root harness changed after sealing')
    for directory, dirs, files in os.walk(harness):
        Path(directory).chmod(0o755)
        for name in files:
            (Path(directory) / name).chmod(0o644)
    (run / 'task.txt').write_text((source / 'instruction.md').read_text())
    (run / 'task.txt').chmod(0o644)
    identity = {'task': config['task'], 'run_id': secrets.token_hex(16)}
    cell = {'task': task, 'workdir': workdir, 'driver': args.driver, 'driver_model': driver['model'],
            'embed_model': drivers['embed_model'], 'wall_sec': wall, 'evaluation_timeout_sec': evaluation_timeout,
            'root_run_id': launch_root['identity']['run_id'], 'root_authority_sha256': args.harness_authority,
            'root_submission_sha256': launch_root['submission_sha256'],
            'harness_inventory_sha256': hashlib.sha256(json.dumps(harness_files, sort_keys=True).encode()).hexdigest(),
            'harness_files': harness_files}
    atomic_json(cell_private / 'cell.json', {'task': task, 'workdir': workdir})
    (external / 'isolation.canary').write_text(secrets.token_hex(32) + '\n')
    names = {stage: 'breakwater-' + identity['run_id'][:12] + '-cell-' + stage for stage in
             ('agent', 'intermediate', 'verifier', 'gateway', 'observe-agent', 'observe-intermediate')}
    network = 'breakwater-' + identity['run_id'][:12] + '-cell'
    ids = {'agent': controller.image(args.cell_image), 'verifier': controller.image(args.verifier_image),
           'observer': controller.image(args.observer_image), 'gateway': controller.image(args.gateway_image)}
    stages, created = {}, []
    protected = [external, authority, reports]
    baseline_dir.chmod(0o755)
    caps = ['--cap-drop=ALL', '--cap-add=SETUID', '--cap-add=SETGID', '--cap-add=CHOWN', '--cap-add=FOWNER',
            '--cap-add=DAC_OVERRIDE', '--cap-add=KILL', '--security-opt=no-new-privileges:true']
    timed_out = None
    try:
        controller.docker('network', 'create', '--internal', network)
        env = {**os.environ, 'UPSTREAM_API_KEY': os.environ[args.driver_key_env]}
        controller.service_gateway(names['gateway'], network, 'driver-gateway', external / 'driver-gateway',
                                   ids['gateway'], ['UPSTREAM_API_KEY', 'GATEWAY_ROLE=driver', 'GATEWAY_SOURCE=driver-gateway',
                                                    'UPSTREAM_BASE_URL=' + drivers['api_base'],
                                                    'DRIVER_MODEL=' + driver['model'],
                                                    'EMBED_MODEL=' + drivers['embed_model']], env, created, 8003)
        # One-token probe through the gateway: a deterministic rejection voids the cell.
        probe = subprocess.run(['docker', 'exec', names['gateway'], 'python3', '-I', '-B', '-c',
            "import json,urllib.request,urllib.error\n"
            "r=urllib.request.Request('http://127.0.0.1:8003/driver/v1/chat/completions',data=json.dumps("
            "{'model':'driver','max_tokens':1,'messages':[{'role':'user','content':'ok'}]}).encode(),"
            "headers={'Content-Type':'application/json'})\n"
            "try:\n print(urllib.request.urlopen(r,timeout=180).status)\nexcept urllib.error.HTTPError as e:\n print(e.code)\n"
            "except Exception as e:\n print(0)"], capture_output=True, text=True)
        status = int((probe.stdout or '0').strip().splitlines()[-1] or 0)
        cell['driver_probe_status'] = status
        if 400 <= status < 500 and status not in (408, 429):
            atomic_json(reports / 'cell-void.json', {'status': 'VOID', 'reason': f'driver endpoint rejected the probe: {status}'})
            raise RuntimeError(f'Driver endpoint rejected the probe ({status}); the cell is void')
        agent = names['agent']
        controller.docker('create', '--name', agent, '--label', 'breakwater.stage=cell-agent', '--ipc=private',
            '--network', network, *gpu_options, '--shm-size=' + config['shm_size'], *caps,
            '-e', 'PYTHONDONTWRITEBYTECODE=1', '-e', 'BREAKWATER_DRIVER_BASE=http://driver-gateway:8003',
            '-e', f'HARNESS_WALL_SEC={wall}', '-e', 'HARNESS_SESSIONS=1',
            *controller.mount(harness, '/harness'), *controller.mount(run / 'task.txt', '/run/breakwater-cell-task/task.txt'),
            *controller.mount(local_live, '/logs/agent', False), *controller.mount(baseline_dir, '/opt/environment-baseline'),
            # Only the two public cell files: private expectations and adapters never enter this container.
            *controller.mount(HERE / 'cell_drive.py', '/opt/breakwater-cell/cell_drive.py'),
            *controller.mount(HERE / 'cell_setup.py', '/opt/breakwater-cell/cell_setup.py'),
            '--entrypoint', 'sleep', ids['agent'], 'infinity')
        created.append(agent)
        inspect_boundary(controller.inspection(agent), 'agent', external, protected, gpu_count=config['gpus'])
        controller.docker('start', agent)
        controller.docker('exec', '--user=0', agent, 'python3', '-I', '-B', '/opt/breakwater-cell/cell_setup.py',
                          '--workdir', workdir)
        atomic_json(external / 'gpu-allocation.json', controller.check_gpus(agent, config))
        from environment_guard import PACKAGES, VERSION, describe, scan, shadow_policy, shadows
        policy = shadow_policy({'packages': [], 'hooks': [], 'environments': False})
        specification = describe(agent, workdir, PACKAGES, policy)
        baseline = {'version': VERSION, 'identity': identity, 'image_id': ids['agent'], 'specification': specification,
                    'files': scan(agent, specification), 'workspace_shadows': {}}
        atomic_json(baseline_dir / 'environment.json', baseline)
        (baseline_dir / 'environment.json').chmod(0o444)
        baseline_dir.chmod(0o555)
        shutil.copyfile(baseline_dir / 'environment.json', external / 'environment-baseline.json')
        baseline_sha = digest(baseline_dir / 'environment.json')
        workspace_baseline = tar_inventory(agent, workdir)
        atomic_json(external / 'cell-workspace-baseline.json', workspace_baseline)
        probe_report = controller.boundary_probe(agent, external, 'agent')
        controller.observer(names['observe-agent'], agent, external / 'agent', ids['observer'], {**identity, 'stage': 'agent'},
                            source_roots=[workdir])
        created.append(names['observe-agent'])
        stages['agent'] = {'inspection': controller.inspection(agent), 'probe': probe_report,
                           'started_ns': time.monotonic_ns(), 'started_wall_ns': time.time_ns()}
        code = controller.captured_agent(agent, ['python3', '-I', '-B', '/opt/breakwater-runtime/run_worker.py', 'python3',
                                                 '-I', '-B', '/opt/breakwater-cell/cell_drive.py', '/harness',
                                                 '/run/breakwater-cell-task/task.txt', workdir],
                                         external, wall + config['wall_grace_sec'],
                                         channels=(('stdout', 'native.stdout'), ('stderr', 'native.stderr')),
                                         user='0:0', on_deadline='stop')
        timed_out = code is None
        controller.docker('stop', '--time', '10', agent)
        stages['agent'].update(stopped_ns=time.monotonic_ns(), stopped_wall_ns=time.time_ns(),
                               exit_code=0 if timed_out else code, wall_clock_exhausted=timed_out)
        stages['agent']['process_drain'] = controller.no_lingering_processes(agent, names['observe-agent'], external / 'agent')
        runtime_agent = controller.runtime_check(agent, baseline, external / 'environment-after-agent.json')
        controller.stop_observer(names['observe-agent'], external / 'agent')
        controller.docker('stop', '--time', '30', names['gateway'])
        require(timed_out or code == 0, 'Cell driver failed; retaining evidence without a score')
        # Only workspace changes cross: the overlay is computed by the controller from Docker's archive.
        (submission / 'files').mkdir(parents=True, mode=0o700)
        final = tar_inventory(agent, workdir, extract_to=submission / 'files', baseline=workspace_baseline['files'])
        before, after = workspace_baseline['files'], final['files']
        def kept(name):
            return not any(part in IGNORED for part in PurePosixPath(name).parts) and not name.endswith(('.pyc', '.pyo'))
        changes = {'workdir': workdir,
                   'added': sorted(name for name in after if name not in before and kept(name)),
                   'modified': sorted(name for name in after if name in before and before[name]['sha256'] != after[name]['sha256']
                                      and kept(name)),
                   'deleted': sorted(name for name in before if name not in after),
                   'executable': sorted(name for name in after if after[name]['exec'] and kept(name)
                                        and (name not in before or before[name]['sha256'] != after[name]['sha256']))}
        atomic_json(submission / 'changes.json', changes)
        atomic_json(external / 'cell-workspace-final.json', {
            'files_sha256': hashlib.sha256(json.dumps(after, sort_keys=True).encode()).hexdigest(), 'files': after,
            'ignored_cache_changes': sorted(name for name in after if not kept(name) and before.get(name) != after[name]),
            'symlinks_added_or_changed': {name: target for name, target in final['symlinks'].items()
                                          if workspace_baseline['symlinks'].get(name) != target},
            'symlinks_removed': sorted(set(workspace_baseline['symlinks']) - set(final['symlinks']))})
        controller.snapshot(agent, '/logs/agent', local, [])
        record_dir = run / 'agent-record'
        controller.snapshot(agent, '/run/breakwater-cell/record', record_dir, [])
        for path in record_dir.iterdir():
            shutil.copyfile(path, local / ('drive-' + path.name))
        submission_sha = seal(submission, identity)
        local_sha = seal(local, identity)
        for directory, dirs, files in os.walk(submission):
            Path(directory).chmod(0o555)
            for name in files:
                (Path(directory) / name).chmod(0o444)
        intermediate = names['intermediate']
        controller.docker('create', '--name', intermediate, '--label', 'breakwater.stage=cell-intermediate', '--ipc=private',
            '--network=none', *gpu_options, '--shm-size=' + config['shm_size'], *caps,
            '-e', 'BREAKWATER_STAGE=intermediate', '-e', 'PYTHONDONTWRITEBYTECODE=1',
            *controller.mount(baseline_dir, '/opt/environment-baseline'), *controller.mount(submission, '/submission'),
            *controller.mount(evaluation, '/evaluation', False), *controller.mount(source / 'tests', '/private/tests'),
            # Trusted coordinator code is reachable only through the root-only /private parent;
            # submitted processes can read the public worker runners and nothing else of it.
            *controller.mount(HERE, '/private/breakwater'), *controller.mount(HERE / 'worker', '/opt/breakwater-worker'),
            *controller.mount(cell_private, '/private/cell'),
            '--entrypoint', 'sleep', ids['agent'], 'infinity')
        created.append(intermediate)
        inspect_boundary(controller.inspection(intermediate), 'intermediate', external, protected, gpu_count=config['gpus'])
        controller.docker('start', intermediate)
        runtime_intermediate = controller.runtime_check(intermediate, baseline,
                                                        external / 'environment-before-intermediate.json', fresh_runtime=True)
        atomic_json(external / 'runtime-integrity.json', {'status': 'PASS', 'baseline_sha256': baseline_sha,
            'baseline_image_id': ids['agent'], 'agent_check': runtime_agent, 'intermediate_check': runtime_intermediate,
            'workspace_shadow_check': 'PASS'})
        probe_report = controller.boundary_probe(intermediate, external, 'intermediate')
        controller.observer(names['observe-intermediate'], intermediate, external / 'intermediate', ids['observer'],
                            {**identity, 'stage': 'intermediate'})
        created.append(names['observe-intermediate'])
        stages['intermediate'] = {'inspection': controller.inspection(intermediate), 'probe': probe_report,
                                  'started_ns': time.monotonic_ns(), 'started_wall_ns': time.time_ns()}
        with (external / 'intermediate.stdout').open('xb') as out, (external / 'intermediate.stderr').open('xb') as err:
            result = subprocess.run(['docker', 'exec', intermediate, 'python3', '-I', '-B', '/private/breakwater/cell_observed.py'],
                                    stdin=subprocess.DEVNULL, stdout=out, stderr=err, timeout=evaluation_timeout)
        controller.docker('stop', '--time', '10', intermediate)
        stages['intermediate'].update(stopped_ns=time.monotonic_ns(), stopped_wall_ns=time.time_ns(), exit_code=result.returncode)
        stages['intermediate']['process_drain'] = controller.no_lingering_processes(
            intermediate, names['observe-intermediate'], external / 'intermediate')
        controller.stop_observer(names['observe-intermediate'], external / 'intermediate')
        require(result.returncode == 0, 'Cell evaluation failed')
        require(read_json(evaluation / 'complete.json').get('complete') is True, 'Missing cell evaluation completion')
        sealed_evaluation = run / 'evaluation-sealed'
        controller.snapshot(intermediate, '/evaluation', sealed_evaluation, [])
        evaluation = sealed_evaluation
        evaluation_sha = seal(evaluation, identity)
        atomic_json(external / 'cell.json', cell)
        external_sha = seal(external, identity, finalized=True, stage_order=['agent', 'intermediate'])
        for root in (external, local, evaluation):
            for directory, dirs, files in os.walk(root):
                Path(directory).chmod(0o555)
                for name in files:
                    (Path(directory) / name).chmod(0o444)
        verifier = names['verifier']
        controller.docker('create', '--name', verifier, '--label', 'breakwater.stage=cell-verifier', '--ipc=private',
            '--user=1102:1102', '--read-only', '--cap-drop=ALL', '--security-opt=no-new-privileges:true',
            '--tmpfs', '/tmp:rw,nosuid,nodev,mode=1777,size=2g', '-e', 'BREAKWATER_STAGE=verifier', '-e', 'FIREWORKS_API_KEY',
            '-e', 'HOME=/tmp/verifier-home', '-e', 'XDG_CACHE_HOME=/tmp/verifier-home/cache',
            *controller.mount(external, '/external-evidence'), *controller.mount(authority, '/authority'),
            *controller.mount(local, '/agent-local'), *controller.mount(submission, '/submission'),
            *controller.mount(evaluation, '/evaluation'), *controller.mount(reports, '/reports', False),
            '--entrypoint', 'sleep', ids['verifier'], 'infinity')
        created.append(verifier)
        inspect_boundary(controller.inspection(verifier), 'verifier', external, protected, gpu_count=config['gpus'])
        stages['verifier'] = {'inspection': controller.inspection(verifier), 'started_ns': time.monotonic_ns(),
                              'probe': {'status': 'INSPECTED', 'method': 'live read-only write probe is required before scoring'}}
        authority_sha = seal(authority, identity, stages=stages, external_sha256=external_sha,
            evaluation_sha256=evaluation_sha, submission_sha256=submission_sha, local_log_sha256=local_sha,
            external_host_root=str(external), protected_host_roots=[str(path) for path in protected],
            pipeline={'task': identity['task'], 'kind': 'harness_cell', 'gpus': config['gpus'],
                      'review_profile': config['review_profile'], 'workdir': workdir},
            cell={key: cell[key] for key in cell if key != 'harness_files'})
        authority.chmod(0o555)
        for path in authority.iterdir():
            path.chmod(0o444)
        controller.docker('start', verifier)
        code = subprocess.run(['docker', 'exec', '--user=1102:1102', verifier, 'python3', '-I', '-B',
                               '/opt/breakwater/verifier.py', '--authority', authority_sha],
                              timeout=config['judge_timeout_sec'] + 600).returncode
        print(json.dumps({'task': task, 'driver': args.driver, 'run_id': identity['run_id'], 'verifier_exit_code': code,
                          'wall_clock_exhausted': timed_out, 'reports': str(reports), 'authority_sha256': authority_sha}))
        return code
    except BaseException as error:
        for stage in ('agent', 'intermediate'):
            if names[stage] in created:
                controller.docker('stop', '--time', '10', names[stage], check=False)
            observer_name = names['observe-' + stage]
            if observer_name in created:
                try:
                    controller.stop_observer(observer_name, external / stage)
                except Exception:
                    controller.docker('stop', '--time', '60', observer_name, check=False)
        if names['gateway'] in created:
            controller.docker('stop', '--time', '30', names['gateway'], check=False)
        if isinstance(error, RulesViolation):
            violation = config.get('rule_violation_reward', 0.0)
            atomic_json(reports / 'reward.json', {'reward': violation})
            atomic_json(reports / 'rule-violation.json', {'reward': violation, 'reason': 'violation of rules',
                                                          'detail': str(error), 'judge_skipped': True})
        atomic_json(reports / 'pipeline-error.json', {'status': 'RULE_VIOLATION' if isinstance(error, RulesViolation) else 'VOID',
                    'error': f'{type(error).__name__}: {error}', 'completed_stages': list(stages)})
        return 2
    finally:
        for name in reversed(created):
            controller.docker('rm', '-f', name, check=False)
        controller.docker('network', 'rm', network, check=False)


if __name__ == '__main__':
    raise SystemExit(main())
