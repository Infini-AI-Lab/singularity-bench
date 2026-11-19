#!/usr/bin/env python3
"""Trusted Docker controller for three workload containers plus isolated logging services.

No external-evidence path, journal volume or provider credential is mounted into
an execution container. Run on the Docker host; never from an agent workspace.

Usage: ``python3 breakwater/run.py --task tasks/<name> ...`` (see breakwater/README.md).

The task contract lives in the ``[breakwater]`` tables of ``tasks/<name>/task.toml`` (breakwater/SPEC.md, read by
spec.py). ``DEFAULTS`` gives every optional key; a task declares what it needs beyond them (its GPUs, its workspace
source, model access, endpoints, host hooks).
"""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import re
import secrets
import selectors
import shutil
import subprocess
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parent))
import host_hooks
import plan as plan_tools
import spec
from spec import (DEFAULTS, EVALUATION_HOSTS, HOME_MOUNT, RESERVED_ENV, RESOURCE_FIELDS, SEALED_PIPELINE,
                  check_resources, cpu_list, relative_paths)
from contract import (CREDENTIAL_ENV, FRAMEWORK_REVISION, RulesViolation, atomic_json, digest, inspect_boundary,
                      inventory, overlaps, read_json, require, seal, tests_digest, write_canary)

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
# The task directory (tasks/<name>), set from --task by main().
TASK = None
# The authoring worker's HOME (author/task_config.py, worker_environment).
WORKER_HOME = '/home/worker'
# Where the sealed worker home is mounted for evaluation and review.

def load_config(task=None):
    """The resolved contract of a task directory (spec.load)."""
    return spec.load(task or TASK)


def docker(*args, check=True, env=None, **kwargs):
    result = subprocess.run(['docker', *map(str, args)], stdin=subprocess.DEVNULL,
                            capture_output=True, check=False, env=env, **kwargs)
    if check and result.returncode:
        raise RuntimeError('Docker operation failed: ' + result.stderr.decode(errors='replace')[:2000])
    return result.stdout.decode().strip()


def network_gateway(name):
    """The host's address on a Docker bridge network."""
    config = json.loads(docker('network', 'inspect', name))[0]['IPAM']['Config']
    gateway = next((item['Gateway'] for item in config if item.get('Gateway') and ':' not in item['Gateway']), None)
    require(gateway, f'Docker network {name} has no IPv4 gateway')
    return gateway


def image(value):
    # Resolve every tag once. All actual creates use immutable IDs.
    result = json.loads(docker('image', 'inspect', value))[0]
    require(not result.get('Config', {}).get('Volumes'), 'Images may not declare implicit volumes')
    return result['Id']


VERIFIER_STAMP = ('import json, sys; sys.path.insert(0, "/opt/breakwater"); import contract; '
                  'print(json.dumps({"revision": getattr(contract, "FRAMEWORK_REVISION", 1), "tests": '
                  'contract.tests_digest("/tests") if hasattr(contract, "tests_digest") else None}))')


def check_verifier_image(image_id, tests):
    """Before authoring, not after it: a verifier image bakes the framework and the task's tests, so one built before
    an upgrade, or before the tests (a rubric, a scorer) changed, is refused now rather than when it verifies."""
    out = subprocess.run(['docker', 'run', '--rm', '--network=none', '--user=1102:1102', '--entrypoint', 'python3',
                          image_id, '-I', '-B', '-c', VERIFIER_STAMP], capture_output=True, text=True, timeout=600)
    require(out.returncode == 0, 'Cannot read the verifier image\'s framework revision: ' + out.stderr.strip()[-500:])
    stamp = json.loads(out.stdout.strip().splitlines()[-1])
    require(stamp.get('revision') == FRAMEWORK_REVISION,
            f'The verifier image has Breakwater revision {stamp.get("revision")}, this launcher {FRAMEWORK_REVISION}: '
            'rebuild it')
    require(stamp.get('tests') == tests_digest(tests),
            "The verifier image's /tests differ from the task's tests/: rebuild it")


def inspection(name):
    value = json.loads(docker('inspect', name))[0]
    # Keep names only; an inspection receipt must never retain a provider key.
    value['Config']['Env'] = [item.partition('=')[0] + '=[REDACTED]' for item in value['Config'].get('Env', [])]
    return {key: value[key] for key in ('Id', 'Image', 'Config', 'HostConfig', 'Mounts', 'State')}


def mount(source, target, readonly=True):
    require(',' not in str(source) and ',' not in str(target), 'Commas are not allowed in bind paths')
    return ['--mount', f'type=bind,source={Path(source).resolve()},target={target}' + (',readonly' if readonly else '')]


def wait_for(predicate, seconds, label):
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(.1)
    raise RuntimeError('Timeout waiting for ' + label)


def boundary_probe(name, external_root, stage):
    # Execute before authoring/evaluation starts. The probes run as both possible
    # untrusted identities, with no evidence path supplied to the agent process.
    program = '''import os,sys
from pathlib import Path
for value in sys.argv[1:]:
 p=Path(value)
 if p.exists(): raise SystemExit("unexpected external evidence path")
 try:
  with p.open("ab") as f: f.write(b"forbidden")
 except (PermissionError,FileNotFoundError,NotADirectoryError,IsADirectoryError): pass
 else: raise SystemExit("unexpected writable evidence path")
'''
    for uid in ('1100:1100', '1101:1200'):
        docker('exec', '--user', uid, name, 'python3', '-I', '-B', '-c', program,
               '/external-evidence/isolation.canary', str(external_root / 'isolation.canary'))
    return {'status': 'PASS', 'stage': stage, 'identities': ['1100', '1101']}


def observer(name, target, folder, image_id, identity, source_roots=(), immutable_roots=(), dataset_manifest=None,
             dataset_roots=()):
    folder.mkdir(mode=0o700)
    if identity.get('stage') == 'intermediate':
        (folder / 'validation-phase').touch()
    pid = json.loads(docker('inspect', target))[0]['State']['Pid']
    docker('run', '-d', '--name', name, '--network=none', '--pid=host', '--cgroupns=host', '--cap-add=SYS_PTRACE',
           '--security-opt=seccomp=unconfined', '--security-opt=apparmor=unconfined',
           *mount(folder, '/capture', False), *mount(HERE / 'process_guard.py', '/opt/process_guard.py'), image_id, '--identity', json.dumps(identity),
           '--pid', str(pid), '--uid', str(os.getuid()), '--gid', str(os.getgid()),
           '--source-roots', json.dumps(list(source_roots)), '--dataset-roots', json.dumps(list(dataset_roots)),
           *(['--dataset-manifest', dataset_manifest] if dataset_manifest else []),
           '--source-excludes', json.dumps([root + '/out' for root in source_roots]),
           '--immutable-roots', json.dumps(list(immutable_roots)), '--snapshot-interval-sec', '3600')
    def ready():
        state = json.loads(docker('inspect', name))[0]['State']
        require(state['Running'], 'External collector exited before execution')
        return (folder / 'ready.json').is_file()
    wait_for(ready, 30, 'external observer')
    container_id = json.loads(docker('inspect', target))[0]['Id']
    docker('exec', name, 'python3', '-I', '-B', '/opt/process_guard.py', 'capture', '--pid', str(pid),
           '--container-id', container_id, '--identity', json.dumps(identity))


def no_lingering_processes(target, watcher, folder):
    state = json.loads(docker('inspect', target))[0]['State']
    if state.get('Running') or state.get('Restarting') or state.get('Pid') != 0:
        raise RulesViolation('Container has not completely stopped before the next stage')
    try:
        docker('exec', watcher, 'python3', '-I', '-B', '/opt/process_guard.py', 'drain')
        report = read_json(folder / 'process-drain.json')
    except Exception as error:
        raise RulesViolation('Unable to prove an empty process namespace/cgroup') from error
    if report.get('status') != 'PASS' or report.get('remaining_pids') != []:
        raise RulesViolation('Lingering processes remain after the container ended')
    report['docker_state'] = {'running': False, 'pid': 0}
    report['controller_drained_ns'] = time.monotonic_ns()
    return report


def runtime_check(target, baseline, report_path, *, fresh_runtime=False):
    from environment_guard import compare, scan
    try:
        actual = scan(target, baseline['specification'])
        result = compare(baseline['files'], actual, fresh_runtime=fresh_runtime)
    except Exception as error:
        raise RulesViolation('Unable to verify unchanged environment binaries/packages') from error
    atomic_json(report_path, result)
    if result['status'] != 'PASS':
        raise RulesViolation('Environment binaries, package source, metadata or permissions changed')
    return result


def stop_observer(name, folder):
    (folder / 'finalize-sources').touch()
    wait_for(lambda: (folder / 'sources-finalized.json').is_file(), 60, 'collector finalization')
    require(read_json(folder / 'sources-finalized.json').get('complete') is True, 'Source collection failed')
    docker('stop', '--time', '60', name)
    state = json.loads(docker('inspect', name))[0]['State']
    require(state['ExitCode'] == 0, 'External observer failed to finalize')
    coverage = read_json(folder / 'coverage.json')
    require(coverage.get('complete') is True and not coverage.get('gaps'), 'External capture has gaps')


def captured_agent(name, command, external, budget, *, channels=(('jsonl', 'native.jsonl'), ('stderr', 'native.stderr')),
                   user='1100:1100', on_deadline='raise'):
    """Host-owned pipes independently copy native CLI bytes and writable local mirrors.

    ``on_deadline='stop'`` treats the wall clock as the budget it is: the copy ends,
    ``None`` is returned and the caller stops the container as after a normal exit.
    """
    copy_script = 'import os,sys;os.umask(2);f=open(sys.argv[1],"wb");\nwhile True:\n b=sys.stdin.buffer.read1(65536)\n if not b:break\n f.write(b);f.flush()\n'
    writers, files = {}, {}
    process = subprocess.Popen(['docker', 'exec', '--user', user, name, *command],
                               stdout=subprocess.PIPE, stderr=subprocess.PIPE, stdin=subprocess.DEVNULL)
    selector = selectors.DefaultSelector()
    selector.register(process.stdout, selectors.EVENT_READ, channels[0][0])
    selector.register(process.stderr, selectors.EVENT_READ, channels[1][0])
    try:
        for channel, filename in channels:
            writers[channel] = subprocess.Popen(['docker', 'exec', '-i', '--user', '1100:1200', name,
                'python3', '-I', '-B', '-c', copy_script, '/logs/agent/' + filename],
                stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            files[channel] = (external / ('agent-native.' + channel)).open('xb')
        deadline = time.monotonic() + budget
        while selector.get_map():
            if time.monotonic() >= deadline:
                require(on_deadline == 'stop', 'Agent deadline exceeded')
                return None
            for key, _ in selector.select(timeout=.25):
                chunk = os.read(key.fileobj.fileno(), 65536)
                if not chunk:
                    selector.unregister(key.fileobj)
                    continue
                files[key.data].write(chunk)
                files[key.data].flush()
                writers[key.data].stdin.write(chunk)
                writers[key.data].stdin.flush()
        return process.wait(timeout=30)
    finally:
        selector.close()
        for writer in writers.values():
            writer.stdin.close()
            require(writer.wait(timeout=30) == 0, 'Agent-local mirror delivery failed')
        for stream in files.values():
            stream.flush()
            os.fsync(stream.fileno())
            stream.close()
        if process.poll() is None:
            process.kill()
            process.wait()
        process.stdout.close()
        process.stderr.close()


def snapshot(container, source, destination, excludes, links=None):
    """Stream a stopped container's files into new host-owned files, without tar links.

    With a ``links`` dict, symbolic links are recorded (relative path -> link text) and
    skipped; without one they fail the snapshot. Hard links and special files always fail.
    """
    import tarfile
    from pathlib import PurePosixPath
    destination.mkdir(mode=0o700)
    process = subprocess.Popen(['docker', 'cp', container + ':' + source.rstrip('/') + '/.', '-'],
                               stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    try:
        with tarfile.open(fileobj=process.stdout, mode='r|*') as archive:
            for member in archive:
                parts = PurePosixPath(member.name).parts
                require(not member.name.startswith('/') and '..' not in parts, 'Unsafe snapshot archive path')
                relative = PurePosixPath(*[part for part in parts if part != '.'])
                if not relative.parts or any(str(relative) == value or str(relative).startswith(value.rstrip('/') + '/') for value in excludes):
                    continue
                if member.issym() and links is not None:
                    links[str(relative)] = member.linkname
                    continue
                require(member.isdir() or member.isfile(), 'Links/special entries in submitted snapshot')
                path = destination / str(relative)
                if member.isdir():
                    path.mkdir(parents=True, exist_ok=True)
                else:
                    path.parent.mkdir(parents=True, exist_ok=True)
                    with archive.extractfile(member) as reader, path.open('xb') as writer:
                        shutil.copyfileobj(reader, writer, length=1024**2)
                    path.chmod(0o755 if member.mode & 0o111 else 0o644)
        require(process.wait(timeout=30) == 0, 'Container snapshot failed')
    finally:
        if process.poll() is None:
            process.kill()
            process.wait()
        process.stdout.close()
        process.stderr.close()


def gpu_devices(selection, count=2):
    """Exactly ``count`` distinct native NVIDIA CDI devices; ``none`` for a CPU task."""
    require(type(count) is int and 0 <= count <= 4, 'Unsupported GPU count')
    if count == 0:
        require(selection in ('none', '', None), 'This task declares zero GPUs')
        return []
    match = re.fullmatch(r'device=(\d+)' + r',(\d+)' * (count - 1), selection or '')
    require(match is not None and len(set(match.groups())) == count, f'Select exactly {count} distinct GPU device indices')
    return [item for index in match.groups() for item in ('--device', 'nvidia.com/gpu=' + index)]


def resource_options(config, resources=None):
    """Docker limits for ``resources`` (by default the shared ``resources`` of the contract)."""
    resources, options = config['resources'] if resources is None else resources, []
    if resources.get('cpus'):
        options += ['--cpus', str(resources['cpus'])]
    if resources.get('cpuset'):
        options += ['--cpuset-cpus', str(resources['cpuset'])]
    if resources.get('memory_mb'):
        options += ['--memory', f"{int(resources['memory_mb'])}m", '--memory-swap', f"{int(resources['memory_mb'])}m"]
    if resources.get('pids'):
        options += ['--pids-limit', str(int(resources['pids']))]
    if resources.get('nofile'):
        options += ['--ulimit', f"nofile={int(resources['nofile'])}:{int(resources['nofile'])}"]
    for name in ('memlock', 'stack'):
        if resources.get(name) is not None:
            options += ['--ulimit', f'{name}={int(resources[name])}:{int(resources[name])}']
    if resources.get('shm_size'):
        options += ['--shm-size=' + str(resources['shm_size'])]
    return options


def agent_resources(config):
    """The authoring container's limits: the shared resources with the agent-only overrides."""
    return {**config['resources'], **config['agent_resources']}


def check_gpus(container, config):
    """Inspect the allocated devices as the worker, before authoring."""
    if not config['gpus']:
        return {'names': [], 'type': 'none'}
    program = ("import torch,json,sys; count,patterns=int(sys.argv[1]),json.loads(sys.argv[2]); "
               "names=[torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]; "
               "assert len(names)==count and all(any(p in n for p in patterns) for n in names), names; "
               "print(json.dumps(names))")
    names = docker('exec', '--user=1101:1200', container, 'python3', '-I', '-B', '-c', program,
                   str(config['gpus']), json.dumps(config['gpu_name_patterns']))
    return {'names': json.loads(names), 'type': config['gpu_type']}


def seed_workspace(config, image_id, destination):
    """The agent-visible workspace: the task tree or the image's own workdir.

    Image seeding never exposes the task's tests, graded evaluation set, solution or
    run directories: only what the task image itself places in its workdir.
    """
    excludes = config.get('workspace_excludes', [])
    if config['workspace_source'] == 'task':
        names = {item for item in excludes if not item.startswith('/')} | {
            'tests', 'private', '.git', '__pycache__', '.pytest_cache', 'task.toml'}
        # The task's trusted runners are mounted for evaluation; the agent's copy would be a decoy.
        anchored = {item.strip('/') for item in excludes if item.startswith('/')} | (
            {config['runners'].strip('/')} if config.get('runners') else set())
        def ignore(directory, entries):
            relative = Path(directory).relative_to(TASK)
            return {entry for entry in entries if entry in names or (relative / entry).as_posix() in anchored}
        shutil.copytree(TASK, destination, ignore=ignore, symlinks=True)
        return
    temporary = 'breakwater-seed-' + secrets.token_hex(6)
    docker('create', '--name', temporary, '--entrypoint', 'true', image_id)
    try:
        snapshot(temporary, config['workdir'], destination, [item.strip('/') for item in excludes])
    finally:
        docker('rm', '-f', temporary, check=False)


def protected_paths_check(baseline_files, root, paths):
    """Task-declared read-only workspace paths must be byte-identical after authoring."""
    if not paths:
        return {'status': 'PASS', 'paths': []}
    final = inventory(root)
    def covered(name):
        return any(name == item or name.startswith(item.rstrip('/') + '/') for item in paths)
    before = {name: value for name, value in baseline_files.items() if covered(name)}
    after = {name: value for name, value in final.items() if covered(name)}
    changed = sorted(name for name in set(before) | set(after) if before.get(name) != after.get(name))
    if changed:
        raise RulesViolation('Protected task files were modified, added or deleted: ' + ', '.join(changed[:20]))
    return {'status': 'PASS', 'paths': list(paths), 'files_checked': len(before)}


def service_gateway(name, network, alias, folder, image_id, env_names, env, created, port):
    """A credentialed sidecar on the internal network, with a private host journal."""
    folder.mkdir(mode=0o700)
    options = []
    for item in env_names:
        options += ['-e', item]
    docker('create', '--name', name, '--network', network, '--network-alias', alias,
           '--user', f'{os.getuid()}:{os.getgid()}', *mount(folder, '/logs/gateway', False),
           '-e', 'GATEWAY_LOG=/logs/gateway/requests.jsonl', '-e', 'GATEWAY_STATE=/logs/gateway/budget.json',
           '-e', f'PORT={port}', *options, image_id, env=env)
    created.append(name)
    docker('network', 'connect', 'bridge', name)
    docker('start', name)
    wait_for(lambda: subprocess.run(['docker', 'exec', name, 'python3', '-I', '-B', '-c',
        f"import urllib.request;urllib.request.urlopen('http://127.0.0.1:{port}/health',timeout=2)"],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL).returncode == 0, 30, alias)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--task', type=Path, required=True, help='Task directory, for example examples/word-count')
    parser.add_argument('--agent-image', required=True, help='Prepared protected-author overlay, not a bare task runtime')
    parser.add_argument('--intermediate-image', help='Prepared runtime with run_worker.py and stdio_capture.so; defaults to agent image')
    parser.add_argument('--verifier-image', required=True)
    parser.add_argument('--observer-image', required=True)
    parser.add_argument('--gateway-image', required=True)
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--asset-root', type=Path, required=True)
    parser.add_argument('--bindings', type=Path, help="moved asset sources, relative to --asset-root (default: the task's)")
    parser.add_argument('--author-model', required=True)
    parser.add_argument('--author-cli', choices=('codex', 'claude-code'), required=True)
    parser.add_argument('--author-upstream', required=True)
    parser.add_argument('--author-key-env', required=True)
    parser.add_argument('--author-auth', choices=('bearer', 'anthropic'), default='bearer',
                        help="how the author gateway sends the provider key: a bearer token, or Anthropic's x-api-key")
    parser.add_argument('--gpus', default=None, help='device=I[,J...] for the declared native NVIDIA CDI devices; none for CPU tasks')
    parser.add_argument('--smoke', action='store_true',
                        help="a smoke run: every stage runs, but the task's evaluation uses its reduced smoke settings "
                             '(breakwater/SPEC.md); the run is sealed as smoke and its reward is not a benchmark score')
    # [breakwater.model_access]: the budgeted development gateway (its own provider key, never in the agent).
    parser.add_argument('--dev-gateway-image', help='breakwater-dev-gateway image (tasks with model access)')
    parser.add_argument('--dev-key-env', help="the development gateway's upstream key variable (default: the task's "
                                              'model_access key_env)')
    parser.add_argument('--judge-concurrency', type=int, choices=range(1, 65), metavar='1-64',
                        help="Judge calls in flight (default 8): raise it, within the provider's rate limit, so a long "
                             "run's review fits the task's judge budget")
    parser.add_argument('--endpoint', action='append', default=[], metavar='NAME=HOST:PORT',
                        help="where one of the task's [[breakwater.endpoints]] really is (the same port on Docker)")
    parser.add_argument('--input', action='append', default=[], metavar='NAME=PATH',
                        help="a file the task's [breakwater.operator] inputs declare (its host hooks read it)")
    parser.add_argument('--value-env', action='append', default=[], metavar='NAME=VAR',
                        help="the variable holding a declared operator value (default: the task's)")
    parser.add_argument('--secret-env', action='append', default=[], metavar='NAME=VAR',
                        help="the variable holding a declared operator secret (default: the task's); held, never in a "
                             'container')
    args = parser.parse_args()
    global TASK
    TASK = args.task.resolve()
    require((TASK / 'task.toml').is_file(), 'Not a Breakwater task directory: ' + str(args.task))
    config = load_config()
    selection = args.gpus if args.gpus is not None else ('none' if config['gpus'] == 0 else
                                                         'device=' + ','.join(str(i) for i in range(config['gpus'])))
    gpu_options = gpu_devices(selection, config['gpus'])
    require(args.author_upstream.startswith('https://'), 'Author upstream must use HTTPS')
    require(args.author_cli != 'codex' or args.author_auth == 'bearer', 'Codex needs --author-auth bearer')
    require(os.environ.get(args.author_key_env) and os.environ.get('FIREWORKS_API_KEY'), 'Missing author/judge credentials')
    # Fail before authoring, not after it: the evaluation stage needs these host variables.
    missing = [name for name in config['intermediate_env'] if not os.environ.get(name)]
    require(not missing, 'Missing evaluation-stage variables: ' + ', '.join(missing))
    development = config['development_gateway']
    dev_key_env = args.dev_key_env or (development or {}).get('key_env')
    if development:
        require(args.dev_gateway_image and os.environ.get(dev_key_env), 'Missing development gateway image or key')
    run = args.run_dir.resolve()
    # The task's host side: the operator's inputs, values and secrets (secrets leave this environment now) and its
    # preflight hook's plan, before the run directory exists.
    task_host = host_hooks.TaskHost(TASK, config, 'docker', args.smoke, args.input, args.value_env, args.secret_env,
                                    avoid=(TASK, run))
    task_host.preflight(reserved=spec.reserved_names(config))
    # Its credential variables, as the verifier checks them, include any the operator renamed.
    config = {**config, 'credential_env': sorted({*config['credential_env'], *task_host.operator.credential_names})}
    # The task's static endpoints with the operator's overrides, healthy before authoring starts, and its hooks'.
    static = plan_tools.static_endpoints(config, args.endpoint)
    endpoints = [*static, *task_host.endpoints()]
    require(not any(row['port_map'] for row in endpoints),
            'Docker cannot remap an endpoint port: serve it on the port the task declares')
    roles = plan_tools.by_role(endpoints)
    require(not (roles['coordinator'] or roles['submission']) or config['intermediate_network'] == 'bridge',
            'Endpoints the evaluation reaches need [breakwater.evaluation] network = "bridge" on Docker')
    require(all(plan_tools.launcher_host(row['host']) for row in roles['agent']) or config['agent_extra_networks'],
            'Endpoints on other hosts need [breakwater.authoring] extra_networks = ["bridge"] on Docker')
    plan_tools.check_health(static)
    require(not run.exists() and not overlaps(run, TASK) and not overlaps(run, args.asset_root), 'Use a fresh, separate run directory')
    bindings = spec.bindings(config, args.asset_root, run, args.bindings)
    # Agent-bound inputs are never part of the submission; neither are the task's snapshot excludes.
    excludes = [spec['target'][len(config['workdir']) + 1:] for spec in bindings
                if 'agent' in spec['stages'] and spec['target'].startswith(config['workdir'] + '/')]
    excludes += config['snapshot_excludes']
    run.mkdir(parents=True, mode=0o700)
    task_host.adopt(run)
    external, authority, local, workspace, evaluation, reports = [run / key for key in
        ('external', 'authority', 'agent-local', 'submission', 'evaluation', 'reports')]
    home = run / 'submission-home'
    local_live, workspace_live = run / 'local-live', run / 'workspace-live'
    baseline_dir = run / 'environment-baseline'
    for folder in (external, authority, local_live, evaluation, reports, baseline_dir):
        folder.mkdir(mode=0o700)
    # The resolved contract the coordinator and the verifier read (sealed into the authority by its digest).
    atomic_json(run / 'contract.json', config)
    (run / 'contract.json').chmod(0o444)
    task_plan_sha = task_host.record(external, static)
    # These are separate writable output mounts, never parents of trusted storage.
    local_live.chmod(0o777)
    evaluation.chmod(0o777)
    reports.chmod(0o777)
    ids = {'agent': image(args.agent_image), 'intermediate': image(args.intermediate_image or args.agent_image),
           'verifier': image(args.verifier_image), 'observer': image(args.observer_image), 'gateway': image(args.gateway_image)}
    if development:
        ids['dev_gateway'] = image(args.dev_gateway_image)
    check_verifier_image(ids['verifier'], TASK / 'tests')
    seed_workspace(config, ids['agent'], workspace_live)
    task_host.write_workspace_files(workspace_live)
    workspace_baseline = inventory(workspace_live)  # Also rejects symlinks/special files before exposure.
    atomic_json(external / 'workspace-baseline.json', {'source': config['workspace_source'], 'files': workspace_baseline})
    write_canary(external)
    identity = {'task': config['task'], 'run_id': secrets.token_hex(16)}
    names = {stage: 'breakwater-' + identity['run_id'][:12] + '-' + stage for stage in
             ('agent', 'intermediate', 'verifier', 'gateway', 'dev-gateway', 'observe-agent', 'observe-intermediate')}
    network = 'breakwater-' + identity['run_id'][:12]
    # Endpoints on the launcher's host and inbound ports: the agent container reaches the launcher's host as this
    # bridge network's gateway, and servers its commands run are published on the host. (Endpoints on other hosts are
    # reached through the task's extra networks.)
    host_network = (network + '-host' if any(plan_tools.launcher_host(row['host']) for row in roles['agent'])
                    or config['inbound_ports'] else None)
    stages, created = {}, []
    baseline_dir.chmod(0o755)
    protected = [external, authority, reports]
    def assets(stage):
        result = []
        for spec in bindings:
            if stage in spec['stages']:
                result += mount(spec['source'], spec['target'])
        return result
    def create(stage, options, image_id, command):
        name = names[stage]
        docker('create', '--name', name, '--label', 'breakwater.stage=' + stage, '--ipc=private',
               *options, image_id, *command)
        created.append(name)
        inspect_boundary(inspection(name), stage, external, protected, gpu_count=config['gpus'],
                         credential_names=config['credential_env'])
        return name
    try:
        if ids['intermediate'] != ids['agent']:
            raise RulesViolation('Agent and intermediate must use the same immutable runtime image in distinct containers')
        task_host.before_authoring(external)
        docker('network', 'create', '--internal', network)
        if host_network:
            docker('network', 'create', host_network)
        gateway_env = {**os.environ, 'AUTHOR_UPSTREAM_API_KEY': os.environ[args.author_key_env]}
        (external / 'gateway').mkdir(mode=0o700)
        docker('create', '--name', names['gateway'], '--network', network, '--network-alias', 'author-gateway',
               '--user', f'{os.getuid()}:{os.getgid()}', *mount(external / 'gateway', '/logs/author', False),
               '-e', 'AUTHOR_UPSTREAM_API_KEY', '-e', 'AUTHOR_MODEL=' + args.author_model,
               '-e', 'AUTHOR_UPSTREAM_BASE_URL=' + args.author_upstream, '-e', 'AUTHOR_UPSTREAM_AUTH=' + args.author_auth,
               '-e', 'AUTHOR_PROTOCOL=' + ('responses' if args.author_cli == 'codex' else 'messages'),
               '-e', 'AUTHOR_GATEWAY_LOG=/logs/author/requests.jsonl', ids['gateway'], env=gateway_env)
        created.append(names['gateway'])
        docker('network', 'connect', 'bridge', names['gateway'])
        docker('start', names['gateway'])
        wait_for(lambda: subprocess.run(['docker', 'exec', names['gateway'], 'python3', '-I', '-B', '-c',
            "import urllib.request;urllib.request.urlopen('http://127.0.0.1:8001/health',timeout=2)"],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL).returncode == 0, 30, 'author gateway')
        if development:
            # Only this sidecar holds the development key; worker commands reach it at the task's model-access
            # host and port (gateway:8000 by default).
            dev_env = {**os.environ, 'UPSTREAM_API_KEY': os.environ[dev_key_env]}
            values = ['UPSTREAM_API_KEY', *[f'{name}={value}' for name, value in spec.model_access_env(development).items()]]
            service_gateway(names['dev-gateway'], network, development['host'], external / 'dev-gateway',
                            ids['dev_gateway'], values, dev_env, created, development['port'])
        agent_env = ['-e', 'PYTHONDONTWRITEBYTECODE=1', '-e', 'BREAKWATER_AUTHOR_MODEL=' + args.author_model,
                     '-e', 'BREAKWATER_AUTHORING_PLAN']
        # What the agent's commands get beyond the workspace and the author gateway; the author image applies it. An
        # endpoint on the launcher's host is the host network's gateway inside the container.
        host_address = network_gateway(host_network) if host_network else None
        def where(host):
            return host_address if plan_tools.launcher_host(host) else host
        agent_endpoints = [{'name': row['name'], 'host': where(row['host']), 'ports': plan_tools.ranges(row['ports'])}
                           for row in roles['agent']]
        inbound = config['inbound_ports']
        plan = spec.plan_text(spec.authoring_plan(
            config, endpoints=agent_endpoints, hosts=[[config['inbound_host'], host_address]] if inbound else [],
            reply_ports=inbound.replace('-', ':') if inbound else None,
            rejects=[{**row, 'host': where(row['host'])} for row in plan_tools.rejects(roles)],
            env={**config['authoring_env'], **task_host.agent_env()}))
        task_host.check_no_secrets(plan)
        environment = {**os.environ, 'BREAKWATER_AUTHORING_PLAN': plan}
        (external / 'authoring-plan.json').write_text(plan + '\n')
        publish = []
        if inbound:
            # On the host's loopback and on the default bridge's gateway (what host.docker.internal is in containers
            # the task's own tools start).
            publish = ['--publish', f'127.0.0.1:{inbound}:{inbound}',
                       '--publish', f"{network_gateway('bridge')}:{inbound}:{inbound}"]
        name = names['agent']
        docker('create', '--name', name, '--label', 'breakwater.stage=agent', '--ipc=private',
               '--network', host_network or network, *publish, *gpu_options, *resource_options(config, agent_resources(config)), '--cap-add=NET_ADMIN', *agent_env,
               *mount(workspace_live, config['workdir'], False), *mount(local_live, '/logs/agent', False),
               *mount(baseline_dir, '/opt/environment-baseline'), *assets('agent'),
               '--entrypoint', 'python3', ids['agent'], '-I', '-B', '/opt/breakwater-runtime/bootstrap.py', env=environment)
        created.append(name)
        if host_network:
            docker('network', 'connect', network, name)  # the author gateway's network
        for extra in config['agent_extra_networks']:
            # Endpoints on other hosts only; the protected bootstrap firewall admits their exact ports.
            require(extra == 'bridge', 'Only the default bridge may be added to authoring')
            docker('network', 'connect', extra, name)
        inspect_boundary(inspection(name), 'agent', external, protected, gpu_count=config['gpus'],
                         credential_names=config['credential_env'])
        agent = name
        docker('start', agent)
        wait_for(lambda: subprocess.run(['docker', 'exec', agent, 'test', '-f', '/run/breakwater/ready'],
                                        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL).returncode == 0,
                 60, 'protected author bootstrap')
        # The handshake: the bootstrap applied this plan, not an older image's idea of the task.
        runtime = json.loads(docker('exec', agent, 'cat', '/run/breakwater/runtime.json'))
        require(runtime.get('plan_sha256') == hashlib.sha256(plan.encode()).hexdigest(),
                'The author image did not apply this authoring plan: rebuild it from this Breakwater')
        docker('exec', agent, 'touch', '/run/breakwater/external-capture-enabled')
        # The protected launcher constructs a minimal child environment rather
        # than inheriting Docker's env. Avoid legitimate import-cache writes too.
        docker('exec', '--user=0', agent, 'python3', '-I', '-B', '-c',
               "from pathlib import Path;p=Path('/run/breakwater/worker.env');"
               "entries=[x for x in p.read_bytes().split(b'\\0') if x and not x.startswith(b'PYTHONDONTWRITEBYTECODE=')];"
               "p.write_bytes(b'\\0'.join(entries+[b'PYTHONDONTWRITEBYTECODE=1'])+b'\\0')")
        atomic_json(external / 'gpu-allocation.json', check_gpus(agent, config))
        from environment_guard import PACKAGES, VERSION, describe, scan, shadow_policy, shadows
        policy = shadow_policy(config['shadow_policy'])
        specification = describe(agent, config['workdir'], PACKAGES, policy)
        # The sealed snapshot leaves out the excluded paths; so does the comparison baseline.
        baseline = {'version': VERSION, 'identity': identity, 'image_id': ids['agent'],
                    'specification': specification, 'files': scan(agent, specification),
                    'workspace_shadows': shadows(workspace_live, policy, excludes)}
        atomic_json(baseline_dir / 'environment.json', baseline)
        (baseline_dir / 'environment.json').chmod(0o444)
        baseline_dir.chmod(0o555)
        shutil.copyfile(baseline_dir / 'environment.json', external / 'environment-baseline.json')
        baseline_sha = digest(baseline_dir / 'environment.json')
        readonly = "import os;from pathlib import Path;p=Path('/opt/environment-baseline/environment.json');assert p.is_file() and not os.access(p,os.W_OK)"
        docker('exec', '--user=1101:1200', agent, 'python3', '-I', '-B', '-c', readonly)
        probe = boundary_probe(agent, external, 'agent')
        observer(names['observe-agent'], agent, external / 'agent', ids['observer'], {**identity, 'stage': 'agent'},
                 source_roots=[config['workdir']], immutable_roots=[row['target'] for row in bindings if 'agent' in row['stages']],
                 dataset_manifest=config['dataset_manifest'], dataset_roots=config['dataset_roots'])
        created.append(names['observe-agent'])
        stages['agent'] = {'inspection': inspection(agent), 'probe': probe, 'started_ns': time.monotonic_ns(),
                           'started_wall_ns': time.time_ns()}
        # 'evaluate': the task grades what exists when the budget ends, so a CLI still running at
        # the deadline is stopped like a normal exit, and any CLI exit status proceeds.
        evaluate_at_deadline = config['agent_deadline'] == 'evaluate'
        code = captured_agent(agent, ['/opt/breakwater-runtime/author-exec', '--model', args.author_model,
                                     '--instruction', (TASK / 'instruction.md').read_text()], external,
                              config['agent_timeout_sec'], on_deadline='stop' if evaluate_at_deadline else 'raise')
        docker('stop', '--time', '10', agent)
        stages['agent'].update(stopped_ns=time.monotonic_ns(), stopped_wall_ns=time.time_ns(), exit_code=code,
                               deadline_reached=code is None)
        stages['agent']['process_drain'] = no_lingering_processes(agent, names['observe-agent'], external / 'agent')
        runtime_agent = runtime_check(agent, baseline, external / 'environment-after-agent.json')
        stop_observer(names['observe-agent'], external / 'agent')
        docker('stop', '--time', '30', names['gateway'])
        if development:
            docker('stop', '--time', '30', names['dev-gateway'])
        # Authoring is dead before snapshot sealing; no background process can race it.
        if not evaluate_at_deadline:
            require(code == 0, 'Agent failed; retaining evidence without issuing a passing reward')
        links, home_links = {}, {}
        snapshot(agent, config['workdir'], workspace, excludes, links=links)
        if config['snapshot_worker_home']:
            snapshot(agent, WORKER_HOME, home, [], links=home_links)
        snapshot(agent, '/logs/agent', local, [])
        # Symbolic links are not sealed as files: the judge sees each one, and installed copies
        # recreate the links that stay inside the submission.
        atomic_json(external / 'workspace-links.json', {'workdir': config['workdir'], 'symlinks': links,
                    **({'home': {'path': WORKER_HOME, 'symlinks': home_links}} if config['snapshot_worker_home'] else {})})
        if shadows(workspace, policy, links=links) != baseline['workspace_shadows']:
            raise RulesViolation('Task workspace shadows a protected package or installs an import/startup hook')
        protected_report = protected_paths_check(workspace_baseline, workspace, config['protected_workspace_paths'])
        atomic_json(external / 'protected-workspace.json', protected_report)
        submission_sha = seal(workspace, identity, symlinks=links)
        home_sha = seal(home, identity, symlinks=home_links) if config['snapshot_worker_home'] else None
        local_sha = seal(local, identity)
        for root in [workspace] + ([home] if home_sha else []):
            for directory, dirs, files in os.walk(root):
                Path(directory).chmod(0o555)
                for name in files:
                    path = Path(directory) / name
                    path.chmod(0o555 if path.stat().st_mode & 0o111 else 0o444)
        submission_target = config['submission_mount'] or config['workdir']
        intermediate_options = [f"--network={config['intermediate_network']}", *gpu_options, *resource_options(config),
            '--cap-drop=ALL', '--cap-add=SETUID', '--cap-add=SETGID', '--cap-add=CHOWN', '--cap-add=FOWNER',
            # KILL: the root coordinator must stop UID-1101 workers (servers, timed-out children).
            '--cap-add=DAC_OVERRIDE', '--cap-add=KILL', *[f'--cap-add={cap}' for cap in config['intermediate_cap_add']],
            '--security-opt=no-new-privileges:true',
            '-e', 'BREAKWATER_STAGE=intermediate', '-e', 'PYTHONDONTWRITEBYTECODE=1',
            *(['-e', 'BREAKWATER_SMOKE=1'] if args.smoke else []),
            # Named host variables only (never credentials), and only the host-gateway alias.
            *[item for variable in config['intermediate_env'] for item in ('-e', variable)],
            *[item for host in config['intermediate_add_hosts'] for item in ('--add-host', host)],
            *mount(baseline_dir, '/opt/environment-baseline'), *mount(workspace, submission_target),
            *(mount(home, HOME_MOUNT) if home_sha else []),
            *mount(evaluation, '/evaluation', False), *mount(TASK / 'tests', '/private/tests'),
            *mount(run / 'contract.json', '/trusted-task/contract.json'),
            *mount(HERE, '/opt/breakwater'),
            *(mount(TASK / config['runners'], '/opt/breakwater-drivers') if config['runners'] else []),
            *assets('intermediate')]
        task_private = task_host.write_private_files(run)
        if task_private:
            intermediate_options += mount(task_private, '/private/operator')
        intermediate = create('intermediate', [*intermediate_options, '--entrypoint', 'sleep'], ids['intermediate'], ['infinity'])
        # Native CDI creates driver mountpoints/cache files at start. Initialize
        # only the trusted idle process before comparing like-for-like runtimes;
        # no submitted code runs until the inventory and observer checks pass.
        docker('start', intermediate)
        runtime_intermediate = runtime_check(intermediate, baseline, external / 'environment-before-intermediate.json', fresh_runtime=True)
        atomic_json(external / 'runtime-integrity.json', {'status': 'PASS', 'baseline_sha256': baseline_sha,
            'baseline_image_id': ids['agent'], 'agent_check': runtime_agent, 'intermediate_check': runtime_intermediate,
            'workspace_shadow_check': 'PASS'})
        probe = boundary_probe(intermediate, external, 'intermediate')
        observer(names['observe-intermediate'], intermediate, external / 'intermediate', ids['observer'],
                 {**identity, 'stage': 'intermediate'})
        created.append(names['observe-intermediate'])
        stages['intermediate'] = {'inspection': inspection(intermediate), 'probe': probe, 'started_ns': time.monotonic_ns(),
                                  'started_wall_ns': time.time_ns()}
        with (external / 'intermediate.stdout').open('xb') as out, (external / 'intermediate.stderr').open('xb') as err:
            result = subprocess.run(['docker', 'exec', intermediate, 'python3', '-I', '-B', '/opt/breakwater/observed.py'],
                                    stdin=subprocess.DEVNULL, stdout=out, stderr=err, timeout=config['evaluation_timeout_sec'])
        docker('stop', '--time', '10', intermediate)
        stages['intermediate'].update(stopped_ns=time.monotonic_ns(), stopped_wall_ns=time.time_ns(), exit_code=result.returncode)
        stages['intermediate']['process_drain'] = no_lingering_processes(intermediate, names['observe-intermediate'], external / 'intermediate')
        stop_observer(names['observe-intermediate'], external / 'intermediate')
        # Both execution containers are stopped and drained: the task's host side may now collect its evidence.
        task_host.after_evaluation(external, {stage: [stages[stage]['started_wall_ns'], stages[stage]['stopped_wall_ns']]
                                              for stage in ('agent', 'intermediate')})
        require(result.returncode == 0, 'Intermediate evaluation failed')
        require(read_json(evaluation / 'complete.json').get('complete') is True, 'Missing intermediate completion')
        eval_snapshot = run / 'evaluation-sealed'
        snapshot(intermediate, '/evaluation', eval_snapshot, [])
        evaluation = eval_snapshot
        evaluation_sha = seal(evaluation, identity)
        # chmod above made evaluation immutable; sealing adds only a host-owned manifest.
        external_sha = seal(external, identity, finalized=True, stage_order=['agent', 'intermediate'])
        # Bind roots are traversable inside the verifier but remain under a private host parent.
        for root in (external, local, evaluation):
            for directory, dirs, files in os.walk(root):
                Path(directory).chmod(0o555)
                for name in files:
                    (Path(directory) / name).chmod(0o444)
        verifier = create('verifier', ['--user=1102:1102', '--read-only', '--cap-drop=ALL',
            '--security-opt=no-new-privileges:true', '--tmpfs', '/tmp:rw,nosuid,nodev,mode=1777,size=2g',
            '-e', 'BREAKWATER_STAGE=verifier', '-e', 'FIREWORKS_API_KEY',
            *(['-e', f'BREAKWATER_JUDGE_CONCURRENCY={args.judge_concurrency}'] if args.judge_concurrency else []),
            '-e', 'HOME=/tmp/verifier-home', '-e', 'XDG_CACHE_HOME=/tmp/verifier-home/cache',
            *mount(external, '/external-evidence'), *mount(authority, '/authority'), *mount(local, '/agent-local'),
            *mount(workspace, config['workdir']), *(mount(home, HOME_MOUNT) if home_sha else []),
            *mount(evaluation, '/evaluation'),
            *mount(reports, '/reports', False), *mount(run / 'contract.json', '/trusted-task/contract.json'),
            *assets('verifier'), '--entrypoint', 'sleep'], ids['verifier'], ['infinity'])
        stages['verifier'] = {'inspection': inspection(verifier), 'started_ns': time.monotonic_ns(),
                              'probe': {'status': 'INSPECTED', 'method': 'live read-only write probe is required before scoring'}}
        authority_sha = seal(authority, identity, stages=stages, external_sha256=external_sha,
            evaluation_sha256=evaluation_sha, submission_sha256=submission_sha, local_log_sha256=local_sha,
            **({'home_sha256': home_sha} if home_sha else {}),
            external_host_root=str(external), protected_host_roots=[str(path) for path in protected],
            pipeline={key: config[key] for key in SEALED_PIPELINE}, contract_sha256=digest(run / 'contract.json'),
            framework_revision=FRAMEWORK_REVISION, tests_sha256=tests_digest(TASK / 'tests'),
            recording=spec.recording(config), **task_host.sealed(task_plan_sha),
            task_directory=TASK.name, **({'smoke': True} if args.smoke else {}))
        authority.chmod(0o555)
        for path in authority.iterdir():
            path.chmod(0o444)
        docker('start', verifier)
        # This is the only stage that can read external evidence or call the judge model.
        code = subprocess.run(['docker', 'exec', '--user=1102:1102', verifier, 'python3', '-I', '-B',
                               '/opt/breakwater/verifier.py', '--authority', authority_sha],
                              timeout=config['judge_timeout_sec'] + 600).returncode
        print(json.dumps({'task': config['task'], 'run_id': identity['run_id'], 'verifier_exit_code': code,
                          **({'smoke': True} if args.smoke else {}),
                          'reports': str(reports), 'external_sha256': external_sha, 'authority_sha256': authority_sha}))
        return code
    except BaseException as error:
        task_host.on_failure()
        # Do not discard failed attempts or silently turn incomplete capture into a pass.
        for stage in ('agent', 'intermediate'):
            if names[stage] in created:
                docker('stop', '--time', '10', names[stage], check=False)
            observer_name = names['observe-' + stage]
            if observer_name in created:
                try:
                    stop_observer(observer_name, external / stage)
                except Exception:
                    docker('stop', '--time', '60', observer_name, check=False)
        for service in ('gateway', 'dev-gateway'):
            if names[service] in created:
                docker('stop', '--time', '30', names[service], check=False)
        if isinstance(error, RulesViolation):
            violation = config['rule_violation_reward']
            atomic_json(reports / 'reward.json', {'reward': violation})
            atomic_json(reports / 'rule-violation.json', {'reward': violation, 'reason': 'violation of rules',
                'detail': str(error), 'judge_skipped': True})
        elif not (reports / 'reward.json').exists():
            atomic_json(reports / 'reward.json', {'reward': config['failure_reward']})
        atomic_json(reports / 'pipeline-error.json', {'status': 'RULE_VIOLATION' if isinstance(error, RulesViolation) else 'INCONCLUSIVE',
                    'error': f'{type(error).__name__}: {error}', 'completed_stages': list(stages)})
        return 2
    finally:
        for name in reversed(created):
            docker('rm', '-f', name, check=False)
        docker('network', 'rm', network, check=False)
        if host_network:
            docker('network', 'rm', host_network, check=False)


if __name__ == '__main__':
    raise SystemExit(main())
