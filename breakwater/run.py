#!/usr/bin/env python3
"""Trusted Docker controller for three workload containers plus isolated logging services.

No external-evidence path, journal volume or provider credential is mounted into
an execution container. Run on the Docker host; never from an agent workspace.

Usage: ``python3 breakwater/run.py --task tasks/<name> ...`` (see breakwater/README.md).

The task contract lives in ``tasks/<name>/breakwater.json``. ``DEFAULTS`` gives every optional key
(four GB300 GPUs, task-tree workspace, no development gateway, no fleet);
coding-agent-harness and cluster-serverless declare one GPU / zero GPUs, an image-seeded
workspace, the budgeted development gateway, and the remote inference fleet.
"""
import argparse
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
from contract import (CREDENTIAL_ENV, RulesViolation, atomic_json, digest, inspect_boundary, inventory, overlaps,
                      read_json, require, seal)

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
# The task directory (tasks/<name>), set from --task by main().
TASK = None
# The authoring worker's HOME (author/task_config.py, worker_environment).
WORKER_HOME = '/home/worker'
# Where the sealed worker home is mounted for evaluation and review.
HOME_MOUNT = '/submission-home'

DEFAULTS = {
    'gpus': 4,
    'gpu_name_patterns': ['GB300', 'B300'],
    'gpu_type': 'GB300',
    'workspace_source': 'task',
    'workspace_excludes': [],
    'submission_mount': None,
    'agent_extra_networks': [],
    'intermediate_network': 'none',
    'intermediate_cap_add': [],
    'intermediate_env': [],
    'intermediate_add_hosts': [],
    'resources': {'cpus': None, 'memory_mb': None, 'nofile': None, 'shm_size': '16g', 'memlock': None,
                  'stack': None, 'pids': None, 'cpuset': None},
    'agent_resources': {},
    'shadow_policy': None,
    'protected_workspace_paths': [],
    'snapshot_excludes': [],
    'snapshot_worker_home': False,
    'dataset_manifest': None,
    'dataset_roots': [],
    'sandbox_worker': None,
    'review_profile': 'exhaustive',
    'review_metadata_paths': [],
    'agent_deadline': 'fail',
    'rule_violation_reward': 0.0,
    'development_gateway': None,
    'cluster': None,
}
RESOURCE_FIELDS = ('cpus', 'memory_mb', 'nofile', 'shm_size', 'memlock', 'stack', 'pids', 'cpuset')
# Launch-contract keys sealed into the authority; verifier.py compares them with its own copy of the contract.
SEALED_PIPELINE = ('task', 'kind', 'gpus', 'review_profile', 'workdir', 'agent_deadline', 'rule_violation_reward',
                   'snapshot_excludes', 'snapshot_worker_home', 'review_metadata_paths')
# Variables the controller sets itself; with credentials and loader/interpreter variables,
# they are never forwarded from the host to the evaluation stage.
RESERVED_ENV = ('BREAKWATER_STAGE', 'PATH', 'HOME', 'PYTHONDONTWRITEBYTECODE')
EVALUATION_HOSTS = ('host.docker.internal:host-gateway',)


def relative_paths(values, label, *, anchored=False):
    """Workspace-relative paths without empty, '.' or '..' components; a leading '/' only if anchored."""
    require(isinstance(values, list), 'Invalid ' + label)
    for value in values:
        require(isinstance(value, str) and value and '\0' not in value, 'Invalid ' + label)
        body = value[1:] if anchored and value.startswith('/') else value
        require(all(part not in ('', '.', '..') for part in body.split('/')), 'Invalid ' + label)
    return values


def cpu_list(cpuset):
    cpus = set()
    for item in cpuset.split(','):
        first, _, last = item.partition('-')
        require(int(first) <= int(last or first) < 65536, 'Invalid cpuset range')
        cpus.update(range(int(first), int(last or first) + 1))
    return sorted(cpus)


def check_resources(values, label):
    require(isinstance(values, dict) and set(values) <= set(RESOURCE_FIELDS), 'Unknown ' + label + ' field')
    for key, value in values.items():
        if value is None:
            continue
        if key == 'cpus':
            valid = type(value) in (int, float) and math.isfinite(value) and value > 0
        elif key == 'shm_size':
            valid = isinstance(value, str) and re.fullmatch(r'[1-9][0-9]*[bkmgBKMG]?', value) is not None
        elif key == 'cpuset':
            valid = (isinstance(value, str) and re.fullmatch(r'[0-9]+(-[0-9]+)?(,[0-9]+(-[0-9]+)?)*', value) is not None
                     and bool(cpu_list(value)))
        elif key in ('memlock', 'stack'):
            valid = type(value) is int and (value == -1 or value > 0)
        else:
            valid = type(value) is int and value > 0
        require(valid, f'Invalid {label} value: {key}')
    if values.get('cpuset') and values.get('cpus'):
        require(len(cpu_list(values['cpuset'])) >= values['cpus'], 'The cpuset has fewer CPUs than the CPU quota')
    return values


def load_config(path=None):
    config = read_json(path or TASK / 'breakwater.json')
    require(isinstance(config.get('resources', {}), dict) and isinstance(config.get('agent_resources', {}), dict),
            'Resources must be objects')
    merged = {**DEFAULTS, **config}
    merged['resources'] = {**DEFAULTS['resources'], **(config.get('resources') or {})}
    merged['agent_resources'] = dict(config.get('agent_resources') or {})
    require(type(merged['gpus']) is int and 0 <= merged['gpus'] <= 4, 'Declared GPU count must be between 0 and 4')
    require(merged['workspace_source'] in ('task', 'image'), 'Unknown workspace source')
    excludes = relative_paths(merged['workspace_excludes'], 'workspace exclude', anchored=True)
    if merged['workspace_source'] == 'task':
        # A plain entry is a name excluded anywhere in the task tree; a leading '/' anchors a
        # path at the task root (so /README.md does not hide nanoGPT/README.md).
        require(all(item.startswith('/') or '/' not in item for item in excludes),
                'A plain workspace exclude is one name; anchor a longer path at the task root with a leading /')
    require(merged['intermediate_network'] in ('none', 'bridge'), 'Unknown intermediate network')
    require(set(merged['intermediate_cap_add']) <= {'NET_ADMIN', 'NET_RAW'}, 'Only network capabilities may be added to the coordinator')
    names = merged['intermediate_env']
    require(isinstance(names, list)
            and all(isinstance(name, str) and re.fullmatch(r'[A-Z][A-Z0-9_]*', name) and name not in CREDENTIAL_ENV
                    and name not in RESERVED_ENV and not name.startswith(('LD_', 'PYTHON')) for name in names)
            and len(set(names)) == len(names),
            'Invalid evaluation-stage variable: credentials and loader/interpreter variables are never forwarded')
    hosts = merged['intermediate_add_hosts']
    require(isinstance(hosts, list) and all(host in EVALUATION_HOSTS for host in hosts if isinstance(host, str))
            and all(isinstance(host, str) for host in hosts) and len(set(hosts)) == len(hosts),
            'Unexpected evaluation host entry')
    require(not hosts or merged['intermediate_network'] == 'bridge', 'Evaluation host entries need the bridge network')
    check_resources(merged['resources'], 'resource')
    check_resources(merged['agent_resources'], 'agent resource')
    check_resources({**merged['resources'], **merged['agent_resources']}, 'agent resource')
    require(merged['review_profile'] in ('exhaustive', 'indexed'), 'Unknown review profile')
    protected = relative_paths(merged['protected_workspace_paths'], 'protected workspace path')
    snapshot_excludes = relative_paths(merged['snapshot_excludes'], 'snapshot exclude')
    relative_paths(merged['review_metadata_paths'], 'review metadata path')
    def covers(prefix, value):
        return value == prefix or value.startswith(prefix + '/')
    require(not any(covers(item, path) or covers(path, item) for item in snapshot_excludes for path in protected),
            'Protected workspace paths may not be excluded from the sealed snapshot')
    require(type(merged['snapshot_worker_home']) is bool, 'snapshot_worker_home must be true or false')
    # The agent-declared training-dataset manifest the observer verifies and archives after authoring.
    inside = str(merged.get('workdir', '')).rstrip('/') + '/'
    manifest, dataset_roots = merged['dataset_manifest'], merged['dataset_roots']
    require(manifest is None or (isinstance(manifest, str) and manifest.startswith(inside)),
            'The dataset manifest must be a path inside the workspace')
    require(isinstance(dataset_roots, list) and all(isinstance(root, str) and root.startswith(inside) for root in dataset_roots)
            and (manifest is not None or not dataset_roots), 'Dataset roots must lie inside the workspace and need a manifest')
    require(merged['agent_deadline'] in ('fail', 'evaluate'), 'Unknown agent deadline policy')
    sandbox = merged['sandbox_worker']
    if sandbox is not None:
        ports = sandbox.get('session_ports') if isinstance(sandbox, dict) else None
        require(isinstance(sandbox, dict) and set(sandbox) == {'port', 'session_ports'} and type(sandbox['port']) is int
                and 1 <= sandbox['port'] <= 65535 and isinstance(ports, str) and re.fullmatch(r'\d{4,5}-\d{4,5}', ports),
                'sandbox_worker needs an integer port and a session_ports range such as "30000-30015"')
        low, high = map(int, ports.split('-'))
        require(1024 <= low <= high <= 65535 and high - low < 64, 'Invalid session port range')
    reward = merged['rule_violation_reward']
    require(type(reward) in (int, float) and math.isfinite(reward), 'Invalid rule-violation reward')
    if merged['cluster'] is not None:
        options = merged['cluster']
        scale = options.get('time_scale', 1) if isinstance(options, dict) else None
        require(isinstance(options, dict) and type(options.get('repeats', 1)) is int and options.get('repeats', 1) >= 1
                and type(scale) in (int, float) and math.isfinite(scale) and scale > 0, 'Invalid cluster options')
    if merged['submission_mount'] is not None:
        require(str(merged['submission_mount']).startswith('/') and merged['submission_mount'] != merged['workdir'],
                'Invalid submission mount')
    return merged


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


def load_bindings(path, asset_root, run_root):
    specs = read_json(path)
    require(isinstance(specs, list), 'Bindings must be a list')
    seen = set()
    for spec in specs:
        require(set(spec) == {'source', 'target', 'stages'}, 'Unknown input binding field')
        source = (asset_root / spec['source']).resolve()
        target = Path(spec['target'])
        require(source.exists() and source.is_relative_to(asset_root.resolve()) and not overlaps(source, run_root),
                'Input must be an existing asset under the dedicated asset root')
        require(target.is_absolute() and '..' not in target.parts and len(target.parts) >= 3,
                'Invalid asset target')
        require(target.parts[1] in ('private', 'models', 'assets', 'app', 'workspace'), 'Unexpected asset destination')
        require(isinstance(spec['stages'], list) and spec['stages'] and
                set(spec['stages']) <= {'agent', 'intermediate', 'verifier'}, 'Invalid binding stages')
        require(not ('agent' in spec['stages'] and target.parts[1] == 'private'), 'Private evaluation input exposed to agent')
        for stage in spec['stages']:
            require((stage, str(target)) not in seen, 'Duplicate asset destination')
            seen.add((stage, str(target)))
        spec['source'] = str(source)
    return specs


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
            'tests', 'private', '.git', '__pycache__', '.pytest_cache', 'breakwater.json', 'bindings.example.json'}
        anchored = {item.strip('/') for item in excludes if item.startswith('/')}
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
    parser.add_argument('--task', type=Path, required=True, help='Task directory, for example tasks/open-ended-nano-gpt')
    parser.add_argument('--agent-image', required=True, help='Prepared protected-author overlay, not a bare task runtime')
    parser.add_argument('--intermediate-image', help='Prepared runtime with run_worker.py and stdio_capture.so; defaults to agent image')
    parser.add_argument('--verifier-image', required=True)
    parser.add_argument('--observer-image', required=True)
    parser.add_argument('--gateway-image', required=True)
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--asset-root', type=Path, required=True)
    parser.add_argument('--bindings', type=Path, required=True)
    parser.add_argument('--author-model', required=True)
    parser.add_argument('--author-cli', choices=('codex', 'claude-code'), required=True)
    parser.add_argument('--author-upstream', required=True)
    parser.add_argument('--author-key-env', required=True)
    parser.add_argument('--gpus', default=None, help='device=I[,J...] for the declared native NVIDIA CDI devices; none for CPU tasks')
    # coding-agent-harness: the budgeted development gateway (its own provider key, never in the agent).
    parser.add_argument('--dev-gateway-image', help='breakwater-dev-gateway image (coding-agent-harness only)')
    parser.add_argument('--dev-key-env', default='FIREWORKS_API_KEY', help='Upstream key variable for the development gateway')
    # cluster-serverless: the operator-staged fleet (private manifest stays host-side).
    parser.add_argument('--fleet', type=Path, help='Private fleet manifest from bin/cluster.sh up (cluster only)')
    parser.add_argument('--public-fleet', type=Path, help='Matching public fleet view for the agent and the plane')
    parser.add_argument('--grader-token-env', default='CLUSTER_GRADER_TOKEN')
    parser.add_argument('--receipt-key-env', default='CLUSTER_RECEIPT_KEY', help='Development receipt key (public by design)')
    args = parser.parse_args()
    global TASK
    TASK = args.task.resolve()
    require((TASK / 'breakwater.json').is_file() and (TASK / 'bindings.example.json').is_file(),
            'Not a Breakwater task directory: ' + str(args.task))
    config = load_config()
    selection = args.gpus if args.gpus is not None else ('none' if config['gpus'] == 0 else
                                                         'device=' + ','.join(str(i) for i in range(config['gpus'])))
    gpu_options = gpu_devices(selection, config['gpus'])
    require(args.author_upstream.startswith('https://'), 'Author upstream must use HTTPS')
    require(os.environ.get(args.author_key_env) and os.environ.get('FIREWORKS_API_KEY'), 'Missing author/judge credentials')
    # Fail before authoring, not after it: the evaluation stage needs these host variables.
    missing = [name for name in config['intermediate_env'] if not os.environ.get(name)]
    require(not missing, 'Missing evaluation-stage variables: ' + ', '.join(missing))
    development = config['development_gateway']
    cluster = config['cluster']
    if development:
        require(args.dev_gateway_image and os.environ.get(args.dev_key_env), 'Missing development gateway image or key')
    fleet = public_fleet = None
    if cluster:
        import cluster_fleet
        require(args.fleet and args.public_fleet, 'The cluster task requires --fleet and --public-fleet')
        require(os.environ.get(args.grader_token_env) and os.environ.get(args.receipt_key_env),
                'Missing fleet grader token or development receipt key')
        fleet, public_fleet = read_json(args.fleet), read_json(args.public_fleet)
        cluster_fleet.check_public_fleet(public_fleet, fleet)
        catalogue = read_json(TASK / 'tests' / 'public' / 'catalogue.json')
        cluster_fleet.check_fleet_shape(fleet, catalogue)
        cluster_fleet.check_fleet_shape(public_fleet, catalogue)
        suite = read_json(TASK / 'tests' / 'suite-graded.json')
        qualified = set(suite['qualification'].values()) == {'target-qualified'}
        # Fail before authoring, not after four hours: a development (unqualified) suite needs an
        # explicit declaration in the task contract.
        require(qualified or cluster.get('allow_unqualified'),
                "The graded suite is not target-qualified; declare cluster.allow_unqualified in the task's "
                'breakwater.json to grade it as a development suite')
    run = args.run_dir.resolve()
    require(not run.exists() and not overlaps(run, TASK) and not overlaps(run, args.asset_root), 'Use a fresh, separate run directory')
    bindings = load_bindings(args.bindings, args.asset_root, run)
    expected = {(row['target'], tuple(sorted(row['stages']))) for row in read_json(TASK / 'bindings.example.json')}
    actual = {(row['target'], tuple(sorted(row['stages']))) for row in bindings}
    require(actual == expected, 'Binding destinations/stage visibility must match the trusted task contract')
    # Agent-bound inputs are never part of the submission; neither are the task's snapshot excludes.
    excludes = [spec['target'][len(config['workdir']) + 1:] for spec in bindings
                if 'agent' in spec['stages'] and spec['target'].startswith(config['workdir'] + '/')]
    excludes += config['snapshot_excludes']
    run.mkdir(parents=True, mode=0o700)
    external, authority, local, workspace, evaluation, reports = [run / key for key in
        ('external', 'authority', 'agent-local', 'submission', 'evaluation', 'reports')]
    home = run / 'submission-home'
    local_live, workspace_live = run / 'local-live', run / 'workspace-live'
    baseline_dir = run / 'environment-baseline'
    for folder in (external, authority, local_live, evaluation, reports, baseline_dir):
        folder.mkdir(mode=0o700)
    # These are separate writable output mounts, never parents of trusted storage.
    local_live.chmod(0o777)
    evaluation.chmod(0o777)
    reports.chmod(0o777)
    ids = {'agent': image(args.agent_image), 'intermediate': image(args.intermediate_image or args.agent_image),
           'verifier': image(args.verifier_image), 'observer': image(args.observer_image), 'gateway': image(args.gateway_image)}
    if development:
        ids['dev_gateway'] = image(args.dev_gateway_image)
    seed_workspace(config, ids['agent'], workspace_live)
    if cluster:
        # The agent and the graded plane see the operator's public fleet, never the private manifest.
        (workspace_live / 'fleet.json').unlink(missing_ok=True)
        (workspace_live / 'fleet.json').write_bytes(args.public_fleet.read_bytes())
    workspace_baseline = inventory(workspace_live)  # Also rejects symlinks/special files before exposure.
    atomic_json(external / 'workspace-baseline.json', {'source': config['workspace_source'], 'files': workspace_baseline})
    (external / 'isolation.canary').write_text(secrets.token_hex(32) + '\n')
    identity = {'task': config['task'], 'run_id': secrets.token_hex(16)}
    names = {stage: 'breakwater-' + identity['run_id'][:12] + '-' + stage for stage in
             ('agent', 'intermediate', 'verifier', 'gateway', 'dev-gateway', 'observe-agent', 'observe-intermediate')}
    network = 'breakwater-' + identity['run_id'][:12]
    sandbox_network = network + '-sandbox'
    stages, created = {}, []
    baseline_dir.chmod(0o755)
    protected = [external, authority, reports]
    cluster_private = None
    audit_before = audit_root = None
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
        inspect_boundary(inspection(name), stage, external, protected, gpu_count=config['gpus'])
        return name
    try:
        if ids['intermediate'] != ids['agent']:
            raise RulesViolation('Agent and intermediate must use the same immutable runtime image in distinct containers')
        if cluster:
            import cluster_fleet
            token = os.environ[args.grader_token_env]
            audit_root = cluster_fleet.audit_directory(fleet)
            cluster_fleet.reset(fleet, token)  # No engine from an earlier session survives into authoring.
            audit_before = cluster_fleet.existing_journals(audit_root)
            cluster_private = run / 'cluster-private'
            cluster_private.mkdir(mode=0o700)
            cluster_fleet.write_private(cluster_private, 'grader-token', token.encode())
            cluster_fleet.write_private(cluster_private, 'fleet.json', args.fleet.read_bytes())
            cluster_fleet.write_private(cluster_private, 'fleet.public.json', args.public_fleet.read_bytes())
            atomic_json(external / 'fleet-manifests.json', {'private_sha256': digest(args.fleet),
                        'public_sha256': digest(args.public_fleet), 'public': public_fleet,
                        'run_id': cluster_fleet.run_id(fleet), 'preexisting_journals': audit_before})
        docker('network', 'create', '--internal', network)
        sandbox = config['sandbox_worker']
        if sandbox:
            # rl-swe-agent: the Harbor sandbox worker runs on the host. The agent container reaches it over this
            # bridge network, and the task sandboxes the worker starts reach the agent's session servers through
            # ports published on the host (host.docker.internal:PORT), as on a host-networked node.
            docker('network', 'create', sandbox_network)
        gateway_env = {**os.environ, 'AUTHOR_UPSTREAM_API_KEY': os.environ[args.author_key_env]}
        (external / 'gateway').mkdir(mode=0o700)
        docker('create', '--name', names['gateway'], '--network', network, '--network-alias', 'author-gateway',
               '--user', f'{os.getuid()}:{os.getgid()}', *mount(external / 'gateway', '/logs/author', False),
               '-e', 'AUTHOR_UPSTREAM_API_KEY', '-e', 'AUTHOR_MODEL=' + args.author_model,
               '-e', 'AUTHOR_UPSTREAM_BASE_URL=' + args.author_upstream, '-e', 'AUTHOR_UPSTREAM_AUTH=bearer',
               '-e', 'AUTHOR_PROTOCOL=' + ('responses' if args.author_cli == 'codex' else 'messages'),
               '-e', 'AUTHOR_GATEWAY_LOG=/logs/author/requests.jsonl', ids['gateway'], env=gateway_env)
        created.append(names['gateway'])
        docker('network', 'connect', 'bridge', names['gateway'])
        docker('start', names['gateway'])
        wait_for(lambda: subprocess.run(['docker', 'exec', names['gateway'], 'python3', '-I', '-B', '-c',
            "import urllib.request;urllib.request.urlopen('http://127.0.0.1:8001/health',timeout=2)"],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL).returncode == 0, 30, 'author gateway')
        if development:
            # Only this sidecar holds the development key; worker commands reach it as gateway:8000.
            dev_env = {**os.environ, 'UPSTREAM_API_KEY': os.environ[args.dev_key_env]}
            values = ['UPSTREAM_API_KEY', 'GATEWAY_ROLE=development', 'GATEWAY_SOURCE=development-gateway',
                      'UPSTREAM_BASE_URL=' + development['upstream'],
                      'CHAT_MODELS=' + json.dumps(development['chat_models'], sort_keys=True),
                      'EMBED_MODEL=' + development['embed_model'],
                      f"MAX_CHAT_TOKENS={int(development['max_chat_tokens'])}",
                      f"MAX_EMBED_TOKENS={int(development['max_embed_tokens'])}"]
            service_gateway(names['dev-gateway'], network, 'gateway', external / 'dev-gateway', ids['dev_gateway'],
                            values, dev_env, created, 8000)
        agent_env = ['-e', 'PYTHONDONTWRITEBYTECODE=1', '-e', 'BREAKWATER_AUTHOR_MODEL=' + args.author_model]
        environment = None
        if cluster:
            environment = {**os.environ, 'CLUSTER_RECEIPT_KEY': os.environ[args.receipt_key_env]}
            agent_env += ['-e', 'CLUSTER_RECEIPT_KEY', '-e', 'CLUSTER_NODES=' + config['workdir'] + '/fleet.json',
                          '-e', 'CLUSTER_CATALOGUE=' + config['workdir'] + '/catalogue.json', '-e', 'CLUSTER_FLEET_MODE=real']
        sandbox_options = []
        if sandbox:
            low, high = sandbox['session_ports'].split('-')
            host_gateway = network_gateway('bridge')  # what host.docker.internal resolves to in the sandboxes
            sandbox_options = ['-e', f"BREAKWATER_SANDBOX_WORKER={network_gateway(sandbox_network)}:{sandbox['port']}",
                               '-e', f'BREAKWATER_SESSION_PORTS={low}:{high}',
                               '--publish', f'127.0.0.1:{low}-{high}:{low}-{high}',
                               '--publish', f'{host_gateway}:{low}-{high}:{low}-{high}']
        name = names['agent']
        docker('create', '--name', name, '--label', 'breakwater.stage=agent', '--ipc=private',
               '--network', sandbox_network if sandbox else network, *sandbox_options, *gpu_options, *resource_options(config, agent_resources(config)), '--cap-add=NET_ADMIN', *agent_env,
               *mount(workspace_live, config['workdir'], False), *mount(local_live, '/logs/agent', False),
               *mount(baseline_dir, '/opt/environment-baseline'), *assets('agent'),
               '--entrypoint', 'python3', ids['agent'], '-I', '-B', '/opt/breakwater-runtime/bootstrap.py', env=environment)
        created.append(name)
        if sandbox:
            docker('network', 'connect', network, name)  # the author gateway's network
        for extra in config['agent_extra_networks']:
            # Remote fleet endpoints only; the protected bootstrap firewall admits exact ports.
            require(extra == 'bridge', 'Only the default bridge may be added to authoring')
            docker('network', 'connect', extra, name)
        inspect_boundary(inspection(name), 'agent', external, protected, gpu_count=config['gpus'])
        agent = name
        docker('start', agent)
        wait_for(lambda: subprocess.run(['docker', 'exec', agent, 'test', '-f', '/run/breakwater/ready'],
                                        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL).returncode == 0,
                 60, 'protected author bootstrap')
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
            # Named host variables only (never credentials), and only the host-gateway alias.
            *[item for variable in config['intermediate_env'] for item in ('-e', variable)],
            *[item for host in config['intermediate_add_hosts'] for item in ('--add-host', host)],
            *mount(baseline_dir, '/opt/environment-baseline'), *mount(workspace, submission_target),
            *(mount(home, HOME_MOUNT) if home_sha else []),
            *mount(evaluation, '/evaluation', False), *mount(TASK / 'tests', '/private/tests'),
            *mount(TASK / 'breakwater.json', '/trusted-task/breakwater.json'),
            *mount(TASK / 'bindings.example.json', '/trusted-task/bindings.example.json'),
            *mount(HERE, '/opt/breakwater'),
            *mount(HERE / 'drivers', '/opt/breakwater-drivers'), *assets('intermediate')]
        if cluster_private:
            intermediate_options += mount(cluster_private, '/private/cluster')
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
        if cluster:
            import cluster_fleet
            # Collected only after both execution containers are dead. The host repeats the
            # coordinator's final reset (idempotent) so every worker incarnation of this run has
            # stopped and fsynced its journal, even when evaluation ended abnormally.
            cluster_fleet.reset(fleet, os.environ[args.grader_token_env])
            copied = cluster_fleet.collect(audit_root, external / 'fleet-inference', audit_before)
            journal_report = cluster_fleet.check_journals(external / 'fleet-inference', cluster_fleet.run_id(fleet))
            atomic_json(external / 'fleet-inference.json', {**journal_report, 'collected': copied,
                        'windows': {stage: [stages[stage]['started_wall_ns'], stages[stage]['stopped_wall_ns']]
                                    for stage in ('agent', 'intermediate')}})
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
            '-e', 'HOME=/tmp/verifier-home', '-e', 'XDG_CACHE_HOME=/tmp/verifier-home/cache',
            *mount(external, '/external-evidence'), *mount(authority, '/authority'), *mount(local, '/agent-local'),
            *mount(workspace, config['workdir']), *(mount(home, HOME_MOUNT) if home_sha else []),
            *mount(evaluation, '/evaluation'),
            *mount(reports, '/reports', False), *assets('verifier'), '--entrypoint', 'sleep'], ids['verifier'], ['infinity'])
        stages['verifier'] = {'inspection': inspection(verifier), 'started_ns': time.monotonic_ns(),
                              'probe': {'status': 'INSPECTED', 'method': 'live read-only write probe is required before scoring'}}
        authority_sha = seal(authority, identity, stages=stages, external_sha256=external_sha,
            evaluation_sha256=evaluation_sha, submission_sha256=submission_sha, local_log_sha256=local_sha,
            **({'home_sha256': home_sha} if home_sha else {}),
            external_host_root=str(external), protected_host_roots=[str(path) for path in protected],
            pipeline={key: config[key] for key in SEALED_PIPELINE},
            task_directory=TASK.name)
        authority.chmod(0o555)
        for path in authority.iterdir():
            path.chmod(0o444)
        docker('start', verifier)
        # This is the only stage that can read external evidence or call the judge model.
        code = subprocess.run(['docker', 'exec', '--user=1102:1102', verifier, 'python3', '-I', '-B',
                               '/opt/breakwater/verifier.py', '--authority', authority_sha],
                              timeout=config['judge_timeout_sec'] + 600).returncode
        print(json.dumps({'task': config['task'], 'run_id': identity['run_id'], 'verifier_exit_code': code,
                          'reports': str(reports), 'external_sha256': external_sha, 'authority_sha256': authority_sha}))
        return code
    except BaseException as error:
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
        if cluster and fleet is not None:
            try:
                import cluster_fleet
                cluster_fleet.reset(fleet, os.environ.get(args.grader_token_env, ''))
            except Exception:
                pass
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
        if config['sandbox_worker']:
            docker('network', 'rm', sandbox_network, check=False)


if __name__ == '__main__':
    raise SystemExit(main())
