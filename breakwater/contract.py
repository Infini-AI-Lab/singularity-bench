"""Host-sealed inputs and Docker boundary checks. No agent-supplied authority."""
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import stat

PROTOCOL = 'breakwater-v1'
STAGES = ('agent', 'intermediate', 'verifier')
MODEL = 'accounts/fireworks/models/deepseek-v4p1-flash-1017'
CHECKS = ('external_integrity', 'three_containers', 'log_isolation', 'capture_complete',
          'local_external_agreement', 'sealed_submission', 'workload_compliance', 'no_tampering',
          'no_lingering_processes', 'runtime_package_integrity')
# Provider/judge/fleet-admin secrets. None may appear in an execution container's
# environment; the gateways, the trusted fleet and the final verifier hold them.
CREDENTIAL_ENV = ('FIREWORKS_API_KEY', 'FIREWORKS_AI_API_KEY', 'OPENAI_API_KEY', 'ANTHROPIC_API_KEY',
                  'BREAKWATER_AUTHOR_API_KEY', 'AUTHOR_UPSTREAM_API_KEY', 'UPSTREAM_API_KEY',
                  'DEV_UPSTREAM_API_KEY', 'DRIVER_UPSTREAM_API_KEY', 'HARBOR_JUDGE_API_KEY',
                  'CLUSTER_GRADER_TOKEN', 'GATEWAY_CTL_TOKEN')


class ContractError(ValueError):
    pass


class RulesViolation(ContractError):
    pass


def require(condition, message):
    if not condition:
        raise ContractError(message)


def read_json(path):
    def pairs(items):
        value = {}
        for key, item in items:
            require(key not in value, 'Duplicate JSON key')
            value[key] = item
        return value
    return json.loads(Path(path).read_bytes(), object_pairs_hook=pairs,
                      parse_constant=lambda value: (_ for _ in ()).throw(ContractError('Nonfinite JSON')))


def atomic_json(path, value):
    path = Path(path)
    pending = path.with_name(path.name + '.pending')
    with pending.open('x') as stream:
        json.dump(value, stream, sort_keys=True, allow_nan=False)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())
    pending.replace(path)


def digest(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024**2), b''):
            value.update(chunk)
    return value.hexdigest()


def regular(root, relative):
    relative = PurePosixPath(relative)
    require(not relative.is_absolute() and relative.parts and
            all(part not in ('', '.', '..') for part in relative.parts), 'Unsafe evidence path')
    path = Path(root)
    require(not path.is_symlink(), 'Symlinked evidence root')
    for part in relative.parts:
        path = path / part
        require(not path.is_symlink(), 'Symlinked evidence entry')
    require(path.is_file() and stat.S_ISREG(path.stat().st_mode), 'Missing evidence file')
    return path


def inventory(root, skip=()):
    root = Path(root)
    require(root.is_dir() and not root.is_symlink(), 'Invalid sealed root')
    files = {}
    for directory, dirs, names in os.walk(root, followlinks=False):
        for name in dirs:
            require(not (Path(directory) / name).is_symlink(), 'Directory symlink in sealed input')
        for name in names:
            path = Path(directory) / name
            relative = path.relative_to(root).as_posix()
            if relative in skip:
                continue
            require(stat.S_ISREG(path.lstat().st_mode) and not path.is_symlink(), 'Nonregular sealed input')
            require(path.stat().st_nlink == 1, 'Hard-linked sealed input')
            files[relative] = {'sha256': digest(path), 'bytes': path.stat().st_size}
    return files


def seal(root, identity, **metadata):
    manifest = {'protocol': PROTOCOL, 'identity': identity, 'files': inventory(root, ('manifest.json',)), **metadata}
    atomic_json(Path(root) / 'manifest.json', manifest)
    return digest(Path(root) / 'manifest.json')


def verify_seal(root, expected, identity=None):
    require(isinstance(expected, str) and len(expected) == 64, 'Missing host manifest authority')
    path = regular(root, 'manifest.json')
    require(digest(path) == expected, 'Manifest differs from trusted host digest')
    value = read_json(path)
    require(value.get('protocol') == PROTOCOL, 'Wrong evidence protocol')
    if identity is not None:
        require(value.get('identity') == identity, 'Wrong trial identity')
    require(value.get('files') == inventory(root, ('manifest.json',)), 'Sealed input changed, missing or added files')
    return value


def overlaps(first, second):
    first, second = Path(first).resolve(), Path(second).resolve()
    return first == second or first.is_relative_to(second) or second.is_relative_to(first)


def inspect_boundary(inspection, stage, external_root, protected_roots=(), gpu_count=2):
    require(stage in STAGES, 'Unknown stage')
    require(inspection.get('Image', '').startswith('sha256:') and len(inspection['Image']) == 71, 'Unpinned execution image')
    host = inspection['HostConfig']
    require(not host.get('Privileged'), 'Privileged workload container')
    require(host.get('PidMode', '') not in ('host',) and not host.get('PidMode', '').startswith('container:'), 'Shared PID namespace')
    require(not host.get('IpcMode', '').startswith(('host', 'container:')), 'Shared IPC namespace')
    require(not host.get('Devices') and not host.get('VolumesFrom'), 'Unexpected device or inherited volumes')
    requests = host.get('DeviceRequests') or []
    require(type(gpu_count) is int and 0 <= gpu_count <= 8, 'Invalid declared GPU count')
    if requests:
        import re
        # The task declares its exact device count (two by default; one for harness
        # authoring/cells; zero for the CPU control-plane task). Never more, never "all".
        require(stage != 'verifier' and gpu_count > 0 and len(requests) == 1, 'Unexpected GPU access')
        request = requests[0]
        ids = request.get('DeviceIDs') or []
        require(request.get('Driver') == 'cdi' and request.get('Count') == 0
                and len(ids) == len(set(ids)) == gpu_count and all(re.fullmatch(r'nvidia\.com/gpu=\d+', item) for item in ids)
                and not request.get('Options') and not request.get('Capabilities'),
                'Only the declared native NVIDIA CDI devices are allowed')
    require(host.get('NetworkMode') != 'host', 'Host network is prohibited')
    mounts = inspection.get('Mounts', [])
    external = []
    for mount in mounts:
        source, target = mount.get('Source', ''), mount['Destination']
        require('docker.sock' not in source and 'docker.sock' not in target and
                target not in ('/', '/proc', '/sys', '/dev', '/var/run', '/run'), 'Host-control mount')
        if source and overlaps(source, external_root):
            external.append(mount)
        if stage != 'verifier' and source:
            require(not any(overlaps(source, path) for path in protected_roots), 'Trusted evidence exposed to execution')
    if stage == 'agent':
        baseline = [item for item in mounts if item['Destination'] == '/opt/environment-baseline']
        require(len(baseline) == 1 and baseline[0]['RW'] is False, 'Agent runtime inventory must be mounted read-only')
    if stage == 'verifier':
        require(len(external) == 1 and external[0]['Destination'] == '/external-evidence'
                and external[0]['RW'] is False, 'Verifier external evidence must be one read-only mount')
        require(host.get('ReadonlyRootfs') is True, 'Writable verifier root filesystem')
        require('ALL' in (host.get('CapDrop') or []) and not host.get('CapAdd'), 'Verifier capabilities')
        require(any(value.startswith('no-new-privileges') for value in host.get('SecurityOpt', [])), 'Verifier can gain privilege')
        require(inspection['Config'].get('User') == '1102:1102', 'Verifier must be an unprivileged identity')
    else:
        require(not external, 'External evidence is mounted in an execution container')
    keys = [value.partition('=')[0] for value in inspection['Config'].get('Env', [])]
    if stage != 'verifier':
        require(not any(name in keys for name in CREDENTIAL_ENV), 'Provider credential in agent/intermediate')
    return {'status': 'PASS', 'stage': stage, 'container_id': inspection['Id']}
