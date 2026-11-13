"""Trusted task plumbing. No provider key or user-written config is inherited."""
import ipaddress
import json
import os
from pathlib import Path
import re
import socket
import stat

AUTHOR, WORKER, SHARED = 1100, 1101, 1200
ENV_KEYS = {
    'PATH', 'VIRTUAL_ENV', 'PYTHONPATH', 'LD_LIBRARY_PATH', 'JAVA_HOME',
    'CUDA_HOME', 'CUDA_PATH', 'CUDA_VISIBLE_DEVICES', 'CUDA_DEVICE_ORDER',
    'NVIDIA_VISIBLE_DEVICES', 'NVIDIA_DRIVER_CAPABILITIES', 'TORCH_CUDA_ARCH_LIST',
    'HF_HUB_OFFLINE', 'TRANSFORMERS_OFFLINE', 'HF_HUB_DISABLE_TELEMETRY', 'HF_HUB_DISABLE_XET',
    'TOKENIZERS_PARALLELISM', 'PYTHONUNBUFFERED', 'OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS',
    'NCCL_SOCKET_IFNAME', 'NCCL_IB_DISABLE', 'NCCL_P2P_DISABLE', 'NCCL_SHM_DISABLE',
    'NCCL_NET_GDR_LEVEL', 'NCCL_CUMEM_ENABLE', 'NCCL_DEBUG', 'PYTORCH_CUDA_ALLOC_CONF',
    'CLUSTER_RECEIPT_KEY', 'CLUSTER_NODES', 'CLUSTER_CATALOGUE', 'CLUSTER_FLEET_MODE',
}


def profile():
    # Every image installs its task-owned policy. Missing policy is a build error.
    return json.loads(Path('/etc/breakwater-task.json').read_text())


def worker_environment(task, inherited):
    env = {key: value for key, value in inherited.items() if key in ENV_KEYS}
    env.update(HOME='/home/worker', USER='worker', LOGNAME='worker', LANG='C.UTF-8',
               SHELL='/bin/bash', BREAKWATER_WORKDIR=task['workdir'], PYTHONUNBUFFERED='1')
    # Keep CUDA and the task virtualenv, never the former author's PATH entries.
    paths = [p for p in env.get('PATH', '/usr/local/bin:/usr/bin:/bin').split(':')
             if p.startswith('/') and not p.startswith(('/root', '/home/author', '/opt/author-'))]
    env['PATH'] = ':'.join(dict.fromkeys(paths + ['/usr/local/bin', '/usr/bin', '/bin']))
    env.update(HF_HOME='/home/worker/.cache/huggingface',
               TORCH_HOME='/home/worker/.cache/torch', TRITON_CACHE_DIR='/home/worker/.triton',
               TORCH_EXTENSIONS_DIR='/home/worker/.cache/torch_extensions')
    if task['development_gateway']:
        env.update(HARNESS_LLM_BASE_URL='http://cell-gateway:8002/v1' if task.get('downstream_full_evaluation') else 'http://gateway:8000/v1', HARNESS_CHAT_MODEL='chat', HARNESS_EMBED_MODEL='embed')
    return env


def write_environment(task):
    env = worker_environment(task, os.environ)
    path = Path('/run/breakwater/worker.env')
    path.write_bytes(b''.join((key + '=' + value).encode() + b'\0' for key, value in sorted(env.items())))
    path.chmod(0o644)
    # Device groups supplement the workspace group, without granting author/root
    # group membership. Device nodes themselves are never chmod'ed or hidden.
    groups = {SHARED}
    for pattern in ('nvidia*', 'nvidia-caps/*', 'dri/*', 'infiniband/*'):
        for device in Path('/dev').glob(pattern):
            info = device.stat()
            if stat.S_ISCHR(info.st_mode) and info.st_gid not in (0, AUTHOR):
                groups.add(info.st_gid)
    Path('/run/breakwater/worker-groups').write_text(' '.join(map(str, sorted(groups))) + '\n')
    Path('/run/breakwater/worker-groups').chmod(0o644)
    return env


def grant_workspaces(task):
    # Never recurse into bound datasets/checkpoints or follow submission symlinks.
    mounts = {line.split()[4].replace('\\040', ' ') for line in Path('/proc/self/mountinfo').read_text().splitlines()}
    for root in task['write_roots']:
        Path(root).mkdir(parents=True, exist_ok=True)
        for directory, dirs, files in os.walk(root, followlinks=False):
            dirs[:] = [name for name in dirs if str(Path(directory, name)) not in mounts
                       and not Path(directory, name).is_symlink()]
            for path in [Path(directory), *(Path(directory, name) for name in files)]:
                if path.is_symlink() or (str(path) in mounts and str(path) != root):
                    continue
                info = path.stat()
                if not (stat.S_ISREG(info.st_mode) or stat.S_ISDIR(info.st_mode)):
                    continue
                os.chown(path, WORKER, SHARED)
                path.chmod((info.st_mode & 0o1777) | (0o2775 if path.is_dir() else 0o660))


def resolve4(host):
    return sorted({row[4][0] for row in socket.getaddrinfo(host, None, socket.AF_INET, socket.SOCK_STREAM)})


POOL_NAME = re.compile(r'^[A-Za-z0-9][A-Za-z0-9.-]{0,62}$')


def pool_endpoints(task):
    """Agent-pool builder access: the login node's and each tray's SSH port and the development
    service, nothing else. Returns ((name, address) for /etc/hosts, (address, port) destinations)."""
    if not task.get('pool_file'):
        return [], []
    pool = json.loads(Path(task['pool_file']).read_text())
    trays, service, login = pool.get('trays'), pool.get('development'), pool.get('login')
    if not trays or not isinstance(service, dict) or (login is not None and not isinstance(login, dict)):
        raise ValueError('Pool needs trays and a development service')
    machines = [login, *trays] if login else trays
    endpoints = [(m['name'], m.get('address') or m['name'], m.get('ssh_port', 22)) for m in machines]
    endpoints.append((service['host'], service['host'], service['port']))
    hosts, destinations = [], []
    for name, host, port in endpoints:
        port = int(port)
        if not POOL_NAME.match(name) or not 1 <= port <= 65535:
            raise ValueError('Invalid pool endpoint')
        for address in resolve4(host):
            ipaddress.IPv4Address(address)
            destinations.append((address, str(port)))
            if name != address:
                hosts.append((name, address))
    return sorted(set(hosts)), sorted(set(destinations))


def worker_destinations(task):
    """Exact destinations only; never Docker subnet/provider-host wildcards."""
    destinations = []
    if task['local_network']:
        # All ranks and local servers must work, including NCCL using eth0.
        destinations += [(ip, None) for ip in sorted({'127.0.0.1', *resolve4(socket.gethostname())})]
    if task.get('fleet_file'):
        fleet = json.loads(Path(task['fleet_file']).read_text())
        if not fleet.get('nodes') or not fleet.get('controller'):
            raise ValueError('Fleet needs nodes and controller')
        for node in fleet['nodes']:
            for ip in resolve4(node.get('host') or fleet['host']):
                ipaddress.IPv4Address(ip)
                destinations.append((ip, '9101:9299'))
        controller = fleet['controller']
        port = int(controller['port'])
        if not 1 <= port <= 65535:
            raise ValueError('Invalid fleet controller port')
        destinations += [(ip, str(port)) for ip in resolve4(controller.get('host') or fleet['host'])]
    destinations += pool_endpoints(task)[1]
    return sorted(set(destinations), key=str)


def fleet_addresses(task):
    hosts, denied = list(pool_endpoints(task)[0]), []
    if not task.get('fleet_file'):
        return sorted(set(hosts)), denied
    fleet = json.loads(Path(task['fleet_file']).read_text())
    for endpoint in [*fleet['nodes'], fleet['controller']]:
        host = endpoint.get('host') or fleet['host']
        for address in resolve4(host):
            hosts.append((host, address))
            if endpoint is not fleet['controller']:
                denied.append((address, str(int(endpoint['port']))))
    return sorted(set(hosts)), sorted(set(denied))
