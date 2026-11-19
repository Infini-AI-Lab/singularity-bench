"""Trusted task plumbing. No provider key or user-written config is inherited."""
import hashlib
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
}


PLAN_KEYS = {'version', 'services', 'endpoints', 'rejects', 'hosts', 'reply_ports', 'env'}
# The worker environment this module fixes (worker_environment), which a plan's environment may not replace.
FIXED_ENV = {'HOME', 'USER', 'LOGNAME', 'LANG', 'SHELL', 'PATH', 'BREAKWATER_WORKDIR', 'PYTHONUNBUFFERED', 'HF_HOME',
             'TORCH_HOME', 'TRITON_CACHE_DIR', 'TORCH_EXTENSIONS_DIR'}
NAME = re.compile(r'(?=.*[A-Za-z])[A-Za-z0-9](?:[A-Za-z0-9.\-]{0,251}[A-Za-z0-9])?')
IPV4 = re.compile(r'\d{1,3}(?:\.\d{1,3}){3}')
VARIABLE = re.compile(r'[A-Z][A-Z0-9_]*')
PORTS = re.compile(r'(\d{1,5})(?::(\d{1,5}))?')


POLICY_KEYS = {'name', 'workdir', 'write_roots', 'local_network'}


def profile():
    """The image's workspace policy (the task's environment/agent-policy.json): its name, workdir, writable roots and
    whether its commands may use the container's own addresses. Everything else comes from the launcher's plan."""
    # Every image installs its task-owned policy. Missing policy is a build error.
    policy = json.loads(Path('/etc/breakwater-task.json').read_text())
    if not isinstance(policy, dict) or set(policy) != POLICY_KEYS:
        raise SystemExit('The image policy has exactly name, workdir, write_roots and local_network')
    return policy


def port(value):
    if type(value) is not int or not 1 <= value <= 65535:
        raise SystemExit('Invalid port in the authoring plan')
    return value


def authoring_plan(environ):
    """The launcher's authoring plan (BREAKWATER_AUTHORING_PLAN: services, endpoints, rejects, hosts, reply ports,
    environment), validated, and the SHA-256 of its exact text, which the bootstrap returns for the launcher to check."""
    text = environ.get('BREAKWATER_AUTHORING_PLAN', '')
    if not text:
        raise SystemExit('BREAKWATER_AUTHORING_PLAN is required; use the protected author launcher')
    plan = json.loads(text)
    if not isinstance(plan, dict) or set(plan) != PLAN_KEYS or plan['version'] != 1:
        raise SystemExit('Unknown authoring plan')
    for service in plan['services']:
        if not isinstance(service, dict) or set(service) != {'host', 'port'} or not NAME.fullmatch(str(service['host'])):
            raise SystemExit('Invalid service in the authoring plan')
        port(service['port'])
    for endpoint in plan['endpoints']:
        if (not isinstance(endpoint, dict) or set(endpoint) != {'name', 'host', 'ports'}
                or not (NAME.fullmatch(str(endpoint['name'])) or IPV4.fullmatch(str(endpoint['name'])))
                or not isinstance(endpoint['host'], str)
                or not isinstance(endpoint['ports'], list) or not endpoint['ports']):
            raise SystemExit('Invalid endpoint in the authoring plan')
        for item in endpoint['ports']:
            match = PORTS.fullmatch(item) if isinstance(item, str) else None
            if not match or not all(1 <= int(part) <= 65535 for part in match.groups() if part) \
                    or (match.group(2) and int(match.group(1)) > int(match.group(2))):
                raise SystemExit('Invalid endpoint ports in the authoring plan')
    for reject in plan['rejects']:
        if not isinstance(reject, dict) or set(reject) != {'host', 'port'} or not isinstance(reject['host'], str):
            raise SystemExit('Invalid reject in the authoring plan')
        port(reject['port'])
    for row in plan['hosts']:
        if not isinstance(row, list) or len(row) != 2 or not NAME.fullmatch(str(row[0])):
            raise SystemExit('Invalid host in the authoring plan')
        ipaddress.IPv4Address(row[1])
    reply = plan['reply_ports']
    if reply is not None and not (isinstance(reply, str) and re.fullmatch(r'\d{4,5}:\d{4,5}', reply)):
        raise SystemExit('Invalid reply ports in the authoring plan')
    env = plan['env']
    if not isinstance(env, dict) or not all(
            isinstance(name, str) and VARIABLE.fullmatch(name) and name not in FIXED_ENV
            and not name.startswith(('LD_', 'PYTHON')) and isinstance(value, str) and '\0' not in value
            and len(value) <= 65536 for name, value in env.items()):
        raise SystemExit('Invalid environment in the authoring plan')
    return plan, hashlib.sha256(text.encode()).hexdigest()


def worker_environment(task, inherited, plan=None):
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
    # The task's authoring environment (its model-access address and aliases, say), from the launcher's plan.
    env.update((plan or {}).get('env') or {})
    return env


def write_environment(task, plan=None):
    env = worker_environment(task, os.environ, plan)
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


def worker_destinations(task, resolve=resolve4, hostname=None):
    """Exact destinations only; never Docker subnet/provider-host wildcards."""
    destinations = []
    if task['local_network']:
        # All ranks and local servers must work, including NCCL using eth0.
        destinations += [(ip, None) for ip in sorted({'127.0.0.1', *resolve((hostname or socket.gethostname)())})]
    return sorted(set(destinations), key=str)


def endpoint_addresses(host, resolve):
    """IPv4 addresses of a plan endpoint's host: the literal address, else what the bootstrap's resolver returns."""
    try:
        return [str(ipaddress.IPv4Address(host))]
    except ValueError:
        addresses = resolve(host)
    for address in addresses:
        ipaddress.IPv4Address(address)
    if not addresses:
        raise SystemExit(f'Cannot resolve the endpoint host {host}')
    return addresses


def firewall_plan(task, plan, resolved, resolve=resolve4, hostname=None):
    """The worker's /etc/hosts lines and the iptables commands, in order, from the image policy (`task`), the launcher's
    authoring plan and the gateways and services the bootstrap resolved and checked (`resolved`: (uid, host, port,
    addresses)). Pure: names are resolved only through `resolve` and `hostname`; nothing is applied here."""
    framework = {address for _, _, _, addresses in resolved for address in addresses}
    endpoint_hosts, endpoint_destinations = [], []
    for endpoint in plan['endpoints']:
        addresses = endpoint_addresses(endpoint['host'], resolve)
        if framework & set(addresses):
            # Otherwise a plan could hand the worker the author gateway or a service under another name.
            raise SystemExit(f"The endpoint {endpoint['name']} resolves to a framework service address")
        if not IPV4.fullmatch(endpoint['name']):  # an address needs no host name
            endpoint_hosts.append((endpoint['name'], addresses[0]))
        endpoint_destinations += [(address, ports) for address in addresses for ports in endpoint['ports']]
    denied = [(address, str(row['port'])) for row in plan['rejects'] for address in endpoint_addresses(row['host'], resolve)]
    hosts = ''.join(f"\n{addresses[0]} {host}\n" for _, host, _, addresses in resolved)
    hosts += ''.join(f"\n{address} {name}\n" for name, address in endpoint_hosts)
    hosts += ''.join(f"\n{address} {name}\n" for name, address in plan['hosts'])
    destinations = worker_destinations(task, resolve, hostname) + endpoint_destinations
    # No private-network, DNS or ESTABLISHED accept rule. Local GPU ranks and task servers get only this container,
    # plus the plan's endpoints.
    commands = []
    for executable in ("iptables", "ip6tables"):
        commands += [[executable, "-P", "OUTPUT", "DROP"], [executable, "-F", "OUTPUT"]]
        if executable == "iptables":
            commands.append([executable, "-A", "OUTPUT", "-d", "127.0.0.11", "-j", "REJECT"])
            # Ports the plan refuses (coordinator-only ports on a host the agent also reaches) precede every accept.
            for address, port in denied:
                commands.append([executable, '-A', 'OUTPUT', '-m', 'owner', '--uid-owner', str(WORKER),
                                 '-p', 'tcp', '-d', address, '--dport', port, '-j', 'REJECT'])
            for uid in (WORKER, 0):
                for address, ports in destinations:
                    rule = [executable, "-A", "OUTPUT", "-m", "owner", "--uid-owner", str(uid), "-d", address]
                    if ports:
                        rule += ["-p", "tcp", "--dport", ports]
                    commands.append(rule + ["-j", "ACCEPT"])
            for uid, _, port, addresses in resolved:
                for address in addresses:
                    commands.append([executable, "-A", "OUTPUT", "-m", "owner", "--uid-owner", str(uid),
                                     "-p", "tcp", "-d", address, "--dport", str(port), "-j", "ACCEPT"])
            if plan['reply_ports']:
                # Inbound connections to servers the agent's commands run: replies only. A new outbound connection
                # from these source ports is still rejected.
                commands.append([executable, "-A", "OUTPUT", "-m", "owner", "--uid-owner", str(WORKER), "-p", "tcp",
                                 "--sport", plan['reply_ports'], "-m", "conntrack", "--ctstate", "ESTABLISHED",
                                 "-j", "ACCEPT"])
        commands.append([executable, "-A", "OUTPUT", "-j", "REJECT"])
    return hosts, commands
