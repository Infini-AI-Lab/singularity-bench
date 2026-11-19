#!/usr/bin/env python3
"""Probe a node for what an unprivileged (Apptainer + namespaces) Breakwater backend needs.

    python3 -I probe.py [--image TASK.sif] [--json OUT.json]

Run it as an ordinary user on a login node and on a GPU compute node inside a Slurm allocation
(srun --pty ...). It only reads system settings and starts short-lived sandboxes under a fresh
temporary directory; it changes nothing on the host. With --image (any SIF, ideally a task
runtime), it also checks the same features inside an Apptainer container.
"""
import argparse
import ctypes
import json
import os
from pathlib import Path
import platform
import resource
import shutil
import subprocess
import sys
import tempfile

RESULTS = {}
# Interfaces of the calling process's network namespace (/sys would show the mount's namespace instead).
NET_DEVICES = "tail -n +3 /proc/net/dev | cut -d: -f1 | tr -d ' ' | tr '\\n' ' '"


def run(command, timeout=60, **kwargs):
    try:
        result = subprocess.run(command, capture_output=True, text=True, timeout=timeout, **kwargs)
        return result.returncode, (result.stdout + result.stderr).strip()[-2000:]
    except FileNotFoundError:
        return 127, 'not found'
    except subprocess.TimeoutExpired:
        return 124, 'timed out'


def record(name, ok, detail=''):
    RESULTS[name] = {'ok': ok, 'detail': detail}
    mark = {True: 'ok  ', False: 'FAIL', None: 'n/a '}[ok]
    print(f'[{mark}] {name}: {detail}'.rstrip(': '), flush=True)


def check(name, command, expect=None, timeout=60, **kwargs):
    code, output = run(command, timeout=timeout, **kwargs)
    ok = code == 0 and (expect is None or expect in output)
    record(name, ok, output.splitlines()[-1] if output else f'exit {code}')
    return ok, output


def read(path):
    try:
        return Path(path).read_text().strip()
    except OSError:
        return None


def system():
    os_release = dict(line.split('=', 1) for line in (read('/etc/os-release') or '').splitlines() if '=' in line)
    record('kernel', True, f"{platform.release()} {platform.machine()} ({os_release.get('PRETTY_NAME', '?').strip(chr(34))})")
    for key, path in (('max_user_namespaces', '/proc/sys/user/max_user_namespaces'),
                      ('unprivileged_userns_clone', '/proc/sys/kernel/unprivileged_userns_clone'),
                      ('apparmor_restrict_unprivileged_userns', '/proc/sys/kernel/apparmor_restrict_unprivileged_userns'),
                      ('apparmor_restrict_unprivileged_unconfined', '/proc/sys/kernel/apparmor_restrict_unprivileged_unconfined'),
                      ('yama_ptrace_scope', '/proc/sys/kernel/yama/ptrace_scope')):
        value = read(path)
        RESULTS['sysctl.' + key] = value
        print(f'       sysctl {key} = {value if value is not None else "absent"}')
    for tool in ('apptainer', 'singularity', 'bwrap', 'unshare', 'nsenter', 'strace', 'pasta', 'slirp4netns', 'nft',
                 'ip', 'squashfuse', 'fuse-overlayfs', 'newuidmap', 'nvidia-smi', 'srun'):
        path = shutil.which(tool)
        version = ''
        if path and tool in ('apptainer', 'singularity', 'bwrap', 'strace', 'pasta', 'nft'):
            version = run([path, '--version'], timeout=10)[1].splitlines()[0] if run([path, '--version'], timeout=10)[1] else ''
        RESULTS['tool.' + tool] = {'path': path, 'version': version}
        print(f'       tool {tool:15s} {path or "-"} {version}')
    user = os.environ.get('USER') or str(os.getuid())
    subuid = [line for line in (read('/etc/subuid') or '').splitlines() if line.split(':')[0] in (user, str(os.getuid()))]
    # Informational: the Apptainer backend needs no subordinate UID range, so its absence is not a failure.
    record('subuid_range', True if subuid else None, subuid[0] if subuid else 'none (root-mapped user namespaces only)')
    limits = {name: resource.getrlimit(getattr(resource, name)) for name in
              ('RLIMIT_NPROC', 'RLIMIT_NOFILE', 'RLIMIT_MEMLOCK', 'RLIMIT_STACK', 'RLIMIT_AS')}
    RESULTS['rlimits'] = limits
    print('       rlimits ' + ', '.join(f'{k[7:]}={v[0]}/{v[1]}' for k, v in limits.items()))
    cgroup = read('/proc/self/cgroup') or ''
    path = cgroup.splitlines()[-1].split(':', 2)[-1] if cgroup else ''
    base = Path('/sys/fs/cgroup') / path.lstrip('/')
    record('cgroup_delegated', os.access(base / 'cgroup.procs', os.W_OK) and os.access(base, os.W_OK),
           f'{"v2" if Path("/sys/fs/cgroup/cgroup.controllers").exists() else "v1"} {path}')
    RESULTS['slurm'] = {'job': os.environ.get('SLURM_JOB_ID'), 'munge_socket': [p for p in
                        ('/run/munge/munge.socket.2', '/var/run/munge/munge.socket.2') if Path(p).exists()],
                        'slurm_conf': [p for p in ('/etc/slurm/slurm.conf', '/etc/slurm-llnl/slurm.conf') if Path(p).exists()]}
    print(f"       slurm job={RESULTS['slurm']['job']} munge={RESULTS['slurm']['munge_socket']} conf={RESULTS['slurm']['slurm_conf']}")
    devices = sorted(p.name for p in Path('/dev').glob('nvidia*'))
    record('gpu_devices', bool(devices) or None, ' '.join(devices[:8]) or 'none')


def landlock():
    try:
        libc = ctypes.CDLL(None, use_errno=True)
        number = {'x86_64': 444, 'aarch64': 444}.get(platform.machine())
        version = libc.syscall(number, None, 0, 1) if number else -1  # LANDLOCK_CREATE_RULESET_VERSION
        record('landlock_abi', version > 0, f'ABI {version}' + (' (network rules need ABI 4)' if 0 < version < 4 else '')
               if version > 0 else f'unavailable (errno {ctypes.get_errno()})')
    except Exception as error:  # noqa: BLE001 - report, never fail
        record('landlock_abi', False, repr(error))
    status = dict(line.split(':', 1) for line in (read('/proc/self/status') or '').splitlines() if ':' in line)
    record('seccomp', 'Seccomp' in status, f"Seccomp field {status.get('Seccomp', '?').strip()}")


def namespaces(work):
    unshare = shutil.which('unshare')
    if not unshare:
        record('userns', False, 'unshare not found')
        return
    check('userns', [unshare, '-Ur', 'id', '-u'], expect='0')
    check('userns_nested', [unshare, '-Ur', unshare, '-Ur', 'id', '-u'], expect='0')
    check('pidns_with_fresh_proc', [unshare, '-Urpf', '--mount-proc', 'sh', '-c', 'echo pid=$$ procs=$(ls /proc | grep -c "^[0-9]")'],
          expect='pid=1 ')
    code, output = run([unshare, '-Urn', 'sh', '-c', NET_DEVICES])
    record('netns_loopback_only', code == 0 and output.split() == ['lo'], output or f'exit {code}')
    # The process wall: a daemonized grandchild dies with its PID namespace's init.
    marker = work / 'survivor'
    run([unshare, '-Urpf', '--mount-proc', 'sh', '-c',
         f'setsid sh -c "sleep 3; touch {marker}" >/dev/null 2>&1 & sleep 0.2; exit 0'], timeout=20)
    run(['sleep', '4'], timeout=10)
    record('pidns_kills_daemonized_children', not marker.exists(), 'survivor absent' if not marker.exists() else 'a detached process outlived its namespace')
    # Read-only bind mounts inherited by a less privileged mount namespace stay read-only.
    source = work / 'ro'
    source.mkdir()
    (source / 'f').write_text('x')
    script = (f'mkdir -p {work}/mnt && mount --bind {source} {work}/mnt && mount -o remount,ro,bind {work}/mnt && '
              f'{unshare} -Urm sh -c "mount -o remount,rw,bind {work}/mnt 2>&1; echo ok > {work}/mnt/f 2>&1; echo done"')
    code, output = run([unshare, '-Urm', 'sh', '-c', script], timeout=20)
    record('ro_bind_locked_in_child_userns', 'done' in output and read(source / 'f') == 'x', output.splitlines()[0] if output else f'exit {code}')
    if shutil.which('nft'):
        check('nftables_in_user_netns', [unshare, '-Urn', 'sh', '-c', 'nft add table inet probe && nft list tables'], expect='probe')
    if shutil.which('strace'):
        check('strace_into_userns_children', ['strace', '-f', '-qq', '-e', 'trace=execve', '-o', str(work / 'trace'),
               unshare, '-Urpf', '--mount-proc', 'sh', '-c', 'true'])
        record('strace_saw_children', 'execve' in (read(work / 'trace') or ''), 'execve events recorded')


def bwrap(work):
    path = shutil.which('bwrap')
    if not path:
        record('bwrap', None, 'not installed (could be shipped inside the image)')
        return
    base = [path, '--unshare-all', '--die-with-parent', '--ro-bind', '/', '/', '--proc', '/proc', '--dev', '/dev',
            '--tmpfs', '/tmp']
    check('bwrap_unshare_all', [*base, 'sh', '-c', 'echo pid=$$'], expect='pid=')
    check('bwrap_nested', [*base, path, '--unshare-all', '--ro-bind', '/', '/', '--proc', '/proc', '--dev', '/dev',
                           'sh', '-c', 'echo nested'], expect='nested')
    nvidia = [item for name in sorted(Path('/dev').glob('nvidia*')) for item in ('--dev-bind', str(name), str(name))]
    if nvidia and shutil.which('nvidia-smi'):
        check('bwrap_gpu', [path, '--unshare-all', '--ro-bind', '/', '/', '--proc', '/proc', '--dev', '/dev', *nvidia,
                            'nvidia-smi', '-L'], expect='GPU')


def apptainer(work, image):
    tool = shutil.which('apptainer') or shutil.which('singularity')
    if not tool:
        record('apptainer', None, 'not installed')
        return
    code, output = run([tool, 'buildcfg'], timeout=20)
    RESULTS['apptainer.buildcfg'] = output[-1500:]
    sysconf = next((Path(line.split('=', 1)[1]) for line in output.splitlines() if line.startswith('SYSCONFDIR=')), Path('/etc'))
    # A relocated unprivileged install reads <prefix>/etc (prefix = the directory above bin/), whatever buildcfg says.
    relocated = Path(tool).resolve().parents[1] / 'etc'
    config = next((path for directory in (relocated, sysconf) for path in (
        directory / 'apptainer/apptainer.conf', directory / 'apptainer.conf', directory / 'singularity/singularity.conf')
        if path.exists()), None)
    RESULTS['apptainer.conf_path'] = str(config) if config else None
    settings = {}
    if config and config.exists():
        for line in config.read_text().splitlines():
            if '=' in line and not line.lstrip().startswith('#'):
                key, value = (part.strip() for part in line.split('=', 1))
                if key in ('allow setuid', 'allow user ns', 'max loop devices', 'allow net users', 'allow net networks',
                           'allow pid ns', 'mount proc', 'mount dev', 'enable overlay', 'enable fusemount',
                           'enable underlay'):
                    settings[key] = value
    RESULTS['apptainer.conf'] = settings
    print(f'       apptainer.conf {config or "not found"} {settings}')
    if not image:
        record('apptainer_exec', None, 'skipped: pass --image FILE.sif to test inside a container')
        return
    common = [tool, 'exec', '--containall', '--no-home', '--cleanenv', '--pwd', '/', '--bind', f'{work}:/probe']
    check('apptainer_exec_containall', [*common, image, 'sh', '-c', 'echo uid=$(id -u) pid=$$'], expect='uid=')
    code, output = run([*common, '--pid', image, 'sh', '-c', 'echo "init=$(cat /proc/1/comm) procs=$(ls /proc | grep -c "^[0-9]")"'])
    last = output.splitlines()[-1] if output else ''
    fields = dict(item.split('=', 1) for item in last.split() if '=' in item)
    record('apptainer_pid_namespace', code == 0 and fields.get('init') not in (None, 'systemd', 'init')
           and fields.get('procs', '999').isdigit() and int(fields['procs']) < 50, last or f'exit {code}')
    code, output = run([*common, '--net', '--network', 'none', image, 'sh', '-c', NET_DEVICES])
    last = output.splitlines()[-1] if output else ''
    record('apptainer_net_none', code == 0 and last.split() == ['lo'], last or f'exit {code}')
    check('apptainer_nested_userns', [*common, image, 'unshare', '-Urpf', '--mount-proc', 'id', '-u'], expect='0')
    # --containall gives a minimal /dev without /dev/full, which bwrap's own --dev /dev would bind; reuse the
    # container's /dev, which is already a private minimal one.
    code, output = run([*common, image, 'bwrap', '--unshare-all', '--ro-bind', '/', '/', '--proc', '/proc',
                        '--dev-bind', '/dev', '/dev', 'true'])
    if 'executable file not found' in output:
        record('apptainer_nested_bwrap', None, 'bwrap is not in this image (add bubblewrap to the image to test)')
    else:
        record('apptainer_nested_bwrap', code == 0, output.splitlines()[-1] if output else f'exit {code}')
    name = 'breakwater-probe-' + str(os.getpid())
    ok, _ = check('apptainer_instance_start', [tool, 'instance', 'start', '--containall', '--no-home', image, name])
    if ok:
        check('apptainer_instance_exec', [tool, 'exec', f'instance://{name}', 'sh', '-c', 'echo joined pid=$$'], expect='joined')
        run([tool, 'instance', 'stop', name], timeout=30)
    if Path('/dev/nvidiactl').exists():
        check('apptainer_gpu', [*common, '--nv', image, 'nvidia-smi', '-L'], expect='GPU')


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--image', help='a SIF image to test inside (optional)')
    parser.add_argument('--json', type=Path, help='also write the results as JSON')
    args = parser.parse_args()
    with tempfile.TemporaryDirectory(prefix='breakwater-probe-') as temporary:
        work = Path(temporary)
        system()
        landlock()
        namespaces(work)
        bwrap(work)
        apptainer(work, args.image)
    if args.json:
        args.json.write_text(json.dumps(RESULTS, indent=1, default=str) + '\n')
    failed = [name for name, value in RESULTS.items() if isinstance(value, dict) and value.get('ok') is False]
    print('\nfailed checks: ' + (', '.join(failed) or 'none'))
    return 0


if __name__ == '__main__':
    sys.exit(main())
