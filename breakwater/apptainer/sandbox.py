"""Stage sandboxes for the unprivileged (Apptainer) backend.

A sandbox is one foreground ``apptainer exec --userns --containall`` per stage: its own user, mount, PID and IPC
namespaces, its own network namespace unless the stage needs the host network, the read-only image, its own
disk-backed /tmp and /var/tmp, and only the binds the launcher names: site-wide binds from apptainer.conf (``bind
path``, ``mount hostfs``), which could include the run directory, never apply. A sandbox without the host network
also has its own hostname (UTS namespace), which its /etc/hosts resolves to the loopback address, as a Docker
container's does to its own address: tools that rendezvous on the hostname (``torchrun --standalone``) then work
with loopback only. Its init waits until the stage ends.

Every process the launcher runs in a sandbox is started by that sandbox's spawner (spawner.py), which also starts
the container. With a Slurm ``prefix`` (``srun --overlap --mem=...``) the spawner, the container and all their
children share the step's cgroup, so the step's memory limit covers the whole sandbox. Slurm kills the whole step
when it runs out of memory; ``alive()`` then turns false and the launcher decides what happens next.

When the init is stopped, the kernel kills every member of the PID namespace; ``prove_empty`` checks that from /proc.
``describe()`` is the sandbox's record for the stage authority: its PID namespace, network, GPUs, the binds asked for,
the mount table the kernel reported when it started, and whether it was proven empty.
"""
from __future__ import annotations

import array
import json
import os
from pathlib import Path
import secrets
import shutil
import signal
import socket
import struct
import subprocess
import sys
import tempfile
import threading
import time

APPTAINER = shutil.which('apptainer') or shutil.which('singularity')
NSENTER = shutil.which('nsenter')
SPAWNER = Path(__file__).resolve().with_name('spawner.py')
# The init: a shell that waits until it is stopped. Its argv carries the sandbox token, which is how the launcher
# finds it among the host's processes.
INIT = 'trap "exit 0" TERM; while :; do sleep 3600 & wait $!; done'
# Apptainer's default binds, kept read-only once the site's bind paths are switched off. A sandbox without the host
# network gets its own /etc/hosts instead of the host's, whose entry for the host's name it could not reach.
SYSTEM_BINDS = ('/etc/hosts', '/etc/resolv.conf', '/etc/localtime')
# Where each sandbox's private /tmp and /var/tmp live on the host (Apptainer's --workdir). Without it they are a
# session tmpfs of `sessiondir max size` (64 MiB by default), too small for compiler and model caches.
SCRATCH = os.environ.get('BREAKWATER_SCRATCH') or tempfile.gettempdir()


class SandboxError(RuntimeError):
    pass


def pid_namespace(pid):
    return os.readlink(f'/proc/{pid}/ns/pid')


def members(namespace):
    """Host PIDs of every process in a PID namespace."""
    found = []
    for entry in Path('/proc').iterdir():
        if entry.name.isdigit():
            try:
                if os.readlink(entry / 'ns/pid') == namespace:
                    found.append(int(entry.name))
            except OSError:
                continue
    return sorted(found)


def mount_table(text):
    """[mount point, ro|rw, filesystem type] for each line of a mountinfo file, in mount order."""
    rows = []
    for line in text.splitlines():
        fields = line.split()
        separator = fields.index('-')
        rows.append([fields[4], 'ro' if 'ro' in fields[5].split(',') else 'rw', fields[separator + 1]])
    return rows


def remove_tree(path):
    """Remove a directory a sandbox wrote, including directories it left without write permission."""
    for directory, dirs, _ in os.walk(path):
        for name in dirs:
            try:
                os.chmod(os.path.join(directory, name), 0o700, follow_symlinks=False)
            except OSError:
                pass
    shutil.rmtree(path, ignore_errors=True)


def session_members(sid):
    """Host PIDs of every process in a session. Apptainer starts the SIF's squashfuse_ll in the spawner's session
    but in a process group of its own, outside the container's PID namespace."""
    found = []
    for entry in Path('/proc').iterdir():
        if entry.name.isdigit():
            try:
                if int((entry / 'stat').read_text().rsplit(')', 1)[1].split()[3]) == sid:
                    found.append(int(entry.name))
            except (OSError, IndexError, ValueError):
                continue
    return sorted(found)


class Spawned:
    """A process started by a spawner: its host PID, and its exit status once it has exited."""

    def __init__(self, connection):
        self.connection, self.stream = connection, connection.makefile('r')
        reply = json.loads(self.stream.readline() or '{"error": "the spawner closed the connection"}')
        if 'pid' not in reply:
            connection.close()
            raise SandboxError('could not start the process: ' + reply.get('error', '?'))
        self.pid = reply['pid']
        self.returncode = None

    def wait(self):
        if self.returncode is None:
            line = self.stream.readline()
            self.returncode = json.loads(line)['exit'] if line else 137  # the spawner died with its sandbox
            self.connection.close()
        return self.returncode


class Sandbox:
    def __init__(self, image, *, binds=(), network=False, gpus=False, env=None, prefix=(), start_timeout=120,
                 role=None, scratch=None, hosts=()):
        if not (APPTAINER and NSENTER):
            raise SandboxError('the Apptainer backend needs apptainer and nsenter on the host')
        self.role, self.network, self.gpus, self.stopped_ns = role, bool(network), bool(gpus), None
        self.stop_error = None
        self.hostname = None if network else secrets.token_hex(6)
        self.binds = [[os.path.realpath(str(source)), str(target), mode] for source, target, mode in binds]
        self.token = 'breakwater-init-' + secrets.token_hex(8)
        self.private = Path(tempfile.mkdtemp(prefix='breakwater-spawner-'))
        self.scratch = Path(tempfile.mkdtemp(prefix='breakwater-tmp-', dir=scratch or SCRATCH))
        self.socket = self.private / 'spawner.sock'
        command = [APPTAINER, 'exec', '--userns', '--containall', '--no-home', '--cleanenv', '--pwd', '/',
                   '--no-mount', 'hostfs,bind-paths,cwd', '--workdir', str(self.scratch)]
        if not network:
            # hosts: names of relayed services, each on 127.0.0.1 or (a mapping) on its own loopback address.
            mapping = hosts if isinstance(hosts, dict) else {name: '127.0.0.1' for name in hosts}
            lines = [f"127.0.0.1\t{' '.join(['localhost', self.hostname, *(n for n, a in mapping.items() if a == '127.0.0.1')])}"]
            lines += [f'{address}\t{name}' for name, address in mapping.items() if address != '127.0.0.1']
            hosts = self.private / 'hosts'
            hosts.write_text('\n'.join(lines) + '\n::1\tlocalhost ip6-localhost ip6-loopback\n')
            command += ['--net', '--network', 'none', '--hostname', self.hostname, '--bind', f'{hosts}:/etc/hosts:ro']
        if gpus:
            command.append('--nv')
        for path in SYSTEM_BINDS:
            if os.path.exists(path) and (network or path != '/etc/hosts'):
                command += ['--bind', f'{path}:{path}:ro']
        for source, target, mode in binds:
            command += ['--bind', f'{source}:{target}:{mode}']
        for key, value in (env or {}).items():
            command += ['--env', f'{key}={value}']
        command += [str(image), 'sh', '-c', INIT, self.token]
        self.process = subprocess.Popen([*prefix, sys.executable, '-I', str(SPAWNER), '--socket', str(self.socket),
                                         '--', *command], stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                                        stderr=subprocess.PIPE, start_new_session=True)
        try:
            self.pid = self._find_init(start_timeout)
            self.namespace = pid_namespace(self.pid)
        except BaseException:
            try:
                os.killpg(self.process.pid, signal.SIGKILL)
            except (ProcessLookupError, PermissionError):
                pass
            self.process.wait()
            for pid in session_members(self.process.pid):
                try:
                    os.kill(pid, signal.SIGKILL)
                except (ProcessLookupError, PermissionError):
                    pass
            self.process.stderr.close()
            shutil.rmtree(self.private, ignore_errors=True)
            remove_tree(self.scratch)
            raise
        self.started_ns = time.monotonic_ns()
        self.mount_rows = mount_table(self.mounts())

    def _find_init(self, timeout):
        host = pid_namespace(os.getpid())
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if self.process.poll() is not None:
                raise SandboxError('sandbox exited while starting: '
                                   + self.process.stderr.read().decode(errors='replace')[-2000:])
            for entry in Path('/proc').iterdir():
                if not entry.name.isdigit():
                    continue
                try:
                    argv = (entry / 'cmdline').read_bytes().split(b'\0')
                    if (os.path.basename(argv[0]) == b'sh' and self.token.encode() in argv
                            and pid_namespace(entry.name) != host and self.socket.exists()):
                        return int(entry.name)
                except OSError:
                    continue
            time.sleep(0.2)
        raise SandboxError('the sandbox init did not appear')

    def alive(self):
        try:
            return pid_namespace(self.pid) == self.namespace and self.process.poll() is None
        except OSError:
            return False

    def enter_command(self, argv, *, cwd='/', namespaces=None):
        """The command that runs argv inside this sandbox. The root and working directory are /proc paths, resolved
        before nsenter joins the mount namespace. A sandbox on the host network is already in the launcher's network
        and UTS namespaces (which its user namespace could not join anyway)."""
        if namespaces is None:
            namespaces = ('-U', '-m', '-p', '-i') if self.network else ('-U', '-m', '-p', '-n', '-i', '-u')
        root = f'/proc/{self.pid}/root'
        return [NSENTER, '-t', str(self.pid), *namespaces, '--preserve-credentials', f'--root={root}',
                f'--wd={root}{cwd}', '--', *argv]

    def network_command(self, argv):
        """Join only this sandbox's network namespace (relays): reachable from inside, invisible to its processes."""
        return [NSENTER, '-t', str(self.pid), '-U', '-n', '--preserve-credentials', '--', *argv]

    def spawn(self, argv, *, stdin, stdout, stderr, extra=None, env=None):
        """Start argv (a host command, usually enter_command(...)) through the spawner, inside the sandbox's step.
        ``extra`` maps descriptor numbers in the child to descriptors of this process."""
        connection = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        try:
            connection.connect(str(self.socket))
        except OSError as error:
            connection.close()
            raise SandboxError(f'the sandbox is gone ({error})') from error
        extra = dict(extra or {})
        request = json.dumps({'argv': [str(item) for item in argv], 'env': env or {}, 'fds': list(extra)}).encode()
        fds = [stdin, stdout, stderr, *extra.values()]
        connection.sendmsg([struct.pack('>I', len(request)) + request],
                           [(socket.SOL_SOCKET, socket.SCM_RIGHTS, array.array('i', fds).tobytes())])
        return Spawned(connection)

    def run(self, argv, *, cwd='/', timeout=60, env=None):
        """Run argv inside the sandbox (trusted launcher operations and tests) and collect its output."""
        out_read, out_write = os.pipe()
        err_read, err_write = os.pipe()
        with open(os.devnull, 'rb') as null:
            try:
                child = self.spawn(self.enter_command(argv, cwd=cwd), stdin=null.fileno(), stdout=out_write,
                                   stderr=err_write, env=env or {'PATH': '/usr/local/bin:/usr/bin:/bin'})
            finally:
                os.close(out_write)
                os.close(err_write)
        output = {}

        def drain(fd, key):
            chunks = []
            while (chunk := os.read(fd, 65536)):
                chunks.append(chunk)
            os.close(fd)
            output[key] = b''.join(chunks).decode(errors='replace')
        readers = [threading.Thread(target=drain, args=(fd, key), daemon=True)
                   for fd, key in ((out_read, 'out'), (err_read, 'err'))]
        for reader in readers:
            reader.start()
        result = [None]
        waiter = threading.Thread(target=lambda: result.__setitem__(0, child.wait()), daemon=True)
        waiter.start()
        waiter.join(timeout)
        if waiter.is_alive():
            try:
                os.killpg(child.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            raise subprocess.TimeoutExpired(argv, timeout)
        for reader in readers:
            reader.join(timeout)
        return result[0], output.get('out', ''), output.get('err', '')

    def mounts(self):
        return Path(f'/proc/{self.pid}/mountinfo').read_text()

    def describe(self):
        return {'role': self.role, 'namespace': self.namespace, 'network': 'host' if self.network else 'none',
                'gpus': self.gpus, 'binds': self.binds, 'mounts': self.mount_rows, 'started_ns': self.started_ns,
                'stopped_ns': self.stopped_ns, 'empty': self.stopped_ns is not None}

    def stop(self, timeout=30):
        """End the stage: kill the container's init (the kernel then empties the namespace), let the spawner and its
        Slurm step end on their own, escalate to SIGTERM and then SIGKILL only if they do not, and prove the
        namespace empty. A Slurm step can stay listed for a few seconds after this returns.

        A sandbox is stopped once: a later call returns the first outcome without looking at the namespace again.
        Once a PID namespace is gone the kernel may give its number to a newer one (a lost command sandbox's
        replacement gets it), so only the first look is about this sandbox."""
        if self.stopped_ns is not None:
            return True
        if self.stop_error is not None:
            raise SandboxError(self.stop_error)
        try:
            if pid_namespace(self.pid) == self.namespace:  # never a reused PID once the sandbox is gone
                os.kill(self.pid, signal.SIGKILL)
        except (OSError, PermissionError):
            pass
        for signum, wait in ((None, 10), (signal.SIGTERM, 10), (signal.SIGKILL, timeout)):
            if signum is not None:
                try:
                    os.killpg(self.process.pid, signum)
                except (ProcessLookupError, PermissionError):
                    pass
            try:
                self.process.wait(timeout=wait)
                break
            except subprocess.TimeoutExpired:
                continue
        # A spawner killed abruptly (by the SIGKILL above or by whoever lost the sandbox) orphans the image's
        # squashfuse_ll, which no process group or PID namespace kill reaches; its session still names the spawner.
        # (With a Slurm prefix the step's cleanup already kills it.) The session ID cannot be reused while a member
        # is alive, so this never reaches an unrelated process.
        for pid in session_members(self.process.pid):
            try:
                os.kill(pid, signal.SIGKILL)
            except (ProcessLookupError, PermissionError):
                pass
        if self.process.stderr:
            self.process.stderr.close()
        shutil.rmtree(self.private, ignore_errors=True)
        try:
            empty = self.prove_empty(timeout)
        except SandboxError as error:
            self.stop_error = str(error)
            raise
        self.stopped_ns = time.monotonic_ns()
        remove_tree(self.scratch)
        return empty

    def stop_proven(self, timeout=30):
        """stop(), reporting a namespace that could not be proven empty as False instead of raising, so that a
        stage's cleanup still stops everything else."""
        try:
            return self.stop(timeout)
        except SandboxError:
            return False

    def prove_empty(self, timeout=30):
        deadline = time.monotonic() + timeout
        while True:
            left = members(self.namespace)
            if not left:
                return True
            if time.monotonic() > deadline:
                raise SandboxError(f'{len(left)} processes outlived the sandbox')
            time.sleep(0.2)
