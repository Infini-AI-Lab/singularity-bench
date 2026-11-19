"""The authoring stage on Apptainer: the agent CLI in the author sandbox, its commands in the command sandbox.

``AuthoringStage(...).run()`` starts, in order:
  1. the author gateway, on a unix socket in a private directory no sandbox can see;
  2. the command sandbox (a memory-limited Slurm step when the contract gives memory and Slurm is in use);
  3. the exec broker, recording every command under ``external/commands``;
  4. the author sandbox, with the agent CLI's files and settings read-only, the shell-prefix client and the broker's
     socket, and the relay that is its only route to the gateway.
The command sandbox sees the workspace and the task's agent assets; the author sandbox also gets the CLI's read-only
settings and runtime files (``author_files``: Claude Code's managed settings, or Codex's settings and codex_forward.py).
It runs the CLI with host-owned pipes that copy its native output to ``external/agent-native.*`` and to the agent's
local mirror (``local/native.*``), until the CLI exits or the authoring budget ends. A command sandbox lost to an
out-of-memory kill is replaced with a fresh one on the same workspace while the stage continues. A task's services for
the agent's commands (``services``: the development gateway of model_access) run on the host, each on a unix socket in
the private directory, relayed into every command sandbox at 127.0.0.1:PORT under the service's host name. The task's
endpoints (``endpoints``: its declared hosts and ports, on this host or others) are relayed into every command sandbox too
(endpoints.py). With ``home_snapshot``, the worker's home is copied there once the sandboxes are gone. At the end it
stops the author sandbox and every command sandbox, proves their PID namespaces empty, drains the broker (every
command answered and its streams closed), and stops the gateway, which then finalizes its journal. The stage record
holds each sandbox's description (sandbox.describe()) and the variable names of every environment it gave a process.
"""
from __future__ import annotations
import functools

import os
from pathlib import Path
import selectors
import shutil
import signal
import subprocess
import sys
import tempfile
import threading
import time

sys.path.insert(0, str(Path(__file__).resolve().parent))
from broker import Broker  # noqa: E402
from endpoints import EndpointRelay, hosts as endpoint_hosts  # noqa: E402
from sandbox import APPTAINER, Sandbox, SandboxError  # noqa: E402

HERE = Path(__file__).resolve().parent
GATEWAY = HERE.parent / 'author_gateway.py'
RELAY = HERE / 'relay.py'
CLIENT = HERE / 'client.py'
RUNTIME_VARIABLES = ('PWD', 'OLDPWD', 'SHLVL', '_', 'HOME', 'USER', 'LOGNAME', 'TERM')


def image_environment(image):
    """The image's own environment, as ``apptainer exec --cleanenv`` gives it to a process. Commands entered with
    nsenter start from this, since they do not run the image's environment scripts."""
    result = subprocess.run([APPTAINER, 'exec', '--userns', '--containall', '--no-home', '--cleanenv', '--pwd', '/',
                             str(image), 'env', '-0'], capture_output=True, timeout=300)
    if result.returncode:
        raise SandboxError('could not read the image environment: ' + result.stderr.decode(errors='replace')[-500:])
    environment = dict(item.split('=', 1) for item in result.stdout.decode().split('\0') if '=' in item)
    return {key: value for key, value in environment.items()
            if key not in RUNTIME_VARIABLES and not key.startswith(('APPTAINER_', 'SINGULARITY_'))}


# How long a command that arrives while a lost command sandbox is being replaced waits for the replacement: the lost
# one's stop (its namespace proven empty) and a new sandbox's start, which can be a new Slurm step on a busy node.
REPLACEMENT_TIMEOUT = 300


@functools.cache
def oom_kill_step():
    """Whether this Slurm's srun takes --oom-kill-step (newer Slurm releases)."""
    try:
        return '--oom-kill-step' in subprocess.run(['srun', '--help'], capture_output=True, text=True, timeout=60).stdout
    except (OSError, subprocess.SubprocessError):
        return False


def slurm_step(memory_mb, cpus=None):
    """The prefix that makes a sandbox its own Slurm step with a memory limit, when the launcher runs on a node of a
    Slurm job. The step stays on that node (in a job of several nodes, srun would otherwise place it on any of them,
    away from the sandbox's private files and sockets). It gets the task's CPU count (Docker's --cpus), since a
    one-task step otherwise defaults to a single CPU, and no CPU binding of its own: a launcher started by srun exports
    its own step's binding, which the new step cannot honour. Where srun has it, --oom-kill-step=0: a site whose
    Slurm sets OOMKillStep would otherwise kill the whole step when one process exceeds the limit; as in a Docker
    container, only the process the kernel picks dies. Outside the job's nodes (an salloc shell on a login node),
    there is no step to make."""
    node = os.environ.get('SLURMD_NODENAME')
    if not memory_mb or not os.environ.get('SLURM_JOB_ID') or not node or not shutil.which('srun'):
        return []
    return ['srun', '--overlap', f'--mem={int(memory_mb)}M', '--nodes=1', f'--nodelist={node}', '--ntasks=1',
            '--cpu-bind=none', *(['--oom-kill-step=0'] if oom_kill_step() else []),
            *([f'--cpus-per-task={int(cpus)}'] if cpus else []), '--quiet']


class AuthoringStage:
    def __init__(self, *, image, workspace, workdir, external, local, cli_dir, cli_argv, cli_env, gateway,
                 budget, author_files=(), agent_binds=(), gpus=False, memory_mb=None, command_env=None, scratch=None,
                 services=(), worker_home_seed=None, cpus=None, endpoints=(), home_snapshot=None, native='jsonl',
                 on_services=None, hosts=None, inbound_ports=()):
        self.image, self.workspace, self.workdir = image, Path(workspace), workdir
        self.external, self.local, self.cli_dir = Path(external), Path(local), Path(cli_dir)
        # gateway None: no author gateway (a submitted agent started by a trusted launcher, which reaches its model only
        # through a service gateway). native: how the CLI's stdout is recorded, 'jsonl' for a CLI's log or 'stdout' for
        # a submitted agent's protected streams.
        # on_services: called with the services' socket paths once they listen, before the CLI starts.
        self.cli_argv, self.cli_env, self.budget = list(cli_argv), dict(cli_env), budget
        self.gateway = dict(gateway) if gateway is not None else None
        self.native, self.on_services = native, on_services
        # author_files: (host file, target) pairs bound read-only into the author sandbox only. A target under
        # /home/author lies inside the author's own writable home, which gets an empty file there as its mount point.
        self.author_files = [(Path(source), target) for source, target in author_files]
        self.agent_binds, self.gpus, self.memory_mb, self.cpus = list(agent_binds), gpus, memory_mb, cpus
        self.scratch = scratch
        self.private = Path(tempfile.mkdtemp(prefix='breakwater-private-'))
        self.worker_home, self.author_home = self.private / 'worker-home', self.private / 'author-home'
        for folder in (self.worker_home, self.author_home, self.private / 'exec'):
            folder.mkdir(mode=0o700)
        if worker_home_seed is not None:  # the image's worker home and prefetched caches (run.worker_home_seed)
            shutil.copytree(worker_home_seed, self.worker_home, symlinks=True, dirs_exist_ok=True)
        self.command_env = {**(command_env if command_env is not None else image_environment(image)),
                            'HOME': '/home/worker', 'BREAKWATER_WORKDIR': workdir, 'PYTHONUNBUFFERED': '1'}
        self.sandboxes, self.record = [], {'command_sandbox_restarts': 0}
        self.stopping = threading.Event()
        # services: dicts with name, argv, env (the service's own, with its key), port and host.
        self.services, self.service_processes, self.service_relays = [dict(item) for item in services], [], []
        self.endpoints, self.endpoint_relays = [dict(item) for item in endpoints], []
        # hosts: more names for the command sandboxes' hosts files. inbound_ports: ports that servers the agent's
        # commands start in the command sandbox serve to the host's loopback ([breakwater.authoring] inbound_ports, for
        # callers such as containers a host-side endpoint starts): a host listener per port, joined through a socket to
        # a relay in each command sandbox's network namespace, replacements included.
        self.hosts, self.inbound_ports = dict(hosts or {}), [int(port) for port in inbound_ports]
        self.inbound_host, self.inbound_sandbox = [], []
        self.home_snapshot = Path(home_snapshot) if home_snapshot else None

    def _command_sandbox(self):
        sandbox = Sandbox(self.image, binds=[(self.workspace, self.workdir, 'rw'), (self.worker_home, '/home/worker', 'rw'),
                                             *self.agent_binds],
                          gpus=self.gpus, prefix=slurm_step(self.memory_mb, self.cpus), role='command', scratch=self.scratch,
                          hosts={**{service['host']: '127.0.0.1' for service in self.services},
                                 **endpoint_hosts(self.endpoints), **self.hosts})
        self.sandboxes.append(sandbox)
        if self.endpoints:
            self.endpoint_relays.append(EndpointRelay(sandbox, self.endpoints).start())
        if self.inbound_ports:
            # A lost sandbox's relays would keep its dead network namespace and the sockets: replace them.
            for relay in self.inbound_sandbox:
                relay.kill()
                relay.wait()
            self.inbound_sandbox = []
            for port in self.inbound_ports:
                path = self.private / 'inbound' / f'{port}.sock'
                path.unlink(missing_ok=True)
                self.inbound_sandbox.append(self._relay(sandbox.network_command(
                    [sys.executable, '-I', str(RELAY), '--connect', str(port), str(path)]), f'session port {port}'))
        for service in self.services:
            relay = subprocess.Popen(sandbox.network_command([sys.executable, '-I', str(RELAY), str(service['port']),
                                                              str(self.private / f"{service['name']}.sock")]),
                                     stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
            self.service_relays.append(relay)
            if relay.stdout.readline() != b'ready\n':
                raise SandboxError(f"the relay for {service['name']} did not start")
            relay.stdout.close()
        return sandbox

    @staticmethod
    def _relay(argv, what):
        relay = subprocess.Popen(argv, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if relay.stdout.readline() != b'ready\n':
            relay.kill()
            error = relay.stderr.read().decode(errors='replace').strip().splitlines()[-1:]
            relay.wait()
            raise SandboxError(f'the relay for {what} did not start' + (f': {error[0]}' if error else ''))
        relay.stdout.close()
        relay.stderr.close()
        return relay

    def _start_service(self, service):
        """A service on the host, listening on its unix socket in the private directory."""
        socket_path, log = self.private / f"{service['name']}.sock", self.private / f"{service['name']}.stderr"
        with log.open('wb') as errors:
            process = subprocess.Popen(service['argv'], env={**service['env'], 'GATEWAY_SOCKET': str(socket_path)},
                                       stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=errors)
        self.service_processes.append(process)
        deadline = time.monotonic() + 30
        while not socket_path.exists():
            if process.poll() is not None or time.monotonic() > deadline:
                raise SandboxError(f"{service['name']} did not start: " + log.read_text(errors='replace')[-500:])
            time.sleep(0.1)

    def _start_gateway(self, socket_path):
        g = self.gateway
        environment = {'PATH': os.environ.get('PATH', '/usr/bin:/bin'), 'AUTHOR_MODEL': g['model'],
                       'AUTHOR_UPSTREAM_API_KEY': g['key'], 'AUTHOR_UPSTREAM_BASE_URL': g['upstream'],
                       'AUTHOR_UPSTREAM_AUTH': g.get('auth', 'bearer'), 'AUTHOR_PROTOCOL': g.get('protocol', 'messages'),
                       'AUTHOR_GATEWAY_LOG': str(self.external / 'gateway' / 'requests.jsonl'),
                       'AUTHOR_GATEWAY_SOCKET': str(socket_path)}
        process = subprocess.Popen([sys.executable, '-I', str(GATEWAY)], env=environment, stdin=subprocess.DEVNULL,
                                   stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
        deadline = time.monotonic() + 30
        while not socket_path.exists():
            if process.poll() is not None or time.monotonic() > deadline:
                raise SandboxError('the author gateway did not start: '
                                   + process.stderr.read().decode(errors='replace')[-500:])
            time.sleep(0.1)
        return process

    def _replace_lost_sandboxes(self, broker):
        """An out-of-memory kill takes the whole Slurm step: record it and swap in a fresh command sandbox. The lost
        one is stopped (its leftovers cleaned up, its namespace proven empty) before the replacement starts, which
        may get the same PID namespace number from the kernel."""
        while not self.stopping.wait(1):
            if broker.sandbox.alive():
                continue
            broker.journal.write(event='command_sandbox_lost', pid=broker.sandbox.pid)
            try:
                broker.sandbox.stop(10)
            except SandboxError as error:
                broker.journal.write(event='command_sandbox_not_empty', error=str(error))
            if self.stopping.is_set():
                return
            replacement = self._command_sandbox()
            broker.replace_sandbox(replacement)
            self.record['command_sandbox_restarts'] += 1
            broker.journal.write(event='command_sandbox_restarted', pid=replacement.pid)

    def _run_cli(self, author):
        out_read, out_write = os.pipe()
        err_read, err_write = os.pipe()
        with open(os.devnull, 'rb') as null:
            try:
                child = author.spawn(author.enter_command(self.cli_argv, cwd=self.workdir), stdin=null.fileno(),
                                     stdout=out_write, stderr=err_write, env=self.cli_env)
            finally:
                os.close(out_write)
                os.close(err_write)
        copies = {out_read: [(self.external / f'agent-native.{self.native}').open('xb'),
                             (self.local / f'native.{self.native}').open('wb')],
                  err_read: [(self.external / 'agent-native.stderr').open('xb'), (self.local / 'native.stderr').open('wb')]}
        selector = selectors.DefaultSelector()
        for fd in copies:
            selector.register(fd, selectors.EVENT_READ)
        deadline, reached = time.monotonic() + self.budget, False
        try:
            while selector.get_map():
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    reached = True
                    break
                for key, _ in selector.select(timeout=min(0.25, remaining)):
                    chunk = os.read(key.fd, 65536)
                    if not chunk:
                        selector.unregister(key.fd)
                        continue
                    for stream in copies[key.fd]:
                        stream.write(chunk)
                        stream.flush()
        finally:
            selector.close()
            if reached:
                try:
                    os.killpg(child.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
            for fd, streams in copies.items():
                os.close(fd)
                for stream in streams:
                    stream.close()
        return child.wait(), reached

    def run(self):
        self.external.mkdir(parents=True, exist_ok=True)
        self.local.mkdir(parents=True, exist_ok=True)
        gateway_socket = self.private / 'author.sock'
        gateway = relay = broker = author = monitor = None
        self.record.update(started_ns=time.monotonic_ns(), started_wall_ns=time.time_ns(),
                           environments={'cli': sorted(self.cli_env), 'command': sorted(self.command_env)})
        try:
            gateway = self._start_gateway(gateway_socket) if self.gateway is not None else None
            for service in self.services:
                self._start_service(service)
            if self.inbound_ports:
                (self.private / 'inbound').mkdir(mode=0o700)
                for port in self.inbound_ports:  # on the host's loopback; a port in use there stops the stage here
                    self.inbound_host.append(self._relay(
                        [sys.executable, '-I', str(RELAY), str(port), str(self.private / 'inbound' / f'{port}.sock')],
                        f'session port {port} on the host (is it in use?)'))
                self.record['inbound_ports'] = list(self.inbound_ports)
            if self.on_services is not None and self.services:
                self.on_services({service['name']: self.private / f"{service['name']}.sock" for service in self.services})
            self.record['services'] = [{key: service[key] for key in ('name', 'host', 'port')} for service in self.services]
            if self.endpoints:
                self.record['endpoints'] = [{'name': item['name'], 'ports': len(item['ports'])} for item in self.endpoints]
            broker = Broker(self.private / 'exec' / 'exec.sock', self._command_sandbox(), self.external / 'commands',
                            env=self.command_env, workspace=self.workdir, replacement_timeout=REPLACEMENT_TIMEOUT)
            binds = [(self.workspace, self.workdir, 'rw'), (self.cli_dir, '/opt/author-cli', 'ro'),
                     (self.private / 'exec', '/run/breakwater', 'ro'),
                     (CLIENT, '/opt/breakwater-runtime/breakwater-exec', 'ro'), (self.local, '/logs/agent', 'rw'),
                     (self.author_home, '/home/author', 'rw'), *self.agent_binds]
            for source, target in self.author_files:
                if target.startswith('/home/author/'):
                    point = self.author_home / target[len('/home/author/'):]
                    point.parent.mkdir(parents=True, exist_ok=True)
                    point.touch()
                binds.append((source, target, 'ro'))
            author = Sandbox(self.image, binds=binds, role='author', scratch=self.scratch)
            if gateway is not None:
                relay = subprocess.Popen(author.network_command([sys.executable, '-I', str(RELAY), '8001',
                                                                 str(gateway_socket)]),
                                         stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            monitor = threading.Thread(target=self._replace_lost_sandboxes, args=(broker,), daemon=True)
            monitor.start()
            exit_code, reached = self._run_cli(author)
            self.record.update(exit_code=exit_code, deadline_reached=reached)
        finally:
            self.stopping.set()
            walls = {}
            if author is not None:
                walls['author'] = author.stop_proven()
            if broker is not None:
                broker.close()
            if monitor is not None:
                monitor.join(10)
            walls['command'] = all([sandbox.stop_proven() for sandbox in self.sandboxes])
            if broker is not None:
                self.record['commands_drained'] = broker.drain()
            self.record['sandboxes'] = [box.describe() for box in [author, *self.sandboxes] if box is not None]
            if relay is not None:
                relay.kill()
                relay.wait()
            for item in self.service_relays:
                item.kill()
                item.wait()
            for item in self.endpoint_relays:
                item.stop()
            for item in self.inbound_host + self.inbound_sandbox:
                item.kill()
                item.wait()
            if self.home_snapshot is not None and walls.get('command'):
                # The worker's home as the agent left it, once no process can change it (workspace.snapshot_home).
                shutil.copytree(self.worker_home, self.home_snapshot, symlinks=True)
            for process in self.service_processes:  # SIGTERM: the gateway finalizes its journal
                process.send_signal(signal.SIGTERM)
                try:
                    process.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
            if gateway is not None:
                gateway.send_signal(signal.SIGTERM)
                try:
                    gateway.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    gateway.kill()
                gateway.stderr.close()
            self.record.update(stopped_ns=time.monotonic_ns(), ended_wall_ns=time.time_ns(), process_wall=walls)
            shutil.rmtree(self.private, ignore_errors=True)
        return self.record
