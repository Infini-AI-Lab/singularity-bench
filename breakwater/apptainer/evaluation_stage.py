"""The evaluation stage on Apptainer: the trusted coordinator and the submitted code in separate sandboxes.

The coordinator sandbox runs ``breakwater/intermediate.py`` (``BREAKWATER_BACKEND=apptainer``) with the private
inputs (tests, labels, validation data), the task contract, the framework and a writable ``/evaluation``. The
submission sandbox, started fresh from the same image, has the sealed submission, the task's runners and the answer-free
``/eval-inputs``, and none of the private paths. The coordinator starts submitted code with ``worker-exec``
(workers.worker_prefix()), which hands the command and its descriptors to the worker broker; the broker runs it in
the submission sandbox under strace. Both sandboxes get the task's GPUs, as the coordinator and its worker shared them
in one Docker container, and the submitted code gets a fresh, writable /home/worker. When the task keeps the sealed
submission apart (``workspace.submission_mount``), both sandboxes also share a writable workdir, where the task's code
installs the copy that the submitted code runs from, and they share the directories the task's code prepares for the
submitted code (``evaluation.shared``). The coordinator's kill_workers stops every process in the submission sandbox
(the worker broker's kill-all). Each port the submitted code serves (``evaluation.ports``) is
relayed from the coordinator's 127.0.0.1 to the submission sandbox's, as their shared loopback was in Docker; each
operator service on the launcher's host that the task names (``host_ports``: loopback URLs in ``evaluation.env``) is
relayed from the coordinator's 127.0.0.1 to the host's, the coordinator's only other network path. At the
end both sandboxes are stopped, their PID namespaces proven empty and the worker broker drained; the record holds both
sandboxes' descriptions and the variable names of both environments.
"""
from __future__ import annotations

import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import tempfile
import threading
import time

sys.path.insert(0, str(Path(__file__).resolve().parent))
from agent_stage import slurm_step  # noqa: E402
from broker import STRACE, Broker, traced  # noqa: E402
from sandbox import Sandbox, SandboxError  # noqa: E402
from endpoints import EndpointRelay, hosts as endpoint_hosts  # noqa: E402
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from spec import HOME_MOUNT  # noqa: E402

HERE = Path(__file__).resolve().parent
FRAMEWORK = HERE.parent
WORKER_CLIENT = HERE / 'worker_client.py'
RELAY = HERE / 'relay.py'


class EvaluationStage:
    def __init__(self, *, image, task, contract, submission, workdir, evaluation, external, worker_env, coordinator_env,
                 timeout, private_binds=(), public_binds=(), gpus=False, memory_mb=None, submission_mount=None,
                 coordinator_argv=None, framework=FRAMEWORK, scratch=None, runners=None, ports=(), host_ports=(),
                 shared=(), worker_home_seed=None, cpus=None, coordinator_endpoints=(), submission_endpoints=(),
                 home=None, workdir_seed=None, submission_binds=()):
        self.image, self.task, self.submission, self.workdir = image, Path(task), Path(submission), workdir
        self.contract = Path(contract)
        self.evaluation, self.external, self.timeout = Path(evaluation), Path(external), timeout
        self.worker_env, self.coordinator_env = dict(worker_env), dict(coordinator_env)
        self.private_binds, self.public_binds = list(private_binds), list(public_binds)
        self.gpus, self.memory_mb, self.framework, self.scratch = gpus, memory_mb, Path(framework), scratch
        self.cpus = cpus
        # The task's endpoints per role (endpoints.py): the coordinator's may include what submitted code must never
        # reach. home: the sealed worker home (workspace.snapshot_home).
        self.endpoints = {'coordinator': [dict(item) for item in coordinator_endpoints],
                          'submission': [dict(item) for item in submission_endpoints]}
        self.home = Path(home) if home else None
        # Binds in the submission sandbox only (to hide parts of the image from submitted code, say).
        self.submission_binds = list(submission_binds)
        # The task's trusted runners of submitted code (spec: evaluation.runners), in both sandboxes.
        self.runner_binds = [(Path(runners), '/opt/breakwater-drivers', 'ro')] if runners else []
        self.target, self.ports, self.host_ports = submission_mount or workdir, list(ports), list(host_ports)
        self.coordinator_argv = coordinator_argv or ['python3', '-I', '-B', '/opt/breakwater/intermediate.py']
        self.private = Path(tempfile.mkdtemp(prefix='breakwater-evaluation-'))
        self.inputs = self.private / 'eval-inputs'
        for folder in (self.inputs, self.private / 'worker', self.private / 'worker-home', self.private / 'relays'):
            folder.mkdir(mode=0o700)
        if worker_home_seed is not None:  # the image's worker home and prefetched caches (run.worker_home_seed)
            shutil.copytree(worker_home_seed, self.private / 'worker-home', symlinks=True, dirs_exist_ok=True)
        # The writable workdir both sandboxes share when the sealed submission is mounted apart from it.
        self.shared = [(self.private / 'workdir', workdir, 'rw')] if submission_mount else []
        # Directories the coordinator prepares for the submitted code (evaluation.shared), as in Docker's one container.
        self.shared += [(self.private / f'shared-{index}', path, 'rw') for index, path in enumerate(shared)]
        for source, _, _ in self.shared:
            source.mkdir(mode=0o700)
        if workdir_seed is not None and submission_mount:
            # A pristine workspace the coordinator applies a sealed change set to (a custom launcher's overlay).
            shutil.copytree(workdir_seed, self.private / 'workdir', symlinks=True, dirs_exist_ok=True)

    def environment(self):
        return {**self.coordinator_env, 'BREAKWATER_STAGE': 'intermediate', 'BREAKWATER_BACKEND': 'apptainer',
                'PYTHONDONTWRITEBYTECODE': '1'}

    def run(self):
        record = {'started_ns': time.monotonic_ns(), 'started_wall_ns': time.time_ns(),
                  'environments': {'coordinator': sorted(self.environment()), 'worker': sorted(self.worker_env)},
                  'relayed_ports': self.ports, 'host_services': self.host_ports,
                  **({'endpoints': {role: [{'name': item['name'], 'ports': len(item['ports'])} for item in items]
                                    for role, items in self.endpoints.items()}} if any(self.endpoints.values()) else {})}
        coordinator = submission = broker = None
        relays, endpoint_relays = [], []
        try:
            submission = Sandbox(self.image, binds=[(self.submission, self.target, 'ro'), *self.shared,
                                                    (self.private / 'worker-home', '/home/worker', 'rw'),
                                                    (self.inputs, '/eval-inputs', 'ro'),
                                                    *self.runner_binds,
                                                    *self.public_binds, *self.submission_binds],
                                 gpus=self.gpus, prefix=slurm_step(self.memory_mb, self.cpus), role='submission',
                                 scratch=self.scratch, hosts=endpoint_hosts(self.endpoints['submission']))
            if self.endpoints['submission']:
                endpoint_relays.append(EndpointRelay(submission, self.endpoints['submission']).start())
            broker = Broker(self.private / 'worker' / 'worker.sock', submission,
                            self.external / 'evaluation-commands', env=self.worker_env, workspace=self.workdir,
                            kill_all=True)
            for name in ('HOME', 'TMPDIR'):
                if self.worker_env.get(name, '').startswith('/tmp/'):
                    submission.run(['mkdir', '-p', self.worker_env[name]], env=self.worker_env)
            private = [(source, target, 'ro') for source, target, _ in self.private_binds]
            binds = [(self.submission, self.target, 'ro'), *self.shared, (self.evaluation, '/evaluation', 'rw'),
                     (self.task / 'tests', '/private/tests', 'ro'), (self.task / 'tests', '/tests', 'ro'),
                     (self.contract, '/trusted-task/contract.json', 'ro'),
                     (self.framework, '/opt/breakwater', 'ro'),
                     *self.runner_binds,
                     (self.inputs, '/eval-inputs', 'rw'), (self.private / 'worker', '/run/breakwater', 'ro'),
                     (WORKER_CLIENT, '/opt/breakwater-runtime/worker-exec', 'ro'), *private,
                     # The submitted code's home, which the coordinator may install, as root could in Docker's container.
                     (self.private / 'worker-home', '/home/worker', 'rw'),
                     *([(self.home, HOME_MOUNT, 'ro')] if self.home else []),
                     *self.public_binds]
            coordinator = Sandbox(self.image, binds=binds, gpus=self.gpus, role='coordinator', scratch=self.scratch,
                                  hosts=endpoint_hosts(self.endpoints['coordinator']))
            if self.endpoints['coordinator']:
                endpoint_relays.append(EndpointRelay(coordinator, self.endpoints['coordinator']).start())
            for port in self.ports:
                socket_path = self.private / 'relays' / f'{port}.sock'
                relays.append(self._relay(submission, ['--connect', str(port), str(socket_path)]))
                relays.append(self._relay(coordinator, [str(port), str(socket_path)]))
            for port in self.host_ports:
                socket_path = self.private / 'relays' / f'host-{port}.sock'
                relays.append(self._relay(None, ['--connect', str(port), str(socket_path)]))
                relays.append(self._relay(coordinator, [str(port), str(socket_path)]))
            record.update(self._coordinate(coordinator))
        finally:
            for relay in relays:
                relay.kill()
                relay.wait()
            for relay in endpoint_relays:
                relay.stop()
            walls = {}
            if coordinator is not None:
                walls['coordinator'] = coordinator.stop_proven()
            if broker is not None:
                broker.close()
            if submission is not None:
                walls['submission'] = submission.stop_proven()
            if broker is not None:
                record['commands_drained'] = broker.drain()
            record['sandboxes'] = [box.describe() for box in (coordinator, submission) if box is not None]
            record.update(stopped_ns=time.monotonic_ns(), ended_wall_ns=time.time_ns(), process_wall=walls)
            shutil.rmtree(self.private, ignore_errors=True)
        return record

    @staticmethod
    def _relay(sandbox, arguments):
        """A relay in ``sandbox``'s network namespace, or on the host (None), once it is listening (relay.py)."""
        argv = [sys.executable, '-I', str(RELAY), *arguments]
        relay = subprocess.Popen(sandbox.network_command(argv) if sandbox is not None else argv,
                                 stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
        if relay.stdout.readline() != b'ready\n':
            relay.kill()
            relay.wait()
            raise SandboxError(f'the relay for port {arguments[-2]} did not start')
        relay.stdout.close()
        return relay

    def _coordinate(self, coordinator):
        argv = coordinator.enter_command(self.coordinator_argv, cwd='/')
        if STRACE:
            argv = traced(argv, self.external / 'intermediate.strace')
        environment = self.environment()
        with open(os.devnull, 'rb') as null, (self.external / 'intermediate.stdout').open('xb') as out, \
                (self.external / 'intermediate.stderr').open('xb') as err:
            child = coordinator.spawn(argv, stdin=null.fileno(), stdout=out.fileno(), stderr=err.fileno(),
                                      env=environment)
        result = [None]
        waiter = threading.Thread(target=lambda: result.__setitem__(0, child.wait()), daemon=True)
        waiter.start()
        waiter.join(self.timeout)
        timed_out = waiter.is_alive()
        if timed_out:
            try:
                os.killpg(child.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            waiter.join(30)
        return {'exit_code': result[0], 'timed_out': timed_out}
