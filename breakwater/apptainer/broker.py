"""The exec broker: runs each agent command inside the command sandbox, outside the agent's reach.

The agent CLI's shell prefix is ``breakwater-exec`` (client.py). It sends the command string, its working directory
and its own stdin, stdout and stderr to this broker over a unix socket that only the author sandbox can reach. The
broker starts ``bash -c COMMAND`` inside the command sandbox under ``strace -f``, copies the command's output to the
client's descriptors and to the evidence directory, and returns the exit status once everything the command wrote
before it exited has reached the client. A command's background jobs keep running in the sandbox, strace keeps
tracing them and their output is still recorded, but no longer copied to the client. If the client goes away first
(the CLI timed the command out), the broker kills the command's process group.

Evidence, per command N: ``exec-N.strace``, ``exec-N.stdout``, ``exec-N.stderr`` (each stream with a ``.sha256``
file); and ``exec.jsonl``, a hash-chained journal that opens with ``broker_started``, records every command's start,
exit (with ``client_gone``: whether its client had stopped waiting first) and closed streams, and ends with
``broker_stopped`` once ``drain()`` has seen every command finish (reconcile.check_exec_journal checks it).
"""
from __future__ import annotations

import array
import fcntl
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import select
import shutil
import signal
import socket
import struct
import termios
import threading
import time

from sandbox import SandboxError

HEADER = struct.Struct('>I')
STATUS = struct.Struct('>i')
MAX_REQUEST = 1 << 20
STRACE = shutil.which('strace')
TRACE = 'process,network,signal,%file,ptrace,mount,prctl'


def traced(argv, path):
    """argv under strace, writing every process's calls to one file in the format the review index reads (the
    observer's options, with each line prefixed by its PID instead of one file per process)."""
    return [STRACE, '-f', '-qq', '-ttt', '-T', '-v', '-xx', '-y', '-s', '256', '-e', f'trace={TRACE}', '-o', str(path),
            *argv]


def receive_request(connection):
    """A length-prefixed JSON request, the client's three standard descriptors and any extra descriptors it
    passes (their numbers in the request's ``fds``)."""
    fds = array.array('i')
    data, ancillary, _, _ = connection.recvmsg(65536, socket.CMSG_SPACE(64 * fds.itemsize))
    for level, kind, payload in ancillary:
        if level == socket.SOL_SOCKET and kind == socket.SCM_RIGHTS:
            fds.frombytes(payload[:len(payload) - len(payload) % fds.itemsize])
    if len(data) < HEADER.size:
        raise ValueError('short request')
    size = HEADER.unpack_from(data)[0]
    if size > MAX_REQUEST:
        raise ValueError('request too large')
    body = data[HEADER.size:]
    while len(body) < size:
        chunk = connection.recv(size - len(body))
        if not chunk:
            raise ValueError('truncated request')
        body += chunk
    request = json.loads(body)
    targets = request.get('fds') or []
    if (len(fds) != 3 + len(targets) or not all(type(n) is int and 3 <= n < 1024 and n != 200 for n in targets)
            or len(set(targets)) != len(targets)):
        for fd in fds:
            os.close(fd)
        raise ValueError('the client must pass stdin, stdout, stderr and the descriptors it names')
    return request, list(fds)


class Journal:
    """Append-only, hash-chained JSONL, written only by the launcher."""

    def __init__(self, path):
        self.path = Path(path)
        self.previous = '0' * 64
        self.sequence = 0
        self.lock = threading.Lock()

    def write(self, **record):
        with self.lock:
            record = {'seq': self.sequence, 'wall_ns': time.time_ns(), **record, 'prev': self.previous}
            line = json.dumps(record, sort_keys=True)
            self.previous = hashlib.sha256(line.encode()).hexdigest()
            self.sequence += 1
            with self.path.open('a') as stream:
                stream.write(line + '\n')


class Stream:
    """A command's stdout or stderr, copied to its evidence file and, until the command is answered, to the client.

    ``settle()`` waits until every byte that is in the pipe when it is called has been delivered, so the client gets
    all the output the command wrote before it exited; ``detach()`` then stops the copies to the client. The stream
    owns a duplicate of the client's descriptor, so the number can never be reused under it."""

    def __init__(self, source, client, path):
        self.source, self.client, self.path = source, os.dup(client), Path(path)
        self.condition = threading.Condition()
        self.taken = self.delivered = 0
        self.busy = self.done = False
        os.set_blocking(source, False)
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def _run(self):
        digest, size = hashlib.sha256(), 0
        poller = select.poll()
        poller.register(self.source, select.POLLIN)
        with self.path.open('wb') as evidence:
            while True:
                poller.poll()
                with self.condition:  # bytes leave the pipe only under the lock settle() takes
                    try:
                        chunk = os.read(self.source, 65536)
                    except BlockingIOError:
                        continue
                    self.taken += len(chunk)
                    client = self.client
                    self.busy = bool(chunk) and client is not None
                if not chunk:
                    break
                evidence.write(chunk)
                evidence.flush()
                digest.update(chunk)
                size += len(chunk)
                if client is not None:
                    try:
                        view = memoryview(chunk)
                        while view:
                            view = view[os.write(client, view):]
                    except OSError:
                        pass  # the client closed its end; keep recording
                with self.condition:
                    self.busy = False
                    self.delivered += len(chunk)
                    self.condition.notify_all()
        self.path.with_suffix(self.path.suffix + '.sha256').write_text(f'{digest.hexdigest()} {size}\n')
        with self.condition:
            os.close(self.source)
            self.done = True
            self.condition.notify_all()
        self.detach(0)

    def settle(self, timeout=10):
        with self.condition:
            if self.done:
                return
            buffered = array.array('i', [0])
            fcntl.ioctl(self.source, termios.FIONREAD, buffered, True)
            target = self.taken + buffered[0]
            self.condition.wait_for(lambda: self.done or self.delivered >= target, timeout)

    def detach(self, timeout=10):
        with self.condition:
            self.condition.wait_for(lambda: not self.busy, timeout)
            client, self.client = self.client, None
        if client is not None:
            os.close(client)


class Broker:
    def __init__(self, socket_path, sandbox, evidence, *, env, workspace='/app', trace=True, kill_all=False,
                 replacement_timeout=0):
        """``kill_all``: the client may ask the broker to stop every process in the sandbox (the evaluation's worker
        broker, for the coordinator's kill_workers). ``replacement_timeout``: the stage replaces a lost sandbox
        (authoring, after an out-of-memory kill takes the command sandbox's whole Slurm step), and a command that
        arrives meanwhile waits up to this many seconds for the replacement instead of failing."""
        self.sandbox, self.evidence, self.env, self.workspace = sandbox, Path(evidence), dict(env), workspace
        self.replacement_timeout = replacement_timeout
        self.sandbox_changed = threading.Condition()
        self.trace, self.allow_kill_all = trace and bool(STRACE), kill_all
        self.evidence.mkdir(parents=True, exist_ok=True)
        self.journal = Journal(self.evidence / 'exec.jsonl')
        self.counter = 0
        self.counter_lock = threading.Lock()
        self.threads = []
        self.journal.write(event='broker_started', trace=self.trace, workspace=workspace)
        self.listener = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self.listener.bind(str(socket_path))
        os.chmod(socket_path, 0o600)
        self.listener.listen(64)
        self.closed = False
        threading.Thread(target=self._accept, daemon=True).start()

    def _accept(self):
        while not self.closed:
            try:
                connection, _ = self.listener.accept()
            except OSError:
                return
            self._start(self._serve, connection)

    def _start(self, target, *args):
        thread = threading.Thread(target=target, args=args, daemon=True)
        with self.counter_lock:
            self.threads = [item for item in self.threads if item.is_alive()] + [thread]
        thread.start()

    def _cwd(self, value):
        path = PurePosixPath(value if isinstance(value, str) and value.startswith('/') else self.workspace)
        inside = path == PurePosixPath(self.workspace) or PurePosixPath(self.workspace) in path.parents
        return str(path) if inside and '..' not in path.parts else self.workspace

    def _serve(self, connection):
        with connection:
            try:
                request, fds = receive_request(connection)
            except (ValueError, OSError, json.JSONDecodeError):
                return
            stdin, stdout, stderr, extra = fds[0], fds[1], fds[2], dict(zip(request.get('fds') or [], fds[3:]))
            try:
                command, argv = request.get('command'), request.get('argv')
                if 'kill_all' in request:
                    grace = request['kill_all']
                    status = (self.kill_all(float(grace)) if self.allow_kill_all and type(grace) in (int, float)
                              and 0 <= grace <= 600 else 126)
                elif isinstance(command, str):
                    status = self.execute(command, self._cwd(request.get('cwd')), stdin, stdout, stderr, connection)
                elif isinstance(argv, list) and argv and all(isinstance(item, str) for item in argv):
                    # Submitted code (the evaluation stage): argv as given, from the workspace, like run_worker.py.
                    status = self.execute(argv, self.workspace, stdin, stdout, stderr, connection, extra=extra)
                else:
                    status = 126
                connection.sendall(STATUS.pack(status))
            except OSError:
                pass
            finally:
                for fd in fds:
                    os.close(fd)

    def kill_all(self, grace=0.0, timeout=60):
        """Stop every process in the sandbox but its init, as Docker's kill_workers stops every process of the worker
        UID: SIGTERM and ``grace`` seconds first when asked, then SIGKILL until none is left. Journaled as
        ``commands_killed``; 0 when none survived."""
        from sandbox import members

        def spared(pid):
            # The sandbox's own processes: Apptainer's init (PID 1 inside, where orphans go), the sandbox init and its
            # sleep. A submitted process can never become a child of the sandbox init.
            if pid == self.sandbox.pid:
                return True
            try:
                fields = dict(line.split(':', 1) for line in Path(f'/proc/{pid}/status').read_text().splitlines()
                              if ':' in line)
            except OSError:
                return True  # already gone
            return fields.get('NSpid', '').split()[-1:] == ['1'] or fields.get('PPid', '').strip() == str(self.sandbox.pid)
        killed, left = set(), []
        for signum, wait in ([(signal.SIGTERM, grace)] if grace > 0 else []) + [(signal.SIGKILL, timeout)]:
            deadline = time.monotonic() + wait
            while True:
                left = [pid for pid in members(self.sandbox.namespace) if not spared(pid)]
                if not left or time.monotonic() > deadline:
                    break
                for pid in left:
                    try:
                        os.kill(pid, signum)
                        killed.add(pid)
                    except ProcessLookupError:
                        pass
                time.sleep(0.1)
            if not left:
                break
        self.journal.write(event='commands_killed', processes=len(killed), survivors=len(left))
        return 0 if not left else 1

    def execute(self, command, cwd, stdin, stdout, stderr, connection=None, extra=None):
        """Run ``command`` (a shell string, or an argv list for submitted code) in the sandbox."""
        with self.counter_lock:
            self.counter += 1
            number = self.counter
        out_read, out_write = os.pipe()
        err_read, err_write = os.pipe()
        status_read, status_write = os.pipe()
        # The spawner hands the status pipe to the command's wrapper as fd 200. The command itself runs with it
        # closed, so a background job cannot forge its exit status; strace keeps its own copy while background
        # jobs run, so the status ends with a newline.
        if isinstance(command, list):
            wrapper = ['/bin/bash', '-c', '"$@" 200>&-; printf "%s\\n" "$?" >&200', 'breakwater-worker', *command]
        else:
            wrapper = ['/bin/bash', '-c', '/bin/bash -c "$1" 200>&-; printf "%s\\n" "$?" >&200', 'breakwater-exec',
                       command]
        sandbox = self.sandbox
        if not sandbox.alive():
            sandbox = self._replacement(sandbox) or sandbox
        self.journal.write(event='exec_started', id=number, cwd=cwd,
                           **({'argv': command} if isinstance(command, list) else {'command': command}))
        while True:
            argv = sandbox.enter_command(wrapper, cwd=cwd)
            if self.trace:
                argv = traced(argv, self.evidence / f'exec-{number}.strace')
            try:
                process = sandbox.spawn(argv, stdin=stdin, stdout=out_write, stderr=err_write,
                                        extra={**(extra or {}), 200: status_write}, env=self.env)
                break
            except SandboxError as error:
                # The spawner was unreachable, so the command never started: run it in the replacement, if one comes.
                replacement = self._replacement(sandbox)
                if replacement is None:
                    for fd in (out_read, out_write, err_read, err_write, status_read, status_write):
                        os.close(fd)
                    self.journal.write(event='sandbox_unavailable', id=number, error=str(error))
                    return 137
                sandbox = replacement
        for fd in (out_write, err_write, status_write):
            os.close(fd)
        streams = [Stream(out_read, stdout, self.evidence / f'exec-{number}.stdout'),
                   Stream(err_read, stderr, self.evidence / f'exec-{number}.stderr')]
        answered, gone = threading.Event(), threading.Event()
        if connection is not None:
            threading.Thread(target=self._watch_client, args=(connection, process, answered, gone), daemon=True).start()
        status = self._status(status_read)
        client_gone = gone.is_set()
        for stream in streams:
            stream.settle()
        for stream in streams:
            stream.detach()
        answered.set()
        # client_gone: the client went away before the command exited (it stopped waiting, so the broker ended it).
        self.journal.write(event='exec_exited', id=number, status=status, client_gone=client_gone)
        # strace (and the stream recordings) end when the command's last background job does.
        self._start(self._close, number, process, streams)
        return status

    @staticmethod
    def _status(fd):
        data = b''
        while not data.endswith(b'\n'):
            chunk = os.read(fd, 64)
            if not chunk:
                break
            data += chunk
        os.close(fd)
        try:
            return int(data)
        except ValueError:
            return 137  # the wrapper was killed before reporting

    @staticmethod
    def _watch_client(connection, process, answered, gone):
        """Kill the command's process group if the client disconnects before the command has exited."""
        try:
            connection.recv(1)
        except OSError:
            pass
        gone.set()
        if answered.wait(0.5):
            return
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass

    def _close(self, number, process, streams):
        process.wait()
        for stream in streams:
            stream.thread.join()
        self.journal.write(event='exec_streams_closed', id=number, tracer_status=process.returncode)

    def replace_sandbox(self, sandbox):
        """Swap in the stage's replacement for a lost sandbox; commands waiting for it go ahead."""
        with self.sandbox_changed:
            self.sandbox = sandbox
            self.sandbox_changed.notify_all()

    def _replacement(self, lost):
        """The sandbox that replaced ``lost``, once the stage has started it; None if none comes in time."""
        if not self.replacement_timeout:
            return None
        with self.sandbox_changed:
            self.sandbox_changed.wait_for(lambda: self.sandbox is not lost or self.closed, self.replacement_timeout)
            return self.sandbox if self.sandbox is not lost else None

    def close(self):
        """Stop accepting commands (commands waiting for a replacement sandbox give up)."""
        self.closed = True
        self.listener.close()
        with self.sandbox_changed:
            self.sandbox_changed.notify_all()

    def drain(self, timeout=60):
        """Once the sandbox is stopped: wait for every command to be answered and its streams closed, then end the
        journal. Returns whether everything finished in time."""
        self.close()
        deadline = time.monotonic() + timeout
        while True:
            with self.counter_lock:
                pending = [item for item in self.threads if item.is_alive()]
            if not pending or time.monotonic() > deadline:
                break
            pending[0].join(max(0.0, deadline - time.monotonic()))
        self.journal.write(event='broker_stopped', commands=self.counter, drained=not pending)
        return not pending
