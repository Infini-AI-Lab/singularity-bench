#!/usr/bin/env python3
"""The per-sandbox spawner: starts a sandbox's container and every process the launcher runs in it.

    spawner.py --socket PATH -- APPTAINER_COMMAND...

It runs on the host, as the user, inside the sandbox's Slurm step when there is one (``srun --overlap --mem=...``),
so the container and everything started in it share the step's cgroup and its memory limit; nsenter alone would
leave a command in the launcher's cgroup. It is outside the sandbox's namespaces, so no sandbox process can see it.

Protocol (unix socket, reachable only by the launcher): one connection per process. The request is a
length-prefixed JSON object ``{"argv": [...], "env": {...}, "fds": [N, ...]}`` with the child's stdin, stdout and
stderr plus one descriptor per entry of ``fds``, passed as SCM_RIGHTS; the child receives each extra descriptor
under the number given in ``fds``. The reply is one JSON line ``{"pid": N}`` when the child has started and
``{"exit": STATUS}`` when it has exited. The spawner exits with the container.
"""
import array
import json
import os
import socket
import struct
import subprocess
import sys
import threading

HEADER = struct.Struct('>I')


def serve(connection):
    with connection:
        fds = array.array('i')
        try:
            data, ancillary, _, _ = connection.recvmsg(1 << 20, socket.CMSG_SPACE(64 * fds.itemsize))
            for level, kind, payload in ancillary:
                if level == socket.SOL_SOCKET and kind == socket.SCM_RIGHTS:
                    fds.frombytes(payload[:len(payload) - len(payload) % fds.itemsize])
            size = HEADER.unpack_from(data)[0]
            body = data[HEADER.size:]
            while len(body) < size:
                chunk = connection.recv(size - len(body))
                if not chunk:
                    raise ValueError('truncated request')
                body += chunk
            request = json.loads(body)
            targets = request.get('fds') or []
            if (not isinstance(request.get('argv'), list) or len(fds) != 3 + len(targets)
                    or not all(type(n) is int and 3 <= n < 1024 for n in targets) or len(set(targets)) != len(targets)):
                raise ValueError('bad request')
            argv, extra = request['argv'], fds[3:]
            if extra:  # give each extra descriptor its requested number, via temporaries so none is overwritten
                temporary = range(900, 900 + len(extra))
                steps = [' '.join(f'{t}>&{source}' for t, source in zip(temporary, extra)),
                         ' '.join(f'{source}>&-' for source in extra),
                         ' '.join(f'{target}>&{t}' for t, target in zip(temporary, targets)),
                         ' '.join(f'{t}>&-' for t in temporary)]
                script = '; '.join('exec ' + step for step in steps) + '; exec "$@"'
                argv = ['/bin/bash', '-c', script, 'spawn', *argv]
            # From /, so that no host directory reaches a sandboxed process (bash would export it as PWD).
            child = subprocess.Popen(argv, stdin=fds[0], stdout=fds[1], stderr=fds[2], pass_fds=tuple(extra),
                                     env=request.get('env') or {}, start_new_session=True, cwd='/')
        except (OSError, ValueError, json.JSONDecodeError) as error:
            try:
                connection.sendall((json.dumps({'error': str(error)}) + '\n').encode())
            except OSError:
                pass
            return
        finally:
            for fd in fds:
                os.close(fd)
        try:
            connection.sendall((json.dumps({'pid': child.pid}) + '\n').encode())
        except OSError:
            pass
        status = child.wait()
        try:
            connection.sendall((json.dumps({'exit': status}) + '\n').encode())
        except OSError:
            pass


def main():
    arguments = sys.argv[1:]
    if len(arguments) < 4 or arguments[0] != '--socket' or arguments[2] != '--':
        raise SystemExit(__doc__)
    path, command = arguments[1], arguments[3:]
    listener = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    listener.bind(path)
    os.chmod(path, 0o600)
    listener.listen(64)
    container = subprocess.Popen(command, stdin=subprocess.DEVNULL)

    def watch():
        container.wait()
        os._exit(0)  # the sandbox is gone: so is its spawner
    threading.Thread(target=watch, daemon=True).start()
    while True:
        connection, _ = listener.accept()
        threading.Thread(target=serve, args=(connection,), daemon=True).start()


if __name__ == '__main__':
    main()
