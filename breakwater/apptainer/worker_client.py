#!/usr/bin/env python3
"""worker-exec: run submitted code in the submission sandbox (the Apptainer backend's run_worker.py).

    worker-exec ARGV...
    worker-exec --kill-all GRACE

The evaluation coordinator starts submitted code with this prefix. It sends ARGV, its stdin, stdout and stderr and
every other descriptor it holds (each under its number, such as a socket named by ``--transport-fd N``) to the
worker broker, which runs ARGV from the workspace inside the submission sandbox, and exits with the command's
status. Killing this process stops the command. ``--kill-all`` asks the broker to stop every process in the submission
sandbox instead (SIGTERM and GRACE seconds first, then SIGKILL), as Docker's kill_workers stops the worker UID's.
"""
import array
import json
import os
import signal
import socket
import struct
import sys

SOCKET = os.environ.get('BREAKWATER_WORKER_SOCKET', '/run/breakwater/worker.sock')


def open_descriptors(skip):
    found = []
    for name in os.listdir('/proc/self/fd'):
        number = int(name)
        if number > 2 and number != skip:
            try:
                os.fstat(number)
            except OSError:
                continue  # the directory listing's own, already closed descriptor
            found.append(number)
    return sorted(found)


def main():
    argv = sys.argv[1:]
    if not argv:
        print('worker-exec: requires a command', file=sys.stderr)
        return 126
    connection = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    try:
        connection.connect(SOCKET)
    except OSError as error:
        print(f'worker-exec: no submission sandbox ({error})', file=sys.stderr)
        return 126
    extra = open_descriptors(connection.fileno())

    def stop(signum, _frame):
        connection.close()
        os._exit(128 + signum)

    for number in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP):
        signal.signal(number, stop)
    if argv[0] == '--kill-all':
        extra, payload = [], {'kill_all': float(argv[1]) if len(argv) > 1 else 0.0, 'fds': []}
    else:
        payload = {'argv': argv, 'fds': extra}
    request = json.dumps(payload).encode()
    connection.sendmsg([struct.pack('>I', len(request)) + request],
                       [(socket.SOL_SOCKET, socket.SCM_RIGHTS, array.array('i', [0, 1, 2, *extra]).tobytes())])
    data = b''
    while len(data) < 4:
        chunk = connection.recv(4 - len(data))
        if not chunk:
            print('worker-exec: the submission sandbox closed the connection', file=sys.stderr)
            return 125
        data += chunk
    return struct.unpack('>i', data)[0]


if __name__ == '__main__':
    sys.exit(main())
