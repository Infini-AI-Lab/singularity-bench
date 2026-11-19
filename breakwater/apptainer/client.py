#!/usr/bin/env python3
"""breakwater-exec: the agent CLI's shell prefix in the Apptainer backend.

    breakwater-exec COMMAND

Sends COMMAND, the working directory and this process's stdin, stdout and stderr to the exec broker, which runs it
in the command sandbox, and exits with the command's status. Killing this process stops the command.
"""
import array
import json
import os
import signal
import socket
import struct
import sys

SOCKET = os.environ.get('BREAKWATER_EXEC_SOCKET', '/run/breakwater/exec.sock')


def main():
    if len(sys.argv) != 2:
        print('breakwater-exec: requires one command string', file=sys.stderr)
        return 126
    request = json.dumps({'command': sys.argv[1], 'cwd': os.getcwd()}).encode()
    connection = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    try:
        connection.connect(SOCKET)
    except OSError as error:
        print(f'breakwater-exec: no command sandbox ({error})', file=sys.stderr)
        return 126

    def stop(signum, _frame):
        connection.close()
        os._exit(128 + signum)

    for number in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP):
        signal.signal(number, stop)
    connection.sendmsg([struct.pack('>I', len(request)) + request],
                       [(socket.SOL_SOCKET, socket.SCM_RIGHTS, array.array('i', [0, 1, 2]).tobytes())])
    data = b''
    while len(data) < 4:
        chunk = connection.recv(4 - len(data))
        if not chunk:
            print('breakwater-exec: the command sandbox closed the connection', file=sys.stderr)
            return 125
        data += chunk
    return struct.unpack('>i', data)[0]


if __name__ == '__main__':
    sys.exit(main())
