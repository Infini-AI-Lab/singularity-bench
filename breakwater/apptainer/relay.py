#!/usr/bin/env python3
"""Relay between a sandbox's loopback port and a host-side unix socket.

    relay.py PORT SOCKET              listen on 127.0.0.1:PORT, connect each client to SOCKET
    relay.py --connect PORT SOCKET    listen on SOCKET, connect each client to 127.0.0.1:PORT

The launcher starts it inside one sandbox's network namespace only (Sandbox.network_command), so the relay stays
outside the sandbox's PID and mount namespaces while it reaches the sandbox's loopback. The model gateway listens on
a SOCKET in a directory no sandbox can see, and the author sandbox reaches it on 127.0.0.1:PORT. A port that
submitted code serves in the submission sandbox reaches the coordinator through two relays: ``--connect`` in the
submission sandbox and the first form in the coordinator sandbox, joined by a SOCKET. Once listening, the relay
prints ``ready``.
"""
import os
import socket
import sys
import threading


def pipe(source, target):
    try:
        while True:
            data = source.recv(65536)
            if not data:
                break
            target.sendall(data)
    except OSError:
        pass
    finally:
        for end in (source, target):
            try:
                end.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass


def main():
    reverse = sys.argv[1:2] == ['--connect']
    port, path = int(sys.argv[-2]), sys.argv[-1]
    if reverse:
        listener = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        listener.bind(path)
        os.chmod(path, 0o600)
    else:
        listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        listener.bind(('127.0.0.1', port))
    listener.listen(64)
    print('ready', flush=True)
    while True:
        client, _ = listener.accept()
        upstream = socket.socket(*((socket.AF_INET, socket.SOCK_STREAM) if reverse else (socket.AF_UNIX, socket.SOCK_STREAM)))
        try:
            upstream.connect(('127.0.0.1', port) if reverse else path)
        except OSError:
            client.close()
            upstream.close()
            continue
        for source, target in ((client, upstream), (upstream, client)):
            threading.Thread(target=pipe, args=(source, target), daemon=True).start()


if __name__ == '__main__':
    main()
