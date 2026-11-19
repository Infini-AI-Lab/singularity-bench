"""Operator endpoints for a sandbox without a network: names on the sandbox's own loopback, joined to real hosts.

A task may need services on the launcher's host or on other machines (its declared endpoints, or those its host hooks
declare). Each endpoint is a name, the real host it stands for, and the ports a sandbox may reach on it. Inside the
sandbox a host name resolves to a loopback address of its own (127.0.1.N, from the sandbox's hosts file,
``addresses()``), and a loopback address (127.x.x.x, a service on the launcher's own machine) is relayed at that same
address, which is local in every sandbox too. A
sandbox cannot be given any other address, so another IP address is refused: name such hosts. ``EndpointRelay`` binds a listening
socket for every allowed port on that address inside the sandbox's network namespace (a helper entered there with
nsenter hands the sockets back over a socket pair and exits) and serves them from the host, joining each connection to
the same port on the real host. Nothing else is reachable, and the sandbox's processes cannot see the relay.
"""
from __future__ import annotations

import array
import json
import os
from pathlib import Path
import selectors
import socket
import subprocess
import sys
import threading

sys.path.insert(0, str(Path(__file__).resolve().parent))
from relay import pipe  # noqa: E402
from sandbox import SandboxError  # noqa: E402

BATCH = 200  # descriptors per message, below the kernel's SCM_RIGHTS limit of 253
# Entered in the sandbox's network namespace: listen on every (address, port) and hand the sockets back over the
# socket pair end whose descriptor number is the second argument.
HELPER = r'''
import array, json, socket, sys
channel = socket.socket(fileno=int(sys.argv[2]))
targets = json.loads(sys.argv[1])
sockets = []
for address, port in targets:
    listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    listener.bind((address, port))
    listener.listen(128)
    sockets.append(listener)
for start in range(0, len(sockets), %d):
    batch = sockets[start:start + %d]
    channel.sendmsg([json.dumps(len(batch)).encode().ljust(16)],
                    [(socket.SOL_SOCKET, socket.SCM_RIGHTS, array.array('i', [s.fileno() for s in batch]).tobytes())])
''' % (BATCH, BATCH)


def loopback(name):
    import ipaddress
    try:
        return ipaddress.ip_address(name).is_loopback
    except ValueError:
        return False


def addresses(endpoints):
    """Each endpoint's address in a sandbox: a loopback address as itself, each host name its own 127.0.1.N."""
    import re
    names = [endpoint['name'] for endpoint in endpoints]
    if len(set(names)) != len(names) or len(names) > 250:
        raise ValueError('endpoint names must be distinct, at most 250')
    result, index = {}, 0
    for name in names:
        if loopback(name):
            result[name] = name
            continue
        if not re.fullmatch(r'(?=.*[A-Za-z])[A-Za-z0-9](?:[A-Za-z0-9.\-]{0,251}[A-Za-z0-9])?', name):
            raise ValueError(f'{name}: a sandbox can reach a host by name or a loopback address only; name this host')
        index += 1
        result[name] = f'127.0.1.{index}'
    if len(set(result.values())) != len(result):
        raise ValueError('a loopback endpoint collides with a name alias')
    return result


def hosts(endpoints):
    """The hosts-file entries a sandbox needs for these endpoints (names only; addresses need none)."""
    return {name: address for name, address in addresses(endpoints).items() if not loopback(name)}


class EndpointRelay:
    def __init__(self, sandbox, endpoints):
        """``endpoints``: dicts with ``name``, ``host`` (the real host) and ``ports`` (the ports allowed), and optionally
        ``port_map``, the real host's port for a sandbox port where they differ (an endpoint the operator runs on
        another port than the one the task declares)."""
        self.sandbox, self.endpoints = sandbox, [dict(endpoint) for endpoint in endpoints]
        self.addresses = addresses(self.endpoints)
        self.listeners = {}  # listening socket -> (real host, port)
        self.selector = selectors.DefaultSelector()
        self.stopped = threading.Event()
        self.thread = None

    def start(self):
        targets = [(self.addresses[endpoint['name']], port, endpoint['host'], (endpoint.get('port_map') or {}).get(port, port))
                   for endpoint in self.endpoints for port in endpoint['ports']]
        parent, child = socket.socketpair(socket.AF_UNIX, socket.SOCK_STREAM)
        helper = subprocess.Popen(self.sandbox.network_command(
            [sys.executable, '-I', '-c', HELPER, json.dumps([[address, port] for address, port, _, _ in targets]),
             str(child.fileno())]),
            pass_fds=(child.fileno(),), stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
        child.close()
        received = []
        try:
            while len(received) < len(targets):
                fds = array.array('i')
                data, ancillary, _, _ = parent.recvmsg(16, socket.CMSG_SPACE(BATCH * fds.itemsize))
                if not data:
                    break
                for level, kind, payload in ancillary:
                    if level == socket.SOL_SOCKET and kind == socket.SCM_RIGHTS:
                        fds.frombytes(payload[:len(payload) - len(payload) % fds.itemsize])
                received += list(fds)
        finally:
            parent.close()
            error = helper.stderr.read().decode(errors='replace')
            helper.stderr.close()
            helper.wait()
        if len(received) != len(targets):
            for fd in received:
                os.close(fd)
            raise SandboxError('could not bind the endpoints in the sandbox: ' + error[-500:])
        for fd, (_, _, host, host_port) in zip(received, targets):
            listener = socket.socket(fileno=fd)
            listener.setblocking(False)
            self.listeners[listener] = (host, host_port)
            self.selector.register(listener, selectors.EVENT_READ)
        self.thread = threading.Thread(target=self._serve, daemon=True)
        self.thread.start()
        return self

    def _serve(self):
        while not self.stopped.is_set():
            for key, _ in self.selector.select(timeout=0.5):
                try:
                    client, _ = key.fileobj.accept()
                except OSError:
                    continue
                client.setblocking(True)
                threading.Thread(target=self._join, args=(client, *self.listeners[key.fileobj]), daemon=True).start()

    @staticmethod
    def _join(client, host, port):
        try:
            upstream = socket.create_connection((host, port), timeout=30)
        except OSError:
            client.close()
            return
        upstream.settimeout(None)
        for source, target in ((client, upstream), (upstream, client)):
            threading.Thread(target=pipe, args=(source, target), daemon=True).start()

    def stop(self):
        self.stopped.set()
        if self.thread is not None:
            self.thread.join(5)
        for listener in self.listeners:
            listener.close()
        self.selector.close()

    def describe(self):
        return [{'name': endpoint['name'], 'address': self.addresses[endpoint['name']],
                 'ports': list(endpoint['ports'])} for endpoint in self.endpoints]
