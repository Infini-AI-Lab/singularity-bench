"""Codex's execution environment in the Apptainer author sandbox: its own executor, with every command forwarded.

    python3 -I /opt/breakwater-runtime/codex-forward.py      (environments.toml ``program``; JSON-RPC on stdio)

Runs Codex's executor (``codex exec-server --listen stdio``, the pinned binary in /opt/author-cli) beside the CLI and
relays the newline-delimited JSON-RPC messages between them. The executor's messages pass unchanged. Of the CLI's:

* ``process/start`` passes with its process replaced by the shell-prefix client running the command, so every
  command the agent runs is a command of the broker in the command sandbox, as with Claude Code. The variables Codex
  sets for commands (``env``: pager, color and locale settings, its session ids) go with the command.
* the file operations (``fs/*``), the other process calls (output, input, terminal size, termination) and the
  session's own calls pass unchanged: the executor carries them out here, on the shared workspace, as Claude Code's
  Read, Write and Edit tools run in the author sandbox.
* any other method is answered with an error and never reaches the executor, so nothing else starts a process here.

The agent's commands run in the command sandbox and cannot reach this sandbox's processes. If the command sandbox is
lost (an out-of-memory kill of its Slurm step), the authoring stage replaces it and the next command runs there.
"""
import json
import os
import re
import shlex
import subprocess
import sys
import threading

CODEX = '/opt/author-cli/codex'
CLIENT = '/opt/breakwater-runtime/breakwater-exec'
NAME = re.compile(r'[A-Za-z_][A-Za-z0-9_]*')
RELAYED = {'initialize', 'initialized', 'environmentConfig/read', 'process/start', 'process/read', 'process/write',
           'process/writeStdin', 'process/resizePty', 'process/terminate'}


def forwarded(message):
    """The message to relay to the executor (process/start rewritten), or None if it may not reach the executor."""
    method = message.get('method')
    if method is None:
        return message  # a reply to a request of the executor's
    if not isinstance(method, str) or not (method in RELAYED or method.startswith('fs/')):
        return None
    params = message.get('params')
    if method != 'process/start':
        return message
    argv = params.get('argv') if isinstance(params, dict) else None
    if not (isinstance(argv, list) and argv and all(isinstance(part, str) for part in argv)):
        return None
    env = params.get('env') if isinstance(params.get('env'), dict) else {}
    assignments = [f'{key}={shlex.quote(str(value))}' for key, value in sorted(env.items()) if NAME.fullmatch(key)]
    command = ' '.join([*assignments, 'exec', shlex.join(argv)])
    return {**message, 'params': {**params, 'argv': [CLIENT, command], 'arg0': None}}


def main():
    executor = subprocess.Popen([CODEX, 'exec-server', '--listen', 'stdio'], stdin=subprocess.PIPE,
                                stdout=subprocess.PIPE)
    out = sys.stdout.buffer
    lock = threading.Lock()

    def send(line):
        with lock:
            out.write(line)
            out.flush()

    def relay_requests():
        try:
            for line in sys.stdin.buffer:
                try:
                    message = json.loads(line)
                    relayed = forwarded(message) if isinstance(message, dict) else None
                except ValueError:
                    relayed = message = None
                if relayed is not None:
                    # Messages that pass unchanged are relayed byte for byte.
                    executor.stdin.write(line.rstrip(b'\n') + b'\n' if relayed is message
                                         else json.dumps(relayed).encode() + b'\n')
                    executor.stdin.flush()
                elif isinstance(message, dict) and 'id' in message:
                    send(json.dumps({'id': message['id'], 'error': {
                        'code': -32601, 'message': f'{message.get("method")!r} is not available in this environment'}}
                        ).encode() + b'\n')
            executor.stdin.close()
        except OSError:
            pass  # the executor is gone; its output ending ends this process

    threading.Thread(target=relay_requests, daemon=True).start()
    try:
        for line in executor.stdout:
            send(line)
    except OSError:
        executor.kill()  # the CLI is gone
    status = executor.wait()
    os._exit(status if status >= 0 else 128 - status)


if __name__ == '__main__':
    main()
