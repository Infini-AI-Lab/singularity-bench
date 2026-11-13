"""Start and stop the local development server by a private launch identity."""
from __future__ import annotations

import argparse
import ctypes
import fcntl
import json
import os
from pathlib import Path
import platform
import re
import secrets
import signal
import subprocess
import time

STATE = Path(os.environ.get('CLUSTER_DEV_STATE_DIR', '/tmp/cluster-control-plane'))
MARKER = 'CLUSTER_DEV_CONTROL_PLANE_ID'


def syscall(number, *args):
    # Some Python builds omit the wrappers even on a kernel with pidfd support.
    if platform.system() != 'Linux' or platform.machine() not in ('x86_64','aarch64'):
        raise RuntimeError('process helpers require Linux x86_64 or aarch64 with pidfd support')
    libc = ctypes.CDLL(None, use_errno=True)
    result = libc.syscall(number, *args)
    if result < 0:
        raise OSError(ctypes.get_errno(), 'pidfd syscall failed')
    return result


def signal_owned(token: str, sig: int) -> int:
    """Match an exact inherited environment field; pidfds avoid PID-reuse races."""
    if not re.fullmatch(r'[0-9a-f]{32}', token):
        raise ValueError('invalid launch identity; refusing to signal processes')
    marker = f'{MARKER}={token}'.encode()
    found = 0
    for entry in Path('/proc').glob('[0-9]*'):
        fd = None
        try:
            if entry.stat().st_uid != os.getuid():
                continue
            fd = syscall(434, int(entry.name), 0)  # pidfd_open
            if marker in (entry/'environ').read_bytes().split(b'\0'):
                syscall(424, fd, sig, 0, 0)  # pidfd_send_signal
                found += 1
        except (FileNotFoundError, ProcessLookupError, PermissionError):
            continue
        finally:
            if fd is not None:
                os.close(fd)
    return found


def stop() -> None:
    path = STATE/'state.json'
    if not path.exists():
        return
    token = json.loads(path.read_text())['token']
    signal_owned(token, signal.SIGTERM)
    deadline = time.monotonic()+5
    while signal_owned(token, 0):
        if time.monotonic() >= deadline:
            break
        time.sleep(.05)
    # Children may have outlived the parent or moved into their own session.
    # They still inherit the launch identity and can be stopped individually.
    for _ in range(40):
        if not signal_owned(token, signal.SIGKILL):
            path.unlink(missing_ok=True)
            return
        time.sleep(.05)
    raise RuntimeError('server processes survived cleanup; state retained for another stop')


def start() -> None:
    stop()
    token = secrets.token_hex(16)
    app = Path(__file__).resolve().parent
    with (STATE/'control-plane.log').open('w') as output:
        proc = subprocess.Popen(['bash', str(app/'run.sh')], cwd=app,
            env=os.environ | {MARKER:token}, stdin=subprocess.DEVNULL,
            stdout=output, stderr=subprocess.STDOUT, start_new_session=True)
    try:
        temporary = STATE/'state.new'
        temporary.write_text(json.dumps({'pid':proc.pid, 'token':token})+'\n')
        temporary.replace(STATE/'state.json')
    except BaseException:
        signal_owned(token, signal.SIGKILL)
        raise
    print(f'Control plane launched (PID {proc.pid}); log: {STATE}/control-plane.log')
    print('Check readiness: curl -fsS http://127.0.0.1:8000/health')


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('start','stop'))
    args = parser.parse_args()
    if not Path('/proc/self/environ').exists():
        parser.error('these helpers are for development; the foreground entrypoint is run.sh')
    STATE.mkdir(mode=0o700, parents=True, exist_ok=True)
    with (STATE/'lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if args.action == 'start':
            start()
        else:
            stop()
            print('Local control plane stopped; remote workers are unchanged.')


if __name__ == '__main__':
    main()
