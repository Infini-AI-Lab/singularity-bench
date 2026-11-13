"""Root grader's launcher for submitted code, after the author CLI is removed."""
import ctypes
import os
from pathlib import Path
import sys
import signal

if os.geteuid() != 0 or not Path('/run/breakwater/finished').exists():
    raise SystemExit('Submitted grading code requires the completed trusted transition')
root = Path('/run/breakwater')
env = dict(entry.decode().split('=', 1) for entry in (root / 'worker.env').read_bytes().split(b'\0') if entry)
groups = [int(g) for g in (root / 'worker-groups').read_text().split()]
capture = (root / 'external-capture-enabled').exists()
if capture:
    if ctypes.CDLL('/opt/breakwater-runtime/stdio_capture.so', use_errno=True).breakwater_capture_start():
        raise OSError(ctypes.get_errno(), 'protected stdio capture')
    os.kill(os.getpid(), signal.SIGSTOP)
os.setgroups(groups)
os.setresgid(1101, 1101, 1101)
os.setresuid(1101, 1101, 1101)
if ctypes.CDLL(None, use_errno=True).prctl(38, 1, 0, 0, 0):
    raise OSError(ctypes.get_errno(), 'no_new_privs')
os.umask(0o002)
os.chdir(env['BREAKWATER_WORKDIR'])
os.execvpe(sys.argv[1], sys.argv[1:], env)
