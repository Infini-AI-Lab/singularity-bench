"""Root-only observation barrier before the trusted downstream-cell coordinator."""
import ctypes
import os
import signal
import sys

if os.geteuid() != 0:
    raise SystemExit('Trusted observer barrier requires root')
if ctypes.CDLL('/opt/breakwater-runtime/stdio_capture.so', use_errno=True).breakwater_capture_start():
    raise OSError(ctypes.get_errno(), 'protected capture startup failed')
os.kill(os.getpid(), signal.SIGSTOP)
os.execv(sys.executable, [sys.executable, '-I', '-B', '/private/breakwater/cell_intermediate.py'])
