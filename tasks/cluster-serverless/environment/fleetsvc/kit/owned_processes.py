"""Stop only processes bearing this launch's unguessable environment identity."""
import os
import ctypes
from pathlib import Path
import re
import signal
import platform
import time


def syscall(number, *args):
    # Older Python/glibc builds omit wrappers even when the Linux kernel supports pidfds.
    if platform.system() != "Linux" or platform.machine() not in ("x86_64", "aarch64"):
        raise RuntimeError("owned cleanup requires Linux x86_64/aarch64 pidfd support")
    libc = ctypes.CDLL(None, use_errno=True)
    result = libc.syscall(number, *args)
    if result < 0:
        raise OSError(ctypes.get_errno(), "pidfd syscall failed")
    return result


def stop(run_id: str) -> None:
    if not re.fullmatch(r"[0-9a-f]{32}", run_id):
        raise ValueError("cleanup requires the 32-hex run identity from fleet.json")
    marker = f"CLUSTER_RUN_ID={run_id}".encode()
    for _ in range(100):
        found = False
        for entry in Path('/proc').glob('[0-9]*'):
            fd = None
            try:
                if entry.stat().st_uid != os.getuid():
                    continue
                fd = syscall(434, int(entry.name), 0)  # pidfd_open
                if marker in (entry / 'environ').read_bytes().split(b'\0'):
                    found = True
                    syscall(424, fd, signal.SIGKILL, 0, 0)  # pidfd_send_signal
            except (OSError, ProcessLookupError):
                pass
            finally:
                if fd is not None:
                    os.close(fd)
        if not found:
            return
        time.sleep(.02)
    raise RuntimeError('owned processes survived cleanup')


if __name__ == '__main__':
    import sys
    stop(sys.argv[1])
