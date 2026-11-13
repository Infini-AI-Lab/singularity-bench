"""CPU-only protected-stream fixture. It never trains or calls a model API."""
import ctypes
import json
import os
import signal
import subprocess

if ctypes.CDLL('/opt/stdio_capture.so', use_errno=True).breakwater_capture_start():
    raise OSError(ctypes.get_errno(), 'capture startup')
os.kill(os.getpid(), signal.SIGSTOP)
os.setgroups([])
os.setresgid(1101, 1101, 1101)
os.setresuid(1101, 1101, 1101)
print(json.dumps({'type': 'item.started', 'item': {'type': 'command_execution', 'id': 'real-cpu-command',
                                                'command': "python3 -I -c print('loss=2.4')"}}), flush=True)
result = subprocess.run(['python3', '-I', '-c', "print('loss=2.4')"], capture_output=True, text=True, check=True)
print(json.dumps({'type': 'item.completed', 'item': {'type': 'command_execution', 'id': 'real-cpu-command',
    'command': "python3 -I -c print('loss=2.4')", 'aggregated_output': result.stdout, 'exit_code': result.returncode}}), flush=True)
