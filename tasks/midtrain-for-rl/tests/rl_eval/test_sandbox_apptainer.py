"""Opt-in Apptainer check of the code sandbox (sandbox.py --apptainer), for hosts without Docker.

    MIDTRAIN_SANDBOX_TEST_SIF=midtrain-code-sandbox.sif python3 -m unittest test_sandbox_apptainer

With the real CPU image (apptainer-sandbox.def) it also runs the NumPy-based expanded test path; any SIF with
python3 checks the isolation.
"""
import os
import shutil
import subprocess
import time
import unittest

from sandbox import Judge

SIF = os.environ.get('MIDTRAIN_SANDBOX_TEST_SIF')


@unittest.skipUnless(SIF and (shutil.which('apptainer') or shutil.which('singularity')),
                     'Set MIDTRAIN_SANDBOX_TEST_SIF to a SIF of the CPU sandbox image')
class SandboxApptainerTests(unittest.TestCase):
    def setUp(self):
        self.judge = Judge(None, apptainer=SIF)

    def test_stdin_tests_timeouts_and_isolation(self):
        self.assertTrue(self.judge.image.startswith('sha256:'))
        self.assertTrue(self.judge.score('import sys\nprint(sum(map(int, sys.stdin.buffer.read().split())))',
                                         {'testtype': 'stdin', 'tests': [{'input': '2 3\n', 'output': '5'}]}))
        self.assertFalse(self.judge.score('print(6)', {'testtype': 'stdin', 'tests': [{'input': '2 3\n', 'output': '5'}]}))
        self.assertIsNone(self.judge.execute('while True: pass', '', timeout=1.0))
        # No network, no host files, no other processes, and the limits hold.
        probe = ('import os, socket\n'
                 'try:\n    socket.create_connection(("1.1.1.1", 53), timeout=2); net = "net"\n'
                 'except OSError:\n    net = "no-net"\n'
                 'home = "home" if os.path.exists(os.path.expanduser("~/.bashrc")) else "no-home"\n'
                 'pids = len([p for p in os.listdir("/proc") if p.isdigit()])\n'
                 'try:\n    bytearray(1 << 30); memory = "1GiB"\n'
                 'except MemoryError:\n    memory = "capped"\n'
                 'print(net, home, "few-processes" if pids < 8 else pids, memory)\n')
        self.assertEqual(self.judge.execute(probe, ''), 'no-net no-home few-processes capped\n')
        # A fork bomb ends at the process limit or the deadline, and leaves nothing behind.
        self.assertIsNone(self.judge.execute('import os\nwhile True:\n    os.fork()', '', timeout=2.0))
        deadline = time.monotonic() + 10  # a killed container's processes take a moment to be reaped
        while (leftover := subprocess.run(["pgrep", "-af", 'sandbox.py", "exec"'], capture_output=True,
                                          text=True).stdout) and time.monotonic() < deadline:
            time.sleep(0.2)
        self.assertEqual(leftover, '')

    @unittest.skipUnless(os.environ.get('MIDTRAIN_SANDBOX_TEST_NUMPY') == '1', 'the real CPU image has NumPy')
    def test_expanded_tests_with_numpy(self):
        tests = ('import numpy as np\n' + '#' + 'x' * 600_000 + '\n'
                 'def check(candidate):\n'
                 '    assert candidate(2) == 4\n'
                 '    np.testing.assert_allclose(candidate(-3), 9)\n')
        label = {'testtype': 'humanevalplus', 'entry_point': 'square', 'test': tests}
        self.assertTrue(self.judge.score('def square(x): return x * x', label))
        self.assertFalse(self.judge.score('def square(x): return 4', label))
        self.assertFalse(self.judge.score('raise SystemExit(0)', label))


if __name__ == '__main__':
    unittest.main()
