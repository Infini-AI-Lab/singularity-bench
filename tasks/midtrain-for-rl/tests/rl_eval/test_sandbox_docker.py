"""Opt-in CPU Docker check for the real sandbox image and expanded-test path."""
import os
import shutil
import sys
import unittest

from sandbox import Judge


@unittest.skipUnless(os.environ.get('MIDTRAIN_RUN_DOCKER_TESTS') == '1'
                     and sys.platform == 'linux' and shutil.which('docker'),
                     'Requires opt-in Linux Docker and the rebuilt CPU sandbox image')
class SandboxDockerTests(unittest.TestCase):
    def test_expanded_tests_numpy_stdin_and_isolated_timeout(self):
        judge = Judge(os.environ.get('MIDTRAIN_SANDBOX_TEST_IMAGE', 'midtrain-code-sandbox:local'))
        tests = ('import numpy as np\n' + '#' + 'x' * 600_000 + '\n'
                 'def check(candidate):\n'
                 '    assert candidate(2) == 4\n'
                 '    np.testing.assert_allclose(candidate(-3), 9)\n')
        label = {'testtype': 'humanevalplus', 'entry_point': 'square', 'test': tests}
        self.assertTrue(judge.score('def square(x): return x * x', label))
        self.assertFalse(judge.score('def square(x): return 4', label))
        self.assertFalse(judge.score('raise SystemExit(0)', label))
        self.assertTrue(judge.score('from __future__ import annotations\nimport sys\n'
                                   'print(sum(map(int, sys.stdin.buffer.read().split())))',
                                   {'testtype': 'stdin', 'tests': [{'input': '2 3\n', 'output': '5'}]}))
        self.assertIsNone(judge.execute('while True: pass', '', timeout=0.2))


if __name__ == '__main__':
    unittest.main()
