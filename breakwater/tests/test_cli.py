"""The breakwater command: Apptainer is the default backend of ``run``; ``--backend docker`` selects Docker. No Docker
or GPUs.
"""
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

FRAMEWORK = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(FRAMEWORK)]
import cli  # noqa: E402


class CommandTests(unittest.TestCase):
    def launched(self, argv):
        with patch.object(cli.os, 'execv') as execv:
            cli.main(argv)
        return execv.call_args.args[1][2:]  # [sys.executable, '-I', script, *arguments]

    def test_run_takes_apptainer_unless_docker_is_asked_for(self):
        self.assertEqual(self.launched(['run', '--task', 'T']), [str(FRAMEWORK / 'apptainer' / 'run.py'), '--task', 'T'])
        self.assertEqual(self.launched(['run', '--backend', 'apptainer', '--task', 'T']),
                         [str(FRAMEWORK / 'apptainer' / 'run.py'), '--task', 'T'])
        self.assertEqual(self.launched(['run', '--task', 'T', '--backend', 'docker']), [str(FRAMEWORK / 'run.py'), '--task', 'T'])
        self.assertEqual(self.launched(['reverify', '--run-dir', 'R']),
                         [str(FRAMEWORK / 'apptainer' / 'reverify.py'), '--run-dir', 'R'])
        for argv in (['run', '--backend'], ['run', '--backend', 'podman', '--task', 'T']):
            with self.subTest(argv=argv), self.assertRaises(SystemExit), patch('sys.stderr'):
                cli.main(argv)


if __name__ == '__main__':
    unittest.main()
