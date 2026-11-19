"""CPU-only tests of the rootless-podman hooks for Harbor (worker/podman), without Harbor, podman or Docker.

python3 -m unittest discover -s tasks/rl-swe-agent/tests/unit -p test_podman_hooks.py -v
"""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[2]
HOOKS = ROOT / 'worker' / 'podman'
sys.path.insert(0, str(HOOKS))
import harbor_podman_hooks  # noqa: E402  (HARBOR_PODMAN_SANDBOXES is unset here, so it patches nothing)

# Stands in for the pinned agent server's docker_ops module: the names the hooks replace.
DOCKER_OPS = '''
import logging
logger = logging.getLogger("docker_ops")
async def _list_compose_projects(): raise AssertionError("not patched")
async def _run_docker(*cmd): raise AssertionError("not patched")
async def _prune_docker(): raise AssertionError("not patched")
def _check_docker_login(): raise AssertionError("not patched")
'''
# Records every podman invocation, one JSON argv per line, and lists one sandbox for "ps".
FAKE_PODMAN = '''#!/usr/bin/env python3
import json, os, sys
with open(os.environ["PODMAN_LOG"], "a") as log:
    log.write(json.dumps(sys.argv[1:]) + "\\n")
if sys.argv[1:2] == ["ps"]:
    print("harbor-trial-1\\nunrelated")
'''
PROBE = '''
import asyncio, json, sys
from agent_server import docker_ops
async def main():
    return {"projects": await docker_ops._list_compose_projects(),
            "down": await docker_ops._run_docker("docker", "compose", "-p", "trial-1", "down"),
            "network": await docker_ops._run_docker("docker", "network", "rm", "x"),
            "other": await docker_ops._run_docker("docker", "build", "."),
            "prune": await docker_ops._prune_docker(), "login": docker_ops._check_docker_login(),
            "hooked": "harbor_podman_hooks" in sys.modules}
print(json.dumps(asyncio.run(main())))
'''


class PodmanHooksTests(unittest.TestCase):
    def setUp(self):
        self.root = Path(tempfile.mkdtemp(prefix='podman-hooks-'))
        (self.root / 'agent_server').mkdir()
        (self.root / 'agent_server' / '__init__.py').write_text('')
        (self.root / 'agent_server' / 'docker_ops.py').write_text(DOCKER_OPS)
        podman = self.root / 'podman'
        podman.write_text(FAKE_PODMAN)
        podman.chmod(0o755)
        self.log = self.root / 'podman.log'

    def run_probe(self, environment):
        environment = {**environment, 'PYTHONPATH': os.pathsep.join(filter(None, [environment.get('PYTHONPATH'),
                                                                                   str(self.root)])),
                       'HARBOR_PODMAN': str(self.root / 'podman'), 'PODMAN_LOG': str(self.log)}
        result = subprocess.run([sys.executable, '-c', PROBE], env=environment, capture_output=True, text=True,
                                cwd=self.root, timeout=60)
        self.assertEqual(result.returncode, 0, result.stderr)
        return json.loads(result.stdout)

    def test_a_subprocess_with_the_environment_runs_the_agent_servers_docker_operations_on_podman(self):
        environment = harbor_podman_hooks.environment({'PATH': os.environ.get('PATH', '/usr/bin:/bin')})
        self.assertEqual(environment['HARBOR_PODMAN_SANDBOXES'], '1')
        shim = Path(environment['PYTHONPATH'].split(os.pathsep)[0])
        self.assertEqual((shim / 'sitecustomize.py').read_text().split()[:2], ['import', 'harbor_podman_hooks'])
        result = self.run_probe(environment)
        self.assertTrue(result['hooked'])
        self.assertEqual(result['projects'], ['trial-1'])  # only the hooks' own sandboxes, without their prefix
        self.assertEqual(result['down'][0], 0)
        self.assertEqual(result['network'], [0, '', ''])  # podman sandboxes create no per-trial networks
        self.assertEqual(result['other'][0], 1)  # anything else is refused, not passed to podman
        self.assertIn('skipped', result['prune'])
        self.assertIsNone(result['login'])
        calls = [json.loads(line) for line in self.log.read_text().splitlines()]
        self.assertEqual(calls, [['ps', '-a', '--format', '{{.Names}}'], ['rm', '--force', '--ignore', 'harbor-trial-1']])

    def test_without_the_variable_nothing_is_patched(self):
        result = subprocess.run([sys.executable, '-c', 'from agent_server import docker_ops; import asyncio; '
                                 'asyncio.run(docker_ops._prune_docker())'],
                                env={'PATH': os.environ.get('PATH', '/usr/bin:/bin'), 'PYTHONPATH': str(self.root)},
                                capture_output=True, text=True, cwd=self.root, timeout=60)
        self.assertIn('not patched', result.stderr)


if __name__ == '__main__':
    unittest.main()
