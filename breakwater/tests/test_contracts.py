"""The Docker controller's contract: GPU device counts, credentials kept out of execution containers, protected paths,
trusted mount points and shadow policies. No API keys, GPUs or Docker.
"""
import copy
import hashlib
import http.client
import http.server
import json
import os
from pathlib import Path
import shutil
import signal
import socket
import subprocess
import sys
import tempfile
import threading
import time
import unittest
import urllib.error
import urllib.request
from unittest.mock import patch

FRAMEWORK = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(FRAMEWORK), str(FRAMEWORK / 'tests')]
from contract import ContractError, RulesViolation, inspect_boundary  # noqa: E402
import run as controller  # noqa: E402
from test_security import inspected  # noqa: E402


class ContractTests(unittest.TestCase):
    def test_declared_gpu_counts_zero_one_two(self):
        self.assertEqual(controller.gpu_devices('device=3', 1), ['--device', 'nvidia.com/gpu=3'])
        self.assertEqual(controller.gpu_devices('none', 0), [])
        self.assertEqual(controller.gpu_devices('device=0,1,2,3', 4)[1::2],
                         ['nvidia.com/gpu=0', 'nvidia.com/gpu=1', 'nvidia.com/gpu=2', 'nvidia.com/gpu=3'])
        with self.assertRaises(ContractError):
            controller.gpu_devices('device=0,1,2,2', 4)
        for value, count in (('device=0,1', 1), ('device=0', 0), ('all', 1), ('device=0', 2), ('none', 1)):
            with self.subTest(value=value, count=count), self.assertRaises(ContractError):
                controller.gpu_devices(value, count)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for count, ids, ok in ((1, ['nvidia.com/gpu=0'], True), (1, ['nvidia.com/gpu=0', 'nvidia.com/gpu=1'], False),
                                   (2, ['nvidia.com/gpu=0', 'nvidia.com/gpu=1'], True), (0, ['nvidia.com/gpu=0'], False)):
                value = inspected('intermediate', root, 'i')
                value['HostConfig']['DeviceRequests'] = [{'Driver': 'cdi', 'Count': 0, 'DeviceIDs': ids,
                                                          'Capabilities': None, 'Options': None}]
                with self.subTest(count=count, ids=ids):
                    if ok:
                        inspect_boundary(value, 'intermediate', root, gpu_count=count)
                    else:
                        with self.assertRaises(ContractError):
                            inspect_boundary(value, 'intermediate', root, gpu_count=count)

    def test_gateway_and_task_secrets_never_enter_execution_containers(self):
        with tempfile.TemporaryDirectory() as temporary:
            for name in ('UPSTREAM_API_KEY', 'DEV_UPSTREAM_API_KEY', 'AUTHOR_UPSTREAM_API_KEY', 'TASK_GRADER_TOKEN'):
                value = inspected('agent', Path(temporary), 'a')
                value['Config']['Env'] = [name + '=[REDACTED]']
                with self.subTest(name=name), self.assertRaises(ContractError):
                    # The task's declared secrets (its contract's credential_env) join the framework's key variables.
                    inspect_boundary(value, 'agent', Path(temporary), gpu_count=1, credential_names=('TASK_GRADER_TOKEN',))
            value = inspected('agent', Path(temporary), 'a')
            value['Config']['Env'] = ['TASK_RECEIPT_KEY=[REDACTED]']  # an operator value, public by design
            inspect_boundary(value, 'agent', Path(temporary), gpu_count=0, credential_names=('TASK_GRADER_TOKEN',))

    def test_protected_paths_detect_edit_add_delete(self):
        from contract import inventory
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / 'spec').mkdir()
            (root / 'src').mkdir()
            (root / 'spec/SPEC.md').write_text('spec')
            (root / 'selfcheck.py').write_text('check')
            (root / 'src/agent.py').write_text('def run(*a): pass')
            baseline = inventory(root)
            paths = ['spec', 'selfcheck.py']
            (root / 'src/agent.py').write_text('def run(*a): return None')  # allowed
            self.assertEqual(controller.protected_paths_check(baseline, root, paths)['status'], 'PASS')
            for mutation in ('edit', 'add', 'delete'):
                trial = root.parent / (root.name + mutation)
                shutil.copytree(root, trial)
                if mutation == 'edit':
                    (trial / 'spec/SPEC.md').write_text('weakened spec')
                if mutation == 'add':
                    (trial / 'spec/extra.md').write_text('x')
                if mutation == 'delete':
                    (trial / 'selfcheck.py').unlink()
                with self.subTest(mutation=mutation), self.assertRaises(RulesViolation):
                    controller.protected_paths_check(baseline, trial, paths)
                shutil.rmtree(trial)

    def test_trusted_mount_points_are_excluded_from_runtime_inventory(self):
        import environment_guard as guard
        spec = {'version': guard.VERSION, 'roots': ['/opt'], 'excludes': guard.EXCLUDES + guard.TRUSTED_MOUNTS}
        self.assertFalse(guard.selected('opt/breakwater', spec))
        self.assertFalse(guard.selected('opt/breakwater/verifier.py', spec))
        # A custom launcher's own mount points (describe(trusted_mounts=...)) are left out the same way.
        self.assertTrue(guard.selected('opt/breakwater-worker/kernel_worker.py', spec))
        spec['excludes'] = spec['excludes'] + ['/opt/breakwater-worker']
        self.assertFalse(guard.selected('opt/breakwater-worker/kernel_worker.py', spec))
        # The protected runtime inside the image is fingerprinted like any other image content.
        self.assertTrue(guard.selected('opt/breakwater-runtime/run_worker.py', spec))

    def test_shadow_policies(self):
        import environment_guard as guard
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / 'pytest').mkdir()
            (root / 'pytest/__init__.py').write_text('fake')
            (root / 'vendor/site-packages/aiohttp').mkdir(parents=True)
            (root / 'vendor/site-packages/aiohttp/__init__.py').write_text('vendored')
            # A task whose evaluation imports its own test tools protects them too...
            extra = guard.shadow_policy({'extra_packages': ['pytest', '_pytest', 'pluggy']})
            self.assertIn('pytest/__init__.py', guard.shadows(root, extra))
            # ...and one whose trusted code never imports the workspace may permit vendored dependencies.
            empty = guard.shadow_policy({'packages': [], 'hooks': [], 'environments': False})
            self.assertEqual(guard.shadows(root, empty), {})


if __name__ == '__main__':
    unittest.main()
