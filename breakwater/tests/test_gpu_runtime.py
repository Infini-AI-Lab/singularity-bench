"""Opt-in actual GB300 boundary test against a built task image.

Set BREAKWATER_GPU_TEST_IMAGE to the image built from capture/Dockerfile.gpu-fixture.
No author/judge model calls or benchmark inputs are used.
"""
import json
import os
from pathlib import Path
import secrets
import sys
import tempfile
import unittest

HERE = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(HERE), str(HERE / 'tests')]
import environment_guard as guard
import run as controller
from test_docker_boundary import available


@unittest.skipUnless(os.environ.get('BREAKWATER_GPU_TEST_IMAGE') and available(),
                     'Requires an explicitly selected GPU fixture image and four GB300 GPUs')
class GpuRuntimeTests(unittest.TestCase):
    def test_real_gpu_compute_recording_drain_fresh_environment_and_tamper_rejection(self):
        image = controller.image(os.environ['BREAKWATER_GPU_TEST_IMAGE'])
        collector = controller.image('breakwater-cpu-observer-fixture:local')
        prefix = 'breakwater-gpu-' + secrets.token_hex(5)
        names = [prefix + '-agent', prefix + '-intermediate']
        watchers = [name + '-observer' for name in names]
        with tempfile.TemporaryDirectory(dir=HERE / 'tests') as temporary:
            root = Path(temporary);root.chmod(0o755)
            baseline_dir = root / 'baseline';baseline_dir.mkdir(mode=0o755)
            baseline_path = baseline_dir / 'environment.json'
            baseline_path.write_text('{}');baseline_path.chmod(0o444)
            try:
                for index, stage in enumerate(('agent', 'intermediate')):
                    target, watcher = names[index], watchers[index]
                    folder = root / stage
                    controller.docker('create', '--name', target, '--network=none', '--ipc=private',
                        '--cap-drop=ALL', '--cap-add=SETUID', '--cap-add=SETGID',
                        '-e', 'PYTHONDONTWRITEBYTECODE=1', *controller.gpu_devices('device=0,1,2,3', 4),
                        *controller.mount(baseline_dir, '/opt/environment-baseline'), image)
                    controller.docker('start', target)
                    if index:
                        comparison = guard.compare(baseline['files'], guard.scan(target, specification), fresh_runtime=True)
                        self.assertEqual(comparison['status'], 'PASS', comparison)
                        print('Fresh intermediate runtime inventory: PASS', flush=True)
                    if not index:
                        specification = guard.describe(target, '/app')
                        baseline = {'specification': specification, 'files': guard.scan(target, specification)}
                        baseline_path.chmod(0o644)
                        baseline_path.write_text(json.dumps(baseline));baseline_path.chmod(0o444)
                        print('Initial runtime inventory:', len(baseline['files']), 'entries', flush=True)
                    controller.docker('exec', '--user=1101:1200', target, 'python3', '-I', '-B', '-c',
                        "from pathlib import Path; assert not Path('/tmp/worker-stage-marker').exists(); "
                        "assert not Path('/external-evidence').exists(); "
                        "assert Path('/opt/environment-baseline/environment.json').read_text()")
                    identity = {'task': 'gb300-runtime-smoke', 'run_id': prefix, 'stage': stage}
                    controller.observer(watcher, target, folder, collector, identity)
                    output = controller.docker('exec', target, 'python3', '-I', '-B', '/opt/gpu_fixture.py')
                    self.assertIn('GPU_COMPUTATION_OK', output)
                    print(stage, output, flush=True)
                    controller.docker('exec', '--detach', '--user=1101:1200', target, 'sleep', '600')
                    controller.docker('stop', '--time', '2', target)
                    self.assertEqual(controller.no_lingering_processes(target, watcher, folder)['remaining_pids'], [])
                    print(stage, 'process drain: PASS', flush=True)
                    controller.stop_observer(watcher, folder)
                    coverage = json.loads((folder / 'coverage.json').read_text())
                    self.assertTrue(coverage['complete'])
                    comparison = guard.compare(baseline['files'], guard.scan(target, specification), fresh_runtime=bool(index))
                    self.assertEqual(comparison['status'], 'PASS', comparison)
                    print(stage, 'stopped runtime inventory: PASS', flush=True)
                    self.assertIn(b'GPU_COMPUTATION_OK', b''.join(p.read_bytes() for p in (folder / 'streams').rglob('stdout.bin')))
                # Replace a genuine installed tensor package, then inspect the stopped
                # container through Docker export rather than trusting target imports.
                controller.docker('start', names[1])
                controller.docker('exec', names[1], 'python3', '-I', '-B', '-c',
                    "import importlib.util;from pathlib import Path;p=Path(importlib.util.find_spec('torch').origin);p.write_text('def noop(*args): pass\\n')")
                controller.docker('stop', '--time', '2', names[1])
                mismatch = guard.compare(baseline['files'], guard.scan(names[1], specification), fresh_runtime=True)
                self.assertEqual(mismatch['status'], 'FAIL')
                self.assertTrue(any(path.endswith('torch/__init__.py') for path in mismatch['changed']))
                print('Tampered Torch rejected:', json.dumps(mismatch), flush=True)
            finally:
                if sys.exc_info()[0] is not None:
                    for stage, watcher in zip(('agent', 'intermediate'), watchers):
                        print(stage, 'observer diagnostic:', controller.docker('logs', '--tail', '60', watcher, check=False), flush=True)
                        for filename in ('failure.json', 'coverage.json'):
                            path = root / stage / filename
                            if path.is_file():
                                print(stage, filename, path.read_text(), flush=True)
                for name in [*watchers, *names]:
                    controller.docker('rm', '-f', name, check=False)


if __name__ == '__main__':
    unittest.main()
