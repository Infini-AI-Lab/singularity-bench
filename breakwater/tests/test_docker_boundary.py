"""Live CPU container regressions. Explicitly skipped when Docker is unavailable.

These use synthetic log fixtures and test mount/credential boundaries plus the
mechanical comparator inside a verifier container. They are not GPU task runs.
"""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

HERE = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(HERE), str(Path(__file__).resolve().parent)]
from test_security import chain, native
from contract import seal


def available():
    try:
        return subprocess.run(['docker', 'info'], capture_output=True, timeout=5).returncode == 0
    except (OSError, subprocess.TimeoutExpired):
        return False


@unittest.skipUnless(available(), 'Docker daemon unavailable; live container isolation is NOT verified')
class DockerBoundaryTests(unittest.TestCase):
    def test_each_task_hides_logs_from_execution_and_readonly_verifier_checks_them(self):
        tasks = ['open-ended-nano-gpt', 'midtrain-for-rl', 'rl-swe-agent', 'serve-glm53-flash',
                 'coding-agent-harness', 'cluster-serverless']
        image = os.environ.get('BREAKWATER_CPU_TEST_IMAGE', 'python:3.12-slim-bookworm')
        for task in tasks:
            with self.subTest(task=task), tempfile.TemporaryDirectory(dir=HERE / 'tests') as temporary:
                root = Path(temporary)
                external, local = root / 'external', root / 'local'
                external.mkdir();local.mkdir()
                identity = {'task': task, 'run_id': 'fixture-only'}
                (external / 'isolation.canary').write_text('hidden fixture')
                for stage in ('agent', 'intermediate'):
                    folder = external / stage;folder.mkdir()
                    fields = {**identity, 'stage': stage}
                    last = chain(folder / 'observer.jsonl', [{**fields, 'event': 'collector_ready'},
                        {**fields, 'event': 'exec_attached', 'protected_stream_supervisor': True},
                        {**fields, 'event': 'collector_stopped', 'complete': True, 'gaps': []}])
                    (folder / 'coverage.json').write_text(json.dumps({'complete': True, 'gaps': [], 'last_event_sha256': last, 'protocol': 'external-evidence-v3'}))
                for stage in ('agent', 'intermediate'):
                    (external / stage / 'ready.json').write_text(json.dumps({'ready': True, 'protocol': 'external-evidence-v3'}))
                (external / 'gateway').mkdir()
                chain(external / 'gateway/requests.jsonl', [{'event': 'gateway_started'},
                    {'event': 'model_request', 'call_id': 'a'}, {'event': 'model_response_end', 'call_id': 'a', 'status': 200},
                    {'event': 'gateway_stopped', 'faulted': False}], gateway=True)
                (external / 'agent-native.jsonl').write_bytes(native())
                (external / 'agent-native.stderr').write_bytes(b'')
                (local / 'native.jsonl').write_bytes(native());(local / 'native.stderr').write_bytes(b'')
                authority = seal(external, identity, finalized=True, stage_order=['agent', 'intermediate'])
                for directory, _, files in os.walk(root):
                    Path(directory).chmod(0o755)
                    for name in files:(Path(directory) / name).chmod(0o444)
                common = ['docker', 'run', '--rm', '--read-only', '--network=none', '--ipc=private',
                          '--cap-drop=ALL', '--security-opt=no-new-privileges:true', '--user=1102:1102']
                for stage in ('agent', 'intermediate'):
                    subprocess.run([*common, image, 'python3', '-I', '-c',
                        "from pathlib import Path; assert not Path('/external-evidence/isolation.canary').exists()"], check=True)
                program = '''import sys
sys.path.insert(0, '/checker')
from reconcile import reconcile
from pathlib import Path
reconcile('/external-evidence', sys.argv[1], '/agent-local')
try:
 with Path('/external-evidence/isolation.canary').open('ab') as f: f.write(b'forbidden')
except PermissionError: pass
except OSError as e:
 assert e.errno == 30
else: raise AssertionError('writable evidence')
'''
                subprocess.run([*common, '--mount', f'type=bind,source={external},target=/external-evidence,readonly',
                    '--mount', f'type=bind,source={local},target=/agent-local,readonly',
                    '--mount', f'type=bind,source={HERE},target=/checker,readonly',
                    image, 'python3', '-I', '-c', program, authority], check=True)
                # Tampering in the formerly agent-writable mirror must fail in the verifier.
                (local / 'native.jsonl').chmod(0o644)
                (local / 'native.jsonl').write_bytes(native().replace(b'loss=2.4', b'loss=0.0'))
                result = subprocess.run([*common, '--mount', f'type=bind,source={external},target=/external-evidence,readonly',
                    '--mount', f'type=bind,source={local},target=/agent-local,readonly',
                    '--mount', f'type=bind,source={HERE},target=/checker,readonly',
                    image, 'python3', '-I', '-c', program, authority], capture_output=True)
                self.assertNotEqual(result.returncode, 0)


if __name__ == '__main__':
    unittest.main()

@unittest.skipUnless(available(), 'Docker unavailable; external recorder integration is NOT verified')
class DockerCollectorTests(unittest.TestCase):
    def test_real_external_recorder_attaches_before_worker_and_finalizes_streams(self):
        import secrets
        import run as controller
        from reconcile import check_chain
        target = 'breakwater-capture-fixture-' + secrets.token_hex(5)
        watcher = target + '-observer'
        worker_image, collector_image = 'breakwater-cpu-capture-fixture:local', 'breakwater-cpu-observer-fixture:local'
        subprocess.run(['docker', 'build', '-q', '-t', worker_image, str(HERE / 'capture')], check=True, capture_output=True)
        subprocess.run(['docker', 'build', '-q', '-t', collector_image, str(HERE / 'collector')], check=True, capture_output=True)
        with tempfile.TemporaryDirectory(dir=HERE / 'tests') as temporary:
            folder = Path(temporary) / 'capture'
            identity = {'task': 'cpu-collector-fixture', 'run_id': target, 'stage': 'intermediate'}
            try:
                controller.docker('run', '-d', '--name', target, '--network=none', '--ipc=private',
                                  '--cap-drop=ALL', '--cap-add=SETUID', '--cap-add=SETGID', worker_image)
                controller.observer(watcher, target, folder, controller.image(collector_image), identity)
                result = subprocess.run(['docker', 'exec', target, 'python3', '-I', '/opt/fixture.py'],
                                        capture_output=True, timeout=30, check=True)
                self.assertIn(b'loss=2.4', result.stdout)
                # A malicious worker deliberately leaves a detached background child.
                controller.docker('exec', '--detach', '--user=1101:1200', target, 'sleep', '600')
                before = json.loads(controller.docker('exec', watcher, 'python3', '-I', '-c',
                    "import sys,json;sys.path.insert(0,'/opt');import process_guard as p;print(json.dumps(p.remaining(json.load(open('/capture/process-boundary.json')))))"))
                self.assertTrue(before)
                controller.docker('stop', '--time', '5', target)
                drain = controller.no_lingering_processes(target, watcher, folder)
                self.assertEqual(drain['status'], 'PASS')
                self.assertEqual(drain['remaining_pids'], [])
                controller.stop_observer(watcher, folder)
                coverage = json.loads((folder / 'coverage.json').read_text())
                self.assertTrue(coverage['complete'])
                check_chain(folder / 'observer.jsonl', identity=identity, blob_root=folder)
                streams = list((folder / 'streams').rglob('stdout.bin'))
                self.assertTrue(streams)
                self.assertIn(b'loss=2.4', b''.join(path.read_bytes() for path in streams))
            finally:
                controller.docker('rm', '-f', watcher, target, check=False)


@unittest.skipUnless(available(), 'Docker unavailable; runtime package integrity is NOT verified')
class DockerEnvironmentTests(unittest.TestCase):
    def test_readonly_baseline_and_external_hashing_detect_noop_package_and_python(self):
        import secrets
        import run as controller
        import environment_guard as guard
        image = 'breakwater-cpu-capture-fixture:local'
        subprocess.run(['docker', 'build', '-q', '-t', image, str(HERE / 'capture')], check=True, capture_output=True)
        target = 'breakwater-environment-fixture-' + secrets.token_hex(5)
        fresh = target + '-fresh'
        spec = {'version': guard.VERSION, 'roots': ['/opt/frozen', '/usr/local/bin'], 'excludes': []}
        with tempfile.TemporaryDirectory(dir=HERE / 'tests') as temporary:
            folder = Path(temporary);folder.chmod(0o755)
            baseline_path = folder / 'environment.json'
            baseline_path.write_text('{}');baseline_path.chmod(0o444)
            try:
                controller.docker('run', '-d', '--name', target, '--network=none',
                    '-e', 'PYTHONDONTWRITEBYTECODE=1', '--mount',
                    f'type=bind,source={folder},target=/opt/environment-baseline,readonly', image)
                baseline = guard.scan(target, spec)
                baseline_path.chmod(0o644);baseline_path.write_text(json.dumps(baseline));baseline_path.chmod(0o444)
                readonly = controller.docker('exec', '--user=1101:1200', target, 'python3', '-I', '-c',
                    "from pathlib import Path;p=Path('/opt/environment-baseline/environment.json');assert p.read_text();\ntry:p.write_text('forged')\nexcept OSError:print('READ_ONLY')\nelse:raise AssertionError('writable baseline')")
                self.assertIn('READ_ONLY', readonly)
                controller.docker('create', '--name', fresh, image)
                self.assertEqual(guard.compare(baseline, guard.scan(fresh, spec))['status'], 'PASS')
                controller.docker('exec', target, 'sh', '-c',
                    "printf 'def genuine(): pass\\n' > /opt/frozen/sglang/__init__.py; "
                    "printf '#!/bin/sh\\nexit 0\\n' > /usr/local/bin/noop-python; "
                    "chmod 755 /usr/local/bin/noop-python; mv /usr/local/bin/noop-python /usr/local/bin/python3.12")
                controller.docker('stop', '--time', '1', target)
                result = guard.compare(baseline, guard.scan(target, spec))
                self.assertEqual(result['status'], 'FAIL')
                self.assertIn('opt/frozen/sglang/__init__.py', result['changed'])
                self.assertIn('usr/local/bin/python3.12', result['changed'])
            finally:
                controller.docker('rm', '-f', target, fresh, check=False)


@unittest.skipUnless(available(), 'Docker unavailable; image workspace seeding is NOT verified')
class DockerWorkspaceSeedTests(unittest.TestCase):
    def test_image_seeding_copies_only_the_image_workdir_and_rejects_links(self):
        """workspace_source=image (harness, cluster, serving): exactly the image's workdir, no task tree."""
        import secrets
        from unittest.mock import patch
        import run as controller
        from contract import ContractError, inventory
        base = os.environ.get('BREAKWATER_CPU_TEST_IMAGE', 'python:3.12-slim-bookworm')
        tag = secrets.token_hex(5)
        images = {'clean': f'breakwater-seed-fixture:{tag}', 'linked': f'breakwater-seed-fixture:{tag}-link'}
        recipes = {'clean': ("RUN mkdir -p /app/agentx /app/out && printf '#!/bin/sh\\nexec true\\n' > /app/serve.sh "
                             "&& chmod 755 /app/serve.sh && echo notes > /app/agentx/code.txt && echo ckpt > /app/out/weights.bin"),
                   'linked': 'RUN mkdir -p /app && echo x > /app/real.txt && ln -s /etc/passwd /app/link'}
        created, original = [], controller.docker
        def recording(*args, **kwargs):
            if args[:1] == ('create',):
                created.append(args[list(args).index('--name') + 1])
            return original(*args, **kwargs)
        with tempfile.TemporaryDirectory(dir=HERE / 'tests') as temporary:
            root = Path(temporary)
            try:
                for kind, recipe in recipes.items():
                    context = root / ('context-' + kind)
                    context.mkdir()
                    (context / 'Dockerfile').write_text(f'FROM {base}\n{recipe}\nWORKDIR /app\n')
                    subprocess.run(['docker', 'build', '-q', '-t', images[kind], str(context)], check=True, capture_output=True)
                config = {'workspace_source': 'image', 'workdir': '/app', 'workspace_excludes': ['out']}
                with patch.object(controller, 'docker', recording):
                    controller.seed_workspace(config, controller.image(images['clean']), root / 'seeded')
                    self.assertEqual(sorted(inventory(root / 'seeded')), ['agentx/code.txt', 'serve.sh'])
                    self.assertTrue(os.access(root / 'seeded/serve.sh', os.X_OK))
                    with self.assertRaises(ContractError):
                        controller.seed_workspace(config, controller.image(images['linked']), root / 'linked')
                self.assertEqual(len(created), 2)
                for name in created:  # The temporary seed containers never outlive seeding.
                    self.assertEqual(original('ps', '-a', '--filter', f'name=^/{name}$', '--format', '{{.Names}}').strip(), '')
            finally:
                subprocess.run(['docker', 'rm', '-f', *created], capture_output=True)
                subprocess.run(['docker', 'rmi', '-f', *images.values()], capture_output=True)
