"""Live CPU integration of the harness validity battery in a protected intermediate container.

Exercises the real path: fresh container with the run.py intermediate restrictions, the
root observation barrier (observed.py), the external strace observer attached from the
host PID namespace, run_worker.py's capture barrier and privilege drop to UID 1101, the
JSON-line model channel THROUGH the protected stdio supervisor, root-side workspace
checks, process drain and complete observer finalization. Skipped without Docker.
No GPU, model, provider key or judge call is used.
"""
import json
import os
from pathlib import Path
import secrets
import shutil
import subprocess
import sys
import tempfile
import unittest

HERE = Path(__file__).resolve().parents[1]
TASK = HERE.parent / 'tasks' / 'coding-agent-harness'
REFERENCE = HERE / 'tests' / 'fixtures' / 'harness' / 'reference_agent.py'
sys.path[:0] = [str(HERE), str(HERE / 'tests')]
from test_docker_boundary import available

DOCKERFILE = '''FROM python:3.12-slim-bookworm AS build
RUN apt-get update && apt-get install -y --no-install-recommends gcc libc6-dev && rm -rf /var/lib/apt/lists/*
COPY stdio_capture.c stdio_capture.h /src/
RUN mkdir /out && gcc -Wall -Wextra -Werror -O2 -fPIC -shared /src/stdio_capture.c -o /out/stdio_capture.so
FROM python:3.12-slim-bookworm
COPY --from=build /out/stdio_capture.so /opt/breakwater-runtime/stdio_capture.so
COPY run_worker.py /opt/breakwater-runtime/run_worker.py
RUN groupadd -g 1200 shared && useradd -u 1101 -g 1200 -M -d /tmp/worker worker \\
    && chmod 755 /opt/breakwater-runtime && chmod 644 /opt/breakwater-runtime/* && mkdir -p /app
CMD ["sleep", "infinity"]
'''


@unittest.skipUnless(available() and REFERENCE.is_file(),
                     'Docker unavailable; protected battery integration is NOT verified')
class DockerHarnessIntermediateTests(unittest.TestCase):
    def test_battery_runs_as_worker_behind_observer_and_recomputes(self):
        import run as controller
        tag = secrets.token_hex(5)
        runtime, observer_image = f'breakwater-harness-int-runtime:{tag}', f'breakwater-harness-int-observer:{tag}'
        target, watcher = f'breakwater-harness-int-{tag}', f'breakwater-harness-int-{tag}-observer'
        with tempfile.TemporaryDirectory(dir=HERE / 'tests') as temporary:
            root = Path(temporary)
            root.chmod(0o755)
            context = root / 'context'
            context.mkdir()
            for name in ('stdio_capture.c', 'stdio_capture.h'):
                shutil.copyfile(HERE / 'capture' / name, context / name)
            shutil.copyfile(HERE / 'author/run_worker.py', context / 'run_worker.py')
            (context / 'Dockerfile').write_text(DOCKERFILE)
            workspace, evaluation, capture = root / 'app', root / 'evaluation', root / 'capture'
            (workspace / 'harness').mkdir(parents=True)
            shutil.copyfile(REFERENCE, workspace / 'harness/agent.py')
            evaluation.mkdir()
            evaluation.chmod(0o777)
            for directory, dirs, files in os.walk(root):
                Path(directory).chmod(0o755)
                for name in files:
                    (Path(directory) / name).chmod(0o644)
            evaluation.chmod(0o777)
            try:
                subprocess.run(['docker', 'build', '-q', '-t', runtime, str(context)], check=True, capture_output=True)
                subprocess.run(['docker', 'build', '-q', '-t', observer_image, str(HERE / 'collector')], check=True,
                               capture_output=True)
                controller.docker('create', '--name', target, '--ipc=private', '--network=none', '--cap-drop=ALL',
                    '--cap-add=SETUID', '--cap-add=SETGID', '--cap-add=CHOWN', '--cap-add=FOWNER', '--cap-add=DAC_OVERRIDE',
                    '--cap-add=KILL',
                    '--security-opt=no-new-privileges:true', '-e', 'BREAKWATER_STAGE=intermediate', '-e', 'PYTHONDONTWRITEBYTECODE=1',
                    *controller.mount(workspace, '/app'), *controller.mount(evaluation, '/evaluation', False),
                    *controller.mount(TASK / 'tests', '/private/tests'), *controller.mount(HERE, '/opt/breakwater'),
                    *controller.mount(TASK / 'breakwater.json', '/trusted-task/breakwater.json'),
                    *controller.mount(HERE / 'drivers', '/opt/breakwater-drivers'),
                    '--entrypoint', 'sleep', runtime, 'infinity')
                controller.docker('start', target)
                identity = {'task': 'coding-agent-harness', 'run_id': 'integration-' + tag, 'stage': 'intermediate'}
                controller.observer(watcher, target, capture, controller.image(observer_image), identity)
                result = subprocess.run(['docker', 'exec', target, 'python3', '-I', '-B', '/opt/breakwater/observed.py'],
                                        capture_output=True, timeout=1500)
                self.assertEqual(result.returncode, 0, result.stderr.decode(errors='replace')[-3000:])
                controller.docker('stop', '--time', '10', target)
                drain = controller.no_lingering_processes(target, watcher, capture)
                self.assertEqual(drain['remaining_pids'], [])
                controller.stop_observer(watcher, capture)
                coverage = json.loads((capture / 'coverage.json').read_text())
                self.assertTrue(coverage['complete'], coverage.get('gaps'))
                sealed = root / 'sealed'
                controller.snapshot(target, '/evaluation', sealed, [])
                battery = json.loads((sealed / 'raw/harness/battery.json').read_text())
                self.assertEqual(battery['summary']['gradeable'], 1.0)
                self.assertEqual(battery['summary']['valid'], 1.0)
                self.assertEqual(battery['summary']['model_calls_total'], 34.0)
                import score_workload
                self.assertEqual(score_workload.harness_score(sealed / 'raw', smoke=TASK / 'tests/smoke',
                                                              module=workspace / 'harness/agent.py'), 1.0)
                # The coordinator was attached behind its protected barrier, and every scenario worker
                # was traced from fork by inheritance: the external traces hold each privilege drop to
                # UID 1101 and each exec of the fixed public runner.
                attached = [json.loads(line) for line in (capture / 'observer.jsonl').read_text().splitlines()
                            if '"exec_attached"' in line]
                self.assertTrue(any(event['protected_stream_supervisor'] for event in attached))
                import review_index
                parser = review_index.syscall_parser()
                drops = runners = 0
                sample = []
                for trace in sorted((capture / 'syscalls').iterdir()):
                    if trace.suffix == '.stderr':
                        continue
                    for _, body in parser.records(trace):
                        text = parser.decoded(body)
                        if len(sample) < 40 and (b'execve' in text or b'prctl' in text):
                            sample.append(trace.name + ': ' + text[:200].decode(errors='replace'))
                        # run_worker: setresuid (untraced %creds) is followed by PR_SET_NO_NEW_PRIVS.
                        drops += b'prctl(PR_SET_NO_NEW_PRIVS, 1' in text
                        runners += text.startswith(b'execve(') and b'harness_child.py' in text
                self.assertGreaterEqual(runners, len(battery['rows']), '\n'.join(sample))
                self.assertGreaterEqual(drops, len(battery['rows']), '\n'.join(sample))
            finally:
                controller.docker('rm', '-f', watcher, target, check=False)
                # Evaluation files were written by the container's root; remove them as root.
                subprocess.run(['docker', 'run', '--rm', '--network=none', '--entrypoint', 'sh',
                                *controller.mount(root, '/cleanup', False), runtime, '-c',
                                'rm -rf /cleanup/evaluation /cleanup/capture /cleanup/sealed'], capture_output=True)
                subprocess.run(['docker', 'rmi', '-f', runtime, observer_image], capture_output=True)


if __name__ == '__main__':
    unittest.main()
