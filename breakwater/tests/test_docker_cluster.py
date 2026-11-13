"""Live mechanics of the cluster grading coordinator against a stub fleet and a stub driver.

A protected intermediate container (run.py restrictions plus the coordinator's NET_ADMIN/NET_RAW
for its egress rules) runs cluster_cells.py behind the real external observer. A stub fleet
container answers the controller and node-manager admin calls and exposes an engine port.
The submitted plane is a stub that probes its own boundary. The task's driver is replaced by a
stub that waits for the plane and records which secrets it received. Verified: the fleet is
four trays of four B300-named GPUs; every cell resets and rekeys every node, starts a fresh
worker-owned /app and /home/worker, runs the plane as UID 1101 with RLIMIT_NPROC 512,
admits engine traffic and rejects node managers and /private, keeps keys out of the plane,
stops every worker process with root's KILL capability, writes the measurement record and
finalizes the observer without gaps. No GPU, model, trace generator or provider key is used.
"""
import hashlib
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
TASK = HERE.parent / 'tasks' / 'cluster-serverless'
sys.path[:0] = [str(HERE), str(HERE / 'tests')]
from test_docker_boundary import available

KIND = json.loads((TASK / 'breakwater.json').read_text()).get('kind')

IMAGE = '''FROM python:3.12-slim-bookworm AS build
RUN apt-get update && apt-get install -y --no-install-recommends gcc libc6-dev && rm -rf /var/lib/apt/lists/*
COPY stdio_capture.c stdio_capture.h /src/
RUN mkdir /out && gcc -Wall -Wextra -Werror -O2 -fPIC -shared /src/stdio_capture.c -o /out/stdio_capture.so
FROM python:3.12-slim-bookworm
RUN apt-get update && apt-get install -y --no-install-recommends iptables && rm -rf /var/lib/apt/lists/*
COPY --from=build /out/stdio_capture.so /opt/breakwater-runtime/stdio_capture.so
COPY run_worker.py /opt/breakwater-runtime/run_worker.py
RUN groupadd -g 1200 shared && useradd -u 1101 -g 1200 -M -d /tmp/worker worker \\
    && chmod 755 /opt/breakwater-runtime && chmod 644 /opt/breakwater-runtime/* && mkdir -p /app
CMD ["sleep", "infinity"]
'''

FLEET = r'''import json, threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
calls = []
NODES = [{'node': f'node{n}', 'hardware': {'devices': [{'name': 'NVIDIA GB300', 'uuid': f'GPU-{n}-{g}'} for g in range(4)]},
          'gpus': [{'index': g, 'uuid': f'GPU-{n}-{g}', 'total_mib': 288000} for g in range(4)],
          'host_ram': {'total_gib': 968.0, 'source': 'stub'}} for n in range(4)]
class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a): pass
    def reply(self, value):
        raw = json.dumps(value).encode(); self.send_response(200)
        self.send_header('Content-Length', str(len(raw))); self.end_headers(); self.wfile.write(raw)
    def do_GET(self):
        if self.path == '/calls': return self.reply(calls)
        if self.path == '/nodes': return self.reply(NODES)
        self.reply({'ok': True, 'port': self.server.server_address[1]})
    def do_POST(self):
        body = self.rfile.read(int(self.headers.get('Content-Length', 0)))
        calls.append({'port': self.server.server_address[1], 'path': self.path,
                      'token_ok': self.headers.get('X-Grader-Token') == 'grader-secret', 'body': json.loads(body or b'{}')})
        self.reply({'engines_remaining': 0} if self.path.endswith('reset') else {})
for port in (9000, 9100, 9150):
    threading.Thread(target=ThreadingHTTPServer(('0.0.0.0', port), Handler).serve_forever, daemon=True).start()
threading.Event().wait()
'''

PLANE = r'''#!/bin/bash
exec python3 - <<'EOF'
import json, os, resource, socket, urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
def probe(port):
    try:
        with socket.create_connection(('fleet', port), timeout=3): return 'connected'
    except OSError as error: return type(error).__name__
facts = {'uid': os.getuid(), 'app_owner': os.stat('/app').st_uid,
         'private': 'readable' if os.access('/private', os.R_OK) else 'denied',
         'engine': probe(9150), 'manager': probe(9100),
         'secrets_in_env': sorted(k for k in os.environ if k in ('CLUSTER_RECEIPT_KEY', 'CLUSTER_GRADER_TOKEN')),
         'catalogue_owner': os.stat('/app/catalogue.json').st_uid, 'home': os.environ.get('HOME'),
         'home_owner': os.stat(os.environ['HOME']).st_uid, 'nproc': resource.getrlimit(resource.RLIMIT_NPROC),
         'link': os.readlink('/app/alias.sh')}
print('PLANE-FACTS ' + json.dumps(facts, sort_keys=True), flush=True)
class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a): pass
    def do_GET(self):
        self.send_response(200); self.send_header('Content-Length', '2'); self.end_headers(); self.wfile.write(b'ok')
ThreadingHTTPServer(('127.0.0.1', int(os.environ['PORT'])), Handler).serve_forever()
EOF
'''

DRIVER = r'''import argparse, json, os, time, urllib.request
p = argparse.ArgumentParser()
for name in ('--trace', '--fleet', '--target', '--out', '--stats-out'): p.add_argument(name)
a = p.parse_args()
deadline = time.time() + 60
while True:
    try:
        urllib.request.urlopen(a.target + '/health', timeout=2); break
    except OSError:
        if time.time() > deadline: raise SystemExit('plane never became healthy')
        time.sleep(.2)
open(a.out, 'w').close()
json.dump({'driver_uid': os.getuid(), 'key': bool(os.environ.get('CLUSTER_RECEIPT_KEY')),
           'token': os.environ.get('CLUSTER_GRADER_TOKEN') == 'grader-secret'}, open(a.stats_out, 'w'))
'''


@unittest.skipUnless(available() and KIND == 'cluster', 'Docker unavailable or not the cluster task; '
                     'cluster coordinator mechanics are NOT verified')
class DockerClusterCoordinatorTests(unittest.TestCase):
    def test_cells_reset_rekey_isolate_and_tear_down_the_worker_plane(self):
        import run as controller
        tag = secrets.token_hex(5)
        image, observer_image = f'breakwater-cluster-int:{tag}', f'breakwater-cluster-int-observer:{tag}'
        network, fleet_name = f'breakwater-cluster-int-{tag}', f'breakwater-cluster-int-{tag}-fleet'
        target, watcher = f'breakwater-cluster-int-{tag}', f'breakwater-cluster-int-{tag}-observer'
        with tempfile.TemporaryDirectory(dir=HERE / 'tests') as temporary:
            root = Path(temporary)
            context = root / 'context'
            context.mkdir()
            for name in ('stdio_capture.c', 'stdio_capture.h'):
                shutil.copyfile(HERE / 'capture' / name, context / name)
            shutil.copyfile(HERE / 'author/run_worker.py', context / 'run_worker.py')
            (context / 'Dockerfile').write_text(IMAGE)
            # Trusted code with a development declaration, the task's real kit plus a stub driver.
            security = root / 'breakwater'
            shutil.copytree(HERE, security, ignore=shutil.ignore_patterns('tests', '__pycache__'))
            contract = root / 'breakwater.json'
            config = json.loads((TASK / 'breakwater.json').read_text())
            config['cluster'] = {'repeats': 1, 'allow_unqualified': True, 'time_scale': 1}
            contract.write_text(json.dumps(config))
            tests = root / 'tests'
            (tests / 'kit').mkdir(parents=True)
            for path in (TASK / 'tests/kit').glob('*.py'):
                shutil.copyfile(path, tests / 'kit' / path.name)
            (tests / 'kit/loadgen.py').write_text(DRIVER)
            (tests / 'public').mkdir()
            shutil.copyfile(TASK / 'tests/public/catalogue.json', tests / 'public/catalogue.json')
            suite = json.loads((TASK / 'tests/suite-graded.json').read_text())
            suite['qualification'] = {'workload': 'development-provisional', 'catalogue': 'development'}
            suite['setup_s'] = 5
            suite.pop('public_catalogue_sha256', None)
            # Synthetic graded traces (one isolated request per model): the generated world traces
            # are not shipped, and the stub driver never sends them.
            for case in suite['cases']:
                rows = [{'request_id': f"{case['name']}-{index}", 'model': model, 'class': 'interactive',
                         'weight': 1.0, 'max_tokens': 16, 'input_tokens': 32, 'slo_ttft_ms': 1000.0,
                         'slo_tbt_ms': 100.0, 'deadline_slack_s': 5.0, 'arrival_s': 0.0}
                        for index, model in enumerate(suite['model_ids'])]
                data = ''.join(json.dumps(row) + '\n' for row in rows).encode()
                (tests / f"trace-graded-{case['name']}.jsonl").write_bytes(data)
                case['trace_sha256'] = hashlib.sha256(data).hexdigest()
            (tests / 'suite-graded.json').write_text(json.dumps(suite))
            private = root / 'cluster-private'
            private.mkdir()
            # The kit writes the fleet run ID at the top level of the private manifest.
            fleet = {'run_id': 'stub', 'host': fleet_name, 'n_nodes': 4, 'gpus_per_node': 4,
                     'controller': {'host': fleet_name, 'port': 9000},
                     'nodes': [{'name': f'node{n}', 'host': fleet_name, 'port': 9100, 'n_gpus': 4} for n in range(4)],
                     'launch': {'inference_audit_dir': '/nonexistent'}}
            public_fleet = {'host': 'fleet', 'n_nodes': 4, 'gpus_per_node': 4, 'controller': {'host': 'fleet', 'port': 9000},
                            'nodes': [{'name': f'node{n}', 'host': 'fleet', 'port': 9100, 'n_gpus': 4} for n in range(4)]}
            (private / 'fleet.json').write_text(json.dumps(fleet))
            (private / 'fleet.public.json').write_text(json.dumps(public_fleet))
            (private / 'grader-token').write_text('grader-secret')
            submission, evaluation, capture = root / 'submission', root / 'evaluation', root / 'capture'
            submission.mkdir()
            (submission / 'run.sh').write_text(PLANE)
            (submission / 'notes.txt').write_text('plane')
            # The host's seal manifest records submitted links; each cell recreates them unfollowed.
            (submission / 'manifest.json').write_text(json.dumps({'symlinks': {'alias.sh': 'run.sh'}}))
            evaluation.mkdir()
            for directory, dirs, files in os.walk(root):
                Path(directory).chmod(0o755)
                for name in files:
                    (Path(directory) / name).chmod(0o644)
            evaluation.chmod(0o777)
            (submission / 'run.sh').chmod(0o755)
            try:
                subprocess.run(['docker', 'build', '-q', '-t', image, str(context)], check=True, capture_output=True)
                subprocess.run(['docker', 'build', '-q', '-t', observer_image, str(HERE / 'collector')], check=True,
                               capture_output=True)
                controller.docker('network', 'create', network)
                controller.docker('run', '-d', '--name', fleet_name, '--network', network, '--network-alias', 'fleet',
                                  '--entrypoint', 'python3', image, '-c', FLEET)
                controller.docker('create', '--name', target, '--ipc=private', '--network', network, '--cap-drop=ALL',
                    '--cap-add=SETUID', '--cap-add=SETGID', '--cap-add=CHOWN', '--cap-add=FOWNER', '--cap-add=DAC_OVERRIDE',
                    '--cap-add=KILL', '--cap-add=NET_ADMIN', '--cap-add=NET_RAW', '--security-opt=no-new-privileges:true',
                    '-e', 'BREAKWATER_STAGE=intermediate', '-e', 'PYTHONDONTWRITEBYTECODE=1',
                    *controller.mount(submission, '/submission'), *controller.mount(evaluation, '/evaluation', False),
                    *controller.mount(tests, '/private/tests'), *controller.mount(private, '/private/cluster'),
                    *controller.mount(security, '/opt/breakwater'),
                    *controller.mount(contract, '/trusted-task/breakwater.json'),
                    *controller.mount(security / 'drivers', '/opt/breakwater-drivers'),
                    '--entrypoint', 'sleep', image, 'infinity')
                controller.docker('start', target)
                identity = {'task': 'cluster-serverless', 'run_id': 'integration-' + tag, 'stage': 'intermediate'}
                controller.observer(watcher, target, capture, controller.image(observer_image), identity)
                result = subprocess.run(['docker', 'exec', target, 'python3', '-I', '-B', '/opt/breakwater/observed.py'],
                                        capture_output=True, timeout=900)
                self.assertEqual(result.returncode, 0, result.stderr.decode(errors='replace')[-4000:])
                controller.docker('stop', '--time', '10', target)
                drain = controller.no_lingering_processes(target, watcher, capture)
                self.assertEqual(drain['remaining_pids'], [])
                controller.stop_observer(watcher, capture)
                self.assertTrue(json.loads((capture / 'coverage.json').read_text())['complete'])
                calls = json.loads(controller.docker('exec', fleet_name, 'python3', '-c',
                                   "import urllib.request;print(urllib.request.urlopen('http://127.0.0.1:9000/calls').read().decode())"))
                sealed = root / 'sealed'
                controller.snapshot(target, '/evaluation', sealed, [])
                measurement = json.loads((sealed / 'raw/cluster/measurement.json').read_text())
                self.assertEqual([cell['label'] for cell in measurement['cells']], ['load-0.75-0', 'load-1.0-0', 'load-1.25-0'])
                self.assertTrue(measurement['development_only'])
                resets = [call for call in calls if call['path'] == '/admin/reset']
                self.assertGreaterEqual(len(resets), 4)
                self.assertTrue(all(call['token_ok'] for call in calls))
                for cell in measurement['cells']:
                    log = (sealed / f"raw/cluster/plane-{cell['label']}.log").read_text()
                    facts = json.loads(log.split('PLANE-FACTS ', 1)[1].splitlines()[0])
                    self.assertEqual(facts, {'uid': 1101, 'app_owner': 1101, 'private': 'denied', 'engine': 'connected',
                                             'manager': 'ConnectionRefusedError', 'secrets_in_env': [],
                                             'catalogue_owner': 0, 'home': '/home/worker', 'home_owner': 1101,
                                             'nproc': [512, 512], 'link': 'run.sh'}, cell['label'])
                    stats = json.loads((sealed / f"raw/cluster/stats-{cell['label']}.json").read_text())
                    self.assertEqual(stats, {'driver_uid': 0, 'key': True, 'token': True})
                # The fleet views the plane got were public; one rekey per node per cell with distinct keys.
                keys = [call['body']['key'] for call in calls if call['path'] == '/admin/rekey']
                self.assertEqual(len(keys), 12)
                self.assertEqual(len(set(keys)), 3)
                self.assertEqual(measurement['fleet_shape'], {'nodes': 4, 'gpus': 16})
                self.assertEqual(len(measurement['hardware_identity']), 4)
            finally:
                controller.docker('rm', '-f', watcher, target, fleet_name, check=False)
                controller.docker('network', 'rm', network, check=False)
                subprocess.run(['docker', 'run', '--rm', '--network=none', '--entrypoint', 'sh',
                                *controller.mount(root, '/cleanup', False), image, '-c',
                                'rm -rf /cleanup/evaluation /cleanup/capture /cleanup/sealed'], capture_output=True)
                subprocess.run(['docker', 'rmi', '-f', image, observer_image], capture_output=True)


if __name__ == '__main__':
    unittest.main()
