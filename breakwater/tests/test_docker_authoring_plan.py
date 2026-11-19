"""The Docker author image applies the launcher's authoring plan (live Docker, CPU; skipped without a daemon).

A minimal author image (the framework's author runtime, iptables, a task policy and a stand-in agent CLI profile) boots
with a plan naming one model-access service and an authoring environment, beside a stand-in author gateway and two
stand-in services on an internal network. As the worker (UID 1101), the agent's commands reach the declared service and
nothing else; the worker environment carries the plan's variables; the bootstrap returns the plan's digest.
"""
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import time
import unittest
import uuid

FRAMEWORK = Path(__file__).resolve().parents[1]
BASE = os.environ.get('BREAKWATER_CPU_TEST_IMAGE', 'python:3.12-slim-bookworm')
STUB = ('import http.server, json, sys\n'
        'body = json.dumps({"ok": True, "model": "test-model", "protocol": "messages", "name": sys.argv[1]}).encode()\n'
        'class H(http.server.BaseHTTPRequestHandler):\n'
        '    def do_GET(self):\n'
        '        self.send_response(200); self.send_header("Content-Length", str(len(body))); self.end_headers()\n'
        '        self.wfile.write(body)\n'
        '    def log_message(self, *_):\n'
        '        pass\n'
        'http.server.ThreadingHTTPServer(("0.0.0.0", int(sys.argv[2])), H).serve_forever()\n')
DOCKERFILE = f'''FROM {BASE}
RUN apt-get update && apt-get install -y --no-install-recommends iptables && rm -rf /var/lib/apt/lists/*
COPY author/ /opt/breakwater-runtime/
COPY policy.json /etc/breakwater-task.json
RUN mkdir -p /opt/author-cli /app && echo claude-code > /opt/author-cli/harness && echo 0 > /opt/author-cli/version
ENTRYPOINT ["python3", "-I", "/opt/breakwater-runtime/bootstrap.py"]
'''
PLAN = {'version': 1, 'services': [{'host': 'models', 'port': 9400}], 'endpoints': [], 'rejects': [], 'hosts': [],
        'reply_ports': None, 'env': {'TASK_LLM_BASE_URL': 'http://models:9400/v1', 'TASK_CHAT_MODEL': 'chat'}}


def available():
    try:
        return subprocess.run(['docker', 'info'], capture_output=True, timeout=5).returncode == 0
    except (OSError, subprocess.TimeoutExpired):
        return False


def docker(*args, check=True):
    result = subprocess.run(['docker', *args], capture_output=True, text=True, timeout=600)
    if check and result.returncode:
        raise AssertionError(f'docker {args[0]} failed: {result.stderr[-2000:]}')
    return result


@unittest.skipUnless(available(), 'Docker daemon unavailable; the author image bootstrap is NOT verified')
class AuthoringPlanTests(unittest.TestCase):
    def setUp(self):
        tag = 'bw-plan-' + uuid.uuid4().hex[:10]
        self.image, self.network, self.containers = tag + ':test', tag, []
        context = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, context, ignore_errors=True)
        shutil.copytree(FRAMEWORK / 'author', context / 'author', ignore=shutil.ignore_patterns('__pycache__'))
        (context / 'policy.json').write_text(json.dumps({'name': 'plan-test', 'workdir': '/app', 'write_roots': ['/app'],
                                                         'local_network': False}))
        (context / 'Dockerfile').write_text(DOCKERFILE)
        docker('build', '-q', '-t', self.image, str(context))
        self.addCleanup(docker, 'rmi', '-f', self.image, check=False)
        docker('network', 'create', '--internal', self.network)
        self.addCleanup(docker, 'network', 'rm', self.network, check=False)

    def tearDown(self):
        for name in reversed(self.containers):
            docker('rm', '-f', name, check=False)

    def start(self, name, alias, port):
        container = f'{self.network}-{name}'
        docker('run', '-d', '--name', container, '--network', self.network, '--network-alias', alias, BASE,
               'python3', '-c', STUB, name, str(port))
        self.containers.append(container)
        return container

    def address(self, container):
        value = json.loads(docker('inspect', container).stdout)[0]
        return value['NetworkSettings']['Networks'][self.network]['IPAddress']

    def test_the_bootstrap_applies_the_plan_and_returns_its_digest(self):
        self.start('author', 'author-gateway', 8001)
        self.start('models', 'models', 9400)
        other = self.start('other', 'other', 9400)
        time.sleep(1)
        plan = json.dumps(PLAN, sort_keys=True, separators=(',', ':'))
        agent = f'{self.network}-agent'
        docker('run', '-d', '--name', agent, '--network', self.network, '--cap-add=NET_ADMIN',
               '-e', 'BREAKWATER_AUTHOR_MODEL=test-model', '-e', 'BREAKWATER_AUTHORING_PLAN=' + plan, self.image)
        self.containers.append(agent)
        for _ in range(120):
            if docker('exec', agent, 'test', '-f', '/run/breakwater/ready', check=False).returncode == 0:
                break
            if docker('inspect', '-f', '{{.State.Running}}', agent).stdout.strip() != 'true':
                self.fail('the bootstrap exited: ' + docker('logs', agent, check=False).stderr[-2000:])
            time.sleep(0.5)
        else:
            self.fail('the bootstrap never became ready')
        runtime = json.loads(docker('exec', agent, 'cat', '/run/breakwater/runtime.json').stdout)
        self.assertEqual(runtime['plan_sha256'], hashlib.sha256(plan.encode()).hexdigest())
        env = dict(item.split('=', 1) for item in
                   docker('exec', agent, 'cat', '/run/breakwater/worker.env').stdout.split('\0') if item)
        self.assertEqual((env['TASK_LLM_BASE_URL'], env['TASK_CHAT_MODEL'], env['HOME']),
                         ('http://models:9400/v1', 'chat', '/home/worker'))

        def fetch(url):
            return docker('exec', '--user', '1101:1200', agent, 'python3', '-c',
                          f'import urllib.request; print(urllib.request.urlopen({url!r}, timeout=5).read().decode())',
                          check=False)
        reached = fetch('http://models:9400/health')
        self.assertEqual((reached.returncode, json.loads(reached.stdout)['name']), (0, 'models'))
        self.assertNotEqual(fetch(f'http://{self.address(other)}:9400/health').returncode, 0)  # undeclared service
        self.assertNotEqual(fetch('http://author-gateway:8001/health').returncode, 0)  # the agent's own gateway

    def test_endpoints_hosts_and_reply_ports_from_the_plan(self):
        self.start('author', 'author-gateway', 8001)
        store = f'{self.network}-store'
        docker('run', '-d', '--name', store, '--network', self.network, '-e', 'STUB=' + STUB, BASE, 'sh', '-c',
               'python3 -c "$STUB" store 9500 & python3 -c "$STUB" store 9501 & wait')
        self.containers.append(store)
        time.sleep(1)
        address = self.address(store)
        plan = json.dumps({**PLAN, 'services': [], 'endpoints': [{'name': 'store', 'host': address, 'ports': ['9500']}],
                           'hosts': [['launcher.internal', address]], 'reply_ports': '30000:30001'})
        agent = f'{self.network}-agent'
        docker('run', '-d', '--name', agent, '--network', self.network, '--cap-add=NET_ADMIN',
               '-e', 'BREAKWATER_AUTHOR_MODEL=test-model', '-e', 'BREAKWATER_AUTHORING_PLAN=' + plan, self.image)
        self.containers.append(agent)
        for _ in range(120):
            if docker('exec', agent, 'test', '-f', '/run/breakwater/ready', check=False).returncode == 0:
                break
            time.sleep(0.5)
        else:
            self.fail('the bootstrap never became ready: ' + docker('logs', agent, check=False).stderr[-2000:])

        def fetch(url):
            return docker('exec', '--user', '1101:1200', agent, 'python3', '-c',
                          f'import urllib.request; print(urllib.request.urlopen({url!r}, timeout=5).read().decode())',
                          check=False)
        self.assertEqual(json.loads(fetch('http://store:9500/health').stdout)['name'], 'store')
        self.assertEqual(json.loads(fetch('http://launcher.internal:9500/health').stdout)['name'], 'store')
        self.assertNotEqual(fetch('http://store:9501/health').returncode, 0)  # an undeclared port of the same host
        rules = docker('exec', agent, 'iptables', '-S', 'OUTPUT').stdout
        self.assertIn('--sport 30000:30001 -m conntrack --ctstate ESTABLISHED -j ACCEPT', rules)

    def test_a_malformed_plan_stops_the_bootstrap(self):
        self.start('author', 'author-gateway', 8001)
        agent = f'{self.network}-agent'
        docker('run', '-d', '--name', agent, '--network', self.network, '--cap-add=NET_ADMIN',
               '-e', 'BREAKWATER_AUTHOR_MODEL=test-model',
               '-e', 'BREAKWATER_AUTHORING_PLAN=' + json.dumps({**PLAN, 'env': {'HOME': '/root'}}), self.image)
        self.containers.append(agent)
        self.assertNotEqual(docker('wait', agent).stdout.strip(), '0')
        self.assertIn('Invalid environment in the authoring plan', docker('logs', agent, check=False).stderr)


if __name__ == '__main__':
    unittest.main()
