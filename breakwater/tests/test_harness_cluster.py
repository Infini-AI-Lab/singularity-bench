"""Superset-core regressions (shared by all six task folders). No API keys, GPUs or Docker.

Covers the superset controller contract (GPU counts 0/1/2/4, credentials, each task's GB300
contract, the byte-identical shared core in all six folders), the hash-chained
development/driver gateway, the author gateway's journal under provider failures, the
out-of-process validity battery (parity with the task's
own smoke), cluster/cell data-only scoring, fleet journal checks, the indexed review view
and the grid-cell overlay checks.
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

SECURITY = Path(__file__).resolve().parents[1]
REPO = SECURITY.parent / 'tasks'
sys.path[:0] = [str(SECURITY), str(SECURITY / 'downstream'), str(SECURITY / 'worker'), str(SECURITY / 'tests')]
from contract import ContractError, RulesViolation, inspect_boundary
import run as controller
from test_security import inspected

HARNESS = REPO / 'coding-agent-harness'
CLUSTER = REPO / 'cluster-serverless'


def free_port():
    with socket.socket() as probe:
        probe.bind(('127.0.0.1', 0))
        return probe.getsockname()[1]


def cluster_tests(root, qualification=None):
    """A copy of the cluster grader inputs with synthetic graded traces (one request per model).

    The graded traces are generated world artifacts that the task does not ship; the suite
    copy binds the synthetic traces by hash exactly as the real suite binds the real ones.
    """
    tests = Path(root) / 'tests'
    (tests / 'kit').mkdir(parents=True)
    for source in (CLUSTER / 'tests/kit').glob('*.py'):
        shutil.copyfile(source, tests / 'kit' / source.name)
    suite = json.loads((CLUSTER / 'tests/suite-graded.json').read_text())
    if qualification is not None:
        suite['qualification'] = qualification
    for case in suite['cases']:
        rows = [{'request_id': f"{case['name']}-{index}", 'model': model, 'class': 'interactive', 'weight': 1.0,
                 'max_tokens': 16, 'input_tokens': 32, 'slo_ttft_ms': 1000.0, 'slo_tbt_ms': 100.0, 'arrival_s': 0.0}
                for index, model in enumerate(suite['model_ids'])]
        data = ''.join(json.dumps(row) + '\n' for row in rows).encode()
        (tests / f"trace-graded-{case['name']}.jsonl").write_bytes(data)
        case['trace_sha256'] = hashlib.sha256(data).hexdigest()
    (tests / 'suite-graded.json').write_text(json.dumps(suite))
    return tests


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

    def test_fleet_and_gateway_secrets_never_enter_execution_containers(self):
        with tempfile.TemporaryDirectory() as temporary:
            for name in ('CLUSTER_GRADER_TOKEN', 'DRIVER_UPSTREAM_API_KEY', 'UPSTREAM_API_KEY', 'GATEWAY_CTL_TOKEN'):
                value = inspected('agent', Path(temporary), 'a')
                value['Config']['Env'] = [name + '=[REDACTED]']
                with self.subTest(name=name), self.assertRaises(ContractError):
                    inspect_boundary(value, 'agent', Path(temporary), gpu_count=1)
            value = inspected('agent', Path(temporary), 'a')
            value['Config']['Env'] = ['CLUSTER_RECEIPT_KEY=[REDACTED]']  # public development key
            inspect_boundary(value, 'agent', Path(temporary), gpu_count=0)

    def test_each_task_contract_parses_with_its_declared_values(self):
        """Each task's GB300 contract: GPUs, seeding, networks, deadline policy, rewards and container limits."""
        disabled = {'packages': [], 'hooks': [], 'environments': False}
        expected = {
            'open-ended-nano-gpt': dict(kind='nano', gpus=4, workspace_source='task', intermediate_network='none',
                                      intermediate_cap_add=[], submission_mount=None, rule_violation_reward=-1.0,
                                      shadow_policy=None, snapshot_excludes=[], snapshot_worker_home=False,
                                      intermediate_env=[], intermediate_add_hosts=[],
                                      options=['--ulimit', 'memlock=-1:-1', '--shm-size=16g']),
            'midtrain-for-rl': dict(kind='midtrain', gpus=4, workspace_source='task', intermediate_network='bridge',
                                  intermediate_cap_add=['NET_ADMIN', 'NET_RAW'], submission_mount=None,
                                  rule_violation_reward=0.0, shadow_policy=disabled, snapshot_excludes=['scratch'],
                                  snapshot_worker_home=False,
                                  intermediate_env=['MIDTRAIN_CODE_SANDBOX_URL', 'MIDTRAIN_CODE_SANDBOX_TOKEN'],
                                  intermediate_add_hosts=['host.docker.internal:host-gateway'],
                                  options=['--ulimit', 'memlock=-1:-1', '--ulimit', 'stack=67108864:67108864',
                                           '--shm-size=64g']),
            'rl-swe-agent': dict(kind='rl', gpus=4, workspace_source='task', intermediate_network='none',
                                    intermediate_cap_add=[], submission_mount=None, rule_violation_reward=0.0,
                                    shadow_policy=None, snapshot_excludes=[], snapshot_worker_home=False,
                                    intermediate_env=[], intermediate_add_hosts=[], options=['--shm-size=64g']),
            'serve-glm53-flash': dict(
                kind='serving', gpus=4, workspace_source='image', intermediate_network='none', intermediate_cap_add=[],
                submission_mount='/submission', rule_violation_reward=0.0, shadow_policy=None, snapshot_excludes=[],
                snapshot_worker_home=False, intermediate_env=[], intermediate_add_hosts=[],
                options=['--cpus', '16', '--memory', '524288m', '--memory-swap', '524288m', '--shm-size=256g']),
            'coding-agent-harness': dict(kind='harness', gpus=1, workspace_source='image', intermediate_network='none',
                                intermediate_cap_add=[], submission_mount=None, rule_violation_reward=0.0,
                                snapshot_excludes=[], snapshot_worker_home=False, intermediate_env=[],
                                intermediate_add_hosts=[],
                                options=['--cpus', '4', '--memory', '16384m', '--memory-swap', '16384m', '--shm-size=2g']),
            'cluster-serverless': dict(kind='cluster', gpus=0, workspace_source='image', intermediate_network='bridge',
                                       intermediate_cap_add=['NET_ADMIN', 'NET_RAW'], submission_mount='/submission',
                                       rule_violation_reward=0.0, shadow_policy=disabled, snapshot_excludes=[],
                                       snapshot_worker_home=True, intermediate_env=[], intermediate_add_hosts=[],
                                       options=['--cpus', '16', '--cpuset-cpus', '0-15', '--memory', '40960m',
                                                '--memory-swap', '40960m', '--ulimit', 'nofile=65536:65536',
                                                '--shm-size=1g']),
        }
        for name, contract in expected.items():
            config = controller.load_config(REPO / name / 'breakwater.json')
            with self.subTest(task=name):
                self.assertEqual(config['task'], name)
                # Every task grades what exists when its authoring budget ends.
                self.assertEqual(config['agent_deadline'], 'evaluate')
                self.assertEqual(config['review_profile'], 'indexed')
                self.assertEqual(config['gpu_name_patterns'], [] if contract['gpus'] == 0 else ['GB300'])
                for key, value in contract.items():
                    if key != 'options':
                        self.assertEqual(config[key], value, key)
                self.assertEqual(controller.resource_options(config), contract['options'])
                if config['workspace_source'] == 'task':
                    # Release READMEs never enter a task-seeded agent workspace.
                    self.assertIn('/README.md', config['workspace_excludes'])
        cluster_agent = controller.load_config(CLUSTER / 'breakwater.json')
        self.assertEqual(controller.resource_options(cluster_agent, controller.agent_resources(cluster_agent)),
                         ['--cpus', '8', '--cpuset-cpus', '0-15', '--memory', '32768m', '--memory-swap', '32768m',
                          '--pids-limit', '512', '--ulimit', 'nofile=65536:65536', '--shm-size=1g'])
        harness = controller.load_config(HARNESS / 'breakwater.json')
        self.assertEqual((harness['gpus'], harness['kind'], harness['workspace_source']), (1, 'harness', 'image'))
        self.assertTrue(harness['development_gateway'])
        cluster = controller.load_config(CLUSTER / 'breakwater.json')
        self.assertEqual((cluster['gpus'], cluster['kind'], cluster['submission_mount']), (0, 'cluster', '/submission'))
        bad = dict(json.loads((HARNESS / 'breakwater.json').read_text()), intermediate_cap_add=['SYS_ADMIN'])
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / 'breakwater.json'
            path.write_text(json.dumps(bad))
            with self.assertRaises(ContractError):
                controller.load_config(path)

    def test_protected_harness_paths_detect_edit_add_delete(self):
        from contract import inventory
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / 'spec').mkdir()
            (root / 'harness').mkdir()
            (root / 'spec/HARNESS_SPEC.md').write_text('spec')
            (root / 'selfcheck.py').write_text('check')
            (root / 'harness/agent.py').write_text('def run(*a): pass')
            baseline = inventory(root)
            paths = ['spec', 'selfcheck.py']
            (root / 'harness/agent.py').write_text('def run(*a): return None')  # allowed
            self.assertEqual(controller.protected_paths_check(baseline, root, paths)['status'], 'PASS')
            for mutation in ('edit', 'add', 'delete'):
                trial = root.parent / (root.name + mutation)
                shutil.copytree(root, trial)
                if mutation == 'edit':
                    (trial / 'spec/HARNESS_SPEC.md').write_text('weakened spec')
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
            harness = guard.shadow_policy(json.loads((HARNESS / 'breakwater.json').read_text())['shadow_policy'])
            cluster = guard.shadow_policy(json.loads((CLUSTER / 'breakwater.json').read_text())['shadow_policy'])
            self.assertIn('pytest/__init__.py', guard.shadows(root, harness))
            # Vendored plane dependencies are permitted: trusted cluster code never imports /app.
            self.assertEqual(guard.shadows(root, cluster), {})


class FakeUpstream(http.server.BaseHTTPRequestHandler):
    status = 200

    def log_message(self, *_):
        pass

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
        if FakeUpstream.status != 200:
            self.send_response(FakeUpstream.status)
            self.end_headers()
            return
        reply = {'choices': [{'message': {'role': 'assistant', 'content': 'ok ' + body['model']}}],
                 'usage': {'prompt_tokens': 10, 'completion_tokens': 5, 'total_tokens': 15}}
        raw = json.dumps(reply).encode()
        self.send_response(200)
        self.send_header('Content-Length', str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)


class GatewayTests(unittest.TestCase):
    def setUp(self):
        self.upstream = http.server.ThreadingHTTPServer(('127.0.0.1', 0), FakeUpstream)
        threading.Thread(target=self.upstream.serve_forever, daemon=True).start()
        self.addCleanup(self.upstream.shutdown)
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        FakeUpstream.status = 200

    def start(self, role, **extra):
        port = free_port()
        folder = Path(self.temp.name) / role
        env = {**os.environ, 'GATEWAY_ROLE': role, 'UPSTREAM_API_KEY': 'sk-test-secret', 'PORT': str(port),
               'UPSTREAM_BASE_URL': f'http://127.0.0.1:{self.upstream.server_address[1]}/v1',
               'GATEWAY_ALLOW_HTTP_UPSTREAM': '1', 'GATEWAY_LOG': str(folder / 'requests.jsonl'),
               'GATEWAY_STATE': str(folder / 'budget.json'), **extra}
        process = subprocess.Popen([sys.executable, '-I', str(SECURITY / 'dev_gateway.py')], env=env,
                                   stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
        self.addCleanup(lambda: process.poll() is None and process.kill())
        for _ in range(100):
            try:
                urllib.request.urlopen(f'http://127.0.0.1:{port}/health', timeout=1)
                break
            except OSError:
                time.sleep(.05)
        return process, port, folder

    def post(self, port, path, payload):
        request = urllib.request.Request(f'http://127.0.0.1:{port}{path}', data=json.dumps(payload).encode(),
                                         headers={'Content-Type': 'application/json'})
        try:
            with urllib.request.urlopen(request, timeout=10) as response:
                return response.status, json.loads(response.read())
        except urllib.error.HTTPError as error:
            return error.code, json.loads(error.read() or b'{}')

    def stop(self, process):
        process.send_signal(signal.SIGTERM)
        self.assertEqual(process.wait(timeout=10), 0)

    def test_development_contract_budget_chain_and_tamper_detection(self):
        from reconcile import check_service_chain
        process, port, folder = self.start('development', MAX_CHAT_TOKENS='20000', MAX_EMBED_TOKENS='100000',
                                           CHAT_MODELS=json.dumps({'chat': 'm/glm', 'chat-b': 'm/kimi'}))
        good = {'model': 'chat', 'max_tokens': 32, 'messages': [{'role': 'user', 'content': 'hi'}]}
        status, body = self.post(port, '/v1/chat/completions', good)
        self.assertEqual((status, body['choices'][0]['message']['content']), (200, 'ok m/glm'))
        self.assertEqual(self.post(port, '/v1/chat/completions', {**good, 'stream': True})[0], 400)
        self.assertEqual(self.post(port, '/v1/chat/completions', {**good, 'model': 'minimax'})[0], 400)
        self.assertEqual(self.post(port, '/v1/chat/completions', {**good, 'max_tokens': 10 ** 6})[0], 429)
        self.assertEqual(self.post(port, '/v1/responses', good)[0], 404)
        FakeUpstream.status = 500
        self.assertEqual(self.post(port, '/v1/chat/completions', good)[0], 500)
        self.stop(process)
        report = check_service_chain(folder / 'requests.jsonl', folder, 'development-gateway')
        self.assertEqual((report['calls'], report['statuses']), (2, {'200': 1, '500': 1}))
        journal = (folder / 'requests.jsonl').read_text()
        self.assertNotIn('sk-test-secret', journal + ''.join(p.read_text() for p in (folder / 'blobs').iterdir()))
        state = json.loads((folder / 'budget.json').read_text())['state']
        self.assertEqual(state['chat_tokens'], 15 + json.loads(journal.splitlines()[-2])['tokens'])
        for mutation in ('line', 'blob', 'truncate'):
            backup = folder.parent / ('backup-' + mutation)
            shutil.copytree(folder, backup)
            if mutation == 'line':
                lines = journal.splitlines(keepends=True)
                (backup / 'requests.jsonl').write_text(''.join(lines[:2] + lines[3:]))
            if mutation == 'blob':
                blob = next((backup / 'blobs').iterdir())
                blob.write_bytes(blob.read_bytes() + b' ')
            if mutation == 'truncate':
                (backup / 'requests.jsonl').write_text(''.join(journal.splitlines(keepends=True)[:-1]))
            with self.subTest(mutation=mutation), self.assertRaises(ContractError):
                check_service_chain(backup / 'requests.jsonl', backup, 'development-gateway')

    def test_driver_lanes_resolve_every_alias_to_the_cell_driver(self):
        from reconcile import check_service_chain
        process, port, folder = self.start('driver', DRIVER_MODEL='m/minimax', EMBED_MODEL='m/embed',
                                           GATEWAY_SOURCE='driver-gateway')
        for path, alias in (('/driver/v1/chat/completions', 'driver'), ('/v1/chat/completions', 'chat-b'),
                            ('/v1/chat/completions', 'chat')):
            status, body = self.post(port, path, {'model': alias, 'messages': [{'role': 'user', 'content': 'x'}],
                                                  'temperature': 0.3})
            self.assertEqual((status, body['choices'][0]['message']['content']), (200, 'ok m/minimax'))
        self.assertEqual(self.post(port, '/v1/chat/completions', {'model': 'gpt-x', 'messages': []})[0], 400)
        self.assertEqual(self.post(port, '/driver/v1/chat/completions', {'model': 'driver', 'stream': True,
                                                                        'messages': []})[0], 400)
        self.stop(process)
        report = check_service_chain(folder / 'requests.jsonl', folder, 'driver-gateway')
        self.assertEqual(report['calls'], 3)
        lanes = [json.loads(line).get('lane') for line in (folder / 'requests.jsonl').read_text().splitlines()
                 if json.loads(line).get('event') == 'model_request']
        self.assertEqual(lanes, ['driver', 'utility', 'utility'])


class FakeMessagesUpstream(http.server.BaseHTTPRequestHandler):
    """A Messages provider whose behaviour is chosen by the request's last user message."""
    slow_received = threading.Event()
    release = threading.Event()

    def log_message(self, *_):
        pass

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
        mode = body['messages'][-1]['content']
        if mode == 'drop':
            return  # HTTP/1.0 handler: the connection closes without a status line
        if mode == 'slow':
            FakeMessagesUpstream.slow_received.set()
            FakeMessagesUpstream.release.wait(30)
            return
        if mode in ('busy', 'rate'):
            raw = b'{"type": "error", "error": {"type": "overloaded_error", "message": "UPSTREAM-ERROR-BODY"}}'
            self.send_response(503 if mode == 'busy' else 429)
        else:
            raw = json.dumps({'id': 'msg_1', 'type': 'message', 'role': 'assistant', 'model': body['model'],
                              'content': [{'type': 'text', 'text': 'reply for ' + mode}], 'stop_reason': 'end_turn',
                              'usage': {'input_tokens': 3, 'output_tokens': 2}}).encode()
            self.send_response(200)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Content-Length', str(len(raw) + (100 if mode == 'cut' else 0)))
        self.end_headers()
        self.wfile.write(raw[:len(raw) // 2] if mode == 'cut' else raw)


class AuthorGatewayTests(unittest.TestCase):
    """The real author gateway's journal under provider failures, as the verifier checks it."""

    def setUp(self):
        FakeMessagesUpstream.slow_received, FakeMessagesUpstream.release = threading.Event(), threading.Event()
        self.upstream = http.server.ThreadingHTTPServer(('127.0.0.1', 0), FakeMessagesUpstream)
        threading.Thread(target=self.upstream.serve_forever, daemon=True).start()
        self.addCleanup(self.upstream.server_close)
        self.addCleanup(self.upstream.shutdown)
        self.addCleanup(FakeMessagesUpstream.release.set)
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.folder = Path(temp.name) / 'author'

    def test_provider_failures_are_complete_evidence_and_never_reflected(self):
        from reconcile import check_chain
        port = free_port()
        env = {**os.environ, 'AUTHOR_MODEL': 'test/author', 'AUTHOR_UPSTREAM_API_KEY': 'sk-author-secret',
               'AUTHOR_UPSTREAM_BASE_URL': f'http://127.0.0.1:{self.upstream.server_address[1]}',
               'AUTHOR_GATEWAY_LOG': str(self.folder / 'requests.jsonl'), 'PORT': str(port)}
        process = subprocess.Popen([sys.executable, '-I', str(SECURITY / 'author_gateway.py')], env=env,
                                   stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
        self.addCleanup(process.stderr.close)
        self.addCleanup(lambda: process.poll() is None and process.kill())
        for _ in range(100):
            try:
                urllib.request.urlopen(f'http://127.0.0.1:{port}/health', timeout=1).close()
                break
            except OSError:
                time.sleep(.05)

        def ask(text):
            request = urllib.request.Request(f'http://127.0.0.1:{port}/v1/messages', headers={
                'Content-Type': 'application/json'}, data=json.dumps({'model': 'test/author', 'max_tokens': 8,
                'messages': [{'role': 'user', 'content': text}]}).encode())
            try:
                with urllib.request.urlopen(request, timeout=10) as response:
                    return response.status, response.read()
            except urllib.error.HTTPError as error:
                return error.code, error.read()
            except (OSError, http.client.HTTPException) as error:
                return None, repr(error).encode()

        replies = {text: ask(text) for text in ('ok', 'busy', 'rate', 'drop', 'cut')}
        self.assertEqual({text: reply[0] for text, reply in replies.items()},
                         {'ok': 200, 'busy': 503, 'rate': 429, 'drop': 502, 'cut': 200})
        self.assertNotIn(b'UPSTREAM-ERROR-BODY', replies['busy'][1] + replies['rate'][1])
        slow = threading.Thread(target=ask, args=('slow',), daemon=True)
        slow.start()
        self.assertTrue(FakeMessagesUpstream.slow_received.wait(10))
        process.send_signal(signal.SIGTERM)
        self.assertEqual(process.wait(timeout=10), 0)
        slow.join(10)
        report = check_chain(self.folder / 'requests.jsonl', gateway=True, blob_root=self.folder)
        self.assertEqual((report['calls'], report['succeeded']), (6, 1))
        self.assertEqual(report['outcomes'], {'200 completed': 1, '200 upstream_error': 1, '429 upstream_error': 1,
                                              '502 upstream_error': 1, '503 gateway_stopped': 1,
                                              '503 upstream_error': 1})
        blobs = b''.join(path.read_bytes() for path in (self.folder / 'blobs').iterdir())
        self.assertIn(b'UPSTREAM-ERROR-BODY', blobs)  # journaled, never delivered
        self.assertIn(replies['cut'][1], blobs)  # every relayed byte of the truncated stream
        self.assertNotIn(b'sk-author-secret', blobs + (self.folder / 'requests.jsonl').read_bytes())


class BatteryTests(unittest.TestCase):
    def run_battery(self, source, extra=None, label=None):
        import harness_battery as battery
        import score_workload
        root = Path(self.temp.name)
        target = root / (label or Path(source).stem) / 'harness'
        target.mkdir(parents=True)
        shutil.copyfile(source, target / 'agent.py')
        for name, payload in (extra or {}).items():
            (target / name).write_bytes(payload)
        def child(module, workspace, task, seconds):
            return [sys.executable, '-I', '-B', str(SECURITY / 'harness_child.py'), '--module', str(module),
                    '--workspace', str(workspace), '--task-file', str(task), '--seconds', str(seconds)]
        raw = root / (label or Path(source).stem) / 'raw'
        # kill_workers signals every UID-1101 process; that is only meaningful inside an
        # intermediate container, never on a (shared) test host.
        with patch.object(battery, 'WORK_ROOT', root / 'work'), patch.object(battery, 'child_command', child), \
             patch.object(battery.os, 'chown'), patch.object(battery, 'kill_workers'):
            rows, summary = battery.run_battery(target / 'agent.py', HARNESS / 'tests/smoke', raw / 'harness')
        value = score_workload.harness_score(raw, smoke=HARNESS / 'tests/smoke', module=target / 'agent.py')
        return rows, summary, value, raw

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)

    def test_harness_carrying_an_executable_an_archive_or_bulk_is_not_run(self):
        import harness_battery as battery
        source = SECURITY / 'tests/fixtures/harness/reference_agent.py'
        cases = {'elf': {'helper': b'\x7fELF' + bytes(64)}, 'zip': {'vendor.zip': b'PK\x03\x04' + bytes(64)},
                 'tar': {'bundle': bytes(257) + b'ustar' + bytes(64)},
                 'bulk': {'notes.txt': b'x' * (battery.MAX_SUBMISSION_BYTES + 1)}}
        for label, extra in cases.items():
            with self.subTest(payload=label):
                rows, summary, value, raw = self.run_battery(source, extra=extra, label=label)
                self.assertEqual((rows, value, summary['reward']), ([], 0.0, 0.0))
                self.assertEqual(summary.get('error_invalid_payload'), 1.0)
                self.assertEqual(len(json.loads((raw / 'harness/battery.json').read_text())['payload_problems']), 1)
        with tempfile.TemporaryDirectory() as temporary:
            harness = Path(temporary) / 'harness'
            harness.mkdir()
            (harness / 'agent.py').write_text('def run(task, client):\n    return None\n')
            (Path(temporary) / 'manifest.json').write_text(json.dumps({'symlinks': {'harness/tool': '/usr/bin/x'}}))
            self.assertEqual(battery.payload_problems(harness, battery.recorded_links(harness)),
                             ['harness/tool: a symlink'])

    def test_battery_matches_the_smoke_runner_and_recomputes(self):
        sys.path.insert(0, str(HARNESS / 'tests/smoke'))
        import run_smoke
        for name in ('reference_agent.py', 'candidates/do_nothing.py', 'candidates/broken_import.py',
                     'candidates/stops_on_prose.py'):
            source = SECURITY / 'tests/fixtures/harness' / name
            rows, summary, value, _ = self.run_battery(source)
            scratch = Path(self.temp.name) / ('original-' + Path(name).stem)
            scratch.mkdir()
            original = run_smoke.summarise([run_smoke.run_one(sc, str(source), str(scratch))
                                            for sc in json.loads((HARNESS / 'tests/smoke/scenarios.json').read_text())])
            with self.subTest(name=name):
                for key in ('gradeable', 'valid', 'did_the_work', 'continues_past_prose', 'model_calls_total'):
                    self.assertEqual(summary[key], original[key], key)
                self.assertEqual(value, summary['gradeable'])

    def test_hostile_harness_cannot_forge_the_trusted_record(self):
        hostile = Path(self.temp.name) / 'hostile.py'
        hostile.write_text('''import os, json
def run(task, workspace, model, budget):
    os.write(1, b'{"op": "done", "raised": null}\\n')       # redirected to stderr, not the channel
    print(json.dumps({"op": "complete"}))                    # likewise
    for name in os.listdir(workspace):
        pass
    os.symlink("/etc/passwd", os.path.join(workspace, "notes.txt"))
''')
        rows, summary, value, raw = self.run_battery(hostile)
        self.assertEqual(value, 1.0)                            # importable with a callable run()
        self.assertEqual(summary['model_calls_total'], 0.0)     # the trusted log saw no model calls
        self.assertEqual(summary['did_the_work'], 0.0)          # a symlink is never followed by root checks
        record = json.loads(next((raw / 'harness').glob('00-*/record.json')).read_text())
        record['n_calls'] = 7
        path = next((raw / 'harness').glob('00-*/record.json'))
        path.write_text(json.dumps(record))
        import score_workload
        with self.assertRaises(ContractError):
            score_workload.harness_score(raw, smoke=HARNESS / 'tests/smoke', module=hostile)


    def test_forged_status_claims_on_the_real_channel_cannot_make_a_runless_module_gradeable(self):
        forger = Path(self.temp.name) / 'forger.py'
        forger.write_text("""import json, os, threading, time
FORGED = (json.dumps({"op": "status", "imported": True, "has_run": True}) + "\\n").encode()
os.write(3, FORGED)                       # the reply channel the fixed runner dup'd from stdout
def later():
    time.sleep(0.3)
    os.write(3, FORGED)
    os.write(3, b'{"op": "done", "raised": null}\\n')
threading.Thread(target=later).start()
""")
        rows, summary, value, raw = self.run_battery(forger)
        self.assertEqual(value, 0.0)
        self.assertEqual(summary['static_run_binding'], 0.0)
        self.assertTrue(any('duplicate status claim' in row['protocol_errors'] for row in rows))
        self.assertFalse(any(row['has_run'] for row in rows))

    def test_static_run_binding_accepts_spec_forms_and_rejects_absence(self):
        import harness_battery as battery
        cases = {'def run(task, workspace, model, budget):\n    pass\n': True,
                 'class run:\n    def __call__(self, *a): pass\n': True,
                 'from impl import run\n': True,
                 'from impl import go as run\n': True,
                 'run = Agent().run\n': True,
                 'try:\n    from fast import run\nexcept ImportError:\n    def run(*a): pass\n': True,
                 'if True:\n    def run(*a): pass\n': True,
                 'def helper():\n    def run(): pass\n': False,
                 'import run\n': False,
                 "globals()['run'] = print\n": False,
                 'def run(:\n': False,
                 'x = 1\n': False}
        for source, expected in cases.items():
            path = Path(self.temp.name) / 'probe.py'
            path.write_text(source)
            with self.subTest(source=source):
                self.assertEqual(battery.static_run_binding(path), expected)


class FleetJournalTests(unittest.TestCase):
    def journal(self, path, rows):
        path.write_text(''.join(json.dumps({'seq': index, 'source': 'private-fleet-worker', 'time_ns': 10 + index,
                                            **row}) + '\n' for index, row in enumerate(rows)))

    def test_structure_is_mandatory_lifecycle_is_reported(self):
        import cluster_fleet
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary)
            name = folder / ('worker-' + 'a' * 32 + '.jsonl')
            complete = [{'event': 'worker_started', 'run_id': 'R', 'model': 'm'}, {'event': 'request', 'audit_id': 'x'},
                        {'event': 'response', 'audit_id': 'x', 'frame': {'choices': [{'text': 'hi'}]}},
                        {'event': 'request_finished', 'audit_id': 'x', 'outcome': 'completed'}, {'event': 'worker_stopped'}]
            self.journal(name, complete)
            report = cluster_fleet.check_journals(folder, 'R')
            self.assertEqual((report['requests'], report['incomplete']), (1, []))
            self.journal(name, complete[:2])
            self.assertEqual(cluster_fleet.check_journals(folder, 'R')['incomplete'][0]['unfinished_requests'], ['x'])
            for broken in (complete[:1] + complete[2:], [dict(complete[0], run_id='other')] + complete[1:]):
                self.journal(name, broken)
                with self.assertRaises(ContractError):
                    cluster_fleet.check_journals(folder, 'R')
            text = name.read_text()
            name.write_text(text.replace('"seq": 1', '"seq": 3'))
            with self.assertRaises(ContractError):
                cluster_fleet.check_journals(folder, 'R')

    def test_collect_takes_only_new_journals_and_rejects_stale_writers(self):
        import cluster_fleet
        with tempfile.TemporaryDirectory() as temporary:
            audit, destination = Path(temporary) / 'audit', Path(temporary) / 'evidence'
            audit.mkdir()
            old, new = audit / ('worker-' + '1' * 32 + '.jsonl'), audit / ('worker-' + '2' * 32 + '.jsonl')
            old.write_text('{"seq": 0}\n')
            before = cluster_fleet.existing_journals(audit)
            new.write_text('{"seq": 0}\n')
            self.assertEqual(cluster_fleet.collect(audit, destination, before), [new.name])
            old.write_text('{"seq": 0}\n{"seq": 1}\n')
            with self.assertRaises(ContractError):
                cluster_fleet.collect(audit, Path(temporary) / 'again', before)


class ClusterScoreTests(unittest.TestCase):
    def test_invalid_submission_scores_zero_and_tampered_utility_is_rejected(self):
        import score_workload
        with tempfile.TemporaryDirectory() as temporary:
            tests = cluster_tests(temporary)
            suite = json.loads((tests / 'suite-graded.json').read_text())
            for name in ('score', 'reference', 'traceunits'):
                sys.modules.pop(name, None)
            sys.path.insert(0, str(tests / 'kit'))
            import score as scorer
            raw = Path(temporary) / 'raw'
            (raw / 'cluster').mkdir(parents=True)
            labels = [f'{name}-0' for name in scorer.CELLS]
            qualified = set(suite['qualification'].values()) == {'target-qualified'}
            measurement = {'labels': labels, 'invalid': '/app/run.sh is missing', 'development_only': not qualified,
                           'measurement_binding': None if not qualified else {}, 'repeats': 1,
                           'score_version': scorer.SCORE_VERSION, 'model_ids': suite['model_ids'],
                           'trace_sha256': {case['name']: case['trace_sha256'] for case in suite['cases']},
                           'cells': [{'label': label, 'trace': label.rsplit('-', 1)[0], 'invalid': 'missing'}
                                     for label in labels]}
            (raw / 'cluster/measurement.json').write_text(json.dumps(measurement))
            config = {'allow_unqualified': True}
            self.assertEqual(score_workload.cluster_score(raw, tests=tests, config=config), 0.0)
            measurement['labels'] = labels[::-1]
            (raw / 'cluster/measurement.json').write_text(json.dumps(measurement))
            with self.assertRaises(ContractError):
                score_workload.cluster_score(raw, tests=tests, config=config)
            if not qualified:
                measurement['labels'] = labels
                (raw / 'cluster/measurement.json').write_text(json.dumps(measurement))
                with self.assertRaises(ContractError):
                    score_workload.cluster_score(raw, tests=tests, config={'allow_unqualified': False})


class CellTests(unittest.TestCase):
    def kernel_measurements(self):
        timings, per_shape, agent, oracle = [], [], [], []
        for eager, candidate, ceiling in ((1.0, .5, .2), (2.0, .8, .5)):
            timings += [{'role': 'trusted', 'seconds': eager},
                        {'role': 'candidate', 'seconds': candidate, 'method': 'parent-clocked-batch-v1',
                         'batch_seconds': [candidate * 10] * 6, 'deferred_completion_suspect': False},
                        {'role': 'trusted', 'seconds': ceiling}]
            agent.append(eager / candidate)
            oracle.append(eager / ceiling)
            per_shape.append({'shape': [1], 'agent_x': round(eager / candidate, 2), 'oracle_x': round(eager / ceiling, 2),
                              'relerr': 0.001})
        import math
        from cell_score import geomean
        a, o = geomean(agent), geomean(oracle)
        reward = round(min(1.0, math.log(a) / math.log(o)), 4)
        return {'task': 'gated-deltanet-forward', 'timings': timings, 'comparisons': [0.001, 0.002, 0.001, 0.001],
                'recompute': [1.9], 'oracle_libraries_refused': [], 'voided': None, 'perf_tolerance': 0.02,
                'recompute_min': 1e-3, 'grader_shapes': 2, 'correct_shapes': 2,
                'reward': {'reward': reward, 'correct': 1.0, 'used_oracle_library': 0.0, 'cached_output': 0.0,
                           'agent_geomean': round(a, 4), 'oracle_geomean': round(o, 4)},
                'details': {'per_shape': per_shape}}

    def write(self, root, value):
        root.mkdir(exist_ok=True)
        (root / 'measurements.json').write_text(json.dumps(value))
        return root

    def test_kernel_reward_recomputes_and_rejects_inconsistent_raw_data(self):
        from cell_score import CellVoid, score_kernel
        with tempfile.TemporaryDirectory() as temporary:
            good = self.kernel_measurements()
            result = score_kernel(self.write(Path(temporary) / 'good', good))
            self.assertEqual(result['reward'], good['reward']['reward'])
            for mutation in ('timing', 'claimed_reward', 'cache', 'seal', 'order'):
                bad = copy.deepcopy(good)
                if mutation == 'timing':
                    bad['timings'][1]['seconds'] = 0.01          # a faster candidate than recorded speedups
                if mutation == 'claimed_reward':
                    bad['reward']['reward'] = 1.0
                if mutation == 'cache':
                    bad['recompute'] = [0.0]                     # cached output must zero the reward
                if mutation == 'seal':
                    bad['oracle_libraries_refused'] = ['fla']
                if mutation == 'order':
                    bad['timings'][0], bad['timings'][1] = bad['timings'][1], bad['timings'][0]
                with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                    score_kernel(self.write(Path(temporary) / mutation, bad))
            void = dict(good, voided='GraderBroken: oracle unavailable')
            with self.assertRaises(CellVoid):
                score_kernel(self.write(Path(temporary) / 'void', void))

    def test_pr_raw_protocol_scores_against_rederived_private_expectations(self):
        import pr33698
        from cell_score import score_pr
        cases, private = pr33698.prepare_cases()
        with tempfile.TemporaryDirectory() as temporary:
            raw = Path(temporary)
            (raw / 'cases.json').write_text(json.dumps(cases))
            (raw / 'private.json').write_text(json.dumps(private))
            rows = [{'example_id': name, 'status': 'returned', 'configs': private[name]} for name in pr33698.EXAMPLE_IDS]
            (raw / 'predictions.jsonl').write_text(''.join(json.dumps(row) + '\n' for row in rows))
            self.assertEqual(score_pr('pr-vllm-33698', raw)['reward'], 1.0)
            rows[1]['configs'][0]['num_blocks'] = 34
            (raw / 'predictions.jsonl').write_text(''.join(json.dumps(row) + '\n' for row in rows))
            self.assertEqual(score_pr('pr-vllm-33698', raw)['reward'], 0.0)
            forged = copy.deepcopy(private)
            forged[pr33698.EXAMPLE_IDS[0]][0]['num_blocks'] = 1
            (raw / 'private.json').write_text(json.dumps(forged))
            with self.assertRaises(ValueError):
                score_pr('pr-vllm-33698', raw)
        self.assertIn('spec_type', json.dumps(private))

    def test_registry_refuses_unprotected_tasks_and_kernel_verifiers_are_instrumentable(self):
        import kernel_parent
        from registry import KERNELS, require_task, task_path
        with self.assertRaises(ValueError):
            require_task('pr-vllm-39391')
        for task in KERNELS:
            source = HARNESS / 'eval/e2e_tasks' / task_path(task) / 'tests/verify_env.py'
            tree = kernel_parent.strip_entrypoint(source)
            self.assertTrue(any(getattr(node, 'name', '') == 'main' for node in tree.body), task)
        remote = object()
        self.assertTrue(kernel_parent.closes_over((lambda: remote), remote))
        self.assertFalse(kernel_parent.closes_over((lambda: 1), remote))

    def test_overlay_must_equal_recorded_change_set(self):
        import verifier
        from contract import digest
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            submission, evidence = root / 'submission', root / 'evidence'
            (submission / 'files/vllm').mkdir(parents=True)
            evidence.mkdir()
            (submission / 'files/vllm/fix.py').write_text('fixed')
            changes = {'workdir': '/app/vllm', 'added': [], 'modified': ['vllm/fix.py'], 'deleted': ['old.py'],
                       'executable': []}
            (submission / 'changes.json').write_text(json.dumps(changes))
            baseline = {'files': {'vllm/fix.py': {'sha256': 'b' * 64}, 'old.py': {'sha256': 'c' * 64}}, 'symlinks': {}}
            final = {'files': {'vllm/fix.py': {'sha256': digest(submission / 'files/vllm/fix.py')}}}
            cell = {'workdir': '/app/vllm', 'harness_inventory_sha256': 'h', 'driver': 'glm53'}
            (evidence / 'cell-workspace-baseline.json').write_text(json.dumps(baseline))
            (evidence / 'cell-workspace-final.json').write_text(json.dumps(final))
            (evidence / 'cell.json').write_text(json.dumps({**cell, 'harness_files': {}}))
            self.assertEqual(verifier.cell_mechanical_checks(cell, evidence, submission)['overlay']['modified'], 1)
            (submission / 'files/vllm/extra.py').write_text('smuggled')
            with self.assertRaises(ContractError):
                verifier.cell_mechanical_checks(cell, evidence, submission)


class ReviewIndexTests(unittest.TestCase):
    def test_trace_index_keeps_security_events_and_journal_shows_deltas(self):
        import review_index
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            hexed = lambda text: ''.join(f'\\x{byte:02x}' for byte in text.encode())
            trace = root / 'exec-1'
            trace.write_text('\n'.join([
                f'1.0 execve("{hexed("/usr/bin/python3")}", ["{hexed("python3")}"], 0x0 /* 3 vars */) = 0',
                f'1.1 openat(AT_FDCWD, "{hexed("/tmp/pytest/__init__.py")}", O_WRONLY|O_CREAT|O_TRUNC, 0644) = 3',
                f'1.2 openat(AT_FDCWD, "{hexed("/usr/lib/libc.so")}", O_RDONLY) = 4',
                f'1.3 newfstatat(AT_FDCWD, "{hexed("/external-evidence")}", 0x7ff, 0) = -1 ENOENT',
                '1.4 write(1, 0x55, 10) = 10']) + '\n')
            text = review_index.index_trace(trace)
            self.assertIn('/tmp/pytest/__init__.py', text)
            self.assertIn('execve("/usr/bin/python3"', text)
            self.assertIn('/external-evidence -> 1', text)
            self.assertNotIn('libc.so', text)
            folder = root / 'gateway'
            (folder / 'blobs').mkdir(parents=True)
            rows = []
            for index, messages in enumerate([[{'role': 'user', 'content': 'A'}],
                                              [{'role': 'user', 'content': 'A'}, {'role': 'assistant', 'content': 'B'},
                                               {'role': 'user', 'content': 'C-new'}]]):
                raw = json.dumps({'model': 'm', 'system': 's', 'messages': messages}).encode()
                value = hashlib.sha256(raw).hexdigest()
                (folder / 'blobs' / value).write_bytes(raw)
                rows.append({'seq': index, 'event': 'model_request', 'call_id': str(index),
                             'output': {'path': f'blobs/{value}', 'sha256': value, 'bytes': len(raw)}})
            (folder / 'requests.jsonl').write_text(''.join(json.dumps(row) + '\n' for row in rows))
            view = review_index.index_journal(folder, 'gateway')
            self.assertIn('first 1 identical', view)
            self.assertIn('C-new', view)
            self.assertEqual(view.count('"content": "A"'), 1)


class TaskContractTests(unittest.TestCase):
    def test_harness_and_cluster_tasks_keep_their_agent_policies(self):
        for root in (HARNESS, CLUSTER):
            with self.subTest(task=root.name):
                policy = json.loads((root / 'environment/agent-policy.json').read_text())
                self.assertEqual(policy['name'], root.name)
                self.assertTrue((root / 'tests/submission_judge.md').is_file())
        self.assertTrue((HARNESS / 'eval/config.json').is_file())

if __name__ == '__main__':
    unittest.main()
