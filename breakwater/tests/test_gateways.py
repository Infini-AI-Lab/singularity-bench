"""The hash-chained development/driver gateway and the author gateway's journal under provider failures. No API keys,
GPUs or Docker.
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
sys.path[:0] = [str(FRAMEWORK)]
from contract import ContractError  # noqa: E402


def free_port():
    with socket.socket() as probe:
        probe.bind(('127.0.0.1', 0))
        return probe.getsockname()[1]


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
        process = subprocess.Popen([sys.executable, '-I', str(FRAMEWORK / 'dev_gateway.py')], env=env,
                                   stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
        self.addCleanup(lambda: process.poll() is None and process.kill())
        self.addCleanup(process.stderr.close)
        for _ in range(100):
            if process.poll() is not None:  # refused its configuration
                break
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

    def test_the_task_declares_every_model_and_an_undeclared_embedding_model_has_no_route(self):
        for bad in ({}, {'CHAT_MODELS': '{}'}, {'CHAT_MODELS': json.dumps({'Chat': 'm'})},
                    {'CHAT_MODELS': json.dumps({'chat': ''})}, {'CHAT_MODELS': json.dumps({'chat': 'm'}), 'MAX_CHAT_TOKENS': '-1'}):
            process, port, folder = self.start('development', **bad)
            with self.subTest(env=bad):
                self.assertNotEqual(process.wait(timeout=10), 0)
                self.assertIn(b'Development gateway' if 'MAX_CHAT_TOKENS' not in bad else b'allowances',
                              process.stderr.read())
        process, port, folder = self.start('development', CHAT_MODELS=json.dumps({'solo': 'm/one'}))
        good = {'model': 'solo', 'max_tokens': 32, 'messages': [{'role': 'user', 'content': 'hi'}]}
        self.assertEqual(self.post(port, '/v1/chat/completions', good)[0], 200)
        self.assertEqual(self.post(port, '/v1/embeddings', {'model': 'embed', 'input': 'x'})[0], 400)
        with urllib.request.urlopen(f'http://127.0.0.1:{port}/v1/models', timeout=5) as response:
            self.assertEqual([row['id'] for row in json.load(response)['data']], ['solo'])
        self.stop(process)

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
        process = subprocess.Popen([sys.executable, '-I', str(FRAMEWORK / 'author_gateway.py')], env=env,
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


if __name__ == '__main__':
    unittest.main()
