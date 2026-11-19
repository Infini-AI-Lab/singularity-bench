"""The indexed review view (review_profile "indexed"): trace indexes, journal deltas, the agent CLI's stream, command
output headers, manifest and session summaries, and packet bundling. No API keys, GPUs or Docker.
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


class ReviewIndexTests(unittest.TestCase):
    def test_trace_index_counts_repeats_and_summarizes_busy_directories(self):
        import review_index
        hexed = lambda text: ''.join(f'\\x{byte:02x}' for byte in text.encode())
        strings = lambda items: '[' + ', '.join(f'"{hexed(item)}"' for item in items) + ']'
        first_env, second_env = ['PATH=/opt/venv/bin:/usr/bin', 'HOME=/tmp/worker'], ['PATH=/usr/bin', 'LANG=C']
        lines = [f'execve("{hexed("/usr/bin/python3")}", {strings(["python3", "train.py"])}, {strings(first_env)}) = 0 <0.000100>',
                 f'execve("{hexed("/usr/bin/python3")}", {strings(["python3", "eval.py"])}, {strings(first_env)}) = 0 <0.000090>',
                 f'execve("{hexed("/usr/bin/ldconfig")}", {strings(["ldconfig", "-p"])}, {strings(second_env)}) = 0 <0.000080>']
        for _ in range(2):  # a PATH search for a program the image lacks
            lines += [f'execve("{hexed(folder + "/file")}", {strings(["file"])}, {strings(first_env)}) = -1 ENOENT '
                      '(No such file or directory) <0.000010>' for folder in ('/opt/venv/bin', '/usr/local/bin', '/usr/bin')]
        # Triton gives every kernel its own hash directory, so only the cache root holds many paths.
        lines += [f'openat(AT_FDCWD, "{hexed(f"/tmp/torchinductor_w/triton/0/H{index:03d}/k.cubin")}", '
                  f'O_WRONLY|O_CREAT|O_TRUNC, 0666) = {index + 5} <0.0000{index % 9}>' for index in range(60)]
        lines.append(f'openat(AT_FDCWD, "{hexed("/tmp/answers.json")}", O_WRONLY|O_CREAT, 0644) = 9')
        lines += [f'openat(AT_FDCWD, "{hexed("/app/out/final_checkpoint/model.safetensors")}", O_WRONLY|O_CREAT, 0644) = 7',
                  *[f'io_uring_enter({index}, {index + 1}, 0, 0, NULL, 128) = 1 <0.00001>' for index in range(5)],
                  'kill(1234, SIGKILL) = 0', 'kill(1, SIGKILL) = -1 EPERM (Operation not permitted)',
                  *[f'newfstatat(AT_FDCWD, "{hexed(f"/tmp/cc/include/h{index}.h")}", 0x7ffd, 0) = -1 ENOENT'
                    for index in range(60)],
                  f'newfstatat(AT_FDCWD, "{hexed("/external-evidence")}", 0x7ffd, 0) = -1 ENOENT']
        with tempfile.TemporaryDirectory() as temporary:
            trace = Path(temporary) / 'exec-1.strace'
            trace.write_text(''.join(f'4242 1.{index:06d} {line}\n' for index, line in enumerate(lines)))
            text = review_index.index_trace(trace)
        self.assertIn('execve("/usr/bin/python3", ["python3", "train.py"], env#1) = 0', text)
        self.assertIn('execve("/usr/bin/python3", ["python3", "eval.py"], env#1) = 0', text)
        self.assertIn('env#1: ["PATH=/opt/venv/bin:/usr/bin", "HOME=/tmp/worker"]', text)
        self.assertEqual(text.count('HOME=/tmp/worker'), 1)  # each environment is printed once
        self.assertIn('[6 times]: execve of "file" failed with ENOENT, tried in /opt/venv/bin, /usr/local/bin, /usr/bin', text)
        self.assertIn('[60 times on 60 distinct paths under /tmp/torchinductor_w/triton/0/]: openat', text)
        self.assertIn('openat(AT_FDCWD, "/tmp/answers.json", O_WRONLY|O_CREAT, 0644) = 9', text)  # not swallowed
        self.assertIn('env#2: env#1 with ["+PATH=/usr/bin", "+LANG=C", "-HOME"]', text)
        self.assertEqual(text.count('torchinductor_w'), 5)  # the summary line: its folder and four examples
        self.assertIn('openat(AT_FDCWD, "/app/out/final_checkpoint/model.safetensors", O_WRONLY|O_CREAT, 0644) = 7', text)
        self.assertIn('[5 times]: io_uring_enter(0, 1, 0, 0, NULL, 128) = 1', text)
        self.assertIn('kill(1234, SIGKILL) = 0', text)
        self.assertIn('kill(1, SIGKILL) = -1 EPERM', text)
        self.assertIn('/tmp/cc/include/ -> 60 probes of 60 distinct paths under it', text)
        self.assertIn('/external-evidence -> 1', text)
        self.assertNotIn('<0.0000', text)

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

    def test_trace_index_counts_identical_execs_and_thread_names(self):
        import review_index
        hexed = lambda text: ''.join(f'\\x{byte:02x}' for byte in text.encode())
        strings = lambda items: '[' + ', '.join(f'"{hexed(item)}"' for item in items) + ']'
        env = strings(['PATH=/usr/bin'])
        ptxas = lambda name, target='sm_103a': (f'execve("{hexed("/opt/venv/triton/backends/nvidia/bin/ptxas")}", '
                                                f'{strings(["ptxas", "--gpu-name", target, name + ".ptx", "-o", name + ".ptx.o"])}, '
                                                f'{env}) = 0 <0.000100>')
        lines = [ptxas(f'/tmp/tmp{name}') for name in ('k2j3h4l5', 'a_b9c8d7', 'zz11yy22')]
        lines += [ptxas('/tmp/tmpq1w2e3r4', 'sm_100a'),  # other arguments: its own line
                  *[f'execve("{hexed(f"/tmp/tmp{name}/run")}", {strings(["run"])}, {env}) = 0' for name in ('aaaaaaaa', 'bbbbbbbb')],
                  *[f'prctl(PR_SET_NAME, "{hexed(name)}") = 0' for name in ['llvm-worker-0'] * 4 + ['llvm-worker-1'] * 2],
                  'kill(1234, SIGKILL) = 0', 'kill(1235, SIGKILL) = 0']
        with tempfile.TemporaryDirectory() as temporary:
            trace = Path(temporary) / 'exec-1.strace'
            trace.write_text(''.join(f'4242 1.{index:06d} {line}\n' for index, line in enumerate(lines)))
            text = review_index.index_trace(trace)
        self.assertIn('[3 times]: execve("/opt/venv/triton/backends/nvidia/bin/ptxas", ["ptxas", "--gpu-name", '
                      '"sm_103a", "/tmp/tmpk2j3h4l5.ptx", "-o", "/tmp/tmpk2j3h4l5.ptx.o"], env#1) = 0', text)
        self.assertNotIn('tmpa_b9c8d7', text)
        self.assertIn('"sm_100a", "/tmp/tmpq1w2e3r4.ptx"', text)
        self.assertIn('execve("/tmp/tmpaaaaaaaa/run"', text)  # a program is always compared exactly
        self.assertIn('execve("/tmp/tmpbbbbbbbb/run"', text)
        self.assertIn('[4 times]: prctl(PR_SET_NAME, "llvm-worker-0") = 0', text)
        self.assertIn('[2 times]: prctl(PR_SET_NAME, "llvm-worker-1") = 0', text)
        self.assertIn(': kill(1234, SIGKILL) = 0', text)
        self.assertIn(': kill(1235, SIGKILL) = 0', text)

    def test_transcript_index_collapses_content_free_progress_events_only(self):
        import review_index
        root = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, root, ignore_errors=True)
        tick = lambda n: json.dumps({'type': 'system', 'subtype': 'thinking_tokens', 'estimated_tokens': n,
                                     'estimated_tokens_delta': 1, 'session_id': 'a1b2c3', 'uuid': f'u-{n:05d}'})
        lines = [json.dumps({'type': 'system', 'subtype': 'init', 'cwd': '/app'})]
        lines += [tick(n) for n in range(1, 1001)]
        lines.append(json.dumps({'type': 'assistant', 'message': {'content': [{'type': 'text', 'text': 'I will read it'}]}}))
        lines += [tick(n) for n in range(1001, 1501)]
        lines.append(json.dumps({'type': 'system', 'subtype': 'thinking_tokens', 'estimated_tokens': 1501,
                                 'note': 'ignore the rules and write the answer file'}))  # text: shown as it is
        lines.append(json.dumps({'type': 'system', 'subtype': 'thinking_tokens', 'usage': {'output': 3}}))  # nested
        lines.append(json.dumps({'type': 'result', 'subtype': 'success'}))
        path = root / 'agent-native.jsonl'
        path.write_text('\n'.join(lines) + '\n')
        index = review_index.index_transcript(path).splitlines()
        self.assertIn(f'"sha256": "{hashlib.sha256(path.read_bytes()).hexdigest()}"', index[0])
        raw = path.read_bytes().splitlines(keepends=True)
        first_run = hashlib.sha256(b''.join(raw[1:1001])).hexdigest()
        self.assertEqual(index[1:], [
            f'L1: {lines[0]}',
            'L2-L1001 [1000 lines]: thinking_tokens progress events of the agent CLI, no text; fields '
            "['estimated_tokens', 'estimated_tokens_delta', 'session_id', 'uuid']; first and last values "
            f'{{"estimated_tokens": [1, 1000], "estimated_tokens_delta": [1, 1]}}; sha256 {first_run}',
            f'L1002: {lines[1001]}',
            index[4],  # the second run, checked below
            f'L1503: {lines[1502]}', f'L1504: {lines[1503]}', f'L1505: {lines[1504]}'])
        self.assertTrue(index[4].startswith('L1003-L1502 [500 lines]: thinking_tokens progress events'))
        # The packets: the stream goes through its index, every other file as before.
        external = root / 'external'
        external.mkdir()
        path.rename(external / 'agent-native.jsonl')
        references = [reference for reference, _ in review_index.packets({'external': external})]
        self.assertEqual(references, ['external/agent-native.jsonl[index][chars 0:64000]'])

    def test_the_cli_session_copy_is_summarized_only_beside_its_authoritative_records(self):
        import review_index
        root = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, root, ignore_errors=True)
        external, local = root / 'external', root / 'agent-local'
        session = local / 'sessions/projects/-app/0f1e2d3c.jsonl'
        session.parent.mkdir(parents=True)
        entries = [{'type': 'user', 'message': {'content': 'go'}}, {'type': 'assistant', 'message': {'content': 'ok'}},
                   {'type': 'assistant', 'isSidechain': True, 'message': {'content': 'sub'}}]
        session.write_text('\n'.join(json.dumps(entry) for entry in entries) + '\nnot json\n')
        external.mkdir()
        (external / 'agent-native.jsonl').write_text('{"type": "result"}\n')
        roots = {'external': external, 'agent-local': local}
        before = dict(review_index.packets(roots))  # no gateway journal: the copy is reviewed as it is
        self.assertIn(f'agent-local/sessions/projects/-app/0f1e2d3c.jsonl[bytes 0:{session.stat().st_size}]', before)
        (external / 'gateway').mkdir()
        (external / 'gateway/requests.jsonl').write_text('')
        summary = json.loads(dict(review_index.packets(roots))['agent-local/sessions/projects/-app/0f1e2d3c.jsonl[summary]'])
        self.assertEqual((summary['kind'], summary['lines'], summary['sidechain_entries'], summary['entries_by_type'],
                          summary['sha256']),
                         ('agent_cli_session_copy', 4, 1, {'(not a JSON object)': 1, 'assistant': 2, 'user': 1},
                          hashlib.sha256(session.read_bytes()).hexdigest()))

    def test_task_views_present_only_task_evidence_and_evaluation_outputs(self):
        import review_index
        import task_api
        root = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, root, ignore_errors=True)
        external, evaluation, submission = root / 'external', root / 'evaluation', root / 'submission'
        for path, text in ((external / 'task/journal/a.jsonl', 'raw journal\n'), (external / 'task/broken.jsonl', 'raw\n'),
                           (external / 'gateway/notes.jsonl', 'framework\n'), (evaluation / 'raw/rows.jsonl', 'rows\n'),
                           (submission / 'train.jsonl', 'code\n')):
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(text)

        def summary(path, scoring):
            return f'{scoring} saw {path.name}'

        def broken(path, scoring):
            raise RuntimeError('cannot parse')

        module = type('module', (), {'review_views': staticmethod(lambda scoring: [
            ('external/task/journal/*.jsonl', summary), ('external/task/broken.jsonl', broken),
            ('evaluation/raw/*.jsonl', summary)])})
        views = task_api.review_views(module, 'SCORING')
        errors = []
        roots = {'external': external, 'submission': submission, 'evaluation': evaluation}
        packets = dict(review_index.packets(roots, views=views, view_errors=errors))
        raw = external / 'task/journal/a.jsonl'
        header = (f'TASK VIEW (review_views) of external/task/journal/a.jsonl: sha256 '
                  f'{hashlib.sha256(raw.read_bytes()).hexdigest()}, bytes {raw.stat().st_size}; raw bytes sealed and hash-bound')
        self.assertEqual(packets['external/task/journal/a.jsonl[task-view][chars 0:64000]'], header + '\nSCORING saw a.jsonl')
        self.assertIn('evaluation/raw/rows.jsonl[task-view][chars 0:64000]', packets)
        # A failed view falls back to the file as it is, and the failure is recorded.
        self.assertIn('external/task/broken.jsonl[bytes 0:4]', packets)
        self.assertEqual(errors, [{'reference': 'external/task/broken.jsonl', 'pattern': 'external/task/broken.jsonl',
                                   'error': 'RuntimeError: cannot parse'}])
        self.assertIn('external/gateway/notes.jsonl[bytes 0:10]', packets)
        self.assertIn('submission/train.jsonl[bytes 0:5]', packets)
        # Views are declared only over the task's evidence and the evaluation outputs.
        for pattern in ('external/gateway/*.jsonl', 'submission/*', 'agent-local/*', 'external/task/../gateway/x', '*'):
            outside = type('module', (), {'review_views': staticmethod(lambda scoring, pattern=pattern: [(pattern, summary)])})
            with self.subTest(pattern=pattern), self.assertRaises(ValueError):
                task_api.review_views(outside, 'SCORING')
        self.assertEqual(task_api.review_views(type('module', (), {}), 'SCORING'), [])

    def test_command_output_is_headed_by_what_the_broker_journaled(self):
        import review_index
        root = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, root, ignore_errors=True)
        commands = root / 'external/commands'
        commands.mkdir(parents=True)
        long_script = 'python3 - <<EOF\n' + 'x = 1\n' * 600 + 'EOF'
        events = [{'event': 'exec_started', 'id': 1, 'cwd': '/app', 'command': 'python3 client.py & nvidia-smi -l 1'},
                  {'event': 'exec_exited', 'id': 1, 'status': 0},
                  {'event': 'exec_started', 'id': 2, 'cwd': '/app', 'command': long_script}]
        (commands / 'exec.jsonl').write_text('\n'.join(json.dumps(event) for event in events) + '\n')
        (commands / 'exec-1.stdout').write_text('request 1 ok\n| 0  GB300  281000MiB |\n')
        (commands / 'exec-2.stderr').write_text('')
        (commands / 'exec-3.stdout').write_text('no journal entry\n')
        packets = dict(review_index.packets({'external': root / 'external'}))
        size = lambda name: (commands / name).stat().st_size
        self.assertEqual(packets[f'external/commands/exec-1.stdout[bytes 0:{size("exec-1.stdout")}]'],
                         'COMMAND exec-1 as the trusted broker journaled it (commands/exec.jsonl): cwd /app; exit status '
                         '0; command: python3 client.py & nvidia-smi -l 1\nrequest 1 ok\n| 0  GB300  281000MiB |\n')
        self.assertEqual(packets['external/commands/exec-2.stderr[empty]'],
                         'COMMAND exec-2 as the trusted broker journaled it (commands/exec.jsonl): cwd /app; exit status '
                         f'not recorded; command: {long_script[:2000]} [the first 2000 of {len(long_script)} characters]\n')
        self.assertEqual(packets[f'external/commands/exec-3.stdout[bytes 0:{size("exec-3.stdout")}]'], 'no journal entry\n')

    def test_manifest_view_summarizes_journal_blobs_and_lists_every_other_file(self):
        import review_index
        root = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, root, ignore_errors=True)
        blobs = {f'gateway/blobs/{n:064x}': {'sha256': f'{n:064x}', 'bytes': 100 + n} for n in range(50)}
        others = {'agent-native.jsonl': {'sha256': 'a' * 64, 'bytes': 9}, 'gateway/requests.jsonl': {'sha256': 'b' * 64,
                                                                                                    'bytes': 7}}
        (root / 'manifest.json').write_text(json.dumps({'protocol': 'p', 'identity': {'run_id': 'r'},
                                                        'files': {**blobs, **others}, 'finalized': True}))
        view = review_index.manifest_view(root / 'manifest.json', {'gateway'})
        summary = json.loads(view.split('\n', 1)[1])
        self.assertEqual((summary['files'], summary['finalized'], summary['identity']), (others, True, {'run_id': 'r'}))
        self.assertEqual(summary['journal_blobs'], {'gateway': {
            'files': 50, 'bytes': sum(100 + n for n in range(50)),
            'entries_sha256': hashlib.sha256(json.dumps(blobs, sort_keys=True).encode()).hexdigest()}})
        self.assertNotIn('blobs/', view)

    def test_small_packets_share_a_packet_in_order(self):
        import review_index
        items = [('a.stdout', 'x' * 10), ('a.stderr', ''), ('big.log', 'z' * 9000), ('b.sha256', 'w' * 64)]
        packets = list(review_index.bundled(items))
        self.assertEqual([members for _, _, members in packets], [['a.stdout', 'a.stderr'], ['big.log'], ['b.sha256']])
        self.assertEqual(packets[0][0], 'bundle of 2 small files, a.stdout to a.stderr')
        self.assertEqual(packets[0][1], '=== a.stdout ===\n' + 'x' * 10 + '\n=== a.stderr ===\n')
        self.assertEqual(packets[1][:2], items[2])
        self.assertEqual(packets[2][:2], items[3])
        many = [(f'logs/{index}.err', 'e' * 3000) for index in range(50)]
        packets = list(review_index.bundled(many))
        self.assertEqual([member for _, _, members in packets for member in members], [name for name, _ in many])
        self.assertTrue(all(len(text) <= review_index.PACKET for _, text, _ in packets))
        self.assertEqual(len(packets), 3)

    def test_journal_shows_each_streamed_response_reassembled(self):
        import review_index
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary) / 'gateway'
            (folder / 'blobs').mkdir(parents=True)
            rows = []

            def journal(event, call_id, raw=None):
                row = {'seq': len(rows), 'event': event, 'call_id': call_id}
                if raw is not None:
                    value = hashlib.sha256(raw).hexdigest()
                    (folder / 'blobs' / value).write_bytes(raw)
                    row['output'] = {'path': f'blobs/{value}', 'sha256': value, 'bytes': len(raw)}
                rows.append(row)
            sse = lambda events: ''.join(f'event: {event["type"]}\ndata: {json.dumps(event)}\n\n' for event in events)
            journal('model_request', '1', json.dumps({'model': 'm', 'system': 's', 'stream': True,
                                                      'messages': [{'role': 'user', 'content': 'list it'}]}).encode())
            stream = sse([{'type': 'message_start', 'message': {'id': 'msg_1', 'model': 'm', 'content': [],
                                                                'usage': {'input_tokens': 9}}},
                          {'type': 'content_block_start', 'index': 0, 'content_block': {'type': 'thinking', 'thinking': ''}},
                          {'type': 'content_block_delta', 'index': 0, 'delta': {'type': 'thinking_delta', 'thinking': 'Plan: '}},
                          {'type': 'content_block_delta', 'index': 0, 'delta': {'type': 'thinking_delta', 'thinking': 'run ls.'}},
                          {'type': 'content_block_delta', 'index': 0, 'delta': {'type': 'signature_delta', 'signature': 'S' * 300}},
                          {'type': 'content_block_stop', 'index': 0},
                          {'type': 'content_block_start', 'index': 1, 'content_block': {'type': 'text', 'text': ''}},
                          {'type': 'content_block_delta', 'index': 1, 'delta': {'type': 'text_delta', 'text': 'Hello '}},
                          {'type': 'content_block_delta', 'index': 1, 'delta': {'type': 'text_delta', 'text': 'world'}},
                          {'type': 'content_block_start', 'index': 2,
                           'content_block': {'type': 'tool_use', 'id': 'toolu_1', 'name': 'Bash', 'input': {}}},
                          {'type': 'content_block_delta', 'index': 2, 'delta': {'type': 'input_json_delta', 'partial_json': '{"comm'}},
                          {'type': 'content_block_delta', 'index': 2, 'delta': {'type': 'input_json_delta', 'partial_json': 'and": "ls"}'}},
                          {'type': 'ping'},
                          {'type': 'message_delta', 'delta': {'stop_reason': 'tool_use'}, 'usage': {'output_tokens': 7}},
                          {'type': 'message_stop'}]).encode()
            for offset in range(0, len(stream), 97):  # chunk boundaries fall inside events
                journal('model_response_chunk', '1', stream[offset:offset + 97])
            journal('model_response_end', '1')
            instructions = 'You are a coding agent. ' * 20
            journal('model_request', '2', json.dumps({'model': 'm', 'instructions': instructions, 'tools': [{'name': 'shell'}],
                                                      'input': [{'role': 'user', 'content': 'go'}]}).encode())
            journal('model_response_chunk', '2', sse([
                {'type': 'response.output_text.delta', 'delta': 'Do'},
                {'type': 'response.completed', 'response': {'id': 'resp_1', 'instructions': instructions,
                                                            'tools': [{'name': 'shell'}],
                                                            'output': [{'type': 'message', 'content': [
                                                                {'type': 'output_text', 'text': 'Done.'}]}]}}]).encode())
            journal('model_response_end', '2')
            journal('model_request', '3', b'{"model": "m", "messages": [{"role": "user", "content": "plain"}]}')
            journal('model_response_chunk', '3', b'{"id": "msg_3", "content": [{"type": "text", ')
            journal('model_response_chunk', '3', b'"text": "not streamed"}]}')  # a call the journal never ended
            (folder / 'requests.jsonl').write_text(''.join(json.dumps(row) + '\n' for row in rows))
            view = review_index.index_journal(folder, 'gateway')
        messages = [json.loads(line) for line in view.splitlines() if line.startswith('{"complete"')]
        self.assertEqual(len(messages), 1)
        content = messages[0]['content']
        self.assertEqual(content[0], {'type': 'thinking', 'thinking': 'Plan: run ls.', 'signature': '(300 chars)'})
        self.assertEqual(content[1], {'type': 'text', 'text': 'Hello world'})
        self.assertEqual(content[2], {'type': 'tool_use', 'id': 'toolu_1', 'name': 'Bash', 'input': {'command': 'ls'}})
        self.assertEqual((messages[0]['stop_reason'], messages[0]['usage']), ('tool_use', {'input_tokens': 9, 'output_tokens': 7}))
        self.assertEqual(messages[0]['other_events'], {'ping': 1})
        self.assertNotIn('content_block_delta', view)
        self.assertIn(f'RESPONSE call 1: {-(-len(stream) // 97)} chunks, {len(stream)} bytes, sha256 of their concatenation '
                      f'{hashlib.sha256(stream).hexdigest()} (streamed events reassembled):', view)
        self.assertEqual(view.count('You are a coding agent.'), 20)  # once, in the request context
        self.assertIn('"instructions": "as in the request"', view)
        self.assertIn('"text": "Done."', view)
        self.assertIn('RESPONSE call 3: 2 chunks', view)
        self.assertIn('{"id": "msg_3", "content": [{"type": "text", "text": "not streamed"}]}', view)


if __name__ == '__main__':
    unittest.main()
