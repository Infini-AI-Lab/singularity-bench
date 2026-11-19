"""Live test of the Apptainer evaluation stage with a fixture coordinator in place of intermediate.py.

    BREAKWATER_APPTAINER_IMAGE=IMAGE.sif python3 -m unittest discover -s breakwater/apptainer/tests -v

The fixture does what a coordinator does: it reads private data, writes answer-free inputs, and starts submitted
code through worker-exec, handing it a socket by descriptor number (as the pre-training PPL driver does).
"""
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
import unittest

HERE = Path(__file__).resolve().parents[1]
REPO = HERE.parents[1]
sys.path[:0] = [str(HERE), str(HERE / 'tests'), str(HERE.parent)]
from test_agent_stage import IMAGE, READY  # noqa: E402

WORKER = r'''
import json, os, socket, sys
from pathlib import Path
channel = socket.socket(fileno=int(sys.argv[1]))
channel.sendall(b'pong:' + channel.recv(100))
print(json.dumps({'private': Path('/private/validation').exists(), 'tests': Path('/tests').exists(),
                  'evaluation': Path('/evaluation').exists(), 'cwd': os.getcwd(),
                  'inputs': Path('/eval-inputs/generation.input').read_text(), 'app': sorted(os.listdir('/app'))}))
'''
COORDINATOR = r'''
import json, socket, subprocess, sys
from pathlib import Path
secret = Path('/private/validation/secret.txt').read_text()
Path('/eval-inputs/generation.input').write_text('answer-free')
parent, child = socket.socketpair(socket.AF_UNIX, socket.SOCK_SEQPACKET)
process = subprocess.Popen(['python3', '-I', '/opt/breakwater-runtime/worker-exec', 'python3', '-c', sys.argv[1],
                            str(child.fileno())], pass_fds=(child.fileno(),), stdout=subprocess.PIPE, text=True)
child.close()
parent.sendall(b'ping')
reply = parent.recv(100).decode()
stdout, _ = process.communicate(timeout=120)
Path('/evaluation/result.json').write_text(json.dumps({'secret': secret, 'reply': reply, 'status': process.returncode,
                                                       'worker': json.loads(stdout)}))
Path('/evaluation/complete.json').write_text(json.dumps({'complete': True}))
'''


# A serving task's coordinator: install the sealed submission (mounted apart, at /submission) as a writable copy at
# /app, start the submitted server from it, and reach it on 127.0.0.1:8000 (relayed to the submission sandbox).
SERVING = r'''
import json, os, signal, subprocess, sys, time, urllib.request
from pathlib import Path
sys.path.insert(0, '/opt/breakwater')
from workers import install_tree, worker_prefix
install_tree([(Path('/submission'), Path('/app'))])
server = subprocess.Popen([*worker_prefix(), 'bash', '-c', 'touch /home/worker/written && echo started > /app/server.log '
                           '&& exec python3 -m http.server 8000 --bind 127.0.0.1 --directory /app'],
                          start_new_session=True)
deadline, page = time.monotonic() + 60, None
while page is None and time.monotonic() < deadline:
    try:
        page = urllib.request.urlopen('http://127.0.0.1:8000/index.txt', timeout=5).read().decode()
    except OSError:
        time.sleep(0.5)
os.killpg(server.pid, signal.SIGTERM)
server.wait(timeout=60)
Path('/evaluation/result.json').write_text(json.dumps({
    'page': page, 'log': Path('/app/server.log').read_text(), 'sealed': sorted(os.listdir('/submission'))}))
'''


@unittest.skipUnless(READY, 'set BREAKWATER_APPTAINER_IMAGE to a SIF with bash and python3; the stage is NOT verified')
class EvaluationStageTests(unittest.TestCase):
    def test_a_submitted_server_runs_from_a_writable_copy_and_its_port_reaches_the_coordinator(self):
        from evaluation_stage import EvaluationStage
        root = Path(tempfile.mkdtemp(prefix='breakwater-evaluation-test-'))
        self.addCleanup(shutil.rmtree, root, True)
        submission, evaluation, external = (root / name for name in ('submission', 'evaluation', 'external'))
        for folder in (submission, evaluation, external):
            folder.mkdir()
        (submission / 'index.txt').write_text('served from the copy\n')
        contract = root / 'contract.json'
        contract.write_text('{}')
        path = {'PATH': '/usr/local/bin:/usr/bin:/bin'}
        record = EvaluationStage(image=IMAGE, task=REPO / 'examples/word-count', contract=contract,
                                 submission=submission, workdir='/app', evaluation=evaluation, external=external,
                                 worker_env=path, coordinator_env=path, timeout=300, submission_mount='/submission',
                                 ports=[8000], coordinator_argv=['python3', '-c', SERVING]).run()
        self.assertEqual((record['exit_code'], record['timed_out']), (0, False), (external / 'intermediate.stderr').read_text())
        self.assertEqual(record['process_wall'], {'coordinator': True, 'submission': True})
        result = json.loads((evaluation / 'result.json').read_text())
        # The sealed submission is untouched; the server ran from the copy, which both sandboxes share.
        self.assertEqual(result, {'page': 'served from the copy\n', 'log': 'started\n', 'sealed': ['index.txt']})
        journal = [json.loads(line) for line in (external / 'evaluation-commands/exec.jsonl').read_text().splitlines()]
        self.assertEqual([row['argv'][:2] for row in journal if row['event'] == 'exec_started'], [['bash', '-c']])
        self.assertEqual(record['relayed_ports'], [8000])

    def test_the_coordinator_reaches_a_host_service_through_a_relay_and_nothing_else(self):
        import http.server
        import threading
        from evaluation_stage import EvaluationStage

        class Service(http.server.BaseHTTPRequestHandler):
            def do_GET(self):
                self.send_response(200)
                self.end_headers()
                self.wfile.write(b'scored by the host service')

            def log_message(self, *_):
                pass
        service = http.server.ThreadingHTTPServer(('127.0.0.1', 0), Service)
        threading.Thread(target=service.serve_forever, daemon=True).start()
        self.addCleanup(service.server_close)
        self.addCleanup(service.shutdown)
        port = service.server_address[1]
        root = Path(tempfile.mkdtemp(prefix='breakwater-evaluation-test-'))
        self.addCleanup(shutil.rmtree, root, True)
        submission, evaluation, external = (root / name for name in ('submission', 'evaluation', 'external'))
        for folder in (submission, evaluation, external):
            folder.mkdir()
        contract = root / 'contract.json'
        contract.write_text('{}')
        # The coordinator gets the service's URL as the task forwards it; the submitted code gets no relay.
        coordinator = (f"import json, os, subprocess, sys, urllib.request\n"
                       f"from pathlib import Path\n"
                       f"page = urllib.request.urlopen(os.environ['SERVICE_URL'], timeout=10).read().decode()\n"
                       f"worker = subprocess.run(['python3', '-I', '/opt/breakwater-runtime/worker-exec', 'python3', '-c', "
                       f"'import urllib.request\\ntry:\\n    urllib.request.urlopen(\"http://127.0.0.1:{port}\", timeout=5)"
                       f"\\n    print(\"reached\")\\nexcept OSError:\\n    print(\"unreachable\")'], capture_output=True, "
                       f"text=True).stdout.strip()\n"
                       f"Path('/evaluation/result.json').write_text(json.dumps({{'page': page, 'worker': worker}}))\n")
        path = {'PATH': '/usr/local/bin:/usr/bin:/bin'}
        record = EvaluationStage(image=IMAGE, task=REPO / 'examples/word-count', contract=contract,
                                 submission=submission, workdir='/app', evaluation=evaluation, external=external,
                                 worker_env=path, coordinator_env={**path, 'SERVICE_URL': f'http://127.0.0.1:{port}/'},
                                 timeout=300, host_ports=[port], coordinator_argv=['python3', '-c', coordinator]).run()
        self.assertEqual((record['exit_code'], record['timed_out']), (0, False), (external / 'intermediate.stderr').read_text())
        self.assertEqual(json.loads((evaluation / 'result.json').read_text()),
                         {'page': 'scored by the host service', 'worker': 'unreachable'})
        self.assertEqual(record['host_services'], [port])

    def test_kill_workers_stops_every_submitted_process(self):
        from evaluation_stage import EvaluationStage
        root = Path(tempfile.mkdtemp(prefix='breakwater-evaluation-test-'))
        self.addCleanup(shutil.rmtree, root, True)
        submission, evaluation, external = (root / name for name in ('submission', 'evaluation', 'external'))
        for folder in (submission, evaluation, external):
            folder.mkdir()
        contract = root / 'contract.json'
        contract.write_text('{}')
        # Submitted code leaves a detached process behind (as a harness's or server's children can); kill_workers
        # stops it, as Docker's kills every process of the worker UID.
        coordinator = (
            "import json, subprocess, sys\n"
            "from pathlib import Path\n"
            "sys.path.insert(0, '/opt/breakwater')\n"
            "from workers import kill_workers, worker_prefix\n"
            "run = lambda command: subprocess.run([*worker_prefix(), 'bash', '-c', command], capture_output=True, "
            "text=True).stdout.strip()\n"
            "run('setsid sleep 907 > /dev/null 2>&1 < /dev/null &')\n"
            "before = run('grep -la \"slee[p].907\" /proc/[0-9]*/cmdline 2>/dev/null | wc -l')\n"
            "kill_workers(grace=1)\n"
            "after = run('grep -la \"slee[p].907\" /proc/[0-9]*/cmdline 2>/dev/null | wc -l')\n"
            "Path('/evaluation/result.json').write_text(json.dumps({'before': before, 'after': after}))\n")
        path = {'PATH': '/usr/local/bin:/usr/bin:/bin'}
        record = EvaluationStage(image=IMAGE, task=REPO / 'examples/word-count', contract=contract,
                                 submission=submission, workdir='/app', evaluation=evaluation, external=external,
                                 worker_env=path, coordinator_env=path, timeout=300,
                                 coordinator_argv=['python3', '-c', coordinator]).run()
        self.assertEqual((record['exit_code'], record['timed_out']), (0, False), (external / 'intermediate.stderr').read_text())
        self.assertEqual(json.loads((evaluation / 'result.json').read_text()), {'before': '1', 'after': '0'})
        journal = [json.loads(line) for line in (external / 'evaluation-commands/exec.jsonl').read_text().splitlines()]
        killed = [row for row in journal if row['event'] == 'commands_killed']
        self.assertEqual(len(killed), 1)
        self.assertGreaterEqual(killed[0]['processes'], 1)
        self.assertEqual(killed[0]['survivors'], 0)
        from reconcile import check_exec_journal
        self.assertEqual(check_exec_journal(external / 'evaluation-commands')['status'], 'PASS')

    def test_submitted_code_runs_apart_from_the_private_inputs(self):
        from evaluation_stage import EvaluationStage
        root = Path(tempfile.mkdtemp(prefix='breakwater-evaluation-test-'))
        self.addCleanup(shutil.rmtree, root, True)
        submission, evaluation, external, validation = (root / name for name in
                                                        ('submission', 'evaluation', 'external', 'validation'))
        for folder in (submission, evaluation, external, validation):
            folder.mkdir()
        (submission / 'generation_model.py').write_text('# submitted\n')
        contract = root / 'contract.json'
        contract.write_text('{}')
        (validation / 'secret.txt').write_text('labels')
        path = {'PATH': '/usr/local/bin:/usr/bin:/bin'}
        record = EvaluationStage(image=IMAGE, task=REPO / 'examples/word-count', contract=contract,
                                 submission=submission,
                                 workdir='/app', evaluation=evaluation, external=external, worker_env=path,
                                 coordinator_env=path, timeout=300,
                                 private_binds=[(validation, '/private/validation', 'ro')],
                                 coordinator_argv=['python3', '-c', COORDINATOR, WORKER]).run()
        self.assertEqual((record['exit_code'], record['timed_out']), (0, False), (external / 'intermediate.stderr').read_text())
        self.assertEqual(record['process_wall'], {'coordinator': True, 'submission': True})
        result = json.loads((evaluation / 'result.json').read_text())
        self.assertEqual((result['secret'], result['reply'], result['status']), ('labels', 'pong:ping', 0))
        worker = result['worker']
        self.assertEqual((worker['private'], worker['tests'], worker['evaluation']), (False, False, False))
        self.assertEqual((worker['cwd'], worker['inputs'], worker['app']), ('/app', 'answer-free', ['generation_model.py']))
        journal = [json.loads(line) for line in (external / 'evaluation-commands/exec.jsonl').read_text().splitlines()]
        self.assertEqual([row['event'] for row in journal][::len(journal) - 1], ['broker_started', 'broker_stopped'])
        self.assertEqual(journal[1]['argv'][:2], ['python3', '-c'])
        self.assertEqual({box['role'] for box in record['sandboxes']}, {'coordinator', 'submission'})
        self.assertTrue(record['commands_drained'])
        self.assertTrue((external / 'intermediate.strace').read_text())


if __name__ == '__main__':
    unittest.main()
