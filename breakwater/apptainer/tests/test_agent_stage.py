"""Live tests of the Apptainer authoring stage with a stand-in agent CLI and a stand-in model provider.

    BREAKWATER_APPTAINER_IMAGE=IMAGE.sif python3 -m unittest discover -s breakwater/apptainer/tests -v

IMAGE.sif needs bash and python3. The stand-in CLI calls the real author gateway through the relay and runs commands
through the real shell-prefix client and broker, so everything but the model and the CLI itself is exercised.
"""
import http.server
import json
import os
from pathlib import Path
import shutil
import signal
import sys
import tempfile
import threading
import time
import unittest

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
IMAGE = os.environ.get('BREAKWATER_APPTAINER_IMAGE')
READY = bool(IMAGE and Path(IMAGE).is_file() and (shutil.which('apptainer') or shutil.which('singularity'))
             and shutil.which('nsenter') and shutil.which('strace'))
FAKE_CLI = r'''
import json, os, subprocess, sys, time, urllib.request
mode = sys.argv[1]
def emit(**event):
    print(json.dumps(event), flush=True)
emit(type='system', subtype='init', cwd=os.getcwd())
if mode == 'hang':
    time.sleep(3600)
request = urllib.request.Request('http://127.0.0.1:8001/v1/messages', headers={'Content-Type': 'application/json'},
    data=json.dumps({'model': 'test/author', 'max_tokens': 8, 'messages': [{'role': 'user', 'content': 'ok'}]}).encode())
emit(type='assistant', reply=json.loads(urllib.request.urlopen(request, timeout=30).read())['content'][0]['text'])
calls = []
def run(command):
    # Each command as Claude Code records it (the verifier compares these), then a summary line for the tests.
    calls.append(f'toolu_{len(calls) + 1}')
    emit(type='assistant', message={'content': [{'type': 'tool_use', 'id': calls[-1], 'name': 'Bash',
                                                 'input': {'command': command}}]})
    result = subprocess.run(['/opt/breakwater-runtime/breakwater-exec', command], capture_output=True, text=True)
    emit(type='user', message={'content': [{'type': 'tool_result', 'tool_use_id': calls[-1], 'content': result.stdout}]})
    emit(type='tool_result', command=command, status=result.returncode, stdout=result.stdout)
run('echo trained > model.txt; echo done')
run('setsid sleep 901 >/dev/null 2>&1 < /dev/null &')
if mode == 'restart':
    # The next command right after the loss, as an agent sends it: it waits for the replacement sandbox.
    open('kill-me', 'w').close()
    while not os.path.exists('killed'):
        time.sleep(0.05)
    run('echo after-restart')
if mode == 'service':
    fetch = 'python3 -c "import os, urllib.request; print(urllib.request.urlopen(os.environ[\\"TASK_URL\\"] + \\"/models\\", timeout=10).read().decode())"'
    run(fetch)
    open('kill-me', 'w').close()
    time.sleep(8)
    run(fetch)
if mode == 'launcher':
    run('grep -c image_sha256 /opt/environment-baseline/environment.json; '
        'test -w /opt/environment-baseline/environment.json || echo read-only')
if mode == 'oom':
    run('python3 -c "b = bytearray(6 * 2**30); print(len(b))"')
    run('echo after-restart')
emit(type='result', subtype='success')
'''


class Provider(http.server.BaseHTTPRequestHandler):
    """A stand-in model provider behind the real author gateway."""
    protocol_version = 'HTTP/1.0'

    def log_message(self, *_):
        pass

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
        raw = json.dumps({'id': 'msg_1', 'type': 'message', 'role': 'assistant', 'model': body['model'],
                          'content': [{'type': 'text', 'text': 'reply for ' + body['messages'][-1]['content']}],
                          'stop_reason': 'end_turn', 'usage': {'input_tokens': 3, 'output_tokens': 2}}).encode()
        self.send_response(200)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Content-Length', str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)


def sleepers(argument):
    found = []
    for entry in Path('/proc').iterdir():
        if entry.name.isdigit():
            try:
                if (entry / 'cmdline').read_bytes().split(b'\0')[:2] == [b'sleep', argument.encode()]:
                    found.append(int(entry.name))
            except OSError:
                continue
    return found


@unittest.skipUnless(READY, 'set BREAKWATER_APPTAINER_IMAGE to a SIF with bash and python3; the stage is NOT verified')
class AuthoringStageTests(unittest.TestCase):
    def setUp(self):
        self.provider = http.server.ThreadingHTTPServer(('127.0.0.1', 0), Provider)
        threading.Thread(target=self.provider.serve_forever, daemon=True).start()
        self.addCleanup(self.provider.server_close)
        self.addCleanup(self.provider.shutdown)
        self.root = Path(tempfile.mkdtemp(prefix='breakwater-stage-'))
        self.addCleanup(shutil.rmtree, self.root, True)
        self.workspace, self.cli = self.root / 'workspace', self.root / 'author-cli'
        for folder in (self.workspace, self.cli):
            folder.mkdir()
        (self.cli / 'fake-cli.py').write_text(FAKE_CLI)

    def stage(self, mode, budget=180, command_env=None, memory_mb=None, services=()):
        from agent_stage import AuthoringStage
        return AuthoringStage(image=IMAGE, workspace=self.workspace, workdir='/app', external=self.root / 'external',
                              local=self.root / 'local', cli_dir=self.cli,
                              cli_argv=['python3', '/opt/author-cli/fake-cli.py', mode],
                              cli_env={'PATH': '/usr/local/bin:/usr/bin:/bin', 'HOME': '/home/author'},
                              gateway={'model': 'test/author', 'key': 'sk-test-provider-key',
                                       'upstream': f'http://127.0.0.1:{self.provider.server_address[1]}'},
                              budget=budget, command_env=command_env, memory_mb=memory_mb, services=services)

    def events(self, path):
        return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]

    def test_a_full_stage_records_everything_and_leaves_nothing_running(self):
        record = self.stage('normal').run()  # the command environment comes from the image itself
        self.assertEqual((record['exit_code'], record['deadline_reached']), (0, False))
        self.assertEqual(record['process_wall'], {'author': True, 'command': True})
        native = (self.root / 'external/agent-native.jsonl').read_text()
        self.assertEqual(native, (self.root / 'local/native.jsonl').read_text())
        lines = [json.loads(line) for line in native.splitlines()]
        self.assertEqual(lines[1]['reply'], 'reply for ok')
        self.assertEqual([line['status'] for line in lines if line['type'] == 'tool_result'], [0, 0])
        self.assertEqual((self.workspace / 'model.txt').read_text(), 'trained\n')
        gateway = [event['event'] for event in self.events(self.root / 'external/gateway/requests.jsonl')]
        self.assertIn('model_request', gateway)
        self.assertIn('model_response_end', gateway)
        self.assertIn('gateway_stopped', gateway)
        commands = [event['event'] for event in self.events(self.root / 'external/commands/exec.jsonl')]
        self.assertEqual(commands.count('exec_started'), 2)
        self.assertNotIn(b'sk-test-provider-key', (self.root / 'external/agent-native.jsonl').read_bytes())
        self.assertEqual(sleepers('901'), [])  # the background job died with its sandbox

    def test_the_deadline_stops_a_cli_that_never_exits(self):
        started = time.monotonic()
        record = self.stage('hang', budget=5, command_env={'PATH': '/usr/local/bin:/usr/bin:/bin'}).run()
        self.assertTrue(record['deadline_reached'])
        self.assertEqual(record['process_wall'], {'author': True, 'command': True})
        self.assertLess(time.monotonic() - started, 90)

    def test_a_service_reaches_every_command_sandbox_under_its_host_name(self):
        # A host service on a unix socket (as the development gateway runs), relayed into the command sandbox at the
        # host and port the task declares (models:9400), which the task's authoring environment names; after the
        # command sandbox is lost and replaced, the replacement reaches it too.
        service = ('import http.server, os, socketserver\n'
                   'class H(http.server.BaseHTTPRequestHandler):\n'
                   '    def do_GET(self):\n'
                   '        body = ("served " + self.path).encode()\n'
                   '        self.send_response(200); self.send_header("Content-Length", str(len(body))); self.end_headers()\n'
                   '        self.wfile.write(body)\n'
                   '    def address_string(self):\n'
                   '        return "unix"\n'
                   '    def log_message(self, *_):\n'
                   '        pass\n'
                   'class S(socketserver.ThreadingMixIn, socketserver.UnixStreamServer):\n'
                   '    def get_request(self):\n'
                   '        return super().get_request()[0], ("unix", 0)\n'
                   'S(os.environ["GATEWAY_SOCKET"], H).serve_forever()\n')
        stage = self.stage('service', command_env={'PATH': '/usr/local/bin:/usr/bin:/bin', 'TASK_URL': 'http://models:9400/v1'},
                           services=[{'name': 'svc', 'port': 9400, 'host': 'models', 'argv': [sys.executable, '-c', service],
                                      'env': {'PATH': '/usr/bin:/bin'}}])

        def kill_when_asked():
            deadline = time.monotonic() + 120
            while not (self.workspace / 'kill-me').exists() and time.monotonic() < deadline:
                time.sleep(0.1)
            victim = stage.sandboxes[-1]
            os.killpg(victim.process.pid, signal.SIGKILL)
            os.kill(victim.pid, signal.SIGKILL)
        threading.Thread(target=kill_when_asked, daemon=True).start()
        record = stage.run()
        self.assertEqual((record['exit_code'], record['command_sandbox_restarts'], record['services']),
                         (0, 1, [{'name': 'svc', 'host': 'models', 'port': 9400}]))
        self.assertEqual(record['process_wall'], {'author': True, 'command': True})
        results = [line for line in self.events(self.root / 'external/agent-native.jsonl') if line['type'] == 'tool_result']
        self.assertEqual([(line['stdout'], line['status']) for line in results[-2:]], [('served /v1/models\n', 0)] * 2)

    def test_a_lost_command_sandbox_is_replaced_and_the_stage_continues(self):
        stage = self.stage('restart', command_env={'PATH': '/usr/local/bin:/usr/bin:/bin'})

        def kill_when_asked():  # what Slurm does to an out-of-memory step
            deadline = time.monotonic() + 120
            while not (self.workspace / 'kill-me').exists() and time.monotonic() < deadline:
                time.sleep(0.1)
            victim = stage.sandboxes[-1]
            os.killpg(victim.process.pid, signal.SIGKILL)
            os.kill(victim.pid, signal.SIGKILL)
            (self.workspace / 'killed').touch()
        threading.Thread(target=kill_when_asked, daemon=True).start()
        record = stage.run()
        self.assertEqual(record['exit_code'], 0)
        self.assertEqual(record['command_sandbox_restarts'], 1)
        results = [line for line in self.events(self.root / 'external/agent-native.jsonl') if line['type'] == 'tool_result']
        self.assertEqual((results[-1]['stdout'], results[-1]['status']), ('after-restart\n', 0))
        journal = [event['event'] for event in self.events(self.root / 'external/commands/exec.jsonl')]
        self.assertNotIn('sandbox_unavailable', journal)  # the command sent during the swap waited for it
        self.assertIn('command_sandbox_lost', journal)
        self.assertIn('command_sandbox_restarted', journal)

    @unittest.skipUnless(os.environ.get('SLURM_JOB_ID') and os.environ.get('SLURMD_NODENAME') and shutil.which('srun'),
                         'needs to run on a node of a Slurm allocation')
    def test_an_out_of_memory_command_is_killed_and_the_stage_continues(self):
        # The command sandbox runs as its own 4 GiB Slurm step, and a command allocates 6 GiB. With --oom-kill-step=0
        # only that command dies and the sandbox stays, as in Docker; without it, Slurm kills the whole step and the
        # stage replaces the sandbox.
        import agent_stage
        record = self.stage('oom', memory_mb=4096, command_env={'PATH': '/usr/local/bin:/usr/bin:/bin'}).run()
        self.assertEqual(record['exit_code'], 0)
        if agent_stage.oom_kill_step():
            self.assertEqual(record['command_sandbox_restarts'], 0)
        else:
            self.assertGreaterEqual(record['command_sandbox_restarts'], 1)
        results = [line for line in self.events(self.root / 'external/agent-native.jsonl') if line['type'] == 'tool_result']
        self.assertNotEqual(results[-2]['status'], 0)
        self.assertNotIn(str(6 * 2**30), results[-2]['stdout'])
        self.assertEqual((results[-1]['stdout'], results[-1]['status']), ('after-restart\n', 0))



class SlurmStepTests(unittest.TestCase):
    """A sandbox's memory-limited Slurm step stays on the launcher's node, whatever the job spans."""

    def step(self, oom_kill_step=True, **env):
        from unittest import mock
        import agent_stage
        with mock.patch.dict(os.environ, env, clear=True), mock.patch.object(shutil, 'which', lambda name: '/usr/bin/' + name), \
                mock.patch.object(agent_stage, 'oom_kill_step', lambda: oom_kill_step):
            return agent_stage.slurm_step(4096, cpus=8)

    def test_the_step_is_pinned_to_the_launchers_node(self):
        argv = self.step(SLURM_JOB_ID='7', SLURMD_NODENAME='gpu-017')
        self.assertEqual(argv[:2], ['srun', '--overlap'])
        for flag in ('--mem=4096M', '--nodes=1', '--ntasks=1', '--nodelist=gpu-017', '--cpu-bind=none', '--cpus-per-task=8',
                     '--oom-kill-step=0'):
            self.assertIn(flag, argv)
        # An older Slurm without the option: the step is made without it (and Slurm's own OOM policy applies).
        self.assertNotIn('--oom-kill-step=0', self.step(False, SLURM_JOB_ID='7', SLURMD_NODENAME='gpu-017'))
        self.assertEqual(self.step(SLURM_JOB_ID='7'), [])  # an salloc shell on a login node: a step would land elsewhere
        self.assertEqual(self.step(SLURMD_NODENAME='gpu-017'), [])  # outside a job: no step at all


if __name__ == '__main__':
    unittest.main()
