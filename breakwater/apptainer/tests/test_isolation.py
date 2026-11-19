"""Live tests of the Apptainer backend's isolation core.

    BREAKWATER_APPTAINER_IMAGE=IMAGE.sif python3 -m unittest discover -s breakwater/apptainer/tests -v

IMAGE.sif needs bash and python3. Without it (or without apptainer, nsenter and strace) the tests skip; a skip is
not evidence of isolation.
"""
import hashlib
import json
import os
from pathlib import Path
import shutil
import socket
import subprocess
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
COMMAND_ENV = {'PATH': '/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin', 'HOME': '/tmp',
               'LANG': 'C.UTF-8'}
HTTP = ("import sys, urllib.request\n"
        "try:\n    print(urllib.request.urlopen('http://127.0.0.1:8001/', timeout=3).read().decode())\n"
        "except Exception as error:\n    print('refused', type(error).__name__); sys.exit(1)\n")


def gateway(path):
    """A stand-in model gateway: HTTP over a unix socket answering 'gateway'."""
    server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    server.bind(str(path))
    server.listen(8)

    def serve():
        while True:
            try:
                connection, _ = server.accept()
            except OSError:
                return
            with connection:
                connection.recv(65536)
                connection.sendall(b'HTTP/1.0 200 OK\r\nContent-Length: 7\r\n\r\ngateway')
    threading.Thread(target=serve, daemon=True).start()
    return server


@unittest.skipUnless(READY, 'set BREAKWATER_APPTAINER_IMAGE to a SIF with bash and python3; isolation is NOT verified')
class IsolationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from broker import Broker
        from sandbox import Sandbox
        cls.root = Path(tempfile.mkdtemp(prefix='breakwater-isolation-'))
        cls.workspace, cls.evidence = cls.root / 'workspace', cls.root / 'evidence'
        cls.exec_dir, cls.gateway_dir, cls.cli = cls.root / 'exec', cls.root / 'gateway', cls.root / 'author-cli'
        for folder in (cls.workspace, cls.exec_dir, cls.gateway_dir, cls.cli):
            folder.mkdir(mode=0o700)
        (cls.cli / 'settings.json').write_text('{}')
        cls.server = gateway(cls.gateway_dir / 'author.sock')
        cls.worker = Sandbox(IMAGE, binds=[(cls.workspace, '/app', 'rw')])
        cls.broker = Broker(cls.exec_dir / 'exec.sock', cls.worker, cls.evidence, env=COMMAND_ENV)
        cls.author = Sandbox(IMAGE, binds=[(cls.workspace, '/app', 'rw'), (cls.cli, '/opt/author-cli', 'ro'),
                                           (cls.exec_dir, '/run/breakwater', 'ro'),
                                           (HERE / 'client.py', '/opt/breakwater-runtime/breakwater-exec', 'ro')])
        cls.relay = subprocess.Popen(cls.author.network_command([sys.executable, '-I', str(HERE / 'relay.py'), '8001',
                                                                 str(cls.gateway_dir / 'author.sock')]),
                                     stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        time.sleep(1)

    @classmethod
    def tearDownClass(cls):
        cls.relay.kill()
        for box in (cls.author, cls.worker):  # already stopped by test_zz when the whole class runs
            if box.alive():
                box.stop()
        cls.broker.close()
        cls.server.close()
        shutil.rmtree(cls.root, ignore_errors=True)

    def command(self, text, timeout=60):
        """What the agent CLI does: run the shell prefix, inside the author sandbox."""
        return self.author.run(['python3', '/opt/breakwater-runtime/breakwater-exec', text], cwd='/app',
                               timeout=timeout)

    def journal(self):
        return [json.loads(line) for line in (self.evidence / 'exec.jsonl').read_text().splitlines()]

    def sleepers(self, argument, wait=5):
        """PIDs of `sleep ARGUMENT` in the command sandbox. A background job can still be starting (just after
        fork or execve, or stopped under strace) when its command returns, so this waits up to `wait` seconds."""
        from sandbox import members
        deadline = time.monotonic() + wait
        while True:
            found = []
            for pid in members(self.worker.namespace):
                try:
                    if Path(f'/proc/{pid}/cmdline').read_bytes().split(b'\0')[:2] == [b'sleep', argument.encode()]:
                        found.append(pid)
                except OSError:
                    continue
            if found or time.monotonic() > deadline:
                return found
            time.sleep(0.1)

    def test_status_streams_and_evidence(self):
        code, out, err = self.command('echo out; echo err >&2; exit 3')
        self.assertEqual((code, out, err), (3, 'out\n', 'err\n'))
        number = max(r['id'] for r in self.journal() if r['event'] == 'exec_exited' and r['status'] == 3)
        deadline = time.monotonic() + 10
        while not (self.evidence / f'exec-{number}.stdout.sha256').exists() and time.monotonic() < deadline:
            time.sleep(0.1)
        self.assertEqual((self.evidence / f'exec-{number}.stdout').read_text(), 'out\n')
        self.assertEqual((self.evidence / f'exec-{number}.stderr').read_text(), 'err\n')
        self.assertIn('execve', (self.evidence / f'exec-{number}.strace').read_text())
        records = self.journal()
        previous = '0' * 64
        for record in records:  # the journal's hash chain is unbroken
            self.assertEqual(record['prev'], previous)
            previous = hashlib.sha256(json.dumps(record, sort_keys=True).encode()).hexdigest()

    def test_all_output_written_before_exit_reaches_the_client(self):
        # The status and the output travel on separate pipes; the answer must wait for the output.
        for _ in range(10):
            code, out, _ = self.command("head -c 300000 /dev/zero | tr '\\0' x; echo; echo end-of-output")
            self.assertEqual((code, len(out), out[-15:]), (0, 300015, '\nend-of-output\n'))
        # A background job's later output is recorded, but no longer sent to the client.
        code, out, _ = self.command('(sleep 2; echo late) & echo now')
        self.assertEqual((code, out), (0, 'now\n'))
        number = max(r['id'] for r in self.journal() if r['event'] == 'exec_exited')
        deadline = time.monotonic() + 20
        while not (self.evidence / f'exec-{number}.stdout.sha256').exists() and time.monotonic() < deadline:
            time.sleep(0.2)
        self.assertEqual((self.evidence / f'exec-{number}.stdout').read_text(), 'now\nlate\n')

    def test_commands_reach_their_own_hostname_on_loopback(self):
        # What torchrun --standalone does: resolve the hostname and rendezvous on it, with loopback only.
        probe = ('import socket; name = socket.gethostname(); address = socket.gethostbyname(name); '
                 'server = socket.socket(); server.bind((address, 0)); server.listen(1); '
                 'socket.create_connection((name, server.getsockname()[1]), timeout=3); print(name, address)')
        code, out, _ = self.command(f'python3 -c "{probe}"')
        self.assertEqual((code, out.split()), (0, [self.worker.hostname, '127.0.0.1']))

    def test_commands_run_in_the_command_sandbox(self):
        code, out, _ = self.command('ls /opt/author-cli /run/breakwater 2>&1; echo cwd=$PWD; echo built > model.txt')
        self.assertEqual(code, 0)
        self.assertEqual(out.count('No such file or directory'), 2, out)
        self.assertIn('cwd=/app', out)
        self.assertEqual((self.workspace / 'model.txt').read_text(), 'built\n')
        code, out, _ = self.command(f'kill -0 {self.author.pid} 2>&1 || echo invisible')
        self.assertIn('invisible', out)

    def test_only_the_author_reaches_the_gateway(self):
        code, out, _ = self.author.run(['python3', '-c', HTTP])
        self.assertEqual((code, out.strip()), (0, 'gateway'))
        code, out, _ = self.command(f"python3 -c \"{HTTP.replace(chr(34), chr(39))}\"")
        self.assertNotEqual(code, 0)
        self.assertIn('refused', out)

    def test_author_files_and_image_stay_read_only(self):
        code, out, _ = self.author.run(['sh', '-c', 'echo x > /opt/author-cli/settings.json 2>&1; '
                                        'mount -o remount,rw /opt/author-cli 2>&1; touch /usr/bin/planted 2>&1; true'])
        self.assertEqual((self.cli / 'settings.json').read_text(), '{}')
        code, out, _ = self.command('touch /usr/bin/planted 2>&1; ls /usr/bin/planted 2>&1')
        self.assertIn('No such file or directory', out)

    def test_background_job_returns_promptly_and_persists(self):
        started = time.monotonic()
        code, out, _ = self.command('setsid sleep 600 >/dev/null 2>&1 < /dev/null & echo started')
        self.assertEqual((code, out.strip()), (0, 'started'))
        self.assertLess(time.monotonic() - started, 15)
        self.assertTrue(self.sleepers('600'))

    def test_killing_the_client_stops_the_command(self):
        client = subprocess.Popen(self.author.enter_command(['python3', '/opt/breakwater-runtime/breakwater-exec',
                                                             'sleep 777'], cwd='/app'),
                                  stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        from sandbox import members

        def running():
            return [pid for pid in members(self.worker.namespace)
                    if Path(f'/proc/{pid}/cmdline').read_bytes().split(b'\0')[:2] == [b'sleep', b'777']]
        deadline = time.monotonic() + 20
        while not running() and time.monotonic() < deadline:
            time.sleep(0.2)
        self.assertTrue(running())
        # Kill the client itself, inside the author sandbox, as the CLI would.
        clients = [pid for pid in members(self.author.namespace)
                   if b'sleep 777' in Path(f'/proc/{pid}/cmdline').read_bytes() and
                   b'breakwater-exec' in Path(f'/proc/{pid}/cmdline').read_bytes()]
        self.assertTrue(clients)
        for pid in clients:
            os.kill(pid, 9)
        client.wait(timeout=20)
        deadline = time.monotonic() + 20
        while running() and time.monotonic() < deadline:
            time.sleep(0.2)
        self.assertFalse(running())

    def test_commands_share_the_sandbox_cgroup(self):
        # With a Slurm prefix, the spawner's cgroup is the step's, so this is what makes the step's memory limit
        # cover every command.
        self.command('setsid sleep 812 >/dev/null 2>&1 < /dev/null &')
        sleepers = self.sleepers('812')
        self.assertTrue(sleepers)
        self.assertEqual(Path(f'/proc/{sleepers[0]}/cgroup').read_text(),
                         Path(f'/proc/{self.worker.process.pid}/cgroup').read_text())
        for pid in sleepers:
            os.kill(pid, 9)

    def test_lost_command_sandbox_is_reported(self):
        # What the launcher sees when Slurm kills an out-of-memory step: the spawner and the container are gone.
        from broker import Broker
        from sandbox import Sandbox
        evidence = self.root / 'lost-evidence'
        lost = Sandbox(IMAGE, binds=[(self.workspace, '/app', 'rw')])
        broker = Broker(self.exec_dir / 'lost.sock', lost, evidence, env=COMMAND_ENV)
        try:
            os.killpg(lost.process.pid, 9)
            os.kill(lost.pid, 9)
            lost.process.wait(timeout=30)
            self.assertFalse(lost.alive())
            with open(os.devnull, 'rb') as null_in, open(os.devnull, 'wb') as null_out:
                status = broker.execute('echo unreachable', '/app', null_in.fileno(), null_out.fileno(),
                                        null_out.fileno())
            self.assertEqual(status, 137)
            events = [json.loads(line)['event'] for line in (evidence / 'exec.jsonl').read_text().splitlines()]
            self.assertIn('sandbox_unavailable', events)
            self.assertTrue(lost.prove_empty())
        finally:
            broker.close()
            lost.stop()

    def test_zz_stopping_the_stage_leaves_no_process(self):
        self.command('setsid sleep 900 >/dev/null 2>&1 < /dev/null &')
        self.assertTrue(self.author.stop())
        self.assertTrue(self.worker.stop())


if __name__ == '__main__':
    unittest.main()
