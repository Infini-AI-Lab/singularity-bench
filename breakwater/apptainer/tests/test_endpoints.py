"""A task's static endpoint and inbound ports on the Apptainer backend ([[breakwater.endpoints]] and [breakwater.authoring]
inbound_ports, through plan.py into agent_stage.AuthoringStage).

    BREAKWATER_APPTAINER_IMAGE=IMAGE.sif python3 -m unittest discover -s breakwater/apptainer/tests -v

A stand-in service listens on the host's loopback at a port of its own (the operator's --endpoint). The task declares
it as store:11000 with a health path; the agent's commands reach it as store:11000 (endpoints.EndpointRelay with a port
map). A server the commands start in the command sandbox on 0.0.0.0:PORT, an inbound port, is reached from the host's
127.0.0.1:PORT, as containers that a host-side service starts reach it through their host name, and from inside the
sandbox as the inbound host's name. After the command sandbox is lost and replaced, a new server on the same port is
reached through the replacement.

With BREAKWATER_TEST_PODMAN_IMAGE set (an image with wget, such as docker.io/library/alpine, in a rootless podman
store), a rootless podman container on slirp4netns also fetches that server at host.docker.internal:PORT.
"""
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
import urllib.request

HERE = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(HERE), str(HERE / 'tests'), str(HERE.parent)]
from test_agent_stage import IMAGE, READY  # noqa: E402

# Runs the agent's commands through the broker; markers in the workspace pace it against the host-side checks.
CLI = r'''
import json, os, subprocess, sys, time
port = int(sys.argv[1])
def emit(**event):
    print(json.dumps(event), flush=True)
def run(command):
    result = subprocess.run(['/opt/breakwater-runtime/breakwater-exec', command], capture_output=True, text=True)
    emit(type='tool_result', command=command, status=result.returncode, stdout=result.stdout)
def wait_for(name):
    while not os.path.exists(name):
        time.sleep(0.05)
fetch = 'python3 -c "import urllib.request,sys; print(urllib.request.urlopen(sys.argv[1], timeout=10).read().decode())" '
serve = ("setsid python3 -c \"import http.server as h, sys\n"
         "class S(h.BaseHTTPRequestHandler):\n"
         "    def do_GET(self):\n"
         "        body = sys.argv[1].encode(); self.send_response(200); self.send_header('Content-Length', str(len(body)));"
         " self.end_headers(); self.wfile.write(body)\n"
         "    def log_message(self, *a): pass\n"
         "h.ThreadingHTTPServer(('0.0.0.0', int(sys.argv[2])), S).serve_forever()\" ")
emit(type='system', subtype='init')
run(fetch + 'http://store:11000/ping')
run(serve + f'session-1 {port} >/dev/null 2>&1 < /dev/null & sleep 1; echo started')
run(fetch + f'http://launcher.internal:{port}/')
open('probe-1', 'w').close(); wait_for('probed-1')
open('kill-me', 'w').close(); wait_for('killed')
run(serve + f'session-2 {port} >/dev/null 2>&1 < /dev/null & sleep 1; echo started')
open('probe-2', 'w').close(); wait_for('probed-2')
emit(type='result', subtype='success')
'''


class Worker(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        body = {'/health': b'ok', '/ping': b'pong'}.get(self.path)
        self.send_response(200 if body else 404)
        self.send_header('Content-Length', str(len(body or b'')))
        self.end_headers()
        self.wfile.write(body or b'')

    def log_message(self, *args):
        pass


def free_port():
    with socket.socket() as probe:
        probe.bind(('127.0.0.1', 0))
        return probe.getsockname()[1]


def config(service_port, inbound):
    """A resolved contract's endpoint and inbound settings, as spec.resolve gives them."""
    return {'endpoints': [{'name': 'store', 'port': 11000, 'service': '127.0.0.1:11000', 'reach': ['agent'],
                           'health': '/health'}], 'inbound_ports': f'{inbound}-{inbound}', 'inbound_host': 'launcher.internal'}


@unittest.skipUnless(READY, 'set BREAKWATER_APPTAINER_IMAGE to a SIF with bash and python3; the stage is NOT verified')
class EndpointStageTests(unittest.TestCase):
    def test_an_endpoint_and_inbound_ports_reach_through_relays_and_survive_a_replacement(self):
        from agent_stage import AuthoringStage
        import plan
        worker = http.server.ThreadingHTTPServer(('127.0.0.1', 0), Worker)
        threading.Thread(target=worker.serve_forever, daemon=True).start()
        self.addCleanup(worker.server_close)
        self.addCleanup(worker.shutdown)
        root = Path(tempfile.mkdtemp(prefix='breakwater-worker-'))
        self.addCleanup(shutil.rmtree, root, True)
        workspace, cli = root / 'workspace', root / 'cli'
        for folder in (workspace, cli):
            folder.mkdir()
        (cli / 'cli.py').write_text(CLI)
        session = free_port()
        declared = config(11000, session)
        endpoints = plan.static_endpoints(declared, [f'store=127.0.0.1:{worker.server_address[1]}'])
        plan.check_health(endpoints)
        stage = AuthoringStage(
            image=IMAGE, workspace=workspace, workdir='/app', external=root / 'external', local=root / 'local',
            cli_dir=cli, cli_argv=['python3', '/opt/author-cli/cli.py', str(session)],
            cli_env={'PATH': '/usr/local/bin:/usr/bin:/bin', 'HOME': '/home/author'}, gateway=None, budget=180,
            command_env={'PATH': '/usr/local/bin:/usr/bin:/bin'},
            endpoints=plan.by_role(endpoints)['agent'], hosts={declared['inbound_host']: '127.0.0.1'},
            inbound_ports=[session])
        probes = {}
        podman_image = os.environ.get('BREAKWATER_TEST_PODMAN_IMAGE')

        def host_side():
            # What an episode sandbox does: connect to the host's loopback on the session port.
            opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
            for number in (1, 2):
                deadline = time.monotonic() + 120
                while not (workspace / f'probe-{number}').exists() and time.monotonic() < deadline:
                    time.sleep(0.05)
                try:
                    probes[number] = opener.open(f'http://127.0.0.1:{session}/', timeout=10).read().decode()
                except OSError as error:
                    probes[number] = f'failed: {error}'
                if podman_image:  # what an episode sandbox of the rootless-podman worker does
                    result = subprocess.run(['podman', 'run', '--rm', '--pid', 'private', '--network',
                                             'slirp4netns:allow_host_loopback=true', '--add-host',
                                             'host.docker.internal:10.0.2.2', podman_image, 'wget', '-qO-',
                                             f'http://host.docker.internal:{session}/'],
                                            capture_output=True, text=True, timeout=120)
                    probes[f'podman-{number}'] = result.stdout.strip() or f'failed: {result.stderr.strip()[-300:]}'
                (workspace / f'probed-{number}').touch()
                if number == 1:
                    deadline = time.monotonic() + 120
                    while not (workspace / 'kill-me').exists() and time.monotonic() < deadline:
                        time.sleep(0.05)
                    victim = stage.sandboxes[-1]  # what Slurm does to an out-of-memory step
                    os.killpg(victim.process.pid, signal.SIGKILL)
                    os.kill(victim.pid, signal.SIGKILL)
                    (workspace / 'killed').touch()

        threading.Thread(target=host_side, daemon=True).start()
        record = stage.run()
        events = [json.loads(line) for line in (root / 'external/agent-native.jsonl').read_text().splitlines() if line]
        results = [event for event in events if event.get('type') == 'tool_result']
        self.assertEqual(record['exit_code'], 0)
        self.assertEqual(record['command_sandbox_restarts'], 1)
        self.assertEqual(record['process_wall'], {'author': True, 'command': True})
        self.assertEqual(record['inbound_ports'], [session])
        self.assertEqual(record['endpoints'], [{'name': 'store', 'ports': 1}])
        self.assertEqual([(line['stdout'].strip(), line['status']) for line in results],
                         [('pong', 0), ('started', 0), ('session-1', 0), ('started', 0)])
        expected = {1: 'session-1', 2: 'session-2'}
        if podman_image:
            expected.update({'podman-1': 'session-1', 'podman-2': 'session-2'})
        self.assertEqual(probes, expected)



class EndpointPlanTests(unittest.TestCase):
    """plan.py: a task's static endpoints with the operator's overrides, their health, roles and port ranges."""

    def test_overrides_health_roles_and_ranges(self):
        import plan
        worker = http.server.ThreadingHTTPServer(('127.0.0.1', 0), Worker)
        threading.Thread(target=worker.serve_forever, daemon=True).start()
        self.addCleanup(worker.server_close)
        self.addCleanup(worker.shutdown)
        declared = config(11000, 30000)
        declared['endpoints'].append({'name': 'store', 'port': 11001, 'service': '127.0.0.1:11001',
                                      'reach': ['coordinator'], 'health': None})
        declared['endpoints'][1]['name'] = 'archive'
        rows = plan.static_endpoints(declared, [f'store=127.0.0.1:{worker.server_address[1]}'])
        self.assertEqual(rows[0], {'name': 'store', 'host': '127.0.0.1', 'ports': [11000],
                                   'port_map': {11000: worker.server_address[1]}, 'reach': ['agent'], 'health': '/health'})
        self.assertIsNone(rows[1]['port_map'])
        plan.check_health(rows)
        roles = plan.by_role(rows)
        self.assertEqual(([row['name'] for row in roles['agent']], [row['name'] for row in roles['coordinator']],
                          roles['submission']), (['store'], ['archive'], []))
        self.assertEqual(plan.ranges([30003, 30000, 30001, 30002, 9000]), ['9000', '30000:30003'])
        self.assertTrue(plan.launcher_host('127.0.0.1') and plan.launcher_host('localhost'))
        self.assertFalse(plan.launcher_host('store.internal'))

    def test_an_unhealthy_endpoint_or_a_bad_override_is_refused(self):
        import plan
        from contract import ContractError
        declared = config(11000, 30000)
        with self.assertRaisesRegex(ContractError, 'start it first'):
            plan.check_health(plan.static_endpoints(declared, [f'store=127.0.0.1:{free_port()}']))
        for values in (['store'], ['store=127.0.0.1'], ['other=127.0.0.1:1'], ['store=h:1', 'store=h:2']):
            with self.subTest(values=values), self.assertRaises(ContractError):
                plan.static_endpoints(declared, values)
        merged = [{'name': 'a', 'host': 'h1', 'ports': [1], 'port_map': None, 'reach': ['agent']},
                  {'name': 'a', 'host': 'h2', 'ports': [2], 'port_map': None, 'reach': ['agent']}]
        with self.assertRaisesRegex(ContractError, 'two hosts'):
            plan.by_role(merged)


if __name__ == '__main__':
    unittest.main()
