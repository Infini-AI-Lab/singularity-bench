"""A coding-agent-harness grid cell (apptainer_cell.py), from authoring through the verifier's checks.

    BREAKWATER_APPTAINER_IMAGE=IMAGE.sif python3 -m unittest discover -s tasks/coding-agent-harness/grid/tests -v

A downstream task needs its GPU image, so a toy one stands in: the test image's /etc/skel is its workspace, a toy
harness (one model call; one file added, one changed, one deleted) is the sealed submission of a stand-in root run
that passed, a stub provider is the driver gateway's upstream, and a stand-in coordinator applies the change set.
The rest is the real cell: the purge check, the driver gateway and its probe, the trusted cell driver as one traced
broker command, the change set, the seals and the verifier sandbox, whose mechanical checks must all pass before it
refuses to score a task outside the downstream registry. A probe the endpoint rejects voids the cell first. The test
image need not be a cell image: the purge check looks for names no image has, and a separate test points it at `sh`,
which every image has, to see the refusal.
"""
import http.server
import importlib.util
import json
import os
from pathlib import Path
import sys
import tempfile
import threading
import unittest
from unittest import mock

GRID = Path(__file__).resolve().parents[1]
FRAMEWORK = GRID.parents[2] / 'breakwater'
# The Apptainer backend's modules, and its test helpers (the test image and whether the backend can run here).
sys.path[:0] = [str(FRAMEWORK / 'apptainer'), str(FRAMEWORK / 'apptainer' / 'tests'), str(FRAMEWORK)]
from test_agent_stage import IMAGE, READY  # noqa: E402

HARNESS = r'''
import os
def run(task, workspace, model, budget):
    reply = model.complete([{'role': 'user', 'content': 'Reply with done.'}])
    with open(os.path.join(workspace, 'toy_result.txt'), 'w') as stream:
        stream.write(reply['choices'][0]['message']['content'] + '\n')
    os.remove(os.path.join(workspace, '.bash_logout'))
    with open(os.path.join(workspace, '.profile'), 'a') as stream:
        stream.write('# edited\n')
    print('harness finished', flush=True)
'''

# Stands in for cell_intermediate.py, which refuses a task outside the downstream registry.
COORDINATOR = r'''
import json, os, shutil
changes = json.load(open('/submission/changes.json'))
for name in changes['added'] + changes['modified']:
    shutil.copyfile('/submission/files/' + name, '/etc/skel/' + name)
for name in changes['deleted']:
    os.remove('/etc/skel/' + name)
os.makedirs('/evaluation/raw')
json.dump({'workspace': sorted(os.listdir('/etc/skel'))}, open('/evaluation/raw/toy.json', 'w'))
json.dump({'complete': True}, open('/evaluation/complete.json', 'w'))
'''


def load_cell():
    spec = importlib.util.spec_from_file_location('breakwater_apptainer_cell', GRID / 'apptainer_cell.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class Upstream(http.server.BaseHTTPRequestHandler):
    """An OpenAI-compatible provider answering every chat call with ``status`` (200: a one-word completion)."""

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
        self.server.calls.append({'model': body.get('model'), 'max_tokens': body.get('max_tokens'),
                                  'authorization': self.headers.get('Authorization')})
        if self.server.status == 200:
            out = {'id': 'x', 'object': 'chat.completion', 'model': body.get('model'),
                   'choices': [{'index': 0, 'message': {'role': 'assistant', 'content': 'done'}, 'finish_reason': 'stop'}],
                   'usage': {'prompt_tokens': 7, 'completion_tokens': 1, 'total_tokens': 8}}
        else:
            out = {'error': {'type': 'invalid_api_key'}}
        data = json.dumps(out).encode()
        self.send_response(self.server.status)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Content-Length', str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, *args):
        pass


@unittest.skipUnless(READY, 'set BREAKWATER_APPTAINER_IMAGE to a SIF with bash and python3; the cell is NOT verified')
class CellTests(unittest.TestCase):
    def run_cell(self, status):
        cell = load_cell()
        from contract import seal
        import spec
        root = Path(tempfile.mkdtemp(prefix='bw-cell-'))
        self.addCleanup(lambda: os.system(f'chmod -R u+w {root} 2>/dev/null; rm -rf {root}'))
        upstream = http.server.ThreadingHTTPServer(('127.0.0.1', 0), Upstream)
        upstream.calls, upstream.status = [], status
        threading.Thread(target=upstream.serve_forever, daemon=True).start()
        self.addCleanup(upstream.server_close)
        self.addCleanup(upstream.shutdown)
        # A root harness run that passed its own verification.
        (root / 'root-run' / 'submission' / 'harness').mkdir(parents=True)
        (root / 'root-run' / 'submission' / 'harness' / 'agent.py').write_text(HARNESS)
        identity = {'task': spec.load(cell.TASK)['task'], 'run_id': 'f' * 32}
        submission_sha = seal(root / 'root-run' / 'submission', identity)
        (root / 'root-run' / 'authority').mkdir()
        authority_sha = seal(root / 'root-run' / 'authority', identity, submission_sha256=submission_sha)
        (root / 'root-run' / 'reports').mkdir()
        (root / 'root-run' / 'reports' / 'reward.json').write_text('{"reward": 1.0}')
        (root / 'root-run' / 'reports' / 'judge.json').write_text('{"verdict": "PASS"}')
        task = root / 'toy-task'
        (task / 'tests').mkdir(parents=True)
        (task / 'task.toml').write_text('[agent]\ntimeout_sec = 120\n[verifier]\ntimeout_sec = 300\n')
        (task / 'instruction.md').write_text('Write toy_result.txt.\n')
        (task / 'judge').mkdir()
        read_json = cell.read_json

        class Authoring(cell.AuthoringStage):
            def __init__(self, **options):
                for service in options['services']:
                    service['env']['GATEWAY_ALLOW_HTTP_UPSTREAM'] = '1'
                super().__init__(**options)

        class Evaluation(cell.EvaluationStage):
            def __init__(self, **options):
                super().__init__(**{**options, 'coordinator_argv': ['python3', '-I', '-c', COORDINATOR]})

        api_base = f'http://127.0.0.1:{upstream.server_port}/v1'
        patches = mock.patch.multiple(
            cell, require_task=lambda name: name, WORKDIRS={'toy': '/etc/skel'}, task_path=lambda name: str(task),
            PURGED=('breakwater-test-absent-cli',),
            read_json=lambda path: ({**read_json(path), 'api_base': api_base} if str(path).endswith('eval/config.json')
                                    else read_json(path)),
            AuthoringStage=Authoring, EvaluationStage=Evaluation)
        argv = ['cell.py', '--task', 'toy', '--driver', 'glm53', '--harness-run', str(root / 'root-run'),
                '--harness-authority', authority_sha, '--image', IMAGE, '--judge-dir', str(task / 'judge'),
                '--run-dir', str(root / 'cell'), '--wall-sec', '120', '--gpus', '0']
        with patches, mock.patch.object(sys, 'argv', argv), mock.patch.dict(os.environ, {'FIREWORKS_API_KEY': 'stub-key'}):
            code = cell.main()
        return code, root / 'cell', upstream.calls

    def test_a_cell_runs_the_harness_and_passes_every_mechanical_check(self):
        code, run, calls = self.run_cell(200)
        error = json.loads((run / 'reports' / 'verification-error.json').read_text())['error']
        self.assertEqual(code, 2)
        self.assertIn("No protected raw protocol for downstream task 'toy'", error)
        # The probe, then the harness's call: both through the gateway, which pinned the driver and held the key.
        self.assertEqual([call['max_tokens'] for call in calls], [1, 8192])
        self.assertEqual({call['model'] for call in calls}, {'accounts/fireworks/models/glm-5p3'})
        self.assertEqual({call['authorization'] for call in calls}, {'Bearer stub-key'})
        changes = json.loads((run / 'submission' / 'changes.json').read_text())
        self.assertEqual(changes, {'workdir': '/etc/skel', 'added': ['toy_result.txt'], 'modified': ['.profile'],
                                   'deleted': ['.bash_logout'], 'executable': []})
        self.assertEqual((run / 'submission' / 'files' / 'toy_result.txt').read_text(), 'done\n')
        workspace = json.loads((run / 'evaluation' / 'raw' / 'toy.json').read_text())['workspace']
        self.assertIn('toy_result.txt', workspace)
        self.assertNotIn('.bash_logout', workspace)
        mechanical = json.loads((run / 'reports' / 'mechanical-check.json').read_text())
        self.assertEqual(mechanical['native_agreement']['status'], 'PASS')
        self.assertEqual(mechanical['runtime_integrity']['status'], 'PASS')
        self.assertEqual((mechanical['driver-gateway']['calls'], mechanical['driver-gateway']['statuses']), (2, {'200': 2}))
        self.assertEqual(mechanical['overlay'], {'added': 1, 'modified': 1, 'deleted': 1, 'executable': 0})
        self.assertEqual(mechanical['driver_probe_status'], 200)
        self.assertIn('the trusted launcher of the agent', mechanical['established']['three_containers'])
        self.assertIn('only through the driver-gateway gateway', mechanical['established']['log_isolation'])
        self.assertIn('harness finished', (run / 'external' / 'agent-native.stdout').read_text())
        self.assertFalse((run / 'workspace-live').exists() or (run / 'workspace-pristine').exists())

    def test_an_image_with_an_agent_cli_or_javascript_runtime_is_refused(self):
        cell = load_cell()
        from contract import ContractError
        with mock.patch.object(cell, 'PURGED', ('sh',)), self.assertRaises(ContractError) as refused:
            cell.inspect_image(IMAGE)
        self.assertIn('sh', str(refused.exception))
        self.assertIn('grid/images/apptainer.def', str(refused.exception))
        with mock.patch.object(cell, 'PURGED', ('breakwater-test-absent-cli',)):
            self.assertIn(cell.inspect_image(IMAGE), (True, False))  # whether the image has the moved HF cache

    def test_a_rejected_probe_voids_the_cell_before_the_harness_runs(self):
        code, run, calls = self.run_cell(401)
        self.assertEqual(code, 2)
        self.assertEqual(len(calls), 1)
        self.assertEqual(json.loads((run / 'reports' / 'void.json').read_text())['status'], 'VOID')
        self.assertEqual(json.loads((run / 'reports' / 'pipeline-error.json').read_text())['status'], 'VOID')
        self.assertFalse((run / 'submission').exists() or (run / 'workspace-live').exists())


if __name__ == '__main__':
    unittest.main()
