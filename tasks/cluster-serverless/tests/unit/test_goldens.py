"""cluster-serverless: what the judge sees of the fleet's inference journals (the task's review view), which fleet
endpoints each role may reach (the host hooks' plan, merged per role), and the hooks against a stand-in fleet. The
goldens were recorded while the framework still did this itself. No GPUs or Docker.
"""
import http.server
import json
from pathlib import Path
import shutil
import sys
import tempfile
import threading
from types import SimpleNamespace
import unittest
from unittest.mock import patch

FIXTURES = Path(__file__).resolve().parent / 'fixtures'
CLUSTER = Path(__file__).resolve().parents[2]
FRAMEWORK = CLUSTER.parents[1] / 'breakwater'
sys.path[:0] = [str(FRAMEWORK)]
sys.path += [str(CLUSTER / 'tests')]
JOURNAL = 'worker-' + 'a' * 32 + '.jsonl'


def load(name):
    import task_api
    return task_api.load_hook(CLUSTER / 'tests' / f'{name}.py', f'cluster_{name}')


class Host:
    """What a host hook is given, for calling the cluster's hooks in this process."""

    def __init__(self, inputs, backend='docker', state=None, evidence=None, **values):
        import host_hooks
        import spec
        self.task, self.backend, self.state = CLUSTER, backend, state
        self.contract = spec.load(CLUSTER)
        self.evidence = host_hooks.Evidence(evidence) if evidence else None
        self.inputs, self.values = inputs, {'receipt-key': 'receipt', 'grader-token': 'token', **values}

    def input(self, name):
        return Path(self.inputs[name])

    def value(self, name):
        return self.values[name]

    secret = value


class GoldenTests(unittest.TestCase):
    def test_the_judge_sees_the_fleet_journal_as_recorded(self):
        with tempfile.TemporaryDirectory() as temporary:
            evidence = Path(temporary)
            (evidence / 'task').mkdir()
            windows = json.loads((FIXTURES / 'fleet_journal_windows.json').read_text())
            (evidence / 'task/fleet-inference.json').write_text(json.dumps({'windows': windows}))
            view = load('breakwater_task').fleet_journal_view(FIXTURES / JOURNAL, SimpleNamespace(evidence=evidence))
        self.assertEqual(view, (FIXTURES / 'fleet_journal_index.txt').read_text())
        patterns = [pattern for pattern, _ in load('breakwater_task').review_views(None)]
        self.assertEqual(patterns, ['external/task/fleet-inference/*.jsonl'])

    def test_each_role_reaches_the_fleet_endpoints_recorded_for_it(self):
        import plan
        expected = json.loads((FIXTURES / 'fleet_endpoints.json').read_text())
        host = Host({'fleet': CLUSTER / 'tests/fleet-graded.json', 'public-fleet': CLUSTER / 'tests/fleet-graded.public.json'})
        hooks = load('breakwater_host')
        with patch.object(hooks.fleet_tools, 'audit_directory', return_value=Path('/')):
            declared = hooks.preflight(host).to_json()
        rows = [{**row, 'port_map': None, 'health': None} for row in declared['endpoints']]
        roles = plan.by_role(rows)
        def plain(items):
            return sorted(({'name': row['name'], 'host': row['host'], 'ports': sorted(row['ports'])} for row in items),
                          key=lambda row: row['name'])
        self.assertEqual(plain(roles['agent']), plain(expected['workers']))
        self.assertEqual(plain(roles['submission']), plain(expected['workers']))
        self.assertEqual(plain(roles['coordinator']), plain(expected['coordinator']))
        self.assertEqual((declared['agent_env']['CLUSTER_RECEIPT_KEY'], declared['agent_env']['CLUSTER_NODES']),
                         ('receipt', '/app/fleet.json'))
        self.assertEqual(declared['private_files']['grader-token'], {'secret': 'grader-token'})
        self.assertEqual(declared['workspace_files'], {'fleet.json': {'input': 'public-fleet'}})


class Controller(http.server.BaseHTTPRequestHandler):
    def do_POST(self):
        self.server.calls.append((self.path, self.headers.get('X-Grader-Token')))
        body = json.dumps({'engines_remaining': 0}).encode()
        self.send_response(200)
        self.send_header('Content-Length', str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args):
        pass


class HookTests(unittest.TestCase):
    def test_the_hooks_reset_the_fleet_and_collect_only_this_runs_journals(self):
        controller = http.server.ThreadingHTTPServer(('127.0.0.1', 0), Controller)
        controller.calls = []
        threading.Thread(target=controller.serve_forever, daemon=True).start()
        self.addCleanup(controller.server_close)
        self.addCleanup(controller.shutdown)
        root = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, root, True)
        audit, state, evidence = root / 'audit', root / 'state', root / 'external' / 'task'
        for folder in (audit, state, evidence):
            folder.mkdir(parents=True)
        fleet = json.loads((CLUSTER / 'tests/fleet-graded.json').read_text())
        fleet['controller'] = {'host': '127.0.0.1', 'port': controller.server_address[1]}
        fleet['launch'] = {**(fleet.get('launch') or {}), 'inference_audit_dir': str(audit)}
        (root / 'fleet.json').write_text(json.dumps(fleet))
        (audit / ('worker-' + 'b' * 32 + '.jsonl')).write_text('')  # an earlier session's
        host = Host({'fleet': root / 'fleet.json', 'public-fleet': CLUSTER / 'tests/fleet-graded.public.json'},
                    state=state, evidence=evidence)
        hooks = load('breakwater_host')
        hooks.before_authoring(host)
        shutil.copyfile(FIXTURES / JOURNAL, audit / JOURNAL)  # this run's
        with patch.object(hooks.fleet_tools, 'check_journals', return_value={'journals': [JOURNAL], 'incomplete': []}):
            hooks.after_evaluation(host, {'agent': [1, 2], 'intermediate': [3, 4]})
        self.assertEqual(controller.calls, [('/admin/reset', 'token')] * 2)
        manifests = json.loads((evidence / 'fleet-manifests.json').read_text())
        self.assertEqual(sorted(manifests['preexisting_journals']), ['worker-' + 'b' * 32 + '.jsonl'])
        collected = json.loads((evidence / 'fleet-inference.json').read_text())
        self.assertEqual((collected['collected'], collected['windows']), ([JOURNAL], {'agent': [1, 2], 'intermediate': [3, 4]}))
        self.assertEqual(sorted(path.name for path in (evidence / 'fleet-inference').iterdir()), [JOURNAL])
        hooks.on_failure(host)
        self.assertEqual(len(controller.calls), 3)

    def test_options_and_qualification_are_checked_before_the_run(self):
        hooks = load('breakwater_host')
        for options in ({'repeats': 0}, {'time_scale': 0}, {'allow_unqualified': 'yes'}, {'unknown': 1}):
            with self.subTest(options=options), self.assertRaises(ValueError):
                hooks.check_options(options)
        hooks.check_options({'repeats': 2, 'time_scale': 0.5, 'allow_unqualified': True})


if __name__ == '__main__':
    unittest.main()
