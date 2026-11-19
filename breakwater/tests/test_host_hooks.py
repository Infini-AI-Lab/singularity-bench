"""Host hooks and operator inputs (host_hooks.py): phases in their own processes, in order, within a timeout; the plan
validated and applied; secrets kept out of everything a sandbox sees; the phases' times checked by the verifier. A
stand-in task (fixtures/hook-task) does what its 'mode' value says. No Docker or GPUs.
"""
import hashlib
import json
import os
from pathlib import Path
import shutil
import stat
import sys
import tempfile
import unittest
from unittest.mock import patch

FRAMEWORK = Path(__file__).resolve().parents[1]
TASK = Path(__file__).resolve().parent / 'fixtures' / 'hook-task'
sys.path[:0] = [str(FRAMEWORK)]
from contract import ContractError, RulesViolation, check_host_hooks  # noqa: E402
import host_hooks  # noqa: E402
import spec  # noqa: E402

EXAMPLE = FRAMEWORK.parent / 'examples' / 'word-count'
SECRET = 'tok-0123456789abcdef'


def contract(**changes):
    return spec.resolve({**spec.read(EXAMPLE), 'host_module': 'tests/breakwater_host.py', 'host_timeout_sec': 20,
                         'operator_inputs': ['fleet'], 'operator_values': {'mode': 'HOOK_MODE'},
                         'operator_secrets': {'token': 'TASK_TOKEN'}, **changes})


class HostHookTests(unittest.TestCase):
    def setUp(self):
        self.root = Path(tempfile.mkdtemp(prefix='breakwater-hooks-test-'))
        self.addCleanup(shutil.rmtree, self.root, True)
        self.fleet = self.root / 'fleet.json'
        self.fleet.write_text('{"nodes": ["n1"]}')

    def host(self, mode='ok', backend='docker', config=None, **options):
        environ = {'HOOK_MODE': mode, 'TASK_TOKEN': SECRET}
        with patch.dict(os.environ, environ):
            host = host_hooks.TaskHost(TASK, config or contract(), backend, inputs=[f'fleet={self.fleet}'],
                                       avoid=(TASK, self.root / 'run'), **options)
            self.assertNotIn('TASK_TOKEN', os.environ)  # the secret left the launcher's environment
        self.addCleanup(lambda: host.state and shutil.rmtree(host.state, True))
        return host

    def test_phases_run_in_order_and_the_plan_is_applied(self):
        host = self.host()
        plan = host.preflight(reserved={'author-gateway'})
        self.assertEqual(plan['metadata'], {'fleet_id': 'f1'})
        self.assertEqual(host.endpoints(), [{'name': 'store', 'host': '127.0.0.1', 'ports': [9000, 9001], 'port_map': None,
                                             'reach': ['agent', 'coordinator'], 'health': None}])
        self.assertEqual(host.agent_env(), {'TASK_STORE': 'http://store:9000'})
        run = self.root / 'run'
        external, workspace = run / 'external', run / 'workspace'
        for folder in (external, workspace):
            folder.mkdir(parents=True)
        host.adopt(run)
        self.assertEqual(host.state, run / 'host-state')
        host.write_workspace_files(workspace)
        self.assertEqual((workspace / 'config/fleet.json').read_text(), self.fleet.read_text())
        self.assertEqual((workspace / 'note.txt').read_text(), 'hello')
        private = host.write_private_files(run)
        self.assertEqual((private / 'token').read_text(), SECRET)
        self.assertEqual(stat.S_IMODE((private / 'token').stat().st_mode), 0o400)
        host.before_authoring(external)
        before = json.loads((external / 'task/before.json').read_text())
        # The hook's process had only PATH, LANG and the declared operator variables.
        self.assertEqual(set(before['environment']) - {'LC_CTYPE'}, {'PATH', 'LANG', 'HOOK_MODE', 'TASK_TOKEN'})
        windows = {'agent': [1, 2], 'intermediate': [3, 4]}
        host.after_evaluation(external, windows)
        after = json.loads((external / 'task/journals/after.json').read_text())
        self.assertEqual((after['windows'], after['order']), (windows, 'preflight\nbefore_authoring\nafter_evaluation\n'))
        self.assertEqual(list(host.times), ['preflight', 'before_authoring', 'after_evaluation'])
        digest = host.record(external)
        recorded = json.loads((external / 'task-plan.json').read_text())
        self.assertEqual(recorded['private_files']['token'],
                         {'secret': 'token', 'sha256': hashlib.sha256(SECRET.encode()).hexdigest()})
        self.assertNotIn(SECRET, (external / 'task-plan.json').read_text())
        self.assertEqual(recorded['operator'], {'inputs': {'fleet': hashlib.sha256(self.fleet.read_bytes()).hexdigest()},
                                                'values': ['mode'], 'secrets': ['token']})
        sealed = host.sealed(digest)
        self.assertEqual((sealed['task_plan_sha256'], sealed['task_metadata'], sorted(sealed['host_hooks'])),
                         (digest, {'fleet_id': 'f1'}, ['after_evaluation', 'before_authoring', 'preflight']))
        host.on_failure()  # its own error is ignored
        self.assertTrue((host.state / 'order').read_text().endswith('on_failure\n'))

    def test_a_failing_or_slow_phase_stops_the_run(self):
        with self.assertRaisesRegex(ContractError, '(?s)preflight hook failed.*preflight refused'):
            self.host('fail-preflight').preflight()
        with self.assertRaisesRegex(ContractError, 'ran out of time'):
            self.host('slow', config=contract(host_timeout_sec=1)).preflight()
        host = self.host('fail-after')
        host.preflight()
        with self.assertRaisesRegex(ContractError, '(?s)after_evaluation hook failed.*collection failed'):
            host.after_evaluation(self.root, {'agent': [1, 2], 'intermediate': [3, 4]})
        host = self.host('evidence-outside')
        host.preflight()
        with self.assertRaisesRegex(ContractError, 'before_authoring hook failed'):
            host.before_authoring(self.root)

    def test_plans_that_would_weaken_isolation_are_refused(self):
        for mode in ('credential-env', 'secret-env', 'secret-metadata', 'loopback-name', 'bad-reach', 'bad-port',
                     'reserved', 'secret-workspace-source', 'bad-path', 'not-a-plan'):
            with self.subTest(mode=mode), self.assertRaises(ContractError):
                self.host(mode).preflight(reserved={'author-gateway'})
        # An address as the name: a loopback one on Apptainer (where the relay listens in the sandbox), any other on
        # Docker (which the container reaches as itself).
        self.assertEqual(self.host('loopback-name', backend='apptainer').preflight()['endpoints'][0]['name'], '127.0.0.5')
        self.assertEqual(self.host('ip-name').preflight()['endpoints'][0]['name'], '10.0.0.5')
        with self.assertRaises(ContractError):
            self.host('ip-name', backend='apptainer').preflight()
        host = self.host('secret-file')
        host.preflight()
        with self.assertRaisesRegex(ContractError, 'contains an operator secret'):
            host.write_workspace_files(self.root / 'workspace')
        with self.assertRaisesRegex(ContractError, 'operator secret'):
            host.check_no_secrets('{"env": {"X": "%s"}}' % SECRET)

    def test_operator_inputs_and_variables_are_checked(self):
        config = contract()
        with patch.dict(os.environ, {'HOOK_MODE': 'ok', 'TASK_TOKEN': SECRET}):
            for inputs, message in (([], 'Missing operator inputs'), ([f'other={self.fleet}'], 'Undeclared'),
                                    ([f'fleet={self.root / "missing"}'], 'not a file'),
                                    ([f'fleet={TASK / "tests/breakwater_host.py"}'], 'inside the task'),
                                    ([f'fleet={self.fleet}', f'fleet={self.fleet}'], 'once per name')):
                with self.subTest(inputs=inputs), self.assertRaisesRegex(ContractError, message):
                    host_hooks.Operator(config, inputs, avoid=(TASK,), environ=dict(os.environ))
        with patch.dict(os.environ, {'HOOK_MODE': 'ok'}, clear=False):
            os.environ.pop('TASK_TOKEN', None)
            with self.assertRaisesRegex(ContractError, 'Set the operator variables: TASK_TOKEN'):
                host_hooks.Operator(config, [f'fleet={self.fleet}'], environ=dict(os.environ))
        environ = {'HOOK_MODE': 'ok', 'OTHER_TOKEN': SECRET}
        operator = host_hooks.Operator(config, [f'fleet={self.fleet}'], secret_env=['token=OTHER_TOKEN'], environ=environ)
        self.assertEqual((operator.secrets, operator.credential_names, 'OTHER_TOKEN' in environ),
                         ({'token': SECRET}, ['OTHER_TOKEN'], False))

    def test_a_task_without_a_host_side_is_inert(self):
        host = host_hooks.TaskHost(EXAMPLE, spec.load(EXAMPLE), 'docker')
        self.assertEqual((host.preflight(), host.state, host.endpoints(), host.agent_env()),
                         ({'endpoints': [], 'agent_env': {}, 'workspace_files': {}, 'private_files': {}, 'metadata': {}},
                          None, [], {}))
        host.before_authoring(self.root)
        self.assertIsNone(host.record(self.root))
        self.assertEqual(host.sealed(None), {})
        self.assertFalse((self.root / 'task').exists() and any((self.root / 'task').iterdir()))

    def test_the_verifier_checks_the_hooks_ran_outside_the_execution_stages(self):
        check_host_hooks({'preflight': [1, 2], 'before_authoring': [3, 4], 'after_evaluation': [30, 31]}, 10, 20)
        for hooks in ({'before_authoring': [3, 11]}, {'preflight': [1, 12]}, {'after_evaluation': [19, 31]}):
            with self.subTest(hooks=hooks), self.assertRaises(RulesViolation):
                check_host_hooks(hooks, 10, 20)
        for hooks in ({'on_failure': [1, 2]}, {'preflight': [2, 1]}, {'preflight': ['1', 2]}, []):
            with self.subTest(hooks=hooks), self.assertRaises(ContractError):
                check_host_hooks(hooks, 10, 20)


if __name__ == '__main__':
    unittest.main()
