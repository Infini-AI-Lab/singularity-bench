"""Live test of the Apptainer launcher's three stages on a fixture task.

    BREAKWATER_APPTAINER_IMAGE=IMAGE.sif python3 -m unittest discover -s breakwater/apptainer/tests -v

The fixture is the word-count example with an agent-visible asset inside the workspace (which the submission must
leave out) and private evaluation data, as a training task has. A stand-in CLI plays the agent, a stand-in provider
the model and a fixture coordinator intermediate.py; the verifier is the real one, in its own sandbox.
"""
import http.server
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import threading
import unittest

HERE = Path(__file__).resolve().parents[1]
REPO = HERE.parents[1]
sys.path[:0] = [str(HERE), str(HERE / 'tests'), str(HERE.parent)]
from test_agent_stage import FAKE_CLI, IMAGE, READY, Provider  # noqa: E402
from test_evaluation_stage import COORDINATOR, WORKER  # noqa: E402


def fixture_task(root):
    """The word-count example as the launcher's fixture: a -1.0 failure reward, an agent-only asset inside /app, and a
    private evaluation directory at /private/validation (which the fixture coordinator reads)."""
    shutil.copytree(REPO / 'examples/word-count', root, ignore=shutil.ignore_patterns('__pycache__'))
    toml = (root / 'task.toml').read_text()
    assets = toml.index('[[breakwater.assets]]')
    toml = (toml[:assets].replace('name = "word-count"', 'name = "launcher-fixture"')
            .replace('failure_reward = 0.0', 'failure_reward = -1.0')
            + '[[breakwater.assets]]\nname = "train-tokens"\nsource = "launcher-fixture/data/train.bin"\n'
            'target = "/app/data/train.bin"\nstages = ["agent"]\n\n'
            '[[breakwater.assets]]\nname = "validation"\nsource = "private_validation"\ntarget = "/private/validation"\n'
            'stages = ["evaluation", "verifier"]\n')
    assert 'name = "launcher-fixture"' in toml and 'failure_reward = -1.0' in toml
    (root / 'task.toml').write_text(toml)
    return root


@unittest.skipUnless(READY, 'set BREAKWATER_APPTAINER_IMAGE to a SIF with bash and python3; the launcher is NOT verified')
class LauncherTests(unittest.TestCase):
    def setUp(self):
        self.provider = http.server.ThreadingHTTPServer(('127.0.0.1', 0), Provider)
        threading.Thread(target=self.provider.serve_forever, daemon=True).start()
        self.addCleanup(self.provider.server_close)
        self.addCleanup(self.provider.shutdown)
        self.root = Path(tempfile.mkdtemp(prefix='breakwater-launcher-'))
        from sandbox import remove_tree
        self.addCleanup(remove_tree, self.root)  # the evidence is left read-only
        assets, self.cli = self.root / 'assets', self.root / 'author-cli'
        data = assets / 'launcher-fixture/data'
        for folder in (data, assets / 'private_validation', self.cli):
            folder.mkdir(parents=True)
        (data / 'train.bin').write_bytes(b'\0' * 64)
        (assets / 'private_validation/secret.txt').write_text('labels')
        (self.cli / 'fake-cli.py').write_text(FAKE_CLI)
        self.task = fixture_task(self.root / 'launcher-fixture')

    def launch(self, coordinator, smoke=False, extra_args=(), extra_env=None):
        """Run the launcher on the task with the stand-in CLI, provider and judge runner, and `coordinator` in place
        of intermediate.py."""
        run, judge = self.root / 'run', self.root / 'judge'
        judge.mkdir()
        (judge / 'codex').write_text('#!/bin/sh\nexit 1\n')  # never reached: the placeholder outputs do not score
        cli = [['python3', '/opt/author-cli/fake-cli.py', 'launcher'], {'PATH': '/usr/local/bin:/usr/bin:/bin'}]
        result = subprocess.run([sys.executable, '-I', str(HERE / 'run.py'), '--task', str(self.task),
                                 '--image', IMAGE, '--cli-dir', str(self.cli), '--judge-dir', str(judge),
                                 '--run-dir', str(run), '--asset-root', str(self.root / 'assets'), '--author-model', 'test/author',
                                 '--author-upstream', f'http://127.0.0.1:{self.provider.server_address[1]}',
                                 '--author-key-env', 'TEST_PROVIDER_KEY', '--judge-key-env', 'TEST_JUDGE_KEY',
                                 '--cli-command', json.dumps(cli), '--coordinator-command', json.dumps(coordinator),
                                 *(['--smoke'] if smoke else []), *extra_args],
                                capture_output=True, text=True, timeout=900,
                                env={**os.environ, 'TEST_PROVIDER_KEY': 'sk-test-provider-key',
                                     'TEST_JUDGE_KEY': 'fw-test-judge-key', **(extra_env or {})})
        return run, result

    def test_host_hooks_operator_inputs_and_their_endpoint_go_through_all_three_stages(self):
        # The fixture with the stand-in host module of breakwater/tests (fixtures/hook-task): its preflight declares an
        # endpoint for the agent's commands and the coordinator, an environment, two workspace files (one from the
        # operator's input) and two private files (one from the operator's secret); its other phases write evidence.
        import checks
        from contract import verify_seal
        shutil.copyfile(HERE.parent / 'tests/fixtures/hook-task/tests/breakwater_host.py',
                        self.task / 'tests/breakwater_host.py')
        toml = (self.task / 'task.toml').read_text()
        (self.task / 'task.toml').write_text(toml + '\n[breakwater.host]\nmodule = "tests/breakwater_host.py"\n'
                                             '\n[breakwater.operator]\ninputs = ["fleet"]\n'
                                             'values = { mode = "HOOK_MODE" }\nsecrets = { token = "TASK_TOKEN" }\n')
        fleet = self.root / 'operator-fleet.json'
        fleet.write_text('{"nodes": ["n1"]}')
        run, result = self.launch(['python3', '-c', COORDINATOR, WORKER], extra_args=['--input', f'fleet={fleet}'],
                                  extra_env={'HOOK_MODE': 'ok', 'TASK_TOKEN': 'tok-0123456789abcdef'})
        error = (run / 'reports/pipeline-error.json')
        self.assertTrue(result.stdout.strip(), result.stderr[-3000:] + (error.read_text() if error.exists() else ''))
        summary = json.loads(result.stdout.strip().splitlines()[-1])
        launch = verify_seal(run / 'authority', summary['authority_sha256'])
        identity = launch['identity']
        external = verify_seal(run / 'external', launch['external_sha256'], identity)
        # The hooks' evidence and the task's plan are sealed with the external evidence; the authority has the rest.
        for name in ('task/before.json', 'task/journals/after.json', 'task-plan.json', 'authoring-plan.json'):
            self.assertIn(name, external['files'])
        self.assertEqual(sorted(launch['host_hooks']), ['after_evaluation', 'before_authoring', 'preflight'])
        self.assertEqual(launch['task_metadata'], {'fleet_id': 'f1'})
        self.assertEqual(launch['task_plan_sha256'], external['files']['task-plan.json']['sha256'])
        self.assertNotIn('tok-0123456789abcdef', (run / 'external/task-plan.json').read_text())
        authoring = json.loads((run / 'external/authoring-plan.json').read_text())
        self.assertEqual((authoring['endpoints'], authoring['env']),
                         ([{'name': 'store', 'host': '127.0.0.1', 'ports': ['9000:9001']}], {'TASK_STORE': 'http://store:9000'}))
        # The workspace files are part of the workspace (and of the submission); the private files reach only the
        # coordinator, at /private/operator.
        self.assertEqual((run / 'submission/note.txt').read_text(), 'hello')
        self.assertEqual((run / 'submission/config/fleet.json').read_text(), fleet.read_text())
        boxes = [box for stage in ('agent', 'intermediate') for box in launch['stages'][stage]['sandboxes']]
        bound = {box['role'] for box in boxes if any(row[1] == '/private/operator' for row in box['binds'])}
        self.assertEqual(bound, {'coordinator'})
        self.assertEqual(launch['stages']['agent']['endpoints'], [{'name': 'store', 'ports': 2}])
        self.assertEqual(launch['stages']['intermediate']['endpoints']['coordinator'], [{'name': 'store', 'ports': 2}])
        self.assertEqual(checks.check_launch(launch, external), identity)
        mechanical = json.loads((run / 'reports/mechanical-check.json').read_text())
        self.assertEqual(mechanical['status'], 'PASS', mechanical)

    def test_a_failed_evaluation_stops_before_the_verifier_and_reports(self):
        run, result = self.launch(['python3', '-c', 'raise SystemExit(3)'])
        self.assertEqual(result.returncode, 2, result.stderr[-3000:])
        line = json.loads(result.stdout.strip().splitlines()[-1])
        self.assertEqual((line['status'], line['error'], line['completed_stages']),
                         ('INCONCLUSIVE', 'ContractError: Intermediate evaluation failed', ['agent', 'intermediate']))
        self.assertIn('Traceback', result.stderr)
        self.assertEqual(json.loads((run / 'reports/reward.json').read_text())['reward'], -1.0)
        self.assertTrue((run / 'reports/pipeline-error.json').exists())
        self.assertFalse((run / 'reports/mechanical-check.json').exists())
        self.assertFalse((run / 'authority/manifest.json').exists())

    def test_three_stages_end_with_the_verifier_reading_sealed_evidence(self):
        import checks
        from contract import verify_seal
        from reconcile import check_exec_journal
        run, result = self.launch(['python3', '-c', COORDINATOR, WORKER], smoke=True)
        error = (run / 'reports/pipeline-error.json')
        self.assertTrue(result.stdout.strip(), result.stderr[-3000:] + (error.read_text() if error.exists() else ''))
        summary = json.loads(result.stdout.strip().splitlines()[-1])
        self.assertEqual((summary['exit_code'], summary['deadline_reached']), (0, False))
        launch = verify_seal(run / 'authority', summary['authority_sha256'])
        identity = launch['identity']
        self.assertEqual(launch['backend'], 'apptainer')
        self.assertIs(launch['smoke'], True)  # sealed with the run: the verifier scores it as a smoke run
        manifest = verify_seal(run / 'submission', launch['submission_sha256'], identity)
        self.assertEqual((run / 'submission/model.txt').read_text(), 'trained\n')
        self.assertIn('model.txt', manifest['files'])
        self.assertIn('solution.py', manifest['files'])
        # Agent-bound data and the task's excluded paths never become part of the submission.
        self.assertFalse(any(name.startswith('data/') for name in manifest['files']))
        self.assertNotIn('README.md', manifest['files'])
        self.assertFalse(any(name.startswith('tests/') for name in manifest['files']))
        self.assertEqual(json.loads((run / 'external/runtime-integrity.json').read_text())['status'], 'PASS')
        events = [json.loads(line)['event'] for line in (run / 'external/gateway/requests.jsonl').read_text().splitlines()]
        self.assertIn('model_response_end', events)
        self.assertEqual((run / 'external/agent-native.jsonl').read_bytes(), (run / 'agent-local/native.jsonl').read_bytes())
        self.assertFalse((run / 'workspace-live').exists())
        native = [json.loads(line) for line in (run / 'external/agent-native.jsonl').read_text().splitlines()]
        self.assertEqual([line['stdout'] for line in native if line['type'] == 'tool_result'][-1], '1\nread-only\n')
        # The evaluation stage: the coordinator read the private labels, the submitted code ran apart from them.
        self.assertEqual((summary['evaluation_exit_code'], summary['evaluation_timed_out']), (0, False))
        verify_seal(run / 'evaluation', launch['evaluation_sha256'], identity)
        result = json.loads((run / 'evaluation/result.json').read_text())
        self.assertEqual((result['secret'], result['reply']), ('labels', 'pong:ping'))
        self.assertEqual((result['worker']['private'], result['worker']['evaluation']), (False, False))
        self.assertIn('model.txt', result['worker']['app'])
        # The launch record passes the verifier's Apptainer checks, and both brokers' evidence is complete.
        self.assertEqual(checks.check_launch(launch, verify_seal(run / 'external', launch['external_sha256'], identity)),
                         identity)
        self.assertEqual(check_exec_journal(run / 'external/commands')['commands'], 3)
        self.assertEqual(check_exec_journal(run / 'external/evaluation-commands')['commands'], 1)
        self.assertEqual(oct((run / 'external/isolation.canary').stat().st_mode & 0o777), '0o444')
        # The verifier ran in its own sandbox and completed every mechanical check; the placeholder outputs then fail
        # to score, so there is no judge call and the reward is the failure reward.
        mechanical = json.loads((run / 'reports/mechanical-check.json').read_text())
        self.assertEqual((mechanical['status'], mechanical['backend'], mechanical['verifier_evidence_write_probe']),
                         ('PASS', 'apptainer', 'PASS'), mechanical)
        self.assertEqual(set(mechanical['observers']), {'commands', 'evaluation-commands'})
        self.assertEqual(mechanical['native_agreement']['tool_calls_compared'], 3)
        self.assertEqual(mechanical['runtime_integrity']['status'], 'PASS')
        # What the judge reads: a statement per mechanically checked row, and the stages without namespace numbers
        # (which the kernel reuses across stages).
        self.assertEqual(set(mechanical['established']) - {'sealed_submission', 'workload_compliance'},
                         {'external_integrity', 'three_containers', 'log_isolation', 'capture_complete',
                          'local_external_agreement', 'no_lingering_processes', 'runtime_package_integrity'})
        self.assertNotIn('pid:[', json.dumps(mechanical))
        isolation = mechanical['isolation']
        self.assertEqual([box['role'] for box in isolation['agent']['sandboxes']], ['author', 'command'])
        self.assertTrue(all(box['proven_empty'] for stage in ('agent', 'intermediate')
                            for box in isolation[stage]['sandboxes']))
        self.assertLessEqual(isolation['agent']['stopped_s'], isolation['intermediate']['started_s'])
        self.assertEqual(isolation['verifier']['sandboxes'][0]['state'], 'running: this is the verifier')
        self.assertEqual(summary['verifier_exit_code'], 2)
        self.assertTrue((run / 'reports/verification-error.json').exists())
        self.assertEqual(json.loads((run / 'reports/reward.json').read_text())['reward'], -1.0)
        # Re-verification: the same sealed run, a fresh verifier sandbox, a new authority and new reports.
        from contract import digest
        before = (run / 'reports/mechanical-check.json').read_bytes()
        again = subprocess.run([sys.executable, '-I', str(HERE / 'reverify.py'), '--run-dir', str(run),
                                '--task', str(self.task), '--image', IMAGE,
                                '--asset-root', str(self.root / 'assets'),
                                '--judge-dir', str(self.root / 'judge'), '--judge-key-env', 'TEST_JUDGE_KEY'],
                               capture_output=True, text=True, timeout=600,
                               env={**os.environ, 'TEST_JUDGE_KEY': 'fw-test-judge-key'})
        self.assertTrue(again.stdout.strip(), again.stderr[-3000:])
        second = json.loads(again.stdout.strip().splitlines()[-1])
        self.assertEqual((second['reverify_of'], second['verifier_exit_code']), (digest(run / 'authority/manifest.json'), 2))
        relaunch = verify_seal(run / 'authority-reverify-1', second['authority_sha256'])
        self.assertEqual((relaunch['reverify_of'], relaunch['submission_sha256']),
                         (second['reverify_of'], launch['submission_sha256']))
        self.assertIs(relaunch['smoke'], True)  # re-verified as the smoke run it was
        self.assertIs(second['smoke'], True)
        self.assertNotEqual(relaunch['stages']['verifier'], launch['stages']['verifier'])
        self.assertEqual(json.loads((run / 'reports-reverify-1/mechanical-check.json').read_text())['status'], 'PASS')
        self.assertEqual((run / 'reports/mechanical-check.json').read_bytes(), before)


if __name__ == '__main__':
    unittest.main()
