"""The verifier's Apptainer checks (no Apptainer needed): the broker journal and the launch record.

    python3 -m unittest discover -s breakwater/apptainer/tests -v
"""
import copy
import hashlib
import json
from pathlib import Path
import shutil
import sys
import tempfile
import unittest

HERE = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(HERE), str(HERE.parent)]
from broker import Journal  # noqa: E402
import checks  # noqa: E402
from contract import ContractError, RulesViolation  # noqa: E402
from reconcile import check_exec_journal  # noqa: E402


class JournalTests(unittest.TestCase):
    def setUp(self):
        self.folder = Path(tempfile.mkdtemp(prefix='breakwater-journal-'))
        self.addCleanup(shutil.rmtree, self.folder, True)

    def command(self, journal, number, status=0, output=b'out\n'):
        journal.write(event='exec_started', id=number, cwd='/app', command='echo out')
        journal.write(event='exec_exited', id=number, status=status)
        (self.folder / f'exec-{number}.strace').write_text('12:00:00 execve("/usr/bin/nsenter", ...) = 0\n')
        for stream, data in (('stdout', output), ('stderr', b'')):
            (self.folder / f'exec-{number}.{stream}').write_bytes(data)
            (self.folder / f'exec-{number}.{stream}.sha256').write_text(f'{hashlib.sha256(data).hexdigest()} {len(data)}\n')
        journal.write(event='exec_streams_closed', id=number, tracer_status=status)

    def journal(self, *, trace=True, stop=True, commands=2):
        journal = Journal(self.folder / 'exec.jsonl')
        journal.write(event='broker_started', trace=trace, workspace='/app')
        for number in range(1, commands + 1):
            self.command(journal, number)
        if stop:
            journal.write(event='broker_stopped', commands=commands, drained=True)
        return journal

    def lines(self):
        return (self.folder / 'exec.jsonl').read_text().splitlines()

    def test_a_complete_journal_passes(self):
        journal = self.journal(stop=False)
        journal.write(event='exec_started', id=3, cwd='/app', command='true')
        journal.write(event='sandbox_unavailable', id=3, error='the sandbox is gone')
        journal.write(event='command_sandbox_lost', pid=7)
        journal.write(event='command_sandbox_restarted', pid=8)
        journal.write(event='broker_stopped', commands=3, drained=True)
        report = check_exec_journal(self.folder)
        self.assertEqual((report['commands'], report['unavailable'], report['command_sandbox_restarts']), (3, 1, 1))

    def test_an_edited_or_removed_record_breaks_the_chain(self):
        self.journal()
        lines = self.lines()
        for changed in ([lines[0], lines[1].replace('echo out', 'echo in'), *lines[2:]], [lines[0], *lines[2:]]):
            (self.folder / 'exec.jsonl').write_text('\n'.join(changed) + '\n')
            with self.assertRaisesRegex(ContractError, 'chain'):
                check_exec_journal(self.folder)

    def test_a_command_without_closed_streams_is_incomplete(self):
        journal = self.journal(stop=False)
        journal.write(event='exec_started', id=3, cwd='/app', command='sleep 1')
        journal.write(event='exec_exited', id=3, status=0)
        journal.write(event='broker_stopped', commands=3, drained=True)
        with self.assertRaisesRegex(ContractError, 'Incomplete'):
            check_exec_journal(self.folder)

    def test_untraced_undrained_or_unfinished_journals_fail(self):
        for options, message in (({'trace': False}, 'trace'), ({'stop': False}, 'drain')):
            shutil.rmtree(self.folder)
            self.folder.mkdir()
            self.journal(**options)
            with self.assertRaisesRegex(ContractError, message):
                check_exec_journal(self.folder)

    def test_changed_or_extra_evidence_fails(self):
        self.journal()
        (self.folder / 'exec-1.stdout').write_bytes(b'changed\n')
        with self.assertRaisesRegex(ContractError, 'digest'):
            check_exec_journal(self.folder)
        shutil.rmtree(self.folder)
        self.folder.mkdir()
        self.journal()
        (self.folder / 'exec-9.stdout').write_bytes(b'')
        with self.assertRaisesRegex(ContractError, 'Unexpected'):
            check_exec_journal(self.folder)


def box(role, start, stop, binds=(), network='none', gpus=True):
    binds = [list(row) for row in binds]
    mounts = [['/', 'ro', 'overlay'], ['/tmp', 'rw', 'xfs']] + [[target, mode, 'xfs'] for _, target, mode in binds]
    return {'role': role, 'namespace': f'pid:[{4026530000 + start}]', 'network': network, 'gpus': gpus, 'binds': binds,
            'mounts': mounts, 'started_ns': start, 'stopped_ns': stop, 'empty': stop is not None}


BASELINE = ('/runs/r/environment-baseline', '/opt/environment-baseline', 'ro')
LAUNCH = {
    'backend': 'apptainer', 'identity': {'task': 't', 'run_id': 'r'}, 'pipeline': {'gpus': 4, 'agent_deadline': 'evaluate'},
    'recording': {'author_gateway': True, 'service_gateways': {}, 'native': 'jsonl'},
    'external_host_root': '/runs/r/external',
    'protected_host_roots': ['/runs/r/external', '/runs/r/authority', '/runs/r/reports'],
    'stages': {
        'agent': {'started_ns': 10, 'stopped_ns': 100, 'started_wall_ns': 5_000, 'exit_code': None, 'deadline_reached': True,
                  'process_wall': {'author': True, 'command': True}, 'commands_drained': True,
                  'environments': {'cli': ['ANTHROPIC_API_KEY', 'HOME', 'PATH'], 'command': ['HOME', 'PATH']},
                  'sandboxes': [box('author', 11, 90, [('/runs/r/workspace-live', '/app', 'rw'), BASELINE], gpus=False),
                                box('command', 12, 95, [('/runs/r/workspace-live', '/app', 'rw'), BASELINE])]},
        'intermediate': {'started_ns': 110, 'stopped_ns': 200, 'ended_wall_ns': 6_000, 'exit_code': 0, 'timed_out': False,
                         'process_wall': {'coordinator': True, 'submission': True}, 'commands_drained': True,
                         'environments': {'coordinator': ['HOME', 'PATH'], 'worker': ['HOME', 'PATH']},
                         'sandboxes': [box('coordinator', 111, 190, [('/runs/r/evaluation', '/evaluation', 'rw')]),
                                       box('submission', 112, 195, [('/runs/r/submission', '/app', 'ro')])]},
        'verifier': {'started_ns': 210, 'started_wall_ns': 7_000,
                     'sandboxes': [box('verifier', 210, None, [('/runs/r/external', '/external-evidence', 'ro'),
                                                               ('/runs/r/reports', '/reports', 'rw')],
                                       network='host', gpus=False)]},
    },
}


class LaunchTests(unittest.TestCase):
    def check(self, change=None):
        launch = copy.deepcopy(LAUNCH)
        if change:
            change(launch)
        return checks.check_launch(launch, {'identity': launch['identity']})

    def test_a_clean_launch_passes(self):
        self.assertEqual(self.check(), LAUNCH['identity'])

    def test_evidence_in_an_execution_sandbox_fails(self):
        def expose(launch):
            launch['stages']['agent']['sandboxes'][1]['binds'].append(['/runs/r/external/commands', '/x', 'ro'])
            launch['stages']['agent']['sandboxes'][1]['mounts'].append(['/x', 'ro', 'xfs'])
        with self.assertRaisesRegex(ContractError, 'Trusted evidence'):
            self.check(expose)

    def test_a_bind_the_kernel_did_not_mount_as_asked_fails(self):
        def writable(launch):
            launch['stages']['intermediate']['sandboxes'][1]['mounts'][-1][1] = 'rw'
        with self.assertRaisesRegex(ContractError, 'wrong mode'):
            self.check(writable)

    def test_a_sandbox_outliving_its_stage_fails(self):
        def linger(launch):
            launch['stages']['agent']['sandboxes'][1]['stopped_ns'] = 120
        with self.assertRaises(RulesViolation):
            self.check(linger)
        def unproven(launch):
            launch['stages']['intermediate']['sandboxes'][0].update(stopped_ns=None, empty=False)
        with self.assertRaises(RulesViolation):
            self.check(unproven)

    def test_overlapping_stages_fail(self):
        def overlap(launch):
            launch['stages']['intermediate']['started_ns'] = 50
        with self.assertRaisesRegex(ContractError, 'overlap'):
            self.check(overlap)
        def early_verifier(launch):  # the verifier is ordered by the wall clock: it may run later, on another node
            launch['stages']['verifier']['started_wall_ns'] = 5_500
        with self.assertRaisesRegex(ContractError, 'Evaluation/verifier overlap'):
            self.check(early_verifier)
        def later_elsewhere(launch):  # another node's monotonic clock: fine as long as the wall clock agrees
            launch['stages']['verifier']['started_ns'] = 3
            launch['stages']['verifier']['sandboxes'][0]['started_ns'] = 3
        self.check(later_elsewhere)

    def test_keys_network_and_gpus_are_checked(self):
        def key(launch):
            launch['stages']['intermediate']['environments']['worker'].append('FIREWORKS_API_KEY')
        def network(launch):
            launch['stages']['agent']['sandboxes'][0]['network'] = 'host'
        def gpu(launch):
            launch['stages']['verifier']['sandboxes'][0]['gpus'] = True
        for change, message in ((key, 'credential'), (network, 'network'), (gpu, 'GPU')):
            with self.assertRaisesRegex(ContractError, message):
                self.check(change)

    def test_the_verifier_needs_its_one_read_only_evidence_bind(self):
        def writable(launch):
            launch['stages']['verifier']['sandboxes'][0]['binds'][0][2] = 'rw'
            launch['stages']['verifier']['sandboxes'][0]['mounts'][2][1] = 'rw'
        with self.assertRaisesRegex(ContractError, 'one read-only mount'):
            self.check(writable)

    def test_an_agent_stopped_at_its_deadline_is_graded(self):
        # With agent_deadline = "evaluate", the host stops a CLI still running at the deadline: its status is the
        # stopped process's (SIGKILL's, -9), and any status of a CLI that exited by itself proceeds too.
        def ended(code, reached):
            def change(launch):
                launch['stages']['agent'].update(exit_code=code, deadline_reached=reached)
            return change
        for code, reached in ((-9, True), (None, True), (0, False), (1, False)):
            with self.subTest(code=code, reached=reached):
                self.check(ended(code, reached))
        for code, reached in ((None, False), ('-9', True)):
            with self.subTest(code=code, reached=reached), self.assertRaisesRegex(ContractError, 'Inconsistent'):
                self.check(ended(code, reached))

    def test_the_declared_exit_policy_is_enforced_at_the_deadline(self):
        # agent_deadline = "evaluate" with agent_exit = "zero": graded at the deadline, but a CLI that exits by itself
        # must exit cleanly (what a launcher of a submitted agent needs); the default, "any", takes any status.
        def ended(code, reached, policy):
            def change(launch):
                launch['pipeline']['agent_exit'] = policy
                launch['stages']['agent'].update(exit_code=code, deadline_reached=reached)
            return change
        for code, reached, policy in ((None, True, 'zero'), (-9, True, 'zero'), (0, False, 'zero'), (1, False, 'any'),
                                      (0, False, 'any')):
            with self.subTest(code=code, reached=reached, policy=policy):
                self.check(ended(code, reached, policy))
        with self.assertRaisesRegex(ContractError, 'agent did not complete'):
            self.check(ended(1, False, 'zero'))
        def strict(launch):  # agent_deadline = "fail": the agent must finish, with status 0, before its deadline
            launch['pipeline'].pop('agent_deadline')
        with self.assertRaisesRegex(ContractError, 'agent did not complete'):
            self.check(strict)

    def test_private_inputs_never_reach_the_agent_or_submitted_code(self):
        def bind(stage, index, target):
            def change(launch):
                sandbox = launch['stages'][stage]['sandboxes'][index]
                sandbox['binds'].append(['/data/private', target, 'ro'])
                sandbox['mounts'].append([target, 'ro', 'xfs'])
            return change
        for stage, index in (('agent', 0), ('agent', 1), ('intermediate', 1)):
            for target in ('/private', '/private/lambada'):
                with self.subTest(stage=stage, index=index, target=target), \
                        self.assertRaisesRegex(ContractError, 'Private evaluation inputs'):
                    self.check(bind(stage, index, target))
        def mounted(launch):  # in the kernel's mount table without a recorded bind
            launch['stages']['agent']['sandboxes'][1]['mounts'].append(['/private/x', 'ro', 'xfs'])
        with self.assertRaisesRegex(ContractError, 'Private evaluation inputs'):
            self.check(mounted)
        self.check(bind('intermediate', 0, '/private/lambada'))  # the coordinator's
        self.check(bind('agent', 1, '/privateer'))

    def test_a_task_declared_credential_is_kept_out_like_the_framework_ones(self):
        def key(launch):
            launch['stages']['agent']['environments']['command'].append('TASK_GRADER_TOKEN')
        self.check(key)
        launch = copy.deepcopy(LAUNCH)
        key(launch)
        with self.assertRaisesRegex(ContractError, 'credential'):
            checks.check_launch(launch, {'identity': launch['identity']}, {'credential_env': ['TASK_GRADER_TOKEN']})

    def test_the_task_authoring_environment_reaches_the_commands_as_on_docker(self):
        sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
        import importlib.util
        spec = importlib.util.spec_from_file_location('breakwater_apptainer_run', Path(__file__).resolve().parents[1] / 'run.py')
        launcher = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(launcher)
        image = {'PATH': '/usr/bin:/bin'}
        self.assertEqual(launcher.command_environment({'authoring_env': {}}, image), image)
        declared = {'TASK_LLM_BASE_URL': 'http://gateway:8000/v1', 'TASK_CHAT_MODEL': 'chat'}
        self.assertEqual(launcher.command_environment({'authoring_env': declared}, image), {**image, **declared})

    def test_a_failed_evaluation_fails(self):
        def failed(launch):
            launch['stages']['intermediate']['exit_code'] = 1
        with self.assertRaisesRegex(ContractError, 'intermediate'):
            self.check(failed)

    def test_the_judge_gets_times_and_statements_but_no_namespace_numbers(self):
        launch = copy.deepcopy(LAUNCH)
        launch['image_sha256'] = 'a' * 64
        summary = checks.summary(launch)
        self.assertNotIn('pid:[', json.dumps(summary))
        self.assertEqual(summary['agent']['started_s'], 0.0)
        self.assertLessEqual(summary['agent']['stopped_s'], summary['intermediate']['started_s'])
        self.assertEqual(summary['verifier']['sandboxes'][0]['state'], 'running: this is the verifier')
        journal = {'commands': 4, 'stream_digests_verified': 8, 'command_sandbox_restarts': 1}
        mechanical = {'observers': {'commands': journal, 'evaluation-commands': {**journal, 'command_sandbox_restarts': 0}},
                      'gateway': {'calls': 7},
                      'native_agreement': {'bytes': 10, 'sha256': 'b' * 64, 'tool_calls_compared': 4,
                                           'tool_results_compared': 4}}
        facts = checks.established(mechanical, launch)
        self.assertIn('16 streams verified', facts['capture_complete'])
        self.assertIn('2 command sandboxes', facts['three_containers'])
        self.assertIn('a' * 64, facts['runtime_package_integrity'])
        self.assertIn('for your review', facts['sealed_submission'])
        # Claude Code is the default; with Codex the statements name its placeholder key and its executor.
        self.assertIn("agent CLI's ANTHROPIC_API_KEY", facts['log_isolation'])
        self.assertNotIn('exec-server', facts['capture_complete'])
        self.assertNotIn('relayed', facts['three_containers'])
        launch['stages']['agent']['agent_cli'] = 'codex'
        launch['stages']['intermediate']['relayed_ports'] = [8000]
        facts = checks.established(mechanical, launch)
        self.assertIn("agent CLI's BREAKWATER_AUTHOR_KEY", facts['log_isolation'])
        self.assertIn('codex exec-server', facts['capture_complete'])
        self.assertIn("served port 8000 in the submission sandbox; the host relayed each to the coordinator",
                      facts['three_containers'])
        self.assertNotIn('development gateway', facts['log_isolation'])
        launch['stages']['agent']['services'] = [{'name': 'dev-gateway', 'host': 'gateway', 'port': 8000}]
        self.assertIn('reached it as gateway:8000', checks.established(mechanical, launch)['log_isolation'])
        launch['stages']['agent']['endpoints'] = [{'name': 'node001', 'ports': 200}]
        launch['stages']['intermediate']['endpoints'] = {'coordinator': [{'name': 'node001', 'ports': 201}],
                                                         'submission': [{'name': 'node001', 'ports': 200}]}
        self.assertIn("from the agent's commands 200 ports on 1 named host (node001), from the submitted code 200 ports on "
                      '1 named host (node001), and from the coordinator 201 ports on 1 named host (node001)',
                      checks.established(mechanical, launch)['three_containers'])
        launch['stages']['agent']['inbound_ports'] = [30000, 30001]
        self.assertIn("only through its relays of ports 30000-30001 into the command sandbox",
                      checks.established(mechanical, launch)['three_containers'])
        launch['stages']['intermediate']['host_services'] = [18081]
        self.assertIn("operator's service on the launcher's host at 127.0.0.1:18081",
                      checks.established(mechanical, launch)['three_containers'])


if __name__ == '__main__':
    unittest.main()
