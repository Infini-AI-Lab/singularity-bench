"""Launch, evidence and review contracts: the authoring deadline, container limits, review
volume, the rule-violation reward, workspace links, the shadow policy and the cluster
submission rules. No Docker daemon, GPU, model or API key is used.
"""
import copy
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import resource
import subprocess
import sys
import tarfile
import tempfile
import threading
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import zipfile

SECURITY = Path(__file__).resolve().parents[1]
REPO = SECURITY.parent / 'tasks'
sys.path[:0] = [str(SECURITY), str(SECURITY / 'tests')]
from contract import ContractError, RulesViolation
import run as controller
import verifier
from reconcile import compare, native_calls
from test_security import inspected, native


def events(*rows):
    return b''.join(json.dumps(row).encode() + b'\n' for row in rows)


def write_config(folder, **changes):
    """A valid breakwater.json (the harness task's) with ``changes`` applied; None removes a key."""
    config = json.loads((REPO / 'coding-agent-harness/breakwater.json').read_text())
    for key, value in changes.items():
        if value is None:
            config.pop(key, None)
        else:
            config[key] = value
    path = Path(folder) / 'breakwater.json'
    path.write_text(json.dumps(config))
    return path


FAKE_DOCKER = '''import os, sys
args = sys.argv[1:]
assert args[0] == 'exec', args
index = 1
while args[index].startswith('-'):
    index += 2 if args[index] in ('--user', '-u') else 1
command = [item.replace('/logs/agent/', os.environ['FAKE_AGENT_LOGS'] + '/') for item in args[index + 1:]]
os.execvp(command[0], command)
'''
# A CLI that is mid-call when the budget ends: one started command, a half-written line, then work.
STILL_RUNNING = ('import sys,time\n'
                 'sys.stdout.write(\'{"type":"item.started","item":{"type":"command_execution","id":"c1",'
                 '"command":"train"}}\\n{"type":"item.upd\');sys.stdout.flush()\n'
                 'sys.stderr.write("working\\n");sys.stderr.flush()\n'
                 'time.sleep(60)')


class AuthoringDeadlineTests(unittest.TestCase):
    """agent_deadline: 'evaluate' grades what exists at the end of the budget; 'fail' keeps the old rule."""

    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)

    def launch(self, agent, policy=None):
        stages = {}
        for index, stage in enumerate(('agent', 'intermediate', 'verifier')):
            stages[stage] = {'inspection': inspected(stage, self.root, stage), 'started_ns': index * 10,
                             'stopped_ns': index * 10 + 5, 'exit_code': 0,
                             'probe': {'status': 'INSPECTED' if stage == 'verifier' else 'PASS'}}
        for stage in ('agent', 'intermediate'):
            stages[stage]['process_drain'] = {'status': 'PASS', 'remaining_pids': [], 'container_id': stage,
                'identity': {'run_id': 'x', 'stage': stage}, 'cgroups': [{'path': '/docker/' + stage}],
                'docker_state': {'running': False, 'pid': 0}, 'controller_drained_ns': stages[stage]['stopped_ns']}
        stages['agent'].update(agent)
        pipeline = {'gpus': 2} if policy is None else {'gpus': 2, 'agent_deadline': policy}
        return {'identity': {'run_id': 'x'}, 'stages': stages, 'pipeline': pipeline,
                'external_host_root': str(self.root), 'protected_host_roots': [str(self.root)]}

    def test_launch_records_accepted_only_under_the_sealed_policy(self):
        identity = {'identity': {'run_id': 'x'}}
        accepted = {'evaluate': [{'exit_code': None, 'deadline_reached': True}, {'exit_code': 1, 'deadline_reached': False},
                                 {'exit_code': 0, 'deadline_reached': False}],
                    'fail': [{'exit_code': 0, 'deadline_reached': False}, {'exit_code': 0}]}
        rejected = {'evaluate': [{'exit_code': None, 'deadline_reached': False}, {'exit_code': 0, 'deadline_reached': True},
                                 {'exit_code': '1', 'deadline_reached': False}, {'exit_code': True, 'deadline_reached': False},
                                 {'exit_code': None}, {'exit_code': 0, 'deadline_reached': 1}],
                    'fail': [{'exit_code': None, 'deadline_reached': True}, {'exit_code': 1, 'deadline_reached': False},
                             {'exit_code': 0, 'deadline_reached': True}]}
        for policy in ('evaluate', 'fail', None):
            for record in accepted[policy or 'fail']:
                with self.subTest(policy=policy, record=record):
                    verifier.check_launch(self.launch(record, policy), identity)
            for record in rejected[policy or 'fail']:
                with self.subTest(policy=policy, record=record), self.assertRaises(ContractError):
                    verifier.check_launch(self.launch(record, policy), identity)
        # The evaluation stage must still complete, whatever the authoring policy.
        launch = self.launch({'exit_code': None, 'deadline_reached': True}, 'evaluate')
        launch['stages']['intermediate']['exit_code'] = 1
        with self.assertRaises(ContractError):
            verifier.check_launch(launch, identity)

    def test_native_log_tolerates_only_the_in_flight_call_and_cut_line_at_the_deadline(self):
        path = self.root / 'native.jsonl'
        started = {'type': 'item.started', 'item': {'type': 'command_execution', 'id': 'cmd-2', 'command': 'python eval.py'}}
        path.write_bytes(native() + events(started) + b'{"type": "item.upd')
        with self.assertRaises(ContractError):
            native_calls(path)
        value = native_calls(path, deadline_reached=True)
        self.assertEqual((value['in_flight'], value['final_line_cut_short']), (['cmd-2'], True))
        self.assertEqual(len(value['calls']), 2)
        # A complete log stays complete; a call still running with a complete last line is fine too.
        path.write_bytes(native() + events(started))
        self.assertEqual(native_calls(path, deadline_reached=True)['in_flight'], ['cmd-2'])
        with self.assertRaises(ContractError):
            native_calls(path)
        second = {'type': 'item.started', 'item': {'type': 'command_execution', 'id': 'cmd-3', 'command': 'ls'}}
        orphan = {'type': 'user', 'message': {'content': [{'type': 'tool_result', 'tool_use_id': 'never-called',
                                                           'content': 'forged'}]}}
        for payload in (native() + events(started, second), native() + events(orphan), b'{"type": "item.upd'):
            path.write_bytes(payload)
            with self.subTest(payload=payload[-40:]), self.assertRaises(ContractError):
                native_calls(path, deadline_reached=True)

    def test_byte_comparison_still_applies_at_the_deadline(self):
        local, external = self.root / 'local', self.root / 'external'
        started = {'type': 'item.started', 'item': {'type': 'command_execution', 'id': 'cmd-2', 'command': 'python eval.py'}}
        payload = native() + events(started) + b'{"type": "item.upd'
        local.write_bytes(payload)
        external.write_bytes(payload)
        report = compare(local, external, deadline_reached=True)
        self.assertEqual((report['in_flight_calls'], report['final_line_cut_short']), (['cmd-2'], True))
        with self.assertRaises(ContractError):
            compare(local, external)
        local.write_bytes(payload[:-3])
        with self.assertRaises(ContractError):
            compare(local, external, deadline_reached=True)

    def run_cli(self, on_deadline):
        tools, logs, external = self.root / 'bin', self.root / 'logs', self.root / 'external'
        for folder in (tools, logs, external):
            folder.mkdir(exist_ok=True)
        docker = tools / 'docker'
        docker.write_text('#!' + sys.executable + ' -I\n' + FAKE_DOCKER)
        docker.chmod(0o755)
        with patch.dict(os.environ, {'PATH': f"{tools}{os.pathsep}{os.environ['PATH']}", 'FAKE_AGENT_LOGS': str(logs)}):
            started = time.monotonic()
            try:
                return controller.captured_agent('agent', [sys.executable, '-I', '-B', '-c', STILL_RUNNING],
                                                 external, 2.0, on_deadline=on_deadline)
            finally:
                self.assertLess(time.monotonic() - started, 30)

    def test_cli_still_running_at_the_deadline_is_stopped_with_identical_copies(self):
        self.assertIsNone(self.run_cli('stop'))
        captured = (self.root / 'external/agent-native.jsonl').read_bytes()
        self.assertTrue(captured.startswith(b'{"type":"item.started"') and captured.endswith(b'{"type":"item.upd'))
        self.assertEqual((self.root / 'logs/native.jsonl').read_bytes(), captured)
        self.assertEqual((self.root / 'logs/native.stderr').read_bytes(),
                         (self.root / 'external/agent-native.stderr').read_bytes())
        self.assertEqual(native_calls(self.root / 'external/agent-native.jsonl', deadline_reached=True)['in_flight'], ['c1'])

    def test_fail_policy_still_raises_at_the_deadline(self):
        with self.assertRaises(ContractError):
            self.run_cli('raise')

    def test_policy_is_validated_and_sealed(self):
        with tempfile.TemporaryDirectory() as folder:
            self.assertEqual(controller.load_config(write_config(folder, agent_deadline=None))['agent_deadline'], 'fail')
            for value in ('evaluate', 'fail'):
                self.assertEqual(controller.load_config(write_config(folder, agent_deadline=value))['agent_deadline'], value)
            for value in ('stop', True, 1):
                with self.subTest(value=value), self.assertRaises(ContractError):
                    controller.load_config(write_config(folder, agent_deadline=value))
        self.assertIn('agent_deadline', controller.SEALED_PIPELINE)


class ResourceLimitTests(unittest.TestCase):
    def test_memlock_stack_pids_and_cpuset_become_docker_limits(self):
        with tempfile.TemporaryDirectory() as folder:
            config = controller.load_config(write_config(folder, resources={
                'cpus': 16, 'cpuset': '0-7,16-23', 'memory_mb': 1024, 'nofile': 4096, 'memlock': -1,
                'stack': 67108864, 'pids': 512, 'shm_size': '64g'}))
            self.assertEqual(controller.resource_options(config), [
                '--cpus', '16', '--cpuset-cpus', '0-7,16-23', '--memory', '1024m', '--memory-swap', '1024m',
                '--pids-limit', '512', '--ulimit', 'nofile=4096:4096', '--ulimit', 'memlock=-1:-1',
                '--ulimit', 'stack=67108864:67108864', '--shm-size=64g'])

    def test_agent_resources_override_only_the_authoring_container(self):
        with tempfile.TemporaryDirectory() as folder:
            config = controller.load_config(write_config(folder, resources={'cpus': 16, 'memory_mb': 40960, 'nofile': 65536},
                                                         agent_resources={'cpus': 8, 'memory_mb': 32768, 'pids': 512}))
            self.assertEqual(controller.resource_options(config),
                             ['--cpus', '16', '--memory', '40960m', '--memory-swap', '40960m',
                              '--ulimit', 'nofile=65536:65536', '--shm-size=16g'])
            self.assertEqual(controller.resource_options(config, controller.agent_resources(config)),
                             ['--cpus', '8', '--memory', '32768m', '--memory-swap', '32768m', '--pids-limit', '512',
                              '--ulimit', 'nofile=65536:65536', '--shm-size=16g'])

    def test_invalid_limits_are_rejected(self):
        bad = [{'memlock': 0}, {'stack': '64m'}, {'pids': -1}, {'pids': True}, {'cpus': 0}, {'shm_size': '16gb'},
               {'cpuset': '0-'}, {'cpuset': '7-3'}, {'cpus': 8, 'cpuset': '0-3'}, {'ulimit': 1}]
        with tempfile.TemporaryDirectory() as folder:
            for resources in bad:
                with self.subTest(resources=resources), self.assertRaises(ContractError):
                    controller.load_config(write_config(folder, resources=resources))
            for agent in ({'gpus': 1}, {'cpus': 4, 'cpuset': '0-1'}, []):
                with self.subTest(agent=agent), self.assertRaises(ContractError):
                    controller.load_config(write_config(folder, agent_resources=agent))


class ReviewVolumeTests(unittest.TestCase):
    """Data payloads by metadata, pickles in full, recomputed outputs summarized, bounded concurrency."""

    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)

    def packets(self, path, **options):
        import review_index
        return list(review_index.file_packets('submission/' + path.name, path, **options))

    def test_data_payloads_are_metadata_and_text_stays_complete(self):
        text = self.root / 'train.py'
        text.write_text('print("x")\n' * 20000)
        self.assertEqual(len(self.packets(text)), 4)
        self.assertEqual(''.join(body for _, body in self.packets(text)), text.read_text())
        for name, data in (('weights.bin', b'abc'), ('blob.dat', b'\x00' * 300000), ('shard.distcp', b'x' * 300000),
                           ('table.parquet', b'PAR1' + b'x' * 300000)):
            path = self.root / name
            path.write_bytes(data)
            packets = self.packets(path)
            with self.subTest(name=name):
                self.assertEqual(len(packets), 1)
                value = json.loads(packets[0][1])
                self.assertEqual((value['kind'], value['bytes'], value['sha256']),
                                 ('data_payload', len(data), hashlib.sha256(data).hexdigest()))
                self.assertLessEqual(len(value['head']), 4 * 1024)
        listed = self.root / 'tokens.jsonl'
        listed.write_text('{"ids": [1, 2, 3]}\n' * 50000)
        self.assertGreater(len(self.packets(listed)), 10)
        self.assertEqual(json.loads(self.packets(listed, metadata=True)[0][1])['kind'], 'data_payload')
        self.assertEqual(self.packets(text, metadata=True), self.packets(text))  # code stays reviewable

    def test_pickles_stay_reviewable_inside_and_outside_archives(self):
        payload = b'\x80\x04\x95\x10\x00cos\nsystem\n(S"id"\ntR.' + b'\x00' * 100
        standalone = self.root / 'state.pkl'
        standalone.write_bytes(payload)
        self.assertIn('system', ''.join(body for _, body in self.packets(standalone)))
        checkpoint = self.root / 'model.pt'
        with zipfile.ZipFile(checkpoint, 'w') as archive:
            archive.writestr('archive/data.pkl', payload)
            archive.writestr('archive/data/0', b'\x00' * 100000)
            archive.writestr('archive/version', '3\n')
        packets = self.packets(checkpoint)
        summary = json.loads(''.join(body for reference, body in packets if '[metadata]' in reference))
        self.assertEqual([member['name'] for member in summary['zip_members']],
                         ['archive/data.pkl', 'archive/data/0', 'archive/version'])
        pickled = [(reference, body) for reference, body in packets if '::archive/data.pkl' in reference]
        self.assertTrue(pickled)
        self.assertIn('system', ''.join(body for _, body in pickled))
        self.assertFalse(any('::archive/data/0' in reference for reference, _ in packets))
        # A legacy (non-zip) torch save: the leading pickle streams in full, the storage bytes as metadata.
        import pickle
        legacy = self.root / 'legacy.pt'
        header = pickle.dumps(119547037146038801333356, protocol=2) + pickle.dumps({'w': 'os.system'}, protocol=2)
        legacy.write_bytes(header + b'\x00' * 500000)
        packets = self.packets(legacy)
        self.assertEqual(json.loads(packets[0][1])['leading_pickle_bytes'], len(header))
        self.assertIn('os.system', ''.join(body for reference, body in packets if '[pickle]' in reference))
        self.assertEqual(len(packets), 2)
        # One that cannot be delimited is reviewed in full.
        broken = self.root / 'broken.bin'
        broken.write_bytes(b'\x80\x04\x95\xff\xff\xff\xff\xff\xff\xff\x7f' + b'\x00' * 200000)
        self.assertEqual(len(self.packets(broken)), 4)

    def test_recomputed_evaluation_outputs_are_summarized_by_kind(self):
        import review_index
        external, evaluation = self.root / 'external', self.root / 'evaluation'
        (evaluation / 'raw').mkdir(parents=True)
        external.mkdir()
        capture = {'records': [{'text': 'x' * 1000, 'ok': True}] * 300, 't_start': 0.0, 'window_s': 900}
        (evaluation / 'raw/traffic-pass-0.json').write_text(json.dumps(capture))
        (evaluation / 'raw/accuracy.generations.json').write_text(json.dumps([{'id': i, 'generation': 'y' * 500}
                                                                               for i in range(560)]))
        (evaluation / 'raw/diagnostics.json').write_text(json.dumps({'reward': 1.5}))
        roots = {'external': external, 'evaluation': evaluation}
        packets = list(review_index.packets(roots, {}, config={'kind': 'serving'}))
        summaries = {reference: json.loads(body) for reference, body in packets if reference.endswith('[recomputed-output]')}
        self.assertEqual({value['file']: value['rows'] for value in summaries.values()},
                         {'raw/traffic-pass-0.json': 300, 'raw/accuracy.generations.json': 560})
        size = (evaluation / 'raw/diagnostics.json').stat().st_size
        self.assertIn(f'evaluation/raw/diagnostics.json[bytes 0:{size}]', [reference for reference, _ in packets])
        # Another kind's outputs are reviewed in full.
        full = list(review_index.packets(roots, {}, config={'kind': 'harness'}))
        self.assertGreater(len(full), len(packets))
        (evaluation / 'raw/rl/evaluation').mkdir(parents=True)
        (evaluation / 'raw/rl/evaluation/amc.generations.jsonl').write_text('{"index": 0}\n{"index": 1}\n')
        midtrain = list(review_index.packets(roots, {}, config={'kind': 'midtrain'}))
        self.assertEqual([json.loads(body)['rows'] for reference, body in midtrain
                          if reference.endswith('amc.generations.jsonl[recomputed-output]')], [2])

    def test_workspace_view_lists_links_and_applies_metadata_paths(self):
        import review_index
        from contract import seal
        external, submission = self.root / 'external', self.root / 'submission'
        (submission / 'out/processed_dataset').mkdir(parents=True)
        external.mkdir()
        (submission / 'out/processed_dataset/part-0.jsonl').write_text('{"text": "row"}\n' * 40000)
        (submission / 'train.py').write_text('print(1)\n')
        seal(submission, {'run_id': 'r'}, symlinks={'latest': 'out/processed_dataset'})
        (external / 'workspace-baseline.json').write_text(json.dumps({'files': {}}))
        roots = {'external': external, 'submission': submission}
        packets = list(review_index.packets(roots, {}, config={'review_metadata_paths': ['out/processed_dataset']}))
        summary = json.loads(''.join(body for reference, body in packets if 'diff-summary' in reference)
                             .split(' ', 6)[-1])
        self.assertEqual(summary['symbolic_links_not_sealed_as_files'], {'latest': 'out/processed_dataset'})
        dataset = [body for reference, body in packets if 'part-0.jsonl' in reference]
        self.assertEqual(len(dataset), 1)
        self.assertEqual(json.loads(dataset[0])['kind'], 'data_payload')
        self.assertIn('submission/train.py[bytes 0:9]', [reference for reference, _ in packets])

    def test_bounded_review_keeps_order_limits_concurrency_and_stops_on_failure(self):
        lock, state = threading.Lock(), {'running': 0, 'peak': 0, 'calls': 0}
        def call(index, item):
            with lock:
                state['running'] += 1
                state['calls'] += 1
                state['peak'] = max(state['peak'], state['running'])
            time.sleep(.02)
            with lock:
                state['running'] -= 1
            if item == 'fail':
                raise ContractError('No valid judge review')
            return item * 2
        items = list(range(40))
        self.assertEqual(verifier.bounded(call, iter(items)), [item * 2 for item in items])
        self.assertLessEqual(state['peak'], verifier.REVIEW_CONCURRENCY)
        self.assertGreater(state['peak'], 1)
        state.update(calls=0)
        with self.assertRaises(ContractError):
            verifier.bounded(call, iter(['fail'] + list(range(100))))
        self.assertLess(state['calls'], 100)

    def test_concurrent_retries_keep_every_rejected_attempt(self):
        attempts = {}
        def judge_call(prompt_path, name, remaining):
            attempts[name] = attempts.get(name, 0) + 1
            if attempts[name] == 1:
                raise ValueError('malformed reply')
            return {'verdict': 'PASS'}
        with tempfile.TemporaryDirectory() as folder, patch.object(verifier, 'REPORTS', Path(folder)), \
                patch.object(verifier, 'judge_call', judge_call):
            deadline = time.monotonic() + 60
            results = verifier.bounded(lambda index, item: verifier.invoke('P', f'packet-{index:06d}', deadline), range(32))
            record = json.loads((Path(folder) / 'judge-retries.json').read_text())
        self.assertEqual(len(results), 32)
        self.assertEqual(sorted(row['call'] for row in record['rejected']), [f'packet-{index:06d}' for index in range(32)])


class RuleViolationRewardTests(unittest.TestCase):
    def run_verifier(self, config):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            launch = {'external_sha256': 'a' * 64, 'identity': {'task': config['task']}}
            with patch.object(verifier, 'REPORTS', directory), patch.object(verifier, 'read_json', return_value=config), \
                 patch.object(verifier, 'verify_seal', side_effect=[launch, {}]), \
                 patch.object(verifier, 'check_launch', side_effect=RulesViolation('a protected package was replaced')), \
                 patch.object(verifier.os, 'geteuid', return_value=1102), patch.dict(os.environ, {'BREAKWATER_STAGE': 'verifier'}), \
                 patch.object(sys, 'argv', ['verifier.py', '--authority', 'a' * 64]), patch.object(verifier, 'audit') as judge:
                self.assertEqual(verifier.main(), 2)
                judge.assert_not_called()
            return (json.loads((directory / 'reward.json').read_text()),
                    json.loads((directory / 'rule-violation.json').read_text()))

    def test_task_declared_violation_reward_is_written(self):
        reward, record = self.run_verifier({'failure_reward': -1.0, 'rule_violation_reward': -1.0, 'task': 'open-ended-nano-gpt'})
        self.assertEqual(reward, {'reward': -1.0})
        self.assertEqual((record['reward'], record['reason'], record['judge_skipped']), (-1.0, 'violation of rules', True))
        reward, record = self.run_verifier({'failure_reward': -1.0, 'task': 'open-ended-nano-gpt'})
        self.assertEqual((reward, record['reward']), ({'reward': 0.0}, 0.0))

    def test_violation_reward_is_validated_and_nano_declares_minus_one(self):
        with tempfile.TemporaryDirectory() as folder:
            for value in (True, '0', [0]):
                with self.subTest(value=value), self.assertRaises(ContractError):
                    controller.load_config(write_config(folder, rule_violation_reward=value))
            self.assertEqual(controller.load_config(write_config(folder))['rule_violation_reward'], 0.0)
        self.assertEqual(controller.load_config(REPO / 'open-ended-nano-gpt/breakwater.json')['rule_violation_reward'], -1.0)
        self.assertIn('rule_violation_reward', controller.SEALED_PIPELINE)


def tar_stream(entries):
    stream = io.BytesIO()
    with tarfile.open(fileobj=stream, mode='w') as archive:
        for name, kind, value in entries:
            member = tarfile.TarInfo(name)
            member.mode = 0o755 if kind == 'dir' else 0o644
            if kind == 'dir':
                member.type = tarfile.DIRTYPE
                archive.addfile(member)
            elif kind == 'file':
                member.size = len(value)
                archive.addfile(member, io.BytesIO(value))
            else:
                member.type = {'symlink': tarfile.SYMTYPE, 'hardlink': tarfile.LNKTYPE, 'fifo': tarfile.FIFOTYPE}[kind]
                member.linkname = value or ''
                archive.addfile(member)
    stream.seek(0)
    return stream


class WorkspaceLinkTests(unittest.TestCase):
    """Anchored seeding excludes, snapshot excludes, recorded links and safe link installation."""

    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)

    def snapshot(self, entries, excludes=(), links=None):
        process = SimpleNamespace(stdout=tar_stream(entries), stderr=io.BytesIO(), wait=lambda **_: 0, poll=lambda: 0)
        destination = self.root / ('snapshot-' + str(len(list(self.root.iterdir()))))
        with patch.object(controller.subprocess, 'Popen', return_value=process):
            controller.snapshot('stopped-agent', '/app', destination, list(excludes), links=links)
        return destination

    def test_task_seeding_anchors_leading_slash_excludes_at_the_task_root(self):
        task = self.root / 'task'
        for name in ('README.md', 'nanoGPT/README.md', 'nanoGPT/train.py', 'out/ckpt.bin', 'sub/out/log.txt',
                     'tests/test.sh', 'docs/README.md', 'breakwater.json', 'bindings.example.json'):
            (task / name).parent.mkdir(parents=True, exist_ok=True)
            (task / name).write_text(name)
        with patch.object(controller, 'TASK', task):
            controller.seed_workspace({'workspace_source': 'task', 'workspace_excludes': ['/README.md', 'out', '/docs']},
                                      None, self.root / 'seeded')
        from contract import inventory
        self.assertEqual(sorted(inventory(self.root / 'seeded')), ['nanoGPT/README.md', 'nanoGPT/train.py'])

    def test_sandbox_worker_endpoint_is_validated(self):
        rl = controller.load_config(REPO / 'rl-swe-agent/breakwater.json')
        self.assertEqual(rl['sandbox_worker'], {'port': 11000, 'session_ports': '30000-30015'})
        with tempfile.TemporaryDirectory() as folder:
            self.assertIsNone(controller.load_config(write_config(folder))['sandbox_worker'])
            for value in ({'port': 11000}, {'port': '11000', 'session_ports': '30000-30015'},
                          {'port': 11000, 'session_ports': '80-90'}, {'port': 11000, 'session_ports': '30000-31000'},
                          {'port': 11000, 'session_ports': '30010-30000'}, {'port': 0, 'session_ports': '30000-30001'},
                          {'port': 11000, 'session_ports': '30000-30015', 'host': '0.0.0.0'}, [11000]):
                with self.subTest(value=value), self.assertRaises(ContractError):
                    controller.load_config(write_config(folder, sandbox_worker=value))

    def test_dataset_manifest_stays_inside_the_workspace(self):
        midtrain = controller.load_config(REPO / 'midtrain-for-rl/breakwater.json')
        self.assertEqual((midtrain['dataset_manifest'], midtrain['dataset_roots']),
                         ('/app/out/training_dataset.json', ['/app/out/processed_dataset']))
        with tempfile.TemporaryDirectory() as folder:
            self.assertIsNone(controller.load_config(write_config(folder))['dataset_manifest'])
            for changes in ({'dataset_manifest': '/etc/dataset.json'}, {'dataset_roots': ['/app/out']},
                            {'dataset_manifest': '/app/out/d.json', 'dataset_roots': ['/tmp/data']},
                            {'dataset_manifest': '/app/out/d.json', 'dataset_roots': '/app/out'}):
                with self.subTest(changes=changes), self.assertRaises(ContractError):
                    controller.load_config(write_config(folder, **changes))

    def test_exclude_syntax_is_validated(self):
        with tempfile.TemporaryDirectory() as folder:
            for value in (['a/b'], ['/'], ['//x'], ['/a/../b'], [''], 'out'):
                with self.subTest(value=value), self.assertRaises(ContractError):
                    controller.load_config(write_config(folder, workspace_source='task', workspace_excludes=value))
            config = controller.load_config(write_config(folder, workspace_source='task',
                                                         workspace_excludes=['/README.md', '/a/b', 'out']))
            self.assertEqual(config['workspace_excludes'], ['/README.md', '/a/b', 'out'])
            controller.load_config(write_config(folder, workspace_source='image', workspace_excludes=['a/b', '/c']))
            for value in (['/scratch'], ['a/../b'], ['spec']):
                with self.subTest(snapshot=value), self.assertRaises(ContractError):
                    controller.load_config(write_config(folder, snapshot_excludes=value))
            for key in ('review_metadata_paths', 'snapshot_excludes'):
                self.assertEqual(controller.load_config(write_config(folder, **{key: ['out/data']}))[key], ['out/data'])

    def test_symbolic_links_are_recorded_and_skipped_but_special_files_fail(self):
        entries = [('./', 'dir', None), ('./pkg/', 'dir', None), ('./pkg/a.py', 'file', b'print(1)\n'),
                   ('./latest', 'symlink', 'pkg/a.py'), ('./abs', 'symlink', '/etc/passwd'),
                   ('./scratch/', 'dir', None), ('./scratch/big', 'file', b'x'), ('./scratch/link', 'symlink', '/x')]
        links = {}
        destination = self.snapshot(entries, excludes=['scratch'], links=links)
        self.assertEqual(links, {'latest': 'pkg/a.py', 'abs': '/etc/passwd'})
        from contract import inventory
        self.assertEqual(sorted(inventory(destination)), ['pkg/a.py'])
        with self.assertRaises(ContractError):
            self.snapshot(entries)  # without a record, links still fail (logs, evaluation, image seeding)
        for kind, value in (('hardlink', 'pkg/a.py'), ('fifo', None)):
            with self.subTest(kind=kind), self.assertRaises(ContractError):
                self.snapshot(entries[:3] + [('./special', kind, value)], links={})

    def sealed(self, files, links, name='sealed'):
        from contract import seal
        source = self.root / name
        for relative, data in files.items():
            (source / relative).parent.mkdir(parents=True, exist_ok=True)
            (source / relative).write_bytes(data)
        source.mkdir(exist_ok=True)
        seal(source, {'run_id': 'r'}, symlinks=links)
        return source

    def install(self, pairs, allowed=()):
        from workers import install_tree
        return install_tree(pairs, allowed=allowed, uid=os.getuid(), gid=os.getgid())

    def test_in_root_links_are_recreated_without_following(self):
        import workers
        source = self.sealed({'pkg/a.py': b'print(1)\n', 'run.sh': b'#!/bin/sh\n'},
                             {'latest': 'pkg/a.py', 'pkg/self': '.', 'dangling': 'missing.txt', 'into-home': '/home/x'})
        target = self.root / 'app'
        target.mkdir()
        (target / 'stale').write_text('from an earlier cell')
        home = self.root / 'home'
        sealed_home = self.sealed({'.bashrc': b'export X=1\n'}, {'app': str(target)}, name='sealed-home')
        with patch.object(workers, 'SEAL_MANIFEST', 'manifest.json'):
            result = self.install([(source, target), (sealed_home, home)], allowed=(Path('/home/x'),))
        self.assertEqual(result['symlinks'], 5)
        self.assertEqual(sorted(path.name for path in target.iterdir()),
                         ['dangling', 'into-home', 'latest', 'pkg', 'run.sh'])
        self.assertEqual(os.readlink(target / 'latest'), 'pkg/a.py')
        self.assertEqual((target / 'latest').read_text(), 'print(1)\n')
        self.assertEqual(os.readlink(home / 'app'), str(target))
        self.assertFalse((target / 'manifest.json').exists())

    def test_links_resolving_outside_the_roots_are_invalid(self):
        from workers import InvalidSubmission
        target = self.root / 'app'
        cases = {'absolute': ({}, {'passwd': '/etc/passwd'}),
                 'relative': ({}, {'up': '../outside'}),
                 # Lexically inside, but 'l' points at the root, so '../..' climbs out of it.
                 'chain': ({'a/b/keep': b''}, {'a/b/l': str(target), 'y': 'a/b/l/../..'}),
                 'through-link': ({}, {'d': 'sub', 'd/x': 'y'}),
                 'unsafe-name': ({}, {'../x': 'y'})}
        for case, (files, links) in cases.items():
            source = self.sealed(files, links, name='sealed-' + case)
            with self.subTest(case=case), self.assertRaises(InvalidSubmission):
                self.install([(source, target)])

    def test_worker_temporary_files_are_wiped_without_following_links(self):
        from workers import wipe_worker_files
        shared, outside = self.root / 'tmp', self.root / 'outside'
        (shared / 'cache/deep').mkdir(parents=True)
        outside.mkdir()
        (outside / 'keep').write_text('not the worker temporary area')
        (shared / 'cache/deep/state').write_text('cell 1')
        (shared / 'escape').symlink_to(outside, target_is_directory=True)
        (shared / 'socketish').write_text('')
        self.assertEqual(wipe_worker_files(roots=(shared,), uid=os.getuid()), 3)
        self.assertEqual(list(shared.iterdir()), [])
        self.assertEqual((outside / 'keep').read_text(), 'not the worker temporary area')

    def test_verifier_requires_the_link_record_to_match_the_sealed_manifest(self):
        evidence = self.root / 'evidence'
        evidence.mkdir()
        (evidence / 'workspace-links.json').write_text(json.dumps({'workdir': '/app', 'symlinks': {'a': 'b'}}))
        result = verifier.task_mechanical_checks({}, evidence, {'symlinks': {'a': 'b'}})
        self.assertEqual(result['workspace_symlinks']['workspace'], 1)
        for submission, home in (({'symlinks': {}}, None), ({'symlinks': {'a': 'b'}}, {'symlinks': {}})):
            with self.subTest(submission=submission, home=home), self.assertRaises(ContractError):
                verifier.task_mechanical_checks({}, evidence, submission, home)
        (evidence / 'workspace-links.json').write_text(json.dumps({'workdir': '/app', 'symlinks': {},
                                                                   'home': {'path': '/home/worker', 'symlinks': {'v': 'w'}}}))
        self.assertEqual(verifier.task_mechanical_checks({}, evidence, {'symlinks': {}}, {'symlinks': {'v': 'w'}})
                         ['workspace_symlinks'], {'sha256': hashlib.sha256((evidence / 'workspace-links.json')
                                                                           .read_bytes()).hexdigest(),
                                                  'workspace': 0, 'home': 1})


class ShadowPolicyTests(unittest.TestCase):
    """Only import-reachable decoys, hooks and environment markers are flagged."""

    def workspace(self, names):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        root = Path(temporary.name)
        for name in names:
            (root / name).parent.mkdir(parents=True, exist_ok=True)
            (root / name).write_text(name)
        return root

    def test_deep_package_names_are_honest_but_top_level_decoys_are_not(self):
        import environment_guard as guard
        honest = self.workspace(['kernels/triton/matmul.py', 'datasets/train.bin', 'cache/torch/inductor/x.py',
                                 'out/model.pth', 'model.pth', 'src/numpy_utils.py', 'triton.md', 'torch_notes.txt',
                                 'datasets/README.md', 'vendor/megatron/__init__.py'])
        self.assertEqual(guard.shadows(honest), {})
        decoys = self.workspace(['triton.py', 'numpy.pyc', 'torch.cpython-312-aarch64-linux-gnu.so',
                                 'datasets/__init__.py', 'datasets/load.py', 'tools/sitecustomize.py',
                                 'usercustomize.py', 'env/lib/python3.12/site-packages/hook.pth', 'venv/pyvenv.cfg'])
        self.assertEqual(sorted(guard.shadows(decoys)), sorted([
            'triton.py', 'numpy.pyc', 'torch.cpython-312-aarch64-linux-gnu.so', 'datasets/__init__.py',
            'datasets/load.py', 'tools/sitecustomize.py', 'usercustomize.py',
            'env/lib/python3.12/site-packages/hook.pth', 'venv/pyvenv.cfg']))
        self.assertEqual(sorted(guard.shadows(decoys, excludes=['datasets', 'venv'])), sorted([
            'triton.py', 'numpy.pyc', 'torch.cpython-312-aarch64-linux-gnu.so', 'tools/sitecustomize.py',
            'usercustomize.py', 'env/lib/python3.12/site-packages/hook.pth']))

    def test_links_count_like_files_because_evaluation_may_recreate_them(self):
        import environment_guard as guard
        root = self.workspace(['fake/__init__.py', 'torch/ops.py', 'notes.md'])
        recorded = {'torch.py': 'fake/__init__.py', 'numpy': 'fake', 'torch/__init__.py': '../fake/__init__.py',
                    'tools/sitecustomize.py': '../notes.md', 'model': '/models/GLM-5.3-Flash', 'data/latest': 'v2'}
        self.assertEqual(guard.shadows(root, links=recorded), {
            'numpy': 'symlink:fake', 'torch.py': 'symlink:fake/__init__.py',
            'torch/__init__.py': 'symlink:../fake/__init__.py', 'torch/ops.py': hashlib.sha256(b'torch/ops.py').hexdigest(),
            'tools/sitecustomize.py': 'symlink:../notes.md'})
        # Links on disk (a seeded workspace) are found without being passed, and excludes apply to them.
        (root / 'triton').symlink_to('fake', target_is_directory=True)
        (root / 'out').mkdir()
        (root / 'out' / 'torch.py').symlink_to('../notes.md')
        self.assertEqual(guard.shadows(root), {'triton': 'symlink:fake'})
        self.assertEqual(guard.shadows(root, excludes=['triton']), {})
        midtrain = guard.shadow_policy({'packages': [], 'hooks': [], 'environments': False})
        self.assertEqual(guard.shadows(root, midtrain, links=recorded), {})

    def test_task_policies_still_apply(self):
        import environment_guard as guard
        root = self.workspace(['triton.py', 'sitecustomize.py', 'pytest/__init__.py', 'lib/site-packages/x.pth',
                               'megatron/__init__.py'])
        midtrain = guard.shadow_policy(json.loads((REPO / 'midtrain-for-rl/breakwater.json').read_text())['shadow_policy'])
        self.assertEqual(guard.shadows(root, midtrain), {})
        hooks_only = guard.shadow_policy({'packages': [], 'environments': False})
        self.assertEqual(sorted(guard.shadows(root, hooks_only)), ['lib/site-packages/x.pth', 'sitecustomize.py'])
        harness = guard.shadow_policy({'extra_packages': ['pytest']})
        self.assertIn('pytest/__init__.py', guard.shadows(root, harness))
        self.assertIn('megatron/__init__.py', guard.shadows(root, harness))


class ClusterSubmissionTests(unittest.TestCase):
    """Fleet identity and shape, the qualification rule, submission limits and per-cell isolation."""

    def test_fleet_run_id_and_shape(self):
        import cluster_fleet
        self.assertEqual(cluster_fleet.run_id({'run_id': 'top', 'launch': {'run_id': 'old'}}), 'top')
        self.assertEqual(cluster_fleet.run_id({'launch': {'run_id': 'old'}}), 'old')
        self.assertIsNone(cluster_fleet.run_id({'launch': {}}))
        fleet = json.loads((REPO / 'cluster-serverless/tests/fleet-graded.json').read_text())
        catalogue = json.loads((REPO / 'cluster-serverless/tests/public/catalogue.json').read_text())
        self.assertEqual(cluster_fleet.check_fleet_shape(fleet, catalogue), {'nodes': 4, 'gpus': 16})
        public = json.loads((REPO / 'cluster-serverless/tests/fleet-graded.public.json').read_text())
        cluster_fleet.check_fleet_shape(public, catalogue)
        bad = {'three nodes': lambda value: value['nodes'].pop(),
               'eight GPUs': lambda value: value['nodes'][0].update(n_gpus=8),
               'declared shape': lambda value: value.update(gpus_per_node=8),
               'catalogue': None}
        for case, mutate in bad.items():
            value, expected = copy.deepcopy(fleet), dict(catalogue)
            if mutate is None:
                expected['n_gpus'] = 32
            else:
                mutate(value)
            with self.subTest(case=case), self.assertRaises(ContractError):
                cluster_fleet.check_fleet_shape(value, expected)

    def test_only_target_qualified_suites_are_qualified(self):
        import score_workload
        from test_harness_cluster import cluster_tests
        with tempfile.TemporaryDirectory() as temporary:
            tests = cluster_tests(temporary, qualification={'workload': 'smoke-qualified', 'catalogue': 'smoke-qualified'})
            raw = Path(temporary) / 'raw'
            (raw / 'cluster').mkdir(parents=True)
            suite = json.loads((tests / 'suite-graded.json').read_text())
            for name in ('score', 'reference', 'traceunits'):
                sys.modules.pop(name, None)
            sys.path.insert(0, str(tests / 'kit'))
            import score as scorer
            labels = [f'{name}-0' for name in scorer.CELLS]
            measurement = {'labels': labels, 'invalid': 'invalid', 'development_only': False, 'measurement_binding': {},
                           'repeats': 1, 'score_version': scorer.SCORE_VERSION, 'model_ids': suite['model_ids'],
                           'trace_sha256': {case['name']: case['trace_sha256'] for case in suite['cases']},
                           'cells': [{'label': label, 'trace': label.rsplit('-', 1)[0], 'invalid': 'invalid'}
                                     for label in labels]}
            (raw / 'cluster/measurement.json').write_text(json.dumps(measurement))
            with self.assertRaises(ContractError):  # a smoke-qualified suite is a development suite
                score_workload.cluster_score(raw, tests=tests, config={'allow_unqualified': True})
            measurement.update(development_only=True, measurement_binding=None)
            (raw / 'cluster/measurement.json').write_text(json.dumps(measurement))
            self.assertEqual(score_workload.cluster_score(raw, tests=tests, config={'allow_unqualified': True}), 0.0)
            # The sealed submission is rechecked: one over its limits must have been graded invalid.
            submission = Path(temporary) / 'submission'
            submission.mkdir()
            (submission / 'run.sh').write_bytes(b'x' * 64)
            import cluster_cells
            with patch.object(cluster_cells, 'MAX_BYTES', 32):
                self.assertEqual(score_workload.cluster_score(raw, tests=tests, config={'allow_unqualified': True},
                                                              roots=[submission]), 0.0)
                measurement.update(invalid=None)
                (raw / 'cluster/measurement.json').write_text(json.dumps(measurement))
                with self.assertRaises(ContractError):
                    score_workload.cluster_score(raw, tests=tests, config={'allow_unqualified': True}, roots=[submission])

    def test_submission_limits_count_every_entry_and_byte_of_both_roots(self):
        import cluster_cells
        from contract import seal
        with tempfile.TemporaryDirectory() as temporary:
            app, home = Path(temporary) / 'app', Path(temporary) / 'home'
            (app / 'pkg').mkdir(parents=True)
            home.mkdir()
            (app / 'run.sh').write_bytes(b'#' * 100)
            (app / 'pkg/a.py').write_bytes(b'#' * 100)
            (home / 'notes').write_bytes(b'#' * 100)
            seal(app, {'run_id': 'r'}, symlinks={'latest': 'pkg/a.py'})
            seal(home, {'run_id': 'r'}, symlinks={})
            self.assertIsNone(cluster_cells.submission_problem([app, home]))
            # Entries: two roots, one directory, three files and one link.
            with patch.object(cluster_cells, 'MAX_ENTRIES', 6):
                self.assertIn('100000', cluster_cells.submission_problem([app, home]))
            with patch.object(cluster_cells, 'MAX_ENTRIES', 7):
                self.assertIsNone(cluster_cells.submission_problem([app, home]))
            with patch.object(cluster_cells, 'MAX_BYTES', 299):
                self.assertIn('8 GiB', cluster_cells.submission_problem([app, home]))
            with patch.object(cluster_cells, 'MAX_BYTES', 300):
                self.assertIsNone(cluster_cells.submission_problem([app, home]))

    def test_plane_process_limit_and_driver_timeout(self):
        import cluster_cells
        hard = resource.getrlimit(resource.RLIMIT_NPROC)[1]
        limit = 64 if hard == resource.RLIM_INFINITY else min(64, hard)
        result = subprocess.run([sys.executable, '-I', '-B', '-c', cluster_cells.NPROC_LIMIT, str(limit),
                                 sys.executable, '-I', '-B', '-c',
                                 'import resource;print(*resource.getrlimit(resource.RLIMIT_NPROC))'],
                                capture_output=True, text=True, check=True)
        self.assertEqual(result.stdout.split(), [str(limit), str(limit)])
        self.assertEqual(cluster_cells.PLANE_PROCESSES, 512)
        self.assertEqual(cluster_cells.driver_timeout(1000.0, 1), 1120.0)
        self.assertEqual(cluster_cells.driver_timeout(1000.0, 4), 370.0)


class ServingScoreTests(unittest.TestCase):
    """The verifier recomputes serving income from the complete capture and the accuracy generations."""

    def test_reward_recomputes_from_the_capture_and_rejects_a_changed_reward(self):
        import score_workload
        tests = REPO / 'serve-glm53-flash/tests'
        sys.path.insert(0, str(tests))
        try:
            def load(name):
                spec = importlib.util.spec_from_file_location('serving_' + name, tests / f'{name}.py')
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
                return module
            simulator, scorer = load('simulator'), load('score')
        finally:
            sys.path.remove(str(tests))
        items = json.loads((tests / 'hidden_eval.json').read_text())
        generations = [{'id': item['id'], 'benchmark': item['benchmark'], 'generation': '\\boxed{' + item['gold'] + '}'}
                       for item in items]
        record = {'primer': False, 'ok': True, 'error': None, 'input_tokens': 1_000_000, 'cached_tokens': 200_000,
                  'reported_output_tokens': 10_000, 'max_tokens': 10_000, 'text': 'x' * 10_000,
                  't_send': 0, 't_first': .1, 't_last': 1, 't_done': 2, 'position': 0}
        with tempfile.TemporaryDirectory() as temporary:
            raw = Path(temporary)
            (raw / 'traffic-pass-0.json').write_text(json.dumps({'records': [record], 't_start': 0.0, 'window_s': 900}))
            (raw / 'accuracy.generations.json').write_text(json.dumps(generations))
            (raw / 'diagnostics.json').write_text(json.dumps({'reward': 0.093}))
            self.assertEqual(score_workload.serving_score(raw, simulator, scorer.score_candidate), 0.093)
            (raw / 'diagnostics.json').write_text(json.dumps({'reward': 0.2}))
            with self.assertRaises(ContractError):
                score_workload.serving_score(raw, simulator, scorer.score_candidate)
            (raw / 'traffic-pass-0.json').write_text(json.dumps({'records': [record], 't_start': 0.0, 'window_s': 300}))
            with self.assertRaises(ContractError):
                score_workload.serving_score(raw, simulator, scorer.score_candidate)


if __name__ == '__main__':
    unittest.main()
