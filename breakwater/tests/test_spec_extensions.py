"""The task specification's extension points (SPEC.md): agent_exit, [breakwater.operator], [breakwater.model_access],
[breakwater.authoring] env and inbound ports, [breakwater.host], [breakwater.options] and [[breakwater.endpoints]], and
the integrity stamps a launcher seals (the framework revision, the digest of the task's tests). No Docker or GPUs.
"""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

FRAMEWORK = Path(__file__).resolve().parents[1]
EXAMPLE = FRAMEWORK.parent / 'examples' / 'word-count'
sys.path[:0] = [str(FRAMEWORK)]
from contract import FRAMEWORK_REVISION, ContractError  # noqa: E402
from contract import tests_digest as digest_of_tests  # noqa: E402  (not collected as a test)
import run as controller  # noqa: E402
import spec  # noqa: E402

ACCESS = {'upstream': 'https://models.example/v1', 'chat_models': {'chat': 'org/model-a', 'chat-b': 'org/model-b'}}


def resolve(**changes):
    """The example task's declared settings with ``changes`` applied, resolved."""
    return spec.resolve({**spec.read(EXAMPLE), **changes})


class AgentExitTests(unittest.TestCase):
    def test_the_exit_policy_follows_the_deadline_policy_unless_declared(self):
        self.assertEqual(resolve(agent_deadline='fail')['agent_exit'], 'zero')
        self.assertEqual(resolve(agent_deadline='evaluate')['agent_exit'], 'any')
        self.assertEqual(resolve(agent_deadline='evaluate', agent_exit='zero')['agent_exit'], 'zero')
        for changes in ({'agent_deadline': 'fail', 'agent_exit': 'any'}, {'agent_exit': 'sometimes'},
                        {'agent_deadline': 'evaluate', 'agent_exit': 1}):
            with self.subTest(changes=changes), self.assertRaises(ContractError):
                resolve(**changes)

    def test_the_exit_policy_and_options_are_sealed(self):
        self.assertIn('agent_exit', spec.SEALED_PIPELINE)
        self.assertIn('options', spec.SEALED_PIPELINE)


class OperatorTests(unittest.TestCase):
    def test_declared_inputs_values_and_secrets_resolve(self):
        config = resolve(operator_inputs=['fleet', 'public-fleet'], operator_values={'receipt-key': 'TASK_RECEIPT_KEY'},
                         operator_secrets={'grader-token': 'TASK_GRADER_TOKEN'})
        self.assertEqual(config['operator_inputs'], ['fleet', 'public-fleet'])
        self.assertEqual(config['credential_env'], ['TASK_GRADER_TOKEN'])

    def test_names_variables_and_overlaps_are_validated(self):
        for changes in ({'operator_inputs': ['Fleet']}, {'operator_inputs': ['fleet', 'fleet']},
                        {'operator_inputs': 'fleet'}, {'operator_values': {'key': 'lower_case'}},
                        {'operator_values': {'key_name': 'TASK_KEY'}}, {'operator_secrets': {'token': 'PATH'}},
                        {'operator_secrets': {'token': 'LD_PRELOAD'}}, {'operator_secrets': {'token': 'PYTHONPATH'}},
                        {'operator_values': {'key': 'TASK_KEY'}, 'operator_secrets': {'other': 'TASK_KEY'}},
                        {'operator_inputs': ['key'], 'operator_values': {'key': 'TASK_KEY'}}):
            with self.subTest(changes=changes), self.assertRaises(ContractError):
                resolve(**changes)


class ModelAccessTests(unittest.TestCase):
    def test_defaults_and_the_key_variable_join_the_credentials(self):
        config = resolve(development_gateway=ACCESS)
        self.assertEqual({key: config['development_gateway'][key] for key in ('host', 'port', 'key_env')},
                         {'host': 'gateway', 'port': 8000, 'key_env': 'FIREWORKS_API_KEY'})
        config = resolve(development_gateway={**ACCESS, 'key_env': 'TASK_MODEL_KEY', 'host': 'models', 'port': 9000},
                         operator_secrets={'token': 'TASK_TOKEN'})
        self.assertEqual(config['credential_env'], ['TASK_MODEL_KEY', 'TASK_TOKEN'])
        self.assertIsNone(resolve()['development_gateway'])
        self.assertEqual(resolve()['credential_env'], [])

    def test_models_budgets_and_addresses_are_validated(self):
        for access in ({'upstream': 'https://x'}, {**ACCESS, 'chat_models': {}}, {**ACCESS, 'chat_models': {'Chat': 'm'}},
                       {**ACCESS, 'chat_models': {'chat': ''}}, {**ACCESS, 'upstream': 'ftp://x'},
                       {**ACCESS, 'max_chat_tokens': 0}, {**ACCESS, 'max_embed_tokens': 1.5}, {**ACCESS, 'port': 0},
                       {**ACCESS, 'host': 'bad host'}, {**ACCESS, 'key_env': 'lower'}, {**ACCESS, 'embed_model': ''},
                       {**ACCESS, 'models': {}}):
            with self.subTest(access=access), self.assertRaises(ContractError):
                resolve(development_gateway=access)


class AuthoringTests(unittest.TestCase):
    def test_environment_names_and_values_are_validated(self):
        env = {'TASK_BASE_URL': 'http://gateway:8000/v1', 'TASK_MODEL': 'chat'}
        self.assertEqual(resolve(authoring_env=env)['authoring_env'], env)
        for bad in ({'FIREWORKS_API_KEY': 'x'}, {'PATH': '/x'}, {'HOME': '/x'}, {'HF_HOME': '/x'}, {'LD_PRELOAD': 'x'},
                    {'PYTHONPATH': 'x'}, {'lower': 'x'}, {'TASK': 1}, {'TASK': 'a\0b'}, {'TASK': 'x' * 65537}):
            with self.subTest(env=bad), self.assertRaises(ContractError):
                resolve(authoring_env=bad)
        with self.assertRaises(ContractError):
            resolve(authoring_env={'TASK_TOKEN': 'x'}, operator_secrets={'token': 'TASK_TOKEN'})

    def test_inbound_ports_are_a_bounded_unprivileged_range_with_a_host_name(self):
        config = resolve(inbound_ports='30000-30015')
        self.assertEqual((config['inbound_ports'], config['inbound_host']), ('30000-30015', 'host.docker.internal'))
        self.assertEqual(resolve(inbound_ports='30000-30015', inbound_host='launcher')['inbound_host'], 'launcher')
        self.assertIsNone(resolve()['inbound_host'])
        for ports in ('30000-30064', '1000-1010', '30015-30000', '30000', 30000):
            with self.subTest(ports=ports), self.assertRaises(ContractError):
                resolve(inbound_ports=ports)


class HostModuleTests(unittest.TestCase):
    def test_the_host_module_is_a_python_file_under_tests(self):
        self.assertEqual(resolve(host_module='tests/breakwater_host.py')['host_module'], 'tests/breakwater_host.py')
        self.assertEqual(resolve()['host_timeout_sec'], 1800)
        for changes in ({'host_module': 'breakwater_host.py'}, {'host_module': 'tests/../host.py'},
                        {'host_module': '/tests/host.py'}, {'host_module': 'tests/host.sh'}, {'host_timeout_sec': 0}):
            with self.subTest(changes=changes), self.assertRaises(ContractError):
                resolve(**changes)

    def test_options_are_a_table(self):
        self.assertEqual(resolve(options={'repeats': 2})['options'], {'repeats': 2})
        with self.assertRaises(ContractError):
            resolve(options=['repeats'])


class EndpointTests(unittest.TestCase):
    def test_static_endpoints_get_their_defaults(self):
        config = resolve(endpoints=[{'name': 'sandbox-worker', 'port': 11000},
                                    {'name': 'store', 'port': 9000, 'service': 'storage.local:19000',
                                     'reach': ['agent', 'coordinator'], 'health': '/health'}])
        self.assertEqual(config['endpoints'], [
            {'name': 'sandbox-worker', 'port': 11000, 'service': '127.0.0.1:11000', 'reach': ['agent'], 'health': None},
            {'name': 'store', 'port': 9000, 'service': 'storage.local:19000', 'reach': ['agent', 'coordinator'],
             'health': '/health'}])

    def test_names_ports_services_and_reach_are_validated(self):
        for endpoints in ([{'name': 'author-gateway', 'port': 1}], [{'name': 'localhost', 'port': 1}],
                          [{'name': 'a', 'port': 1}, {'name': 'a', 'port': 2}], [{'name': 'a', 'port': 0}],
                          [{'name': 'a', 'port': 1, 'service': 'nohost'}], [{'name': 'a', 'port': 1, 'reach': []}],
                          [{'name': 'a', 'port': 1, 'reach': ['verifier']}], [{'name': 'a', 'port': 1, 'health': 'x'}],
                          [{'name': 'a', 'port': 1, 'extra': 1}], [{'name': '-a', 'port': 1}], [{'port': 1}],
                          {'name': 'a', 'port': 1}):
            with self.subTest(endpoints=endpoints), self.assertRaises(ContractError):
                resolve(endpoints=endpoints)
        with self.assertRaises(ContractError):
            resolve(development_gateway=ACCESS, endpoints=[{'name': 'gateway', 'port': 1}])
        with self.assertRaises(ContractError):
            resolve(inbound_ports='30000-30001', endpoints=[{'name': 'host.docker.internal', 'port': 1}])


class WorkerEnvironmentTests(unittest.TestCase):
    def test_submitted_code_always_learns_its_workdir(self):
        import task_api
        declared = task_api.worker_environment({'workdir': '/app', 'worker_env': {'PATH': '/usr/bin', 'HOME': '/tmp/w'}})
        self.assertEqual(declared, {'BREAKWATER_WORKDIR': '/app', 'PATH': '/usr/bin', 'HOME': '/tmp/w'})
        own = task_api.worker_environment({'workdir': '/app', 'worker_env': {'BREAKWATER_WORKDIR': '/srv'}})
        self.assertEqual(own['BREAKWATER_WORKDIR'], '/srv')
        self.assertEqual(task_api.worker_environment({'workdir': '/app', 'worker_env': None})['BREAKWATER_WORKDIR'], '/app')


class StampTests(unittest.TestCase):
    def test_the_tests_digest_covers_paths_and_contents_but_not_python_caches(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / 'tests'
            (root / 'unit').mkdir(parents=True)
            (root / 'score.py').write_text('print(1)\n')
            (root / 'unit' / 'test_score.py').write_text('pass\n')
            before = digest_of_tests(root)
            (root / '__pycache__').mkdir()
            (root / '__pycache__' / 'score.cpython-312.pyc').write_bytes(b'\0')
            (root / 'unit' / 'stale.pyc').write_bytes(b'\0')
            self.assertEqual(digest_of_tests(root), before)
            (root / 'score.py').write_text('print(2)\n')
            self.assertNotEqual(digest_of_tests(root), before)
            (root / 'score.py').write_text('print(1)\n')
            (root / 'score.py').rename(root / 'unit' / 'score.py')
            self.assertNotEqual(digest_of_tests(root), before)
            os.symlink('score.py', root / 'unit' / 'link.py')
            with self.assertRaises(ContractError):
                digest_of_tests(root)

    def test_a_docker_verifier_image_from_another_revision_or_other_tests_is_refused(self):
        with tempfile.TemporaryDirectory() as temporary:
            tests = Path(temporary)
            (tests / 'score.py').write_text('pass\n')
            def stamp(revision, digest):
                return subprocess.CompletedProcess([], 0, json.dumps({'revision': revision, 'tests': digest}) + '\n', '')
            current = digest_of_tests(tests)
            with patch.object(controller.subprocess, 'run', return_value=stamp(FRAMEWORK_REVISION, current)) as run:
                controller.check_verifier_image('sha256:abc', tests)
            argv = run.call_args.args[0]
            self.assertEqual(argv[:4], ['docker', 'run', '--rm', '--network=none'])
            for revision, digest in ((FRAMEWORK_REVISION - 1, current), (FRAMEWORK_REVISION, '0' * 64),
                                     (FRAMEWORK_REVISION, None)):
                with self.subTest(revision=revision, digest=digest), \
                        patch.object(controller.subprocess, 'run', return_value=stamp(revision, digest)), \
                        self.assertRaises(ContractError):
                    controller.check_verifier_image('sha256:abc', tests)
            failed = subprocess.CompletedProcess([], 1, '', 'no python3')
            with patch.object(controller.subprocess, 'run', return_value=failed), self.assertRaises(ContractError):
                controller.check_verifier_image('sha256:abc', tests)

    def test_the_stamp_script_reads_this_framework(self):
        out = subprocess.run([sys.executable, '-I', '-B', '-c',
                              controller.VERIFIER_STAMP.replace('/opt/breakwater', str(FRAMEWORK))
                              .replace('"/tests"', json.dumps(str(EXAMPLE / 'tests')))],
                             capture_output=True, text=True, check=True)
        self.assertEqual(json.loads(out.stdout), {'revision': FRAMEWORK_REVISION, 'tests': digest_of_tests(EXAMPLE / 'tests')})



class Reached(BaseException):
    """Raised past the verifier's error handling when it gets as far as the launch checks."""


class VerifierStampTests(unittest.TestCase):
    def run_verifier(self, launch_checks=None, failure_reward=0.0, **stamps):
        import verifier
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            config = {'failure_reward': failure_reward, 'task': 'word-count'}
            launch = {'external_sha256': 'a' * 64, 'identity': {'task': 'word-count'},
                      'framework_revision': FRAMEWORK_REVISION, 'tests_sha256': 't' * 64, **stamps}
            with patch.object(verifier, 'REPORTS', directory), patch.object(verifier, 'read_json', return_value=config), \
                 patch.object(verifier, 'tests_digest', return_value='t' * 64), \
                 patch.object(verifier, 'verify_seal', side_effect=[launch, {}]), \
                 patch.object(verifier, 'check_launch', side_effect=launch_checks or Reached('reached the launch checks')), \
                 patch.object(verifier.os, 'geteuid', return_value=1102), \
                 patch.dict(os.environ, {'BREAKWATER_STAGE': 'verifier'}), \
                 patch.object(sys, 'argv', ['verifier.py', '--authority', 'a' * 64]), \
                 patch.object(verifier, 'audit') as judge:
                try:
                    code = verifier.main()
                except Reached as error:
                    return str(error), None
                judge.assert_not_called()
                void = directory / 'void.json'
                self.void = json.loads(void.read_text()) if void.exists() else None
                return code, json.loads((directory / 'reward.json').read_text())

    def test_a_run_sealed_by_another_revision_or_with_other_tests_is_not_verified(self):
        self.assertEqual(self.run_verifier()[0], 'reached the launch checks')
        for stamps in ({'framework_revision': FRAMEWORK_REVISION - 1}, {'framework_revision': None},
                       {'tests_sha256': '0' * 64}, {'tests_sha256': None}):
            with self.subTest(stamps=stamps):
                code, reward = self.run_verifier(**stamps)
                self.assertNotEqual(code, 0)
                self.assertEqual(reward, {'reward': 0.0})

    def test_a_void_attempt_keeps_the_failure_reward_and_is_never_judged(self):
        import task_api
        code, reward = self.run_verifier(launch_checks=task_api.Void('the oracle is unavailable'), failure_reward=-1.0)
        self.assertEqual((code, reward), (2, {'reward': -1.0}))
        self.assertEqual(self.void, {'status': 'VOID', 'reason': 'the oracle is unavailable'})


if __name__ == '__main__':
    unittest.main()
