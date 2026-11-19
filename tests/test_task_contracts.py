"""This repository's tasks against the Breakwater framework: every task declares a contract the spec accepts, with its
declared values, and the policies their agents run under. No GPUs or Docker.

    python3 -m pytest tests
"""
import copy
import importlib.util
import io
import resource
import tarfile
from types import SimpleNamespace
import zipfile
import hashlib
import http.client
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
import urllib.error
import urllib.request
from unittest.mock import patch

FRAMEWORK = Path(__file__).resolve().parents[1] / 'breakwater'
TASKS = FRAMEWORK.parent / 'tasks'
HARNESS = TASKS / 'coding-agent-harness'
CLUSTER = TASKS / 'cluster-serverless'
sys.path[:0] = [str(FRAMEWORK)]
from contract import ContractError  # noqa: E402
import run as controller  # noqa: E402


class TaskContractValueTests(unittest.TestCase):
    def test_each_task_contract_parses_with_its_declared_values(self):
        """Each task's GB300 contract: GPUs, seeding, networks, deadline policy, rewards and container limits."""
        disabled = {'packages': [], 'hooks': [], 'environments': False}
        expected = {
            'open-ended-nano-gpt': dict(gpus=4, workspace_source='task', intermediate_network='none', runners='runners',
                                      intermediate_cap_add=[], submission_mount=None, rule_violation_reward=-1.0,
                                      shadow_policy=None, snapshot_excludes=[], snapshot_worker_home=False,
                                      intermediate_env=[], intermediate_add_hosts=[],
                                      options=['--ulimit', 'memlock=-1:-1', '--shm-size=16g']),
            'midtrain-for-rl': dict(gpus=4, workspace_source='task', intermediate_network='bridge',
                                  intermediate_cap_add=['NET_ADMIN', 'NET_RAW'], submission_mount=None,
                                  rule_violation_reward=0.0, shadow_policy=disabled, snapshot_excludes=['scratch'],
                                  snapshot_worker_home=False,
                                  intermediate_env=['MIDTRAIN_CODE_SANDBOX_URL', 'MIDTRAIN_CODE_SANDBOX_TOKEN'],
                                  intermediate_add_hosts=['host.docker.internal:host-gateway'],
                                  options=['--ulimit', 'memlock=-1:-1', '--ulimit', 'stack=67108864:67108864',
                                           '--shm-size=64g']),
            'rl-swe-agent': dict(gpus=4, workspace_source='task', intermediate_network='none',
                                    intermediate_cap_add=[], submission_mount=None, rule_violation_reward=0.0,
                                    shadow_policy=None, snapshot_excludes=[], snapshot_worker_home=False,
                                    intermediate_env=[], intermediate_add_hosts=[], options=['--shm-size=64g']),
            'serve-glm53-flash': dict(
                gpus=4, workspace_source='image', intermediate_network='none', intermediate_cap_add=[],
                submission_mount='/submission', rule_violation_reward=0.0, shadow_policy=None, snapshot_excludes=[],
                snapshot_worker_home=False, intermediate_env=[], intermediate_add_hosts=[],
                options=['--cpus', '16', '--memory', '524288m', '--memory-swap', '524288m', '--shm-size=256g']),
            'coding-agent-harness': dict(gpus=1, workspace_source='image', intermediate_network='none',
                                intermediate_cap_add=[], submission_mount=None, rule_violation_reward=0.0,
                                snapshot_excludes=[], snapshot_worker_home=False, intermediate_env=[],
                                intermediate_add_hosts=[],
                                options=['--cpus', '4', '--memory', '16384m', '--memory-swap', '16384m', '--shm-size=2g']),
            'cluster-serverless': dict(gpus=0, workspace_source='image', intermediate_network='bridge',
                                       intermediate_cap_add=['NET_ADMIN', 'NET_RAW'], submission_mount='/submission',
                                       rule_violation_reward=0.0, shadow_policy=disabled, snapshot_excludes=[],
                                       snapshot_worker_home=True, intermediate_env=[], intermediate_add_hosts=[],
                                       options=['--cpus', '16', '--cpuset-cpus', '0-15', '--memory', '40960m',
                                                '--memory-swap', '40960m', '--ulimit', 'nofile=65536:65536',
                                                '--shm-size=1g']),
        }
        for name, contract in expected.items():
            config = controller.load_config(TASKS / name)
            with self.subTest(task=name):
                self.assertEqual(config['task'], name)
                # Every task grades what exists when its authoring budget ends.
                self.assertEqual(config['agent_deadline'], 'evaluate')
                self.assertEqual(config['review_profile'], 'indexed')
                self.assertEqual(config['gpu_name_patterns'], [] if contract['gpus'] == 0 else ['GB300'])
                for key, value in contract.items():
                    if key != 'options':
                        self.assertEqual(config[key], value, key)
                self.assertEqual(controller.resource_options(config), contract['options'])
                if config['workspace_source'] == 'task':
                    # Release READMEs never enter a task-seeded agent workspace.
                    self.assertIn('/README.md', config['workspace_excludes'])
        cluster_agent = controller.load_config(CLUSTER)
        self.assertEqual(controller.resource_options(cluster_agent, controller.agent_resources(cluster_agent)),
                         ['--cpus', '8', '--cpuset-cpus', '0-15', '--memory', '32768m', '--memory-swap', '32768m',
                          '--pids-limit', '512', '--ulimit', 'nofile=65536:65536', '--shm-size=1g'])
        harness = controller.load_config(HARNESS)
        self.assertEqual((harness['gpus'], harness['workspace_source']), (1, 'image'))
        self.assertTrue(harness['development_gateway'])
        cluster = controller.load_config(CLUSTER)
        self.assertEqual((cluster['gpus'], cluster['submission_mount']), (0, '/submission'))
        import spec
        with self.assertRaises(ContractError):
            spec.resolve(dict(spec.read(HARNESS), intermediate_cap_add=['SYS_ADMIN']))


class DeclaredPolicyTests(unittest.TestCase):
    def test_tasks_declare_their_dataset_records_and_shadow_policies(self):
        import environment_guard as guard
        import spec
        midtrain = spec.load(TASKS / 'midtrain-for-rl')
        self.assertEqual((midtrain['dataset_manifest'], midtrain['dataset_roots']),
                         ('/app/out/training_dataset.json', ['/app/out/processed_dataset']))
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for name in ('pytest/__init__.py', 'vendor/site-packages/aiohttp/__init__.py', 'sitecustomize.py'):
                (root / name).parent.mkdir(parents=True, exist_ok=True)
                (root / name).write_text('x')
            harness = guard.shadow_policy(spec.load(HARNESS)['shadow_policy'])
            self.assertIn('pytest/__init__.py', guard.shadows(root, harness))
            # Vendored dependencies and hooks are permitted where the trusted code never imports the workspace.
            for name in ('cluster-serverless', 'midtrain-for-rl'):
                with self.subTest(task=name):
                    policy = guard.shadow_policy(spec.load(TASKS / name)['shadow_policy'])
                    self.assertEqual(guard.shadows(root, policy), {})
        self.assertEqual(spec.load(TASKS / 'open-ended-nano-gpt')['rule_violation_reward'], -1.0)


class HostSideTaskTests(unittest.TestCase):
    def test_cluster_serverless_declares_its_fleet_as_operator_inputs_and_host_hooks(self):
        import spec
        cluster = spec.load(CLUSTER)
        self.assertEqual((cluster['host_module'], cluster['operator_inputs'], cluster['operator_values'],
                          cluster['operator_secrets'], cluster['credential_env']),
                         ('tests/breakwater_host.py', ['fleet', 'public-fleet'], {'receipt-key': 'CLUSTER_RECEIPT_KEY'},
                          {'grader-token': 'CLUSTER_GRADER_TOKEN'}, ['CLUSTER_GRADER_TOKEN']))
        self.assertEqual(cluster['options'], {'repeats': 1, 'allow_unqualified': True, 'time_scale': 1})
        self.assertEqual(cluster['agent_extra_networks'], ['bridge'])


class EndpointTaskTests(unittest.TestCase):
    def test_rl_swe_agent_declares_its_sandbox_worker_and_session_ports(self):
        import spec
        rl = spec.load(TASKS / 'rl-swe-agent')
        self.assertEqual(rl['endpoints'], [{'name': 'sandbox-worker', 'port': 11000, 'service': '127.0.0.1:11000',
                                            'reach': ['agent'], 'health': '/health'}])
        self.assertEqual((rl['inbound_ports'], rl['inbound_host']), ('30000-30015', 'host.docker.internal'))


class TaskContractTests(unittest.TestCase):
    def test_harness_and_cluster_tasks_keep_their_agent_policies(self):
        for root in (HARNESS, CLUSTER):
            with self.subTest(task=root.name):
                policy = json.loads((root / 'environment/agent-policy.json').read_text())
                self.assertEqual(policy['name'], root.name)
                self.assertTrue((root / 'tests/submission_judge.md').is_file())
        self.assertTrue((HARNESS / 'eval/config.json').is_file())


class AllTaskConfigurationTests(unittest.TestCase):
    def test_every_task_declares_a_breakwater_contract(self):
        tasks = FRAMEWORK.parent / 'tasks'
        names = ['cluster-serverless', 'coding-agent-harness', 'midtrain-for-rl', 'open-ended-nano-gpt', 'rl-swe-agent',
                 'serve-glm53-flash']
        self.assertEqual(sorted(path.name for path in tasks.iterdir() if path.is_dir()), names)
        for name in names:
            root = tasks / name
            with self.subTest(task=name):
                import spec
                cfg = spec.load(root)
                self.assertEqual(cfg['task'], name)
                self.assertTrue((root / 'instruction.md').is_file())
                self.assertTrue((root / 'tests/submission_judge.md').is_file())
                script = (root / 'tests/test.sh').read_text()
                self.assertIn('/opt/breakwater/verifier.py', script)
                self.assertIn('BREAKWATER_STAGE', script)
                self.assertEqual(json.loads((root / 'environment/agent-policy.json').read_text())['name'], name)
                for row in cfg['assets']:
                    self.assertFalse('agent' in row['stages'] and row['target'].startswith('/private/'))

    def test_serving_score_is_dollars_per_gpu_hour_not_token_throughput(self):
        path = FRAMEWORK.parent / 'tasks/serve-glm53-flash/tests/simulator.py'
        spec = importlib.util.spec_from_file_location('monetary_simulator', path)
        simulator = importlib.util.module_from_spec(spec);spec.loader.exec_module(simulator)
        # The graded shape: four GPUs billed for a 900 s window, i.e. one GPU-hour.
        self.assertEqual((simulator.GPU_COUNT, simulator.DURATION_S, simulator.DRAIN_S, simulator.CONCURRENCY),
                         (4, 900, 300, 16))
        record = {'primer': False, 'ok': True, 'error': None, 'input_tokens': 1_000_000, 'cached_tokens': 200_000,
                  'reported_output_tokens': 10_000, 'max_tokens': 10_000, 'text': 'x' * 10_000,
                  't_send': 0, 't_first': .1, 't_last': 1, 't_done': 2, 'position': 0}
        metrics = simulator.score_records([record], 0, simulator.DURATION_S)
        self.assertAlmostEqual(metrics['income_usd'], .093)
        self.assertEqual(metrics['gpu_hours'], 1.0)
        self.assertEqual(metrics['income_per_gpu_hour'], .093)
        self.assertEqual(simulator.score(metrics), .093)
        self.assertNotIn('tokens_per_gpu_hour', metrics)


if __name__ == '__main__':
    unittest.main()
