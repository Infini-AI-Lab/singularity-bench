"""The coding-agent-harness grid: data-only cell scoring, the overlay checks and attempt bookkeeping. No GPUs or Docker.

    python3 -m pytest tasks/coding-agent-harness/grid/tests
"""
import copy
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
from types import SimpleNamespace
from unittest.mock import patch

GRID = Path(__file__).resolve().parents[1]
HARNESS = GRID.parent
FRAMEWORK = GRID.parents[2] / 'breakwater'
TASKS = FRAMEWORK.parent / 'tasks'
sys.path[:0] = [str(GRID), str(GRID / 'downstream'), str(GRID / 'worker'), str(FRAMEWORK)]
from contract import ContractError  # noqa: E402


class CellTests(unittest.TestCase):
    def kernel_measurements(self):
        timings, per_shape, agent, oracle = [], [], [], []
        for eager, candidate, ceiling in ((1.0, .5, .2), (2.0, .8, .5)):
            timings += [{'role': 'trusted', 'seconds': eager},
                        {'role': 'candidate', 'seconds': candidate, 'method': 'parent-clocked-batch-v1',
                         'batch_seconds': [candidate * 10] * 6, 'deferred_completion_suspect': False},
                        {'role': 'trusted', 'seconds': ceiling}]
            agent.append(eager / candidate)
            oracle.append(eager / ceiling)
            per_shape.append({'shape': [1], 'agent_x': round(eager / candidate, 2), 'oracle_x': round(eager / ceiling, 2),
                              'relerr': 0.001})
        import math
        from cell_score import geomean
        a, o = geomean(agent), geomean(oracle)
        reward = round(min(1.0, math.log(a) / math.log(o)), 4)
        return {'task': 'gated-deltanet-forward', 'timings': timings, 'comparisons': [0.001, 0.002, 0.001, 0.001],
                'recompute': [1.9], 'oracle_libraries_refused': [], 'voided': None, 'perf_tolerance': 0.02,
                'recompute_min': 1e-3, 'grader_shapes': 2, 'correct_shapes': 2,
                'reward': {'reward': reward, 'correct': 1.0, 'used_oracle_library': 0.0, 'cached_output': 0.0,
                           'agent_geomean': round(a, 4), 'oracle_geomean': round(o, 4)},
                'details': {'per_shape': per_shape}}

    def write(self, root, value):
        root.mkdir(exist_ok=True)
        (root / 'measurements.json').write_text(json.dumps(value))
        return root

    def test_kernel_reward_recomputes_and_rejects_inconsistent_raw_data(self):
        from cell_score import CellVoid, score_kernel
        with tempfile.TemporaryDirectory() as temporary:
            good = self.kernel_measurements()
            result = score_kernel(self.write(Path(temporary) / 'good', good))
            self.assertEqual(result['reward'], good['reward']['reward'])
            for mutation in ('timing', 'claimed_reward', 'cache', 'seal', 'order'):
                bad = copy.deepcopy(good)
                if mutation == 'timing':
                    bad['timings'][1]['seconds'] = 0.01          # a faster candidate than recorded speedups
                if mutation == 'claimed_reward':
                    bad['reward']['reward'] = 1.0
                if mutation == 'cache':
                    bad['recompute'] = [0.0]                     # cached output must zero the reward
                if mutation == 'seal':
                    bad['oracle_libraries_refused'] = ['fla']
                if mutation == 'order':
                    bad['timings'][0], bad['timings'][1] = bad['timings'][1], bad['timings'][0]
                with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                    score_kernel(self.write(Path(temporary) / mutation, bad))
            void = dict(good, voided='GraderBroken: oracle unavailable')
            with self.assertRaises(CellVoid):
                score_kernel(self.write(Path(temporary) / 'void', void))

    def test_pr_raw_protocol_scores_against_rederived_private_expectations(self):
        import pr33698
        from cell_score import score_pr
        cases, private = pr33698.prepare_cases()
        with tempfile.TemporaryDirectory() as temporary:
            raw = Path(temporary)
            (raw / 'cases.json').write_text(json.dumps(cases))
            (raw / 'private.json').write_text(json.dumps(private))
            rows = [{'example_id': name, 'status': 'returned', 'configs': private[name]} for name in pr33698.EXAMPLE_IDS]
            (raw / 'predictions.jsonl').write_text(''.join(json.dumps(row) + '\n' for row in rows))
            self.assertEqual(score_pr('pr-vllm-33698', raw)['reward'], 1.0)
            rows[1]['configs'][0]['num_blocks'] = 34
            (raw / 'predictions.jsonl').write_text(''.join(json.dumps(row) + '\n' for row in rows))
            self.assertEqual(score_pr('pr-vllm-33698', raw)['reward'], 0.0)
            forged = copy.deepcopy(private)
            forged[pr33698.EXAMPLE_IDS[0]][0]['num_blocks'] = 1
            (raw / 'private.json').write_text(json.dumps(forged))
            with self.assertRaises(ValueError):
                score_pr('pr-vllm-33698', raw)
        self.assertIn('spec_type', json.dumps(private))

    def test_registry_refuses_unprotected_tasks_and_kernel_verifiers_are_instrumentable(self):
        import kernel_parent
        from registry import KERNELS, require_task, task_path
        with self.assertRaises(ValueError):
            require_task('pr-vllm-39391')
        for task in KERNELS:
            source = HARNESS / 'eval/e2e_tasks' / task_path(task) / 'tests/verify_env.py'
            tree = kernel_parent.strip_entrypoint(source)
            self.assertTrue(any(getattr(node, 'name', '') == 'main' for node in tree.body), task)
        remote = object()
        self.assertTrue(kernel_parent.closes_over((lambda: remote), remote))
        self.assertFalse(kernel_parent.closes_over((lambda: 1), remote))

    def verification_task(self):
        """The cell's verification task as the verifier loads it (task_api.load of verify/tests)."""
        import task_api
        return task_api.load_hook(GRID / 'verify' / 'tests' / 'breakwater_task.py', 'cell_verification_task')

    def test_overlay_must_equal_recorded_change_set(self):
        from contract import digest
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            submission, evidence = root / 'submission', root / 'evidence'
            (submission / 'files/vllm').mkdir(parents=True)
            (evidence / 'task').mkdir(parents=True)
            (submission / 'files/vllm/fix.py').write_text('fixed')
            changes = {'workdir': '/app/vllm', 'added': [], 'modified': ['vllm/fix.py'], 'deleted': ['old.py'],
                       'executable': []}
            (submission / 'changes.json').write_text(json.dumps(changes))
            baseline = {'files': {'vllm/fix.py': {'sha256': 'b' * 64}, 'old.py': {'sha256': 'c' * 64}}, 'symlinks': {}}
            final = {'files': {'vllm/fix.py': {'sha256': digest(submission / 'files/vllm/fix.py')}}}
            cell = {'workdir': '/app/vllm', 'harness_inventory_sha256': 'h', 'driver': 'glm53'}
            (evidence / 'task/cell-workspace-baseline.json').write_text(json.dumps(baseline))
            (evidence / 'task/cell-workspace-final.json').write_text(json.dumps(final))
            (evidence / 'task/cell.json').write_text(json.dumps({**cell, 'harness_files': {}}))
            module = self.verification_task()
            scoring = SimpleNamespace(metadata=cell, evidence=evidence, submission=submission)
            facts = module.mechanical(scoring)
            self.assertEqual((facts['overlay']['modified'], facts['driver']), (1, 'glm53'))
            with self.assertRaisesRegex(ContractError, 'differs from the sealed one'):
                module.mechanical(SimpleNamespace(**{**vars(scoring), 'metadata': {**cell, 'driver': 'other'}}))
            (submission / 'files/vllm/extra.py').write_text('smuggled')
            with self.assertRaises(ContractError):
                module.mechanical(scoring)

    def test_the_scorer_recomputes_the_reward_and_a_void_measurement_is_no_score(self):
        import task_api
        module = self.verification_task()
        with tempfile.TemporaryDirectory() as temporary:
            raw = self.write(Path(temporary) / 'raw', self.kernel_measurements())
            records = Path(temporary) / 'deterministic'
            records.mkdir()
            scoring = SimpleNamespace(metadata={'task': 'nvfp4-gemm'}, raw=raw, records=records)
            expected = self.kernel_measurements()['reward']['reward']
            result = module.score(scoring)
            self.assertEqual((result['reward'], result['details']['cell_score']['reward']), (expected, expected))
            self.assertEqual(json.loads((records / 'cell-score.json').read_text())['reward'], expected)
            self.write(raw, {**self.kernel_measurements(), 'voided': 'oracle unavailable'})
            with self.assertRaisesRegex(task_api.Void, 'oracle unavailable'):
                module.score(scoring)

    def test_review_views_summarize_inventories_and_tensor_payloads(self):
        import review_index
        import task_api
        module = self.verification_task()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            external, evaluation = root / 'external', root / 'evaluation'
            (external / 'task').mkdir(parents=True)
            (external / 'task/cell-workspace-final.json').write_text(json.dumps(
                {'files': {'a.py': {'sha256': 'a' * 64}}, 'files_sha256': 'f' * 64, 'symlinks_removed': []}))
            payload = evaluation / 'transport/request/r1'
            payload.mkdir(parents=True)
            (payload / '0.bin').write_bytes(b'\0' * 16)
            (payload / 'tree.json').write_text('{}')
            (evaluation / 'other.bin').write_bytes(b'\1')
            views = task_api.review_views(module, SimpleNamespace())
            errors = []
            packets = dict(review_index.packets({'external': external, 'evaluation': evaluation}, views=views,
                                                view_errors=errors))
            inventory = next(text for reference, text in packets.items()
                             if reference.startswith('external/task/cell-workspace-final.json[task-view]'))
            self.assertEqual(json.loads(inventory.split('\n', 1)[1]),
                             {'kind': 'trusted_workspace_inventory', 'file': 'cell-workspace-final.json', 'files': 1,
                              'files_sha256': 'f' * 64, 'symlinks_removed': []})
            tensor = next(text for reference, text in packets.items()
                          if reference.startswith('evaluation/transport/request/r1/0.bin[task-view]'))
            self.assertIn('"kind": "tensor_payload"', tensor)
            self.assertIn('evaluation/other.bin[metadata]', packets)  # not a tensor payload: the generic presentation
            self.assertEqual(errors, [])


class GridTests(unittest.TestCase):
    def test_attempts_are_the_attempt_directories_not_their_logs(self):
        import grid
        root = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, root, True)
        cell = root / 'nvfp4-gemm__glm53'
        for name in ('attempt-1', 'attempt-2', 'attempt-10'):
            (cell / name).mkdir(parents=True)
            (cell / (name + '.log')).write_text('log')
        (cell / 'attempt-notes').mkdir()
        self.assertEqual([path.name for path in grid.attempts(root, 'nvfp4-gemm', 'glm53')],
                         ['attempt-1', 'attempt-2', 'attempt-10'])
        self.assertEqual(grid.attempts(root, 'mla-decode', 'glm53'), [])


if __name__ == '__main__':
    unittest.main()
