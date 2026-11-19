"""cluster-serverless: data-only cell scoring, submission checks, fleet journals and fleet launch. No GPUs or Docker.

    python3 -m pytest tasks/cluster-serverless/tests/unit
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

FRAMEWORK = Path(__file__).resolve().parents[4] / 'breakwater'
TASKS = FRAMEWORK.parent / 'tasks'
CLUSTER = TASKS / 'cluster-serverless'
sys.path[:0] = [str(FRAMEWORK)]
sys.path += [str(CLUSTER / 'tests')]
from contract import ContractError  # noqa: E402


def task_code(name):
    """A task's trusted code (tasks/NAME/tests/breakwater_task.py)."""
    import task_api
    return task_api.load_hook(TASKS / name / 'tests' / 'breakwater_task.py', 'task_' + name.replace('-', '_'))


def cluster_tests(root, qualification=None):
    """A copy of the cluster grader inputs with synthetic graded traces (one request per model).

    The graded traces are generated world artifacts that the task does not ship; the suite
    copy binds the synthetic traces by hash exactly as the real suite binds the real ones.
    """
    tests = Path(root) / 'tests'
    (tests / 'kit').mkdir(parents=True)
    for source in (CLUSTER / 'tests/kit').glob('*.py'):
        shutil.copyfile(source, tests / 'kit' / source.name)
    suite = json.loads((CLUSTER / 'tests/suite-graded.json').read_text())
    if qualification is not None:
        suite['qualification'] = qualification
    for case in suite['cases']:
        rows = [{'request_id': f"{case['name']}-{index}", 'model': model, 'class': 'interactive', 'weight': 1.0,
                 'max_tokens': 16, 'input_tokens': 32, 'slo_ttft_ms': 1000.0, 'slo_tbt_ms': 100.0, 'arrival_s': 0.0}
                for index, model in enumerate(suite['model_ids'])]
        data = ''.join(json.dumps(row) + '\n' for row in rows).encode()
        (tests / f"trace-graded-{case['name']}.jsonl").write_bytes(data)
        case['trace_sha256'] = hashlib.sha256(data).hexdigest()
    (tests / 'suite-graded.json').write_text(json.dumps(suite))
    return tests


class FleetJournalTests(unittest.TestCase):
    def journal(self, path, rows):
        path.write_text(''.join(json.dumps({'seq': index, 'source': 'private-fleet-worker', 'time_ns': 10 + index,
                                            **row}) + '\n' for index, row in enumerate(rows)))

    def test_structure_is_mandatory_lifecycle_is_reported(self):
        import cluster_fleet
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary)
            name = folder / ('worker-' + 'a' * 32 + '.jsonl')
            complete = [{'event': 'worker_started', 'run_id': 'R', 'model': 'm'}, {'event': 'request', 'audit_id': 'x'},
                        {'event': 'response', 'audit_id': 'x', 'frame': {'choices': [{'text': 'hi'}]}},
                        {'event': 'request_finished', 'audit_id': 'x', 'outcome': 'completed'}, {'event': 'worker_stopped'}]
            self.journal(name, complete)
            report = cluster_fleet.check_journals(folder, 'R')
            self.assertEqual((report['requests'], report['incomplete']), (1, []))
            self.journal(name, complete[:2])
            self.assertEqual(cluster_fleet.check_journals(folder, 'R')['incomplete'][0]['unfinished_requests'], ['x'])
            for broken in (complete[:1] + complete[2:], [dict(complete[0], run_id='other')] + complete[1:]):
                self.journal(name, broken)
                with self.assertRaises(ContractError):
                    cluster_fleet.check_journals(folder, 'R')
            text = name.read_text()
            name.write_text(text.replace('"seq": 1', '"seq": 3'))
            with self.assertRaises(ContractError):
                cluster_fleet.check_journals(folder, 'R')

    def test_collect_takes_only_new_journals_and_rejects_stale_writers(self):
        import cluster_fleet
        with tempfile.TemporaryDirectory() as temporary:
            audit, destination = Path(temporary) / 'audit', Path(temporary) / 'evidence'
            audit.mkdir()
            old, new = audit / ('worker-' + '1' * 32 + '.jsonl'), audit / ('worker-' + '2' * 32 + '.jsonl')
            old.write_text('{"seq": 0}\n')
            before = cluster_fleet.existing_journals(audit)
            new.write_text('{"seq": 0}\n')
            self.assertEqual(cluster_fleet.collect(audit, destination, before), [new.name])
            old.write_text('{"seq": 0}\n{"seq": 1}\n')
            with self.assertRaises(ContractError):
                cluster_fleet.collect(audit, Path(temporary) / 'again', before)


class ClusterScoreTests(unittest.TestCase):
    def test_invalid_submission_scores_zero_and_tampered_utility_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            tests = cluster_tests(temporary)
            suite = json.loads((tests / 'suite-graded.json').read_text())
            for name in ('score', 'reference', 'traceunits'):
                sys.modules.pop(name, None)
            sys.path.insert(0, str(tests / 'kit'))
            import score as scorer
            raw = Path(temporary) / 'raw'
            (raw / 'cluster').mkdir(parents=True)
            labels = [f'{name}-0' for name in scorer.CELLS]
            qualified = set(suite['qualification'].values()) == {'target-qualified'}
            measurement = {'labels': labels, 'invalid': '/app/run.sh is missing', 'development_only': not qualified,
                           'measurement_binding': None if not qualified else {}, 'repeats': 1,
                           'score_version': scorer.SCORE_VERSION, 'model_ids': suite['model_ids'],
                           'trace_sha256': {case['name']: case['trace_sha256'] for case in suite['cases']},
                           'cells': [{'label': label, 'trace': label.rsplit('-', 1)[0], 'invalid': 'missing'}
                                     for label in labels]}
            (raw / 'cluster/measurement.json').write_text(json.dumps(measurement))
            config = {'allow_unqualified': True}
            self.assertEqual(task_code('cluster-serverless').cluster_score(raw, tests=tests, config=config), 0.0)
            measurement['labels'] = labels[::-1]
            (raw / 'cluster/measurement.json').write_text(json.dumps(measurement))
            with self.assertRaises(ContractError):
                task_code('cluster-serverless').cluster_score(raw, tests=tests, config=config)
            if not qualified:
                measurement['labels'] = labels
                (raw / 'cluster/measurement.json').write_text(json.dumps(measurement))
                with self.assertRaises(ContractError):
                    task_code('cluster-serverless').cluster_score(raw, tests=tests, config={'allow_unqualified': False})


class ClusterSubmissionTests(unittest.TestCase):
    """Fleet identity and shape, the qualification rule, submission limits and per-cell isolation."""

    def test_fleet_run_id_and_shape(self):
        import cluster_fleet
        self.assertEqual(cluster_fleet.run_id({'run_id': 'top', 'launch': {'run_id': 'old'}}), 'top')
        self.assertEqual(cluster_fleet.run_id({'launch': {'run_id': 'old'}}), 'old')
        self.assertIsNone(cluster_fleet.run_id({'launch': {}}))
        fleet = json.loads((TASKS / 'cluster-serverless/tests/fleet-graded.json').read_text())
        catalogue = json.loads((TASKS / 'cluster-serverless/tests/public/catalogue.json').read_text())
        self.assertEqual(cluster_fleet.check_fleet_shape(fleet, catalogue), {'nodes': 4, 'gpus': 16})
        public = json.loads((TASKS / 'cluster-serverless/tests/fleet-graded.public.json').read_text())
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
                task_code('cluster-serverless').cluster_score(raw, tests=tests, config={'allow_unqualified': True})
            measurement.update(development_only=True, measurement_binding=None)
            (raw / 'cluster/measurement.json').write_text(json.dumps(measurement))
            self.assertEqual(task_code('cluster-serverless').cluster_score(raw, tests=tests, config={'allow_unqualified': True}), 0.0)
            # The sealed submission is rechecked: one over its limits must have been graded invalid.
            submission = Path(temporary) / 'submission'
            submission.mkdir()
            (submission / 'run.sh').write_bytes(b'x' * 64)
            import cluster_cells
            with patch.object(cluster_cells, 'MAX_BYTES', 32):
                self.assertEqual(task_code('cluster-serverless').cluster_score(raw, tests=tests, config={'allow_unqualified': True},
                                                              roots=[submission]), 0.0)
                measurement.update(invalid=None)
                (raw / 'cluster/measurement.json').write_text(json.dumps(measurement))
                with self.assertRaises(ContractError):
                    task_code('cluster-serverless').cluster_score(raw, tests=tests, config={'allow_unqualified': True}, roots=[submission])

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


class FleetLaunchTests(unittest.TestCase):
    """An Apptainer fleet gets a writable home of its own for each run (vLLM and Triton write caches under HOME), which
    `down` removes."""

    def setUp(self):
        import importlib.util
        spec = importlib.util.spec_from_file_location('cluster_launch', CLUSTER / '_kit' / 'cluster_launch.py')
        sys.path.insert(0, str(CLUSTER / '_kit'))
        self.addCleanup(sys.path.remove, str(CLUSTER / '_kit'))
        self.launch = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.launch)
        self.root = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.root, ignore_errors=True)

    def test_each_run_gets_a_fleet_home_that_down_removes(self):
        import shlex
        import types
        run_id = 'a' * 32
        a = types.SimpleNamespace(run_id=run_id, worker_script='manager.py', run_dir='/runs/r', weights='/raid/weights',
                                  image='/shared/fleet.sif', _first_node='n1', controller_port=9000, receipt_key='k',
                                  grader_token='t', host_ram_gib=968, time_scale=1.0)
        with patch.dict(os.environ, {'CLUSTER_FLEET_HOME': str(self.root)}):
            home, argv = self.launch.fleet_home(a), shlex.split(self.launch.apptainer_cmd(a, 'n1'))
        self.assertEqual(home, str(self.root / f'cluster-fleet-{run_id}'))
        self.assertEqual(argv[:2], ['/bin/bash', '-c'])
        create, fleet = argv[2].split(' && exec ', 1)
        self.assertEqual(shlex.split(create), ['mkdir', '-p', '-m', '700', home])
        fleet = shlex.split(fleet)
        self.assertEqual(fleet[fleet.index('--home') + 1], f'{home}:/fleet/home')
        self.assertNotIn('--no-home', fleet)
        # down stops the run's processes on each node, then removes that run's home, and no other directory.
        os.makedirs(home)
        manifest = self.root / 'fleet.json'
        record = {'run_id': run_id, 'nodes': [{'name': 'n1'}],
                  'launch': {'mode': 'apptainer', 'kit_dir': str(CLUSTER / '_kit'), 'python': sys.executable,
                             'cleanup_nodes': ['n1'], 'fleet_home': home}}
        manifest.write_text(json.dumps(record))
        with patch.dict(os.environ, {'CLUSTER_REMOTE_LAUNCH': 'bash -c {cmd}'}):
            self.assertEqual(self.launch.down(types.SimpleNamespace(fleet=str(manifest))), 0)
            self.assertFalse(os.path.exists(home))
            record['launch']['fleet_home'] = str(self.root)
            manifest.write_text(json.dumps(record))
            with self.assertRaises(ValueError):
                self.launch.down(types.SimpleNamespace(fleet=str(manifest)))
        self.assertTrue(self.root.is_dir())


if __name__ == '__main__':
    unittest.main()
