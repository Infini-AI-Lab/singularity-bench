"""CPU-only tests of the held-out evaluator, the score rule and the data contract (no Docker, GPU or network).

Run without installing Torch/pytest:
python3 -m unittest discover -s tasks/rl-swe-agent/tests/unit -p test_evaluation.py -v
"""
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'evaluation'))
import suites as ev
import evaluate_checkpoint as runner
import prepare_inputs


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


score = load('rl_swe_agent_score', ROOT / 'evaluation/score.py')
contract = load('rl_swe_agent_contract_test', ROOT / 'dataset_contract.py')
IMAGE = 'sha256:' + 'a' * 64
DEV, SWE = 'rl-swe-agent-dev', 'swebench-verified'


class SuiteTests(unittest.TestCase):
    def test_one_deadline_includes_setup_and_reserves_cleanup(self):
        with patch.object(ev.time, 'monotonic', return_value=100):
            budget = ev.Deadline(14400)
        with patch.object(ev.time, 'monotonic', return_value=200):
            self.assertEqual(budget.remaining(), 14240)
        with patch.object(ev.time, 'monotonic', return_value=14440):
            with self.assertRaises(TimeoutError):
                budget.remaining()
            self.assertEqual(budget.remaining(cleanup=True), 60)
        for invalid in [0, 60, float('nan'), float('inf')]:
            with self.assertRaises(ValueError):
                ev.Deadline(invalid)

    def test_image_pinning_preserves_task_timeouts_and_other_sections(self):
        plain = '[agent]\ntimeout_sec = 600\n[verifier]\ntimeout_sec = 600\n'
        expected = ev.tomllib.loads(plain)
        expected['environment'] = {'docker_image': IMAGE}
        self.assertEqual(ev.tomllib.loads(ev.pin_image(plain, IMAGE)), expected)
        for field in ['', 'docker_image = "upstream:latest"\n']:
            original = '[agent]\ntimeout_sec = 3000\n[environment]\n' + field + 'cpus = 1\n[verifier]\ntimeout_sec = 3000\n'
            actual = ev.tomllib.loads(ev.pin_image(original, IMAGE))
            expected = ev.tomllib.loads(original)
            expected['environment']['docker_image'] = IMAGE
            self.assertEqual(actual, expected)

    def test_both_image_types_are_prepared_and_reused_without_rebuild(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            paths, specs = {}, {}
            for suite, image in [('pulled', 'upstream:latest'), ('built', None)]:
                source = root / suite
                task = source / 'task'
                (task / 'environment').mkdir(parents=True)
                (task / 'task.toml').write_text('[environment]\ncpus = 1\n' + (f'docker_image = "{image}"\n' if image else ''))
                (task / 'environment/Dockerfile').write_text('FROM python:3.12\n')
                (task / 'instruction.md').write_text('Fix the task.')
                paths[suite] = source
                specs[suite] = {'commit': 'abc', 'task_count': 1}
            commands = []
            def run(command, **kwargs):
                commands.append(command)
                if command[0] == 'git':
                    return SimpleNamespace(stdout='abc\n' if 'rev-parse' in command else '', returncode=0)
                if command[:3] == ['docker', 'image', 'inspect']:
                    if command[3] == 'upstream:latest' and sum(c[:2] == ['docker', 'pull'] for c in commands) == 0:
                        return SimpleNamespace(stdout='', returncode=1)
                    return SimpleNamespace(stdout=IMAGE, returncode=0)
                return SimpleNamespace(stdout='', returncode=0)
            before, after = root / 'before', root / 'after'
            before.mkdir()
            after.mkdir()
            with patch.object(ev, 'SUITES', specs), patch.object(ev.subprocess, 'run', side_effect=run):
                ev.prepare_suites(paths, before, ev.Deadline(3600), {}, image_workers=1)
                self.assertEqual(sum(c[:2] == ['docker', 'build'] for c in commands), 1)
                self.assertEqual(sum(c[:2] == ['docker', 'pull'] for c in commands), 1)
                commands.clear()
                ev.prepare_suites(paths, after, ev.Deadline(3600), {},
                                  image_manifest=before / 'image-cache.json', image_workers=1)
                self.assertFalse(any(c[:2] in [['docker', 'build'], ['docker', 'pull']] for c in commands))
                self.assertEqual(json.loads((before / 'image-cache.json').read_text()),
                                 json.loads((after / 'image-cache.json').read_text()))
                (paths['built'] / 'task/instruction.md').write_text('Changed task')
                changed = root / 'changed'
                changed.mkdir()
                with self.assertRaisesRegex(ValueError, 'source changed'):
                    ev.prepare_suites(paths, changed, ev.Deadline(3600), {},
                                      image_manifest=before / 'image-cache.json', image_workers=1)

    def test_exclusions_shrink_the_selection_but_not_the_verified_suite(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = root / 'swe'
            for name in ('keep', 'drop'):
                (source / name).mkdir(parents=True)
                (source / name / 'task.toml').write_text(f'[environment]\ndocker_image = "{name}:latest"\n')
            specs = {SWE: {'commit': 'abc', 'task_count': 2}}
            def run(command, **kwargs):
                if command[0] == 'git':
                    return SimpleNamespace(stdout='abc\n' if 'rev-parse' in command else '', returncode=0)
                return SimpleNamespace(stdout=IMAGE, returncode=0)
            with patch.object(ev, 'SUITES', specs), patch.object(ev.subprocess, 'run', side_effect=run):
                out = root / 'out'
                out.mkdir()
                selected = ev.prepare_suites({SWE: source}, out, ev.Deadline(3600), {},
                                             image_workers=1, exclude={SWE: ['drop']})
                self.assertEqual(selected[SWE]['task_names'], ['keep'])
                self.assertEqual(set(json.loads((out / 'image-cache.json').read_text())), {SWE + '/keep'})
                for exclude in ({SWE: ['missing']}, {DEV: ['keep']}):
                    other = root / str(len(list(root.iterdir())))
                    other.mkdir()
                    with self.assertRaises(ValueError):
                        ev.prepare_suites({SWE: source}, other, ev.Deadline(3600), {},
                                          image_workers=1, exclude=exclude)

    def test_dev_suite_prepares_only_the_dev_split(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source, output = root / 'data', root / 'prepared'
            output.mkdir()
            for ident in ('train', 'dev'):
                task = source / 'tasks' / ident
                (task / 'environment').mkdir(parents=True)
                (task / 'instruction.md').write_text(ident)
                (task / 'task.toml').write_text('[environment]\ncpus = 1\n')
                (task / 'environment/Dockerfile').write_text('FROM ubuntu:24.04\n')
            dataset = {'dataset': 'fixture/corpus', 'revision': 'pin', 'dev_ids': ['dev']}
            suite = {'kind': 'dev_split', 'repository': 'fixture/corpus', 'revision': 'pin', 'task_count': 1}
            calls = []
            def run(command, **kwargs):
                calls.append(command)
                return SimpleNamespace(returncode=0, stdout=IMAGE)
            with patch.object(ev, 'SUITES', {DEV: suite}), \
                 patch.object(ev, 'validate_dev_split', return_value=dataset), \
                 patch.object(ev.subprocess, 'run', side_effect=run):
                selected = ev.prepare_suites({DEV: source}, output, ev.Deadline(3600), {}, image_workers=1)
            self.assertEqual(selected[DEV]['task_names'], ['dev'])
            self.assertTrue((output / 'tasks' / DEV / 'dev').is_dir())
            self.assertFalse((output / 'tasks' / DEV / 'train').exists())
            self.assertFalse(any(command[0] == 'git' for command in calls))

    def test_suites_are_the_dev_split_and_swe_bench_verified(self):
        self.assertEqual(set(ev.SUITES), {DEV, SWE})
        self.assertEqual((ev.SUITES[DEV]['task_count'], ev.SUITES[DEV]['agent_timeout_sec']), (73, 600))
        self.assertEqual((ev.SUITES[SWE]['task_count'], ev.SUITES[SWE]['agent_timeout_sec']), (500, 3000))
        self.assertEqual(set(runner.CONCURRENCY), set(ev.SUITES))

    def test_fixed_subset_matches_the_shipped_exclusions(self):
        exclusions = (ROOT / 'evaluation/swe-exclusions.json').read_bytes()
        subset = json.loads((ROOT / 'evaluation/swe-subset.json').read_text())
        self.assertEqual(subset['swe_exclusions_sha256'], hashlib.sha256(exclusions).hexdigest())
        self.assertEqual((subset['size'], len(set(subset['tasks'])), subset['eligible']), (100, 100, 480))
        self.assertFalse(set(subset['tasks']) & set(json.loads(exclusions)['tasks']))
        self.assertEqual(len(json.loads(exclusions)['tasks']), 20)

    def test_partial_and_error_results_do_not_produce_a_full_score(self):
        with tempfile.TemporaryDirectory() as temp:
            job = Path(temp)
            def result(name, reward, exception=None):
                trial = job / name
                trial.mkdir(exist_ok=True)
                ev.write(trial / 'result.json', {'task_name': name, 'finished_at': 'now',
                    'verifier_result': {'rewards': {'reward': reward}}, 'exception_info': exception})
            result('a', 1)
            partial = ev.summarize_trials(job, ['a', 'b'], -2)
            self.assertEqual(partial['unfinished_tasks'], ['b'])
            self.assertEqual(partial['resolved_fraction_of_selected'], .5)
            self.assertIsNone(partial['resolved_rate'])
            result('b', None, {'exception_type': 'EnvironmentStartError'})
            errored = ev.summarize_trials(job, ['a', 'b'], 0)
            self.assertTrue(errored['complete'])
            self.assertFalse(errored['score_valid'])
            self.assertIsNone(errored['resolved_rate'])
            result('b', 0)
            self.assertEqual(ev.summarize_trials(job, ['a', 'b'], 0)['resolved_rate'], .5)

    def test_cleanup_only_removes_this_runs_compose_projects(self):
        with tempfile.TemporaryDirectory() as temp:
            job = Path(temp)
            (job / 'task__abc').mkdir()
            (job / 'task__abc/config.json').write_text('{}')
            rows = '\n'.join(json.dumps({'ID': identity, 'Labels': 'com.docker.compose.project=' + label})
                             for identity, label in [('owned', 'task__abc__env'), ('other', 'unrelated__env')])
            with patch.object(ev.subprocess, 'run', return_value=SimpleNamespace(stdout=rows, returncode=0)) as run:
                ev.cleanup_sandboxes([job], ev.Deadline(3600), {'DOCKER_HOST': 'ssh://sandbox'})
                removes = [call.args[0] for call in run.call_args_list if 'rm' in call.args[0]]
                self.assertEqual(removes, [['docker', 'rm', '-f', 'owned'], ['docker', 'network', 'rm', 'owned']])
                self.assertTrue(all(call.kwargs['env']['DOCKER_HOST'] == 'ssh://sandbox' for call in run.call_args_list))


class RunnerTests(unittest.TestCase):
    def test_two_jobs_share_one_server_and_finish_independently(self):
        self.exercise_runner(False)

    def test_shared_deadline_stops_both_jobs_and_retains_partial_results(self):
        self.exercise_runner(True)

    def test_checkpoint_without_complete_marker_is_not_evaluated(self):
        self.exercise_runner(False, complete=False)

    def exercise_runner(self, timed_out, complete=True):
        processes, audits = [], []
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            model, base = root / 'model', root / 'base'
            for folder in (model, base):
                folder.mkdir()
                for name in ['config.json', 'model.safetensors.index.json']:
                    (folder / name).write_text('{}')
            if complete:
                (model / '.complete').write_text('validated by mini_slime.model.checkpoint\n')
            selection = root / 'selection'
            swe = root / 'swe'
            for name in ('excluded', 'chosen', 'other'):
                (swe / name).mkdir(parents=True)
                (swe / name / 'task.toml').write_text('[environment]\n')
            selection.mkdir()
            exclusions = json.dumps({'tasks': {'excluded': 'reason'}}).encode()
            (selection / 'swe-exclusions.json').write_bytes(exclusions)
            (selection / 'swe-subset.json').write_text(json.dumps({
                'swe_exclusions_sha256': hashlib.sha256(exclusions).hexdigest(), 'strata': {}, 'tasks': ['chosen']}))
            manifest = root / 'image-cache.json'
            manifest.write_text(json.dumps({f'{SWE}/{name}': {} for name in ('excluded', 'chosen', 'other')}))
            output = root / 'output'
            datasets = {name: {'path': str(root / name), 'task_names': ['task'],
                               'result_names': {name + '/task': 'task'}} for name in ev.SUITES}
            prepared = []

            def prepare(paths, out, budget, env, **kwargs):
                prepared.append(kwargs)
                return datasets

            def popen(command, **kwargs):
                is_server = command[0] == 'docker'
                proc = SimpleNamespace(pid=100 + len(processes), returncode=None if is_server or timed_out else 0)
                proc.poll = lambda: proc.returncode
                proc.wait = lambda **kwargs: proc.returncode
                processes.append(proc)
                if is_server:
                    self.assertIn('EVAL_CONTEXT_LIMIT=65536', command)
                    self.assertIn(str(model.resolve()) + ':/eval-model:ro', command)
                else:
                    self.assertEqual(command[:2], ['/usr/bin/uv', 'run'])
                    config = json.loads(Path(command[-1]).read_text())
                    self.assertEqual(config['n_concurrent_trials'], 8 if config['job_name'] == DEV else 24)
                    self.assertEqual(config['agents'][0]['override_timeout_sec'], 600 if config['job_name'] == DEV else 3000)
                    self.assertEqual(config['agents'][0]['kwargs']['version'], '2.4.6')
                    trial = Path(config['jobs_dir']) / config['job_name'] / 'trial'
                    trial.mkdir(parents=True)
                    ev.write(trial / 'result.json', {'task_name': config['job_name'] + '/task', 'finished_at': 'now',
                        'verifier_result': {'rewards': {'reward': 1 if config['job_name'] == DEV else 0}}})
                return proc

            def run(command, **kwargs):
                if command[0] == 'git':
                    return SimpleNamespace(stdout=runner.LOCK['harbor']['commit'] if 'rev-parse' in command else b'',
                                           returncode=0)
                if command[:3] == ['docker', 'rm', '-f'] and processes:
                    processes[0].returncode = 0
                if command[:2] == ['docker', 'run']:
                    audits.append(command)
                return SimpleNamespace(stdout=IMAGE, stderr='', returncode=0)

            class Budget(ev.Deadline):
                def remaining(self, cap=None, cleanup=False):
                    if timed_out and len(processes) == 3 and not cleanup:
                        self.work_end = ev.time.monotonic() - 1
                    return super().remaining(cap, cleanup)

            def kill(pid, sig):
                next(proc for proc in processes if proc.pid == pid).returncode = -sig

            argv = ['evaluate_checkpoint', '--checkpoint', str(model), '--base-model', str(base),
                    '--swe-benchmark', str(swe), '--image-manifest', str(manifest), '--harbor', str(root),
                    '--api-host', '10.0.0.1', '--output', str(output)]
            with patch.object(runner, 'HERE', selection), patch.object(runner, 'prepare_suites', side_effect=prepare), \
                 patch.object(runner, 'Deadline', Budget), patch.object(runner.subprocess, 'Popen', side_effect=popen) as started, \
                 patch.object(runner.subprocess, 'run', side_effect=run), patch.object(runner.urllib.request, 'urlopen'), \
                 patch.object(runner.shutil, 'which', return_value='/usr/bin/uv'), \
                 patch.object(ev.os, 'killpg', side_effect=kill), \
                 patch('sys.stdout', new_callable=io.StringIO), patch.object(sys, 'argv', argv):
                code = runner.main()
            summary = json.loads((output / 'summary.json').read_text())
            self.assertEqual(summary['evaluation_budget_seconds'], 14400)
            if not complete:
                self.assertEqual(code, 1)
                self.assertIn('.complete', summary['error'])
                self.assertEqual((processes, audits), ([], []))
                self.assertIsNone(summary['macro_resolved_rate'])
                return
            self.assertEqual(len(audits), 1)
            self.assertIn('mini_slime.model.checkpoint', audits[0])
            self.assertIn(str(ROOT / 'mini_slime') + ':/trusted/mini_slime:ro', audits[0])
            self.assertIn(str(model.resolve()) + ':/candidate:ro', audits[0])
            self.assertEqual(prepared[0]['exclude'], {SWE: ['excluded', 'other']})
            self.assertEqual(set(json.loads(prepared[0]['image_manifest'].read_text())), {f'{SWE}/chosen'})
            settings = json.loads((output / 'eval-settings.json').read_text())
            self.assertEqual(settings['excluded_tasks'], {SWE: ['excluded']})
            self.assertEqual(len(processes), 3)
            self.assertEqual(started.call_args_list[0].args[0].count(IMAGE), 1)
            if timed_out:
                self.assertEqual(code, 124)
                self.assertFalse(summary['complete'])
                self.assertIsNone(summary['macro_resolved_rate'])
                self.assertTrue(all(proc.poll() is not None for proc in processes))
            else:
                self.assertEqual(code, 0)
                self.assertTrue(summary['complete'] and summary['export_audit_pass'])
                self.assertEqual(summary['macro_resolved_rate'], .5)


class ScoreTests(unittest.TestCase):
    def evaluation(self, root, audit=True, complete=True, rates=(48 / 73, 42 / 100)):
        out = root / 'eval'
        out.mkdir()
        benchmarks = {name: {'selected': 1, 'graded': 1, 'resolved': 1, 'complete': complete, 'score_valid': complete,
                             'errors': {}, 'resolved_rate': rate if complete else None}
                      for name, rate in zip((DEV, SWE), rates)}
        ev.write(out / 'summary.json', {'complete': complete, 'export_audit_pass': audit, 'benchmarks': benchmarks,
                                        'settings': {'excluded_tasks': {}}})
        return out

    def reward(self, root, out, verdict=None):
        trial = root / 'trial'
        trial.mkdir(exist_ok=True)
        integrity = None
        if verdict is not None:
            integrity = root / 'judge.json'
            integrity.write_text(json.dumps({'verdict': verdict}))
        with patch('sys.stdout', new_callable=io.StringIO):
            score.main(trial, out, integrity)
        return json.loads((trial / 'verifier/reward.json').read_text())['reward']

    def test_reward_is_the_mean_resolve_rate_only_with_a_passing_review(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            out = self.evaluation(root)
            self.assertAlmostEqual(self.reward(root, out, 'PASS'), (48 / 73 + 42 / 100) / 2)
            for verdict in ('FAIL', 'INCONCLUSIVE'):
                self.assertEqual(self.reward(root, out, verdict), 0.0)
            (root / 'judge.json').unlink()
            trial = root / 'trial'
            with patch('sys.stdout', new_callable=io.StringIO):
                score.main(trial, out, root / 'judge.json')
            self.assertEqual(json.loads((trial / 'verifier/reward.json').read_text())['reward'], 0.0)

    def test_failed_audit_or_incomplete_evaluation_scores_zero(self):
        for kwargs in ({'audit': None}, {'complete': False}):
            with tempfile.TemporaryDirectory() as temp:
                root = Path(temp)
                self.assertEqual(self.reward(root, self.evaluation(root, **kwargs), 'PASS'), 0.0)


class DataContractTests(unittest.TestCase):
    def fixture(self, root):
        """Synthetic small files with the release's split sizes, not real task data."""
        policy, data = root / 'policy', root / 'data'
        policy.mkdir()
        (data / 'tasks').mkdir(parents=True)
        ids = [f'task_{i}' for i in range(728)]
        tasks, rows = [], {}
        for index, ident in enumerate(ids):
            folder = data / 'tasks' / ident
            folder.mkdir()
            prompt = f'Synthetic split-validation fixture {index}\n'
            (folder / 'instruction.md').write_text(prompt)
            tasks.append({'instance_id': ident, 'archive_sha256': 'a' * 64,
                          'files': {'instruction.md': contract.digest(folder / 'instruction.md')}})
            rows[ident] = {'prompt': prompt, 'metadata': {'instance_id': ident,
                'dataset_revision': 'fixture-revision', 'archive_sha256': 'a' * 64,
                'split': 'train' if index < 655 else 'dev'}}
        manifest = {'dataset': 'fixture/corpus', 'revision': 'fixture-revision',
            'original_rows': 728, 'source_sha256': 'b' * 64,
            'counts': {'train': 655, 'dev': 73, 'smoke': 8}, 'tasks': tasks,
            'train_ids': ids[:655], 'dev_ids': ids[655:], 'smoke_ids': ids[:8]}
        (data / 'manifest.json').write_text(json.dumps(manifest))
        for split, selected in [('train', ids[:655]), ('dev', ids[655:]), ('smoke', ids[:8])]:
            (data / f'{split}.jsonl').write_text(''.join(json.dumps(rows[i]) + '\n' for i in selected))
        (policy / 'sources.lock.json').write_text(json.dumps({'dataset': {
            'repo_id': 'fixture/corpus', 'revision': 'fixture-revision',
            'files': {'tasks.parquet': {'sha256': 'b' * 64}}}}))
        self.refresh_policy(policy, data)
        return policy, data

    def refresh_policy(self, policy, data):
        (policy / 'data_manifest.json').write_text(json.dumps({'prepared_file_sha256': {
            name: contract.digest(data / name) for name in ('manifest.json', 'train.jsonl', 'dev.jsonl', 'smoke.jsonl')}}))

    def test_fixed_split_validates_and_rejects_another_release_even_with_rewritten_hashes(self):
        with tempfile.TemporaryDirectory() as tmp:
            policy, data = self.fixture(Path(tmp))
            with patch.object(contract, 'ROOT', policy):
                self.assertEqual(contract.validate(data)['counts']['train'], 655)
                manifest = json.loads((data / 'manifest.json').read_text())
                manifest['original_rows'] = 5000
                (data / 'manifest.json').write_text(json.dumps(manifest))
                self.refresh_policy(policy, data)
                with self.assertRaisesRegex(ValueError, '728-task'):
                    contract.validate(data)

    def test_dev_never_overlaps_training_even_if_file_hashes_are_updated(self):
        with tempfile.TemporaryDirectory() as tmp:
            policy, data = self.fixture(Path(tmp))
            dev = [json.loads(line) for line in (data / 'dev.jsonl').read_text().splitlines()]
            leaked = json.loads((data / 'train.jsonl').read_text().splitlines()[0])
            leaked['metadata']['split'] = 'dev'
            dev[0] = leaked
            (data / 'dev.jsonl').write_text(''.join(json.dumps(row) + '\n' for row in dev))
            self.refresh_policy(policy, data)
            with patch.object(contract, 'ROOT', policy), self.assertRaisesRegex(ValueError, 'overlap'):
                contract.validate(data)

    def test_mismatched_inputs_are_replaced_only_on_request(self):
        payloads = {'tasks.parquet': b'pinned-source', 'README.md': b'card', 'extract_parquet_tasks.py': b'not executed'}
        lock = {'dataset': {'files': {name: {'sha256': hashlib.sha256(data).hexdigest(), 'url': 'https://fixture.invalid/' + name}
                                      for name, data in payloads.items()}}}
        for replace in (False, True):
            with self.subTest(replace=replace), tempfile.TemporaryDirectory() as tmp:
                task = Path(tmp)
                snapshot, data = task / 'artifacts/dataset_snapshot', task / 'data'
                snapshot.mkdir(parents=True)
                data.mkdir()
                (data / 'manifest.json').write_text(json.dumps({'version': 'other'}))
                (snapshot / 'tasks.parquet').write_bytes(b'other-source')
                def validate(path):
                    if json.loads((path / 'manifest.json').read_text())['version'] != 'pinned':
                        raise ValueError('Wrong dataset')
                def download(url, destination, checksum):
                    if not destination.exists():
                        destination.write_bytes(payloads[destination.name])
                    self.assertEqual(contract.digest(destination), checksum)
                def prepare(source, output):
                    self.assertEqual(source.read_bytes(), payloads['tasks.parquet'])
                    output.mkdir()
                    (output / 'manifest.json').write_text(json.dumps({'version': 'pinned'}))
                args = ['prepare_inputs'] + (['--replace'] if replace else [])
                with patch.object(prepare_inputs, 'ROOT', task), patch.object(prepare_inputs, 'SNAPSHOT', snapshot), \
                     patch.object(prepare_inputs, 'LOCK', lock), \
                     patch.object(prepare_inputs, 'validate', side_effect=validate), \
                     patch.object(prepare_inputs, 'download', side_effect=download), \
                     patch.object(prepare_inputs, 'prepare', side_effect=prepare), \
                     patch('sys.argv', args), patch('sys.stdout', new_callable=io.StringIO):
                    if not replace:
                        with self.assertRaisesRegex(ValueError, '--replace'):
                            prepare_inputs.main()
                        self.assertEqual((snapshot / 'tasks.parquet').read_bytes(), b'other-source')
                        self.assertEqual(json.loads((data / 'manifest.json').read_text())['version'], 'other')
                    else:
                        prepare_inputs.main()
                        self.assertEqual((snapshot / 'tasks.parquet').read_bytes(), payloads['tasks.parquet'])
                        self.assertEqual(json.loads((data / 'manifest.json').read_text())['version'], 'pinned')


if __name__ == '__main__':
    unittest.main()
