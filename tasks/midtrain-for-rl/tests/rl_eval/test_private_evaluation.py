"""Coverage, private preparation, expanded-test sandbox routing and public reward contract."""
import ast
import io
import json
from pathlib import Path
import re
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import evaluate
import prepare_eval
import rewards
import sandbox

ROOT = Path(__file__).resolve().parent
TASK = ROOT.parents[1]


def amc_fixture():
    # Source-local integer IDs are not globally unique; URL/text identify tasks.
    return [{'id': i % 25, 'problem': f'Math problem {i}', 'answer': float(i),
             'url': f'https://example.invalid/problem/{i}'} for i in range(83)]


def code_fixture():
    return [{'task_id': f'HumanEval/{i}', 'prompt': 'def answer(x):\n    """Return x."""\n',
             'entry_point': 'answer', 'canonical_solution': '    return PRIVATE_REFERENCE',
             'test': 'def check(candidate):\n    assert candidate(2) == 2\n    assert candidate(-1) == -1\n'}
            for i in range(164)]


class PreparationTests(unittest.TestCase):
    def test_all_83_math_problems_and_integer_labels_are_preserved(self):
        rows = prepare_eval.amc_rows(amc_fixture())
        self.assertEqual(len({row['id'] for row in rows}), 83)
        self.assertEqual(rows[-1]['label'], '82')
        for bad in (amc_fixture()[:-1], amc_fixture() + amc_fixture()[:1],
                    amc_fixture()[:-1] + amc_fixture()[:1]):
            with self.assertRaises(ValueError):
                prepare_eval.amc_rows(bad)
        for invalid in (float('nan'), float('inf'), 0.5):
            rows = amc_fixture()
            rows[0]['answer'] = invalid
            with self.assertRaises(ValueError):
                prepare_eval.amc_rows(rows)

    def test_every_expanded_test_is_preserved_and_references_stay_out_of_prompts(self):
        source = code_fixture()
        rows = prepare_eval.humanevalplus_rows(reversed(source))
        self.assertEqual(len(rows), 164)
        self.assertEqual(rows[0]['id'], 'HumanEval/0')
        self.assertEqual(rows[0]['label']['test'], source[0]['test'])
        self.assertEqual(rows[0]['label']['entry_point'], 'answer')
        self.assertNotIn('PRIVATE_REFERENCE', json.dumps(rows))
        self.assertNotIn('assert candidate', rows[0]['prompt'])
        self.assertIn(source[0]['prompt'], rows[0]['prompt'])
        for bad in (source[:-1], source + source[:1], source[:-1] + source[:1]):
            with self.assertRaises(ValueError):
                prepare_eval.humanevalplus_rows(bad)

    def test_preparation_writes_only_new_suites_and_unchanged_science(self):
        import csv
        import random
        import pyarrow as pa
        import pyarrow.parquet as pq
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            pq.write_table(pa.Table.from_pylist(amc_fixture()), directory / 'math.parquet')
            (directory / 'code.jsonl').write_text(''.join(json.dumps(row) + '\n' for row in code_fixture()))
            science = directory / 'science.csv'
            with science.open('w', newline='') as stream:
                writer = csv.DictWriter(stream, fieldnames=['Question', 'Correct Answer',
                                                            *[f'Incorrect Answer {i}' for i in range(1, 4)]])
                writer.writeheader()
                writer.writerows({'Question': f'Question {i}', 'Correct Answer': 'right',
                                  **{f'Incorrect Answer {j}': f'wrong {j}' for j in range(1, 4)}}
                                 for i in range(198))
            def download(repo, *_, **kwargs):
                return directory / ('math.parquet' if repo == 'AI-MO/aimo-validation-amc' else
                                    'code.jsonl' if repo == 'evalplus/humanevalplus' else 'science.csv')
            with patch.dict(sys.modules, {'huggingface_hub': SimpleNamespace(hf_hub_download=download)}), \
                    patch.object(sys, 'argv', ['prepare_eval.py', '--output', str(directory / 'prepared')]):
                prepare_eval.main()
            manifest = json.loads((directory / 'prepared/manifest.json').read_text())
            self.assertEqual(set(manifest['datasets']), set(evaluate.EXPECTED))
            for name, (items, samples) in evaluate.EXPECTED.items():
                path = directory / 'prepared' / f'{name}.jsonl'
                self.assertEqual(len(evaluate.jsonl_rows(path)), items)
                self.assertEqual(manifest['datasets'][name]['sha256'], prepare_eval.digest(path))
                self.assertEqual(manifest['datasets'][name]['samples'], samples)
            rows = [json.loads(line) for line in evaluate.jsonl_rows(directory / 'prepared/gpqa_diamond.jsonl')]
            for i, row in enumerate(rows):
                permutation = list(range(4))
                random.Random(f'{manifest["seed"]}:gpqa:{i}').shuffle(permutation)
                self.assertEqual(row['label'], chr(65 + permutation.index(0)))


class SandboxTests(unittest.TestCase):
    def fixture_execute(self, program, stdin, **kwargs):
        # Execute only the tiny, locally authored fixtures in these tests.
        result = subprocess.run([sys.executable, '-I', '-c', program], input=stdin,
                                text=True, capture_output=True, timeout=3)
        return result.stdout if result.returncode == 0 else None

    def test_full_assertions_wrong_answers_and_early_exit(self):
        judge = object.__new__(sandbox.Judge)
        judge.execute = self.fixture_execute
        label = code_fixture()[0]
        label = {'testtype': 'humanevalplus', 'entry_point': label['entry_point'], 'test': label['test']}
        self.assertTrue(judge.score('def answer(x): return x', label))
        # Passes first assertion, fails a later expanded assertion.
        self.assertFalse(judge.score('def answer(x): return 2', label))
        self.assertFalse(judge.score('raise SystemExit(0)', label))
        self.assertFalse(judge.score('def different(x): return x', label))
        judge.execute = lambda *args, **kwargs: None
        self.assertFalse(judge.score('def answer(x): return x', label))

    def test_reward_sends_complete_expanded_suite_to_operator_service(self):
        label = {'testtype': 'humanevalplus', 'entry_point': 'answer', 'test': code_fixture()[0]['test']}
        response = '```python\ndef answer(x): return x\n```'
        with patch.dict('os.environ', {'MIDTRAIN_CODE_SANDBOX_URL': 'http://sandbox.invalid'}), \
                patch.object(rewards.urllib.request, 'urlopen') as send:
            send.return_value = io.StringIO('{"status":"ok","passed":true}')
            self.assertEqual(rewards.code_reward(response, label), 1.0)
            payload = json.loads(send.call_args.args[0].data)
            self.assertEqual(payload, {'code': 'def answer(x): return x', 'label': label})
            send.return_value = io.StringIO('{"status":"infrastructure_error"}')
            with self.assertRaises(RuntimeError):
                rewards.code_reward(response, label)

    def test_large_program_uses_stdin_and_keeps_docker_isolation(self):
        class Process:
            returncode = 0
            def __init__(self, command, *, stdin, stdout, **kwargs):
                payload = json.load(stdin)
                self_outer.assertGreater(len(payload['code']), 500_000)
                self_outer.assertEqual(payload['stdin'], '2 3\n')
                # Model a sandbox success; do not execute supplied source on host.
                stdout.write(b'5\n')
                stdout.flush()
                for flag in ('--read-only', '--cap-drop', '--security-opt', '--pids-limit', '--memory'):
                    self_outer.assertIn(flag, command)
                self_outer.assertEqual(command[command.index('--network') + 1], 'none')
                self_outer.assertEqual(command[command.index('--user') + 1], '65534:65534')
                self_outer.assertLess(len(command[-1]), 1024)
                ast.parse(command[-1])
            def poll(self):
                return 0
        self_outer = self
        judge = object.__new__(sandbox.Judge)
        import threading
        judge.image, judge.timeout, judge.slots = 'sha256:fixture', 6, threading.BoundedSemaphore(1)
        with patch.object(sandbox.subprocess, 'Popen', Process), patch.object(sandbox.subprocess, 'run'):
            self.assertEqual(judge.execute('#' + 'x' * 600_000, '2 3\n'), '5\n')


class PublicContractTests(unittest.TestCase):
    def test_instruction_code_matches_actual_rl_reward_implementation(self):
        blocks = re.findall(r'^```python\n(.*?)^```$', (TASK / 'instruction.md').read_text(), re.M | re.S)
        documented = {}
        for block in blocks:
            tree = ast.parse(block)
            documented.update({node.name: ast.dump(node) for node in ast.walk(tree)
                               if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))})
        for module, names in ((ROOT / 'rewards.py', ['final_boxed', 'math_reward', 'science_reward',
                                                   'extract_code', 'code_reward', 'reward']),
                              (ROOT / 'sandbox.py', ['equivalent', 'functional_program', 'score_rl'])):
            functions = {node.name: ast.dump(node) for node in ast.walk(ast.parse(module.read_text()))
                         if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))}
            for name in names:
                self.assertEqual(documented[name], functions[name], name)

    def test_private_source_identities_are_absent_from_agent_docs(self):
        sources = json.loads((ROOT / 'sources.json').read_text())
        private_ids = [spec['repo'] for spec in sources.values() if isinstance(spec, dict)]
        eval_sources = json.loads((ROOT / 'eval_sources.json').read_text())
        private_ids += [spec['repo'] for spec in eval_sources.values() if isinstance(spec, dict)]
        for path in [TASK / name for name in ('README.md', 'instruction.md')]:
            for repo in private_ids:
                self.assertNotIn(repo, path.read_text(), str(path))
        for path in TASK.rglob('README.md'):
            if 'vendor' in path.parts:
                continue
            for repo in private_ids:
                self.assertNotIn(repo, path.read_text(), str(path))


if __name__ == '__main__':
    unittest.main()
