"""Breakwater stays a framework: no task's name or task-specific feature appears in its code or its documents, they never
name the benchmark it ships with, its paper or its studies, and its tests need none of this repository's tasks.
Task-specific code, data and documentation live in the tasks. No Docker or GPUs.
"""
from pathlib import Path
import re
import unittest

FRAMEWORK = Path(__file__).resolve().parents[1]
TASK_SPECIFIC = re.compile(
    r'fleet|(?<!Slurm )cluster(?!s? ?(with|without)|.*Slurm)|sandbox[-_]worker|\bgrid\b|downstream|\bcells?\b|HARNESS_'
    r'|CLUSTER_'
    r'|coding-agent-harness|rl-swe-agent|cluster-serverless|open-ended-nano-gpt|midtrain|serve-glm', re.IGNORECASE)
# The framework stands on its own: the benchmark it ships with, that benchmark's paper and its studies are not its
# subject. (Apptainer's own legacy name, as in /.singularity.d or singularity.conf, is not the benchmark's.)
PROJECT_SPECIFIC = re.compile(r'singularity-bench|red.?team|\bthe paper\b|\bour (paper|study)\b', re.IGNORECASE)


def framework_files():
    """The framework's code and documents, its tests aside."""
    return sorted(path for path in FRAMEWORK.rglob('*') if path.suffix in ('.py', '.md')
                  and 'tests' not in path.relative_to(FRAMEWORK).parts)


def matches(pattern):
    return [f'{path.relative_to(FRAMEWORK)}:{number}: {line.strip()[:120]}'
            for path in framework_files() for number, line in enumerate(path.read_text().splitlines(), 1)
            if pattern.search(line)]


class FrameworkGenericTests(unittest.TestCase):
    def test_no_task_names_in_the_framework(self):
        found = matches(TASK_SPECIFIC)
        self.assertEqual(found, [], 'task-specific names in the framework:\n' + '\n'.join(found))

    def test_the_framework_stands_on_its_own(self):
        found = matches(PROJECT_SPECIFIC)
        self.assertEqual(found, [], 'the benchmark, its paper or its studies in the framework:\n' + '\n'.join(found))

    def test_the_framework_tests_need_no_task_of_this_repository(self):
        # The framework's tests use its example task and their own fixtures, so Breakwater can be tested on its own.
        names = re.compile(r'\btasks/|[\'"]tasks[\'"]|open-ended-nano-gpt|midtrain-for-rl|rl-swe-agent|serve-glm53-flash'
                           r'|coding-agent-harness|cluster-serverless')
        files = [path for folder in (FRAMEWORK / 'tests', FRAMEWORK / 'apptainer' / 'tests') for path in folder.rglob('*')
                 if path.is_file() and path.suffix in ('.py', '.json', '.md', '.toml') and path != Path(__file__).resolve()]
        found = [f'{path.relative_to(FRAMEWORK)}:{number}: {line.strip()[:120]}'
                 for path in files for number, line in enumerate(path.read_text().splitlines(), 1) if names.search(line)]
        self.assertEqual(found, [], "the framework's tests depend on this repository's tasks:\n" + '\n'.join(found))

    def test_the_pattern_catches_what_it_should(self):
        for text in ('the cluster fleet', 'rl-swe-agent', 'HARNESS_LLM_BASE_URL', 'a grid cell', 'downstream tasks',
                     'CLUSTER_GRADER_TOKEN', 'sandbox_worker'):
            with self.subTest(text=text):
                self.assertTrue(TASK_SPECIFIC.search(text))
        for text in ('a Slurm cluster', 'clusters without Docker', 'the harness file of the installed CLI',
                     'the driver role'):
            with self.subTest(text=text):
                self.assertFalse(TASK_SPECIFIC.search(text))
        for text in ('Singularity-Bench', 'singularity-bench-gateway/2.1', 'the red-team attacks', 'in the paper',
                     'our study'):
            with self.subTest(text=text):
                self.assertTrue(PROJECT_SPECIFIC.search(text))
        for text in ('/.singularity.d/libs', 'singularity/singularity.conf', "'SINGULARITY_'", 'a paper trail'):
            with self.subTest(text=text):
                self.assertFalse(PROJECT_SPECIFIC.search(text))


if __name__ == '__main__':
    unittest.main()
