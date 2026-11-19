"""word-count's scorer on hand-written raw outputs, without running the task (breakwater/WRITING_TASKS.md, "Check and
test the task").

    python3 -m unittest discover -s examples/word-count/tests/unit -v
"""
from pathlib import Path
import sys
import tempfile
import types
import unittest

TASK = Path(__file__).resolve().parents[2]
REPO = TASK.parents[1]
sys.path.insert(0, str(REPO / 'breakwater'))  # Breakwater's helpers, as in the stages
from contract import ContractError  # noqa: E402
from task_api import load_hook  # noqa: E402

# Loaded by path under its own name, as the stages load it, so no other task's breakwater_task can stand in for it.
task = load_hook(TASK / 'tests/breakwater_task.py', 'word_count_task')


def scoring(root, outputs):
    """What score() reads: the sealed raw outputs and the texts asset, here hand-written."""
    (root / 'texts.jsonl').write_text('{"id": "t0", "text": "The cat, the hat."}\n{"id": "t1", "text": "Go!"}\n')
    (root / 'raw').mkdir()
    (root / 'raw/counts.jsonl').write_text(''.join(line + '\n' for line in outputs))
    return types.SimpleNamespace(raw=root / 'raw', asset=lambda name: root / 'texts.jsonl', smoke=False)


class ScoreTests(unittest.TestCase):
    def setUp(self):
        self.root = Path(self.enterContext(tempfile.TemporaryDirectory()))

    def test_correct_counts_score_one(self):
        result = task.score(scoring(self.root, ['{"id": "t0", "counts": {"cat": 1, "hat": 1, "the": 2}}',
                                                '{"id": "t1", "counts": {"go": 1}}']))
        self.assertEqual(result['reward'], 1.0)
        self.assertEqual(result['details'], {'correct': 2, 'texts': 2})

    def test_wrong_and_missing_answers_earn_nothing(self):
        result = task.score(scoring(self.root, ['{"id": "t0", "counts": {"cat": 1, "hat": 1, "the": 1}}']))
        self.assertEqual(result['reward'], 0.0)

    def test_a_text_answered_twice_stops_scoring(self):
        line = '{"id": "t1", "counts": {"go": 1}}'
        with self.assertRaises(ContractError):
            task.score(scoring(self.root, [line, line]))


if __name__ == '__main__':
    unittest.main()
