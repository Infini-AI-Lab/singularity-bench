import unittest
import tempfile
from pathlib import Path
from evaluate import summarize, jsonl_rows


class EvaluationTests(unittest.TestCase):
    def test_unicode_separator_is_not_a_jsonl_record_boundary(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'data.jsonl'
            path.write_text('{"prompt":"a\u2028b"}\n')
            self.assertEqual(len(jsonl_rows(path)), 1)

    def test_average_not_pass_at_4_and_equal_suite_weights(self):
        report = summarize({'amc': {'rewards': [1, 0, 0, 0]},
                            'humanevalplus': {'rewards': [1, 1]},
                            'gpqa_diamond': {'rewards': [0, 0, 0]}},
                           {'amc': (1, 4), 'humanevalplus': (2, 1), 'gpqa_diamond': (3, 1)})
        self.assertEqual(report['suites']['amc']['score'], 1 / 4)
        self.assertEqual(report['suites']['amc']['metric'], 'avg@4')
        self.assertAlmostEqual(report['final_reward'], 5 / 12)

    def test_incomplete_results_cannot_be_scored(self):
        with self.assertRaises(ValueError):
            summarize({'amc': {'rewards': [1]}}, {'amc': (1, 4)})

    def test_missing_suite_cannot_be_scored(self):
        with self.assertRaises(ValueError):
            summarize({})


if __name__ == '__main__':
    unittest.main()
