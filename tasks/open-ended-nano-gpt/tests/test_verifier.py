"""Scoring-helper tests: no GPU, private corpus, or API calls required."""
import importlib.util
import json
import math
from pathlib import Path
import tempfile
import unittest

SPEC = importlib.util.spec_from_file_location("nano_verifier", Path(__file__).with_name("verify.py"))
verifier = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(verifier)


class VerifierTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name).resolve()
        self.addCleanup(self.temp.cleanup)

    def logprobs(self, indices, token_ids, values, logprob_type="float32"):
        import pyarrow as pa
        import pyarrow.parquet as pq
        path = self.root / "logprobs.parquet"
        pq.write_table(pa.table({"token_index": pa.array(indices, pa.int64()),
                                 "token_id": pa.array(token_ids, pa.int64()),
                                 "logprob": pa.array(values, getattr(pa, logprob_type)())}), path)
        return path

    def test_perplexity_scores_every_target_after_the_first_token_once(self):
        tokens = [151643, 10, 20, 30, 151668]
        data = self.root / "test.bin"
        data.write_bytes(b"".join(token.to_bytes(4, "little") for token in tokens))
        values = [-1.0, -2.0, -0.5, -0.25]
        result = verifier.score_logprobs(self.logprobs([1, 2, 3, 4], tokens[1:], values), data)
        self.assertEqual(result["num_scored_tokens"], 4)
        self.assertAlmostEqual(result["validation_loss"], 0.9375)
        self.assertAlmostEqual(result["perplexity"], math.exp(0.9375))
        for indices, token_ids, logprobs, logprob_type in [
            ([1, 2, 3], tokens[1:4], values[:3], "float32"),
            ([1, 2, 3, 3], [10, 20, 30, 30], values, "float32"),
            ([2, 3, 4, 5], tokens[1:], values, "float32"),
            ([1, 2, 3, 4], [10, 20, 31, 151668], values, "float32"),
            ([1, 2, 3, 4], tokens[1:], [-1.0, 0.5, -0.5, -0.25], "float32"),
            ([1, 2, 3, 4], tokens[1:], [-1.0, float("nan"), -0.5, -0.25], "float32"),
            ([1, 2, 3, 4], tokens[1:], [-1, -2, -1, -1], "int64"),
        ]:
            with self.subTest(indices=indices, token_ids=token_ids, logprobs=logprobs), \
                    self.assertRaises(ValueError):
                verifier.score_logprobs(self.logprobs(indices, token_ids, logprobs, logprob_type), data)
        data.write_bytes(b"".join(token.to_bytes(4, "little") for token in [1, 151669]))
        with self.assertRaises(ValueError):
            verifier.score_logprobs(self.logprobs([1], [151669], [-1.0]), data)

    def test_checkpoint_selection_and_path_escape(self):
        (self.root / "out/final_checkpoint").mkdir(parents=True)
        (self.root / "out/final_checkpoint/config.json").write_text('{}')
        record = self.root / "out/best_checkpoint.json"
        record.write_text(json.dumps({"checkpoint_path": "out/final_checkpoint"}))
        self.assertEqual(verifier.select_checkpoint(self.root), self.root / "out/final_checkpoint")
        for path in ("out/checkpoints/step_001000", "out/other", "out/../out/final_checkpoint", "../outside", "/tmp", "", None):
            record.write_text(json.dumps({"checkpoint_path": path}))
            with self.assertRaises(ValueError):
                verifier.select_checkpoint(self.root)
        record.write_text(json.dumps({"checkpoint_path": "out/final_checkpoint"}))
        checkpoint = self.root / "out/final_checkpoint"
        (checkpoint / "config.json").unlink()
        with self.assertRaises(ValueError):
            verifier.select_checkpoint(self.root)
        checkpoint.rmdir()
        with self.assertRaises(ValueError):
            verifier.select_checkpoint(self.root)
        other = self.root / "out/other"
        other.mkdir()
        (other / "config.json").write_text('{}')
        checkpoint.symlink_to(other, target_is_directory=True)
        with self.assertRaises(ValueError):
            verifier.select_checkpoint(self.root)

    def test_duplicate_and_nonfinite_json_rejected(self):
        path = self.root / "best_checkpoint.json"
        for content in ('{"checkpoint_path":"out/other","checkpoint_path":"out/final_checkpoint"}',
                        '{"validation_loss": NaN}'):
            path.write_text(content)
            with self.subTest(content=content), self.assertRaises(ValueError):
                verifier.read_json(path)


if __name__ == '__main__':
    unittest.main()
