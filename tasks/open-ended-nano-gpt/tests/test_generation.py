"""CPU generation protocol tests using invented text, never benchmark examples."""
import importlib.util
import json
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

from tokenizers import Tokenizer

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("generation_protocol", ROOT / "generation_eval.py")
protocol = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(protocol)
SPEC = importlib.util.spec_from_file_location("verifier", Path(__file__).with_name("verify.py"))
verifier = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(verifier)


class GenerationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tokenizer = Tokenizer.from_file(str(ROOT / "models/tokenizer.json"))

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()

    def tokens(self, text):
        return self.tokenizer.encode(text, add_special_tokens=False).ids + [protocol.EOS_ID]

    def test_multitoken_word_is_decoded_as_one_answer(self):
        tokens = self.tokens(" Constantinople, next")
        self.assertGreater(len(self.tokenizer.encode(" Constantinople").ids), 1)
        self.assertEqual(protocol.predicted_word(self.tokenizer, tokens), "Constantinople")
        self.assertEqual(protocol.predicted_word(self.tokenizer, self.tokens(" Καλημέρα!")), "Καλημέρα")
        self.assertEqual(protocol.predicted_word(self.tokenizer, [protocol.EOS_ID]), "")

    def test_normalization_is_case_sensitive_and_keeps_internal_punctuation(self):
        self.assertEqual(protocol.normalize_word(' “O’Brien,” next'), "O’Brien")
        self.assertNotEqual(protocol.normalize_word("Cup"), protocol.normalize_word("cup"))
        self.assertEqual(protocol.normalize_word("   "), "")

    def test_invalid_ids_and_early_stopping_are_rejected(self):
        for tokens in ([], [True, protocol.EOS_ID], [151669, protocol.EOS_ID],
                       [-1, protocol.EOS_ID], [1.0, protocol.EOS_ID], [1],
                       [protocol.EOS_ID, 1], [1] * 17, "cup"):
            with self.subTest(tokens=tokens), self.assertRaises(ValueError):
                protocol.validate_tokens(tokens)
        protocol.validate_tokens([1] * 16)

    def test_runner_supplies_only_prompt_and_fixed_budget_and_loads_once(self):
        task = self.root / "task with spaces"
        task.mkdir()
        for name in ("run_generation.sh", "generation_eval.py"):
            shutil.copyfile(ROOT / name, task / name)
        (task / "models").mkdir()
        shutil.copyfile(ROOT / "models/tokenizer.json", task / "models/tokenizer.json")
        checkpoint = task / "saved model"
        checkpoint.mkdir()
        (checkpoint / "config.json").write_text("{}")
        data = self.root / "toy data.jsonl"
        texts = ["Mira picked up the red cup", "Mira travelled to Constantinople"]
        data.write_text("\n".join(json.dumps({"text": t}) for t in texts) + "\n")
        prompts = [self.tokenizer.encode(t.rsplit(" ", 1)[0], add_special_tokens=False).ids for t in texts]
        outputs = [self.tokens(" cup"), self.tokens(" Constantinople")]
        # A deterministic test double. Assertions ensure no target is supplied
        # and the generation budget does not depend on answer tokenization.
        (task / "generation_model.py").write_text(f'''
loaded = False
def load_model(model_path):
    global loaded
    assert not loaded
    assert model_path == {str(checkpoint)!r}
    loaded = True
    print("fixture loading diagnostic")
    return {{"index": 0}}
def generate(model, prompt_token_ids, max_new_tokens):
    assert max_new_tokens == 16
    index = model["index"]
    assert prompt_token_ids == {prompts!r}[index]
    model["index"] += 1
    return {outputs!r}[index]
''')
        result = subprocess.run(["bash", str(task / "run_generation.sh"),
                                 "--model_path", str(checkpoint), "--data_path", str(data)],
                                cwd=self.root, capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("fixture loading diagnostic", result.stderr)
        rows = [json.loads(line) for line in result.stdout.splitlines()]
        self.assertEqual(rows[-1], protocol.summary(2, 2))
        self.assertEqual([row["generated_token_ids"] for row in rows[:-1]], outputs)

    def test_supplied_stubs_and_missing_arguments_fail(self):
        result = subprocess.run(["bash", str(ROOT / "run_generation.sh")], capture_output=True)
        self.assertNotEqual(result.returncode, 0)
        spec = importlib.util.spec_from_file_location("stub", ROOT / "generation_model.py")
        adapter = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(adapter)
        with self.assertRaises(NotImplementedError):
            adapter.load_model("unused")
        with self.assertRaises(NotImplementedError):
            adapter.generate(None, [1], 16)

    def test_empty_or_malformed_data_fails(self):
        path = self.root / "toy.jsonl"
        for content in ("", "\n", '{}\n', '{"text":"one"}\n', '{"text":"a !!!"}\n'):
            path.write_text(content)
            with self.subTest(content=content), self.assertRaises(ValueError):
                protocol.read_examples(path)

    def test_dataset_checksum_rejects_substitution(self):
        path = self.root / "substitute.jsonl"
        path.write_text('{"text":"A different dataset"}\n')
        with self.assertRaises(ValueError):
            verifier.validate_lambada(path)


if __name__ == "__main__":
    unittest.main()
