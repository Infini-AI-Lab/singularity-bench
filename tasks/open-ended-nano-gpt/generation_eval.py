#!/usr/bin/env python3
"""Supplied LAMBADA generation protocol; agents must not modify this file."""
import argparse
import contextlib
import importlib.util
import json
from pathlib import Path
import string
import sys

NUM_TOKENIZER_IDS = 151669
EOS_ID = 151643
MAX_NEW_TOKENS = 16
PROTOCOL = "lambada_openai_greedy_word_v1"
EDGE_PUNCTUATION = string.punctuation + "“”‘’«»…"


def read_examples(path):
    examples = []
    with Path(path).open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            row = json.loads(line)
            text = row.get("text")
            if not isinstance(text, str):
                raise ValueError(f"Row {line_number} requires a text string")
            parts = text.rstrip().rsplit(maxsplit=1)
            if len(parts) != 2 or not parts[0].strip() or not normalize_word(parts[1]):
                raise ValueError(f"Row {line_number} requires a context and final word")
            # The separating whitespace is omitted; the model generates it.
            examples.append((parts[0], parts[1]))
    if not examples:
        raise ValueError("LAMBADA input must not be empty")
    return examples


def normalize_word(text):
    words = text.split()
    return words[0].strip(EDGE_PUNCTUATION) if words else ""


def validate_tokens(tokens):
    if not isinstance(tokens, list) or not 1 <= len(tokens) <= MAX_NEW_TOKENS:
        raise ValueError("generate must return 1..16 new token IDs")
    if any(type(token) is not int or not 0 <= token < NUM_TOKENIZER_IDS for token in tokens):
        raise ValueError("Generated tokens must be defined integer tokenizer IDs")
    if EOS_ID in tokens:
        if tokens.index(EOS_ID) != len(tokens) - 1:
            raise ValueError("Generation must stop immediately after end-of-text")
    elif len(tokens) != MAX_NEW_TOKENS:
        raise ValueError("Generation stopped before 16 tokens without end-of-text")
    return tokens


def predicted_word(tokenizer, tokens):
    validate_tokens(tokens)
    if tokens[-1] == EOS_ID:
        tokens = tokens[:-1]
    return normalize_word(tokenizer.decode(tokens, skip_special_tokens=False))


def summary(correct, total):
    return {"protocol": PROTOCOL, "num_examples": total, "num_correct": correct,
            "accuracy": correct / total}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model_path", required=True, type=Path)
    parser.add_argument("--data_path", required=True, type=Path)
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    if not args.model_path.is_dir() or not (args.model_path / "config.json").is_file():
        raise ValueError("model_path must contain a checkpoint and config.json")
    examples = read_examples(args.data_path)
    from tokenizers import Tokenizer
    tokenizer = Tokenizer.from_file(str(root / "models/tokenizer.json"))
    tokenizer.no_truncation()
    tokenizer.no_padding()
    # Only this block imports agent code. stdout belongs to the JSONL protocol.
    with contextlib.redirect_stdout(sys.stderr):
        spec = importlib.util.spec_from_file_location("generation_model", root / "generation_model.py")
        adapter = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(adapter)
        model = adapter.load_model(str(args.model_path.resolve()))
    correct = 0
    for example_id, (prompt, answer) in enumerate(examples):
        prompt_ids = tokenizer.encode(prompt, add_special_tokens=False).ids
        with contextlib.redirect_stdout(sys.stderr):
            tokens = adapter.generate(model, list(prompt_ids), MAX_NEW_TOKENS)
        prediction = predicted_word(tokenizer, tokens)
        correct += prediction == normalize_word(answer)
        print(json.dumps({"example_id": example_id, "generated_token_ids": tokens}), flush=True)
    print(json.dumps(summary(correct, len(examples))), flush=True)


if __name__ == "__main__":
    main()
