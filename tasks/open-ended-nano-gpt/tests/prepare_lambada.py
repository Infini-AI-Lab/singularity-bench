#!/usr/bin/env python3
"""Host-only preparation of the pinned English LAMBADA OpenAI test split."""
import argparse
import hashlib
import json
from pathlib import Path
import tempfile
import os
import subprocess

# lambada_prompts sits next to this script; this import also works under python3 -I.
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
import lambada_prompts

DATASET = json.loads(Path(__file__).with_name("lambada_dataset.json").read_text())
SHA256 = DATASET["sha256"]


def validate(content):
    if hashlib.sha256(content).hexdigest() != SHA256:
        raise ValueError("LAMBADA checksum mismatch")
    rows = [json.loads(line) for line in content.splitlines()]
    if len(rows) != DATASET["num_examples"] or any(not isinstance(row.get("text"), str) for row in rows):
        raise ValueError(f"Expected {DATASET['num_examples']} English LAMBADA text rows")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--prompts-output", type=Path,
                        help="answer-free JSONL; defaults to lambada_prompts.jsonl beside the private split")
    parser.add_argument("--check", action="store_true", help="validate local data without downloading")
    args = parser.parse_args()
    output = args.output.resolve()
    task = Path(__file__).resolve().parents[1]
    if output.is_relative_to(task):
        raise ValueError("Private LAMBADA must be stored outside the agent task directory")
    prompts = args.prompts_output or output.with_name("lambada_prompts.jsonl")
    if prompts.is_symlink():
        raise ValueError("LAMBADA prompt output must not be a symlink")
    prompts = prompts.resolve()
    if prompts == output or prompts.is_relative_to(task):
        raise ValueError("Answer-free LAMBADA output must be separate and outside the agent task directory")
    valid = False
    if output.exists():
        try:
            validate(output.read_bytes())
            valid = True
        except (ValueError, TypeError, AttributeError):
            if args.check:
                raise
    if not valid:
        if args.check:
            raise FileNotFoundError(f"Prepared LAMBADA is missing: {output}")
        output.parent.mkdir(parents=True, exist_ok=True)
        # Like OpenWebText, fetch the pinned snapshot using the HOST Hugging Face
        # CLI. Containers stay offline and the agent never receives this directory.
        with tempfile.TemporaryDirectory(prefix=".lambada-download-", dir=output.parent) as directory:
            subprocess.run([
                "hf", "download", DATASET["repository"], DATASET["file"],
                "--repo-type", "dataset", "--revision", DATASET["revision"],
                "--local-dir", directory,
            ], env=dict(os.environ, HF_HUB_OFFLINE="0"), check=True)
            content = (Path(directory) / DATASET["file"]).read_bytes()
            validate(content)
            temporary = Path(directory) / "validated.jsonl"
            temporary.write_bytes(content)
            temporary.chmod(0o644)
            temporary.replace(output)
    lambada_prompts.prepare(prompts, output.read_bytes(), check=args.check)
    print(f"Prepared answer-free audit/generation prompts at {prompts}")
    print(f"Validated {DATASET['num_examples']} LAMBADA examples (sha256={SHA256}) at {output}")


if __name__ == "__main__":
    main()
