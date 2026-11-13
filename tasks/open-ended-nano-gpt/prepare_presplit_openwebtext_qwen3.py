#!/usr/bin/env python3
"""Tokenize a manifest-verified, presplit OpenWebText snapshot with Qwen3."""

import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from tqdm import tqdm
from transformers import AutoTokenizer


DATASET_REPO = "YangZhoumill/openwebtext-presplit"
DATASET_REVISION = "b96bd5a6a3ae68fe05593cac6a36b170296e8f04"
SOURCE_REPO = "Skylion007/openwebtext"
SOURCE_REVISION = "79d93d786212f7344586290adb811d4ae6a1762c"
DOCUMENT_END_TOKEN = "<|endoftext|>"


def arguments():
    task_root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-dir",
        type=Path,
        required=True,
        help="downloaded snapshot directory containing split_manifest.json",
    )
    parser.add_argument("--tokenizer", type=Path, default=task_root / "models")
    parser.add_argument(
        "--train-output-dir",
        type=Path,
        required=True,
        help="output directory of the training split (train.bin, meta.json)",
    )
    parser.add_argument(
        "--test-output-dir",
        type=Path,
        required=True,
        help="output directory of the private validation split (test.bin, meta.json)",
    )
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--skip-source-checksums", action="store_true")
    parser.add_argument("--validate-source-only", action="store_true")
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument(
        "--splits", nargs="+", choices=("train", "test"), default=("train", "test")
    )
    return parser.parse_args()


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_manifest(root, check_hashes):
    manifest_path = root / "split_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    expected = {
        "source_repo": SOURCE_REPO,
        "source_revision": SOURCE_REVISION,
        "shuffle_seed": 2357,
        "test_fraction_requested": 0.0005,
        "total_rows": 8_013_769,
        "train_rows": 8_009_762,
        "test_rows": 4_007,
    }
    for key, value in expected.items():
        if manifest.get(key) != value:
            raise RuntimeError(f"manifest {key} is {manifest.get(key)!r}, expected {value!r}")
    for split, entries in manifest["files"].items():
        for entry in entries:
            path = root / entry["path"]
            if not path.is_file() or path.stat().st_size != entry["bytes"]:
                raise RuntimeError(f"missing or incorrectly sized source file: {path}")
            if check_hashes and sha256(path) != entry["sha256"]:
                raise RuntimeError(f"source checksum mismatch: {path}")
    return manifest, sha256(manifest_path)


def save_progress(path, progress):
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(progress, indent=2) + "\n")
    os.replace(temporary, path)


def tokenize_split(split, entries, input_root, output_dir, tokenizer, eot_id, batch_size):
    output_dir.mkdir(parents=True, exist_ok=True)
    final_path = output_dir / f"{split}.bin"
    partial_path = output_dir / f"{split}.bin.part"
    progress_path = output_dir / f"{split}.progress.json"

    progress = {"completed_files": [], "num_documents": 0, "num_tokens": 0}
    if progress_path.exists():
        progress = json.loads(progress_path.read_text())
    completed = progress["completed_files"]
    expected_completed = [entry["path"] for entry in entries[: len(completed)]]
    if completed != expected_completed:
        raise RuntimeError(f"non-prefix progress state in {progress_path}")
    expected_bytes = progress["num_tokens"] * np.dtype("<u4").itemsize
    if final_path.exists():
        if len(completed) != len(entries):
            raise RuntimeError(f"final output has incomplete progress: {final_path}")
        if final_path.stat().st_size != expected_bytes:
            raise RuntimeError(f"final output size does not match {progress_path}")
        return final_path, progress
    if not partial_path.exists():
        if completed:
            raise RuntimeError(f"partial output is missing for {progress_path}")
    else:
        actual_bytes = partial_path.stat().st_size
        if actual_bytes < expected_bytes:
            raise RuntimeError(f"partial output is shorter than {progress_path}")
        if actual_bytes > expected_bytes:
            print(
                f"discarding {actual_bytes - expected_bytes} bytes from an interrupted "
                f"{split} source shard"
            )
            with partial_path.open("r+b") as partial:
                partial.truncate(expected_bytes)

    with partial_path.open("ab") as output:
        for entry in tqdm(entries[len(completed) :], desc=f"tokenizing {split} shards"):
            parquet = pq.ParquetFile(input_root / entry["path"])
            shard_documents = 0
            shard_tokens = 0
            for batch in parquet.iter_batches(batch_size=batch_size, columns=["text"]):
                texts = batch.column(0).to_pylist()
                encoded = tokenizer(
                    texts,
                    add_special_tokens=False,
                    return_attention_mask=False,
                    return_token_type_ids=False,
                )["input_ids"]
                for document in encoded:
                    values = np.asarray(document + [eot_id], dtype="<u4")
                    output.write(values.tobytes())
                    shard_tokens += len(values)
                shard_documents += len(encoded)
            output.flush()
            os.fsync(output.fileno())
            progress["completed_files"].append(entry["path"])
            progress["num_documents"] += shard_documents
            progress["num_tokens"] += shard_tokens
            save_progress(progress_path, progress)

    if progress["num_documents"] <= 0 or progress["num_tokens"] <= 0:
        raise RuntimeError(f"empty tokenized {split} split")
    os.replace(partial_path, final_path)
    return final_path, progress


def write_metadata(output_dir, split, progress, manifest_hash, tokenizer, tokenizer_dir):
    metadata = {
        "dataset_repository": DATASET_REPO,
        "dataset_revision": DATASET_REVISION,
        "source_repository": SOURCE_REPO,
        "source_revision": SOURCE_REVISION,
        "source_manifest_sha256": manifest_hash,
        "split": split,
        "shuffle_seed": 2357,
        "test_fraction": 0.0005,
        "num_documents": progress["num_documents"],
        "num_tokens": progress["num_tokens"],
        "dtype": "uint32-le",
        "tokenizer_path": str(tokenizer_dir),
        "tokenizer_size": len(tokenizer),
        "document_end_token": DOCUMENT_END_TOKEN,
        "document_end_token_id": tokenizer.convert_tokens_to_ids(DOCUMENT_END_TOKEN),
        "completed_source_files": progress["completed_files"],
    }
    (output_dir / "meta.json").write_text(json.dumps(metadata, indent=2) + "\n")


def validate_tokenized(split, entries, output_dir, manifest, manifest_hash, tokenizer):
    final_path = output_dir / f"{split}.bin"
    progress_path = output_dir / f"{split}.progress.json"
    metadata_path = output_dir / "meta.json"
    if not final_path.is_file() or not progress_path.is_file() or not metadata_path.is_file():
        raise RuntimeError(f"tokenized {split} output is incomplete in {output_dir}")
    progress = json.loads(progress_path.read_text())
    metadata = json.loads(metadata_path.read_text())
    if progress.get("completed_files") != [entry["path"] for entry in entries]:
        raise RuntimeError(f"tokenized {split} source-file list is incomplete")
    if progress.get("num_documents") != manifest[f"{split}_rows"]:
        raise RuntimeError(f"tokenized {split} document count is incorrect")
    if final_path.stat().st_size != progress.get("num_tokens", -1) * 4:
        raise RuntimeError(f"tokenized {split} byte count is incorrect")
    expected_metadata = {
        "dataset_repository": DATASET_REPO,
        "dataset_revision": DATASET_REVISION,
        "source_repository": SOURCE_REPO,
        "source_revision": SOURCE_REVISION,
        "source_manifest_sha256": manifest_hash,
        "split": split,
        "shuffle_seed": 2357,
        "test_fraction": 0.0005,
        "num_documents": progress["num_documents"],
        "num_tokens": progress["num_tokens"],
        "dtype": "uint32-le",
        "tokenizer_size": len(tokenizer),
        "document_end_token": DOCUMENT_END_TOKEN,
        "document_end_token_id": tokenizer.convert_tokens_to_ids(DOCUMENT_END_TOKEN),
        "completed_source_files": progress["completed_files"],
    }
    for key, value in expected_metadata.items():
        if metadata.get(key) != value:
            raise RuntimeError(f"tokenized {split} metadata mismatch for {key}")
    print(
        f"validated {split}: {progress['num_documents']} documents, "
        f"{progress['num_tokens']} tokens"
    )


def main():
    args = arguments()
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive")
    manifest, manifest_hash = validate_manifest(
        args.input_dir, check_hashes=not args.skip_source_checksums
    )
    if args.validate_source_only:
        print(
            f"validated source: {manifest['total_rows']} documents, "
            f"{sum(len(files) for files in manifest['files'].values())} files"
        )
        return
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, local_files_only=True, use_fast=True)
    eot_id = tokenizer.convert_tokens_to_ids(DOCUMENT_END_TOKEN)
    if eot_id == tokenizer.unk_token_id:
        raise RuntimeError(f"tokenizer does not contain {DOCUMENT_END_TOKEN}")

    output_dirs = {"train": args.train_output_dir, "test": args.test_output_dir}
    if args.validate_only:
        for split in args.splits:
            validate_tokenized(
                split,
                manifest["files"][split],
                output_dirs[split],
                manifest,
                manifest_hash,
                tokenizer,
            )
        return
    for split in args.splits:
        output_dir = output_dirs[split]
        _, progress = tokenize_split(
            split,
            manifest["files"][split],
            args.input_dir,
            output_dir,
            tokenizer,
            eot_id,
            args.batch_size,
        )
        expected_rows = manifest[f"{split}_rows"]
        if progress["num_documents"] != expected_rows:
            raise RuntimeError(
                f"{split} produced {progress['num_documents']} documents, expected {expected_rows}"
            )
        write_metadata(output_dir, split, progress, manifest_hash, tokenizer, args.tokenizer)


if __name__ == "__main__":
    main()
