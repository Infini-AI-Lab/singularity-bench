"""Trusted open-ended-nano-gpt scoring helpers, imported as ``verify`` by breakwater/nano_stages.py.

The evaluation coordinator and the verifier use them to select the submitted checkpoint,
hash-check the supplied generation files, check the pinned LAMBADA split and compute
perplexity from the sealed log probabilities. None of them imports submitted code.
"""
import hashlib
import json
import math
from pathlib import Path
import stat
import types

TEST_DIR = Path(__file__).resolve().parent
LAMBADA_SAMPLE_SIZE = 500
LAMBADA_SAMPLE_SEED = 20260911


def generation_contract(root):
    # Hashes live in the verifier-only tests mount. Never import unchecked code
    # as root. The public protocol module imports agent code only inside main().
    for relative, expected in read_json(TEST_DIR / "generation_contract.json").items():
        path = root / relative
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise ValueError(f"Required supplied generation file missing or modified: {relative}")
    if not (root / "generation_model.py").is_file():
        raise ValueError("Required generation_model.py is missing")
    protocol = types.ModuleType("trusted_generation_protocol")
    # Compile verified source directly: never load submission-provided .pyc files.
    path = root / "generation_eval.py"
    exec(compile(path.read_bytes(), str(path), "exec"), protocol.__dict__)
    return protocol


def validate_lambada(path):
    expected = read_json(TEST_DIR / "lambada_dataset.json")
    if hashlib.sha256(path.read_bytes()).hexdigest() != expected["sha256"]:
        raise ValueError("Private LAMBADA split does not match the pinned dataset checksum")


def read_json(path):
    def invalid(value):
        raise ValueError(f"Non-finite JSON value: {value}")
    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"Duplicate JSON key: {key}")
            result[key] = value
        return result
    return json.loads(Path(path).read_text(), parse_constant=invalid, object_pairs_hook=unique)


def select_checkpoint(root):
    root = root.resolve()
    record = read_json(root / "out/best_checkpoint.json")
    if record.get("checkpoint_path") != "out/final_checkpoint":
        raise ValueError("best checkpoint_path must be exactly out/final_checkpoint")
    checkpoint = root / "out/final_checkpoint"
    if checkpoint.resolve() != checkpoint or not checkpoint.is_dir():
        raise ValueError("out/final_checkpoint must be a real directory, not a symlink")
    if not (checkpoint / "config.json").is_file():
        raise ValueError("Best checkpoint lacks config.json")
    return checkpoint


def score_logprobs(parquet_path, data_path):
    """Validate full target coverage and compute PPL from natural-log probabilities.

    Read bounded batches so scoring does not require the corpus in memory.
    Only the probability of each observed target is needed, not the vocabulary
    distribution. Model provenance/causality still require the independent audit.
    """
    import numpy as np
    import pyarrow as pa
    import pyarrow.parquet as pq

    size = Path(data_path).stat().st_size
    if size < 8 or size % 4:
        raise ValueError("Validation data must contain at least two uint32 tokens")
    expected_tokens = size // 4 - 1
    if not stat.S_ISREG(Path(parquet_path).lstat().st_mode):
        raise ValueError("Log probabilities must be a regular Parquet file")
    with pq.ParquetFile(parquet_path) as parquet, Path(data_path).open("rb") as data:
        schema = parquet.schema_arrow
        if (schema.names != ["token_index", "token_id", "logprob"] or
                schema.field("token_index").type != pa.int64() or
                schema.field("token_id").type != pa.int64() or
                schema.field("logprob").type not in (pa.float32(), pa.float64())):
            raise ValueError("Expected token_index:int64, token_id:int64, logprob:float32/float64")
        if parquet.metadata.num_rows != expected_tokens:
            raise ValueError(f"Expected {expected_tokens} logprob rows, received {parquet.metadata.num_rows}")
        data.seek(4)  # The first stream token is context, not a next-token target.
        count = 0
        batch_sums = []
        for batch in parquet.iter_batches(batch_size=65536):
            if any(column.null_count for column in batch.columns):
                raise ValueError("Logprob rows must not contain nulls")
            indices, token_ids, logprobs = [column.to_numpy() for column in batch.columns]
            length = batch.num_rows
            if not np.array_equal(indices, np.arange(count + 1, count + length + 1)):
                raise ValueError("token_index must cover positions 1..N-1 exactly once, in order")
            targets = np.frombuffer(data.read(length * 4), dtype="<u4")
            if not np.array_equal(token_ids, targets) or np.any(targets >= 151669):
                raise ValueError("token_id must match the private target at every position and be <151669")
            if not np.all(np.isfinite(logprobs)) or np.any(logprobs > 0):
                raise ValueError("logprob must be finite and nonpositive")
            batch_sums.append(math.fsum(float(value) for value in logprobs))
            count += length
        if count != expected_tokens:
            raise ValueError(f"Expected {expected_tokens} scored tokens, received {count}")
    loss = -math.fsum(batch_sums) / count
    ppl = math.exp(loss)
    if not math.isfinite(ppl):
        raise ValueError("Perplexity overflow")
    return {"validation_loss": loss, "num_scored_tokens": count, "perplexity": ppl}
