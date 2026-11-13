#!/usr/bin/env python3
"""Download, derive and audit midtrain-for-rl's pinned inputs.

Every input is pinned to a Hugging Face revision and verified against the
repository's own checksums (LFS sha256, or the git blob sha1 for small files).
Each prepared directory receives a completion record (.midtrain-for-rl.json)
listing its files with sizes and sha256 digests. --check validates those
records offline; --check --deep also recomputes every digest.

Roots (defaults live inside the task directory, which Git ignores):
  --artifacts-root  artifacts/  base model                agent (read-only) and verifier
  --data-root       data/       corpora and dev evals      agent (read-only)
"""
from __future__ import annotations

import argparse
import base64
from concurrent.futures import ThreadPoolExecutor, as_completed
import fnmatch
import hashlib
import json
import os
from pathlib import Path
import pickle
import random
import re
import shutil
import statistics
import sys
import threading
import time
import uuid
import zlib

TASK_ROOT = Path(__file__).resolve().parent
SOURCES = TASK_ROOT / "data_sources.json"
RECORD = ".midtrain-for-rl.json"
STEPS = ("model", "corpora", "dev-evals", "readme")
_print_lock = threading.Lock()


def log(message: str) -> None:
    with _print_lock:
        print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


# --------------------------------------------------------------------------- files


def sha256_file(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(16 * 1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def git_blob_sha1(path: Path) -> str:
    hasher = hashlib.sha1()
    hasher.update(f"blob {path.stat().st_size}\0".encode())
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(16 * 1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def remote_files(repo: str, revision: str, patterns: list[str], repo_type: str = "dataset") -> list[dict]:
    """Pinned file inventory with the repository's own checksums."""
    from huggingface_hub import HfApi

    # List only the directories the patterns can match; a pattern without one needs the whole tree.
    heads = [pattern.split("*", 1)[0] for pattern in patterns]
    roots = [None] if any("/" not in head for head in heads) else sorted({h.rsplit("/", 1)[0] for h in heads})
    files = {}
    api = HfApi()
    for root in roots:
        for item in api.list_repo_tree(repo, repo_type=repo_type, revision=revision, recursive=True,
                                       path_in_repo=root):
            if getattr(item, "size", None) is None:
                continue
            if not any(fnmatch.fnmatchcase(item.path, pattern) for pattern in patterns):
                continue
            lfs = getattr(item, "lfs", None)
            files[item.path] = {"path": item.path, "size": item.size,
                                "sha256": lfs.sha256 if lfs else None,
                                "git_sha1": None if lfs else item.blob_id}
    missing = [p for p in patterns if "*" not in p and p not in files]
    if missing:
        raise ValueError(f"{repo}@{revision}: missing pinned files {missing}")
    if not files:
        raise ValueError(f"{repo}@{revision}: no files match {patterns}")
    return [files[path] for path in sorted(files)]


def verify(path: Path, remote: dict) -> str:
    """Return the local sha256 after checking size and the repository checksum."""
    if path.stat().st_size != remote["size"]:
        raise ValueError(f"{path}: size {path.stat().st_size}, expected {remote['size']}")
    digest = sha256_file(path)
    if remote["sha256"] and digest != remote["sha256"]:
        raise ValueError(f"{path}: sha256 mismatch")
    if remote["git_sha1"] and git_blob_sha1(path) != remote["git_sha1"]:
        raise ValueError(f"{path}: git blob sha1 mismatch")
    return digest


def fetch(repo: str, revision: str, remote: dict, destination: Path, repo_type: str = "dataset") -> str:
    """Download one pinned file atomically; an existing verified copy is reused."""
    from huggingface_hub import hf_hub_download

    if destination.is_file():
        try:
            return verify(destination, remote)
        except ValueError as error:
            log(f"replacing invalid local copy: {error}")
            destination.unlink()
    destination.parent.mkdir(parents=True, exist_ok=True)
    for attempt in range(1, 4):
        staging = destination.parent / f".tmp-{destination.name}-{uuid.uuid4().hex[:8]}"
        try:
            # force_download: a corrupt cached blob must never satisfy a pinned checksum.
            path = Path(hf_hub_download(repo_id=repo, filename=remote["path"], revision=revision,
                                        repo_type=repo_type, local_dir=staging, force_download=True))
            digest = verify(path, remote)
            os.replace(path, destination)
            return digest
        except Exception as error:  # network and checksum failures are retried
            if attempt == 3:
                raise
            log(f"retrying {repo}/{remote['path']} after {type(error).__name__}: {error}")
            time.sleep(5 * attempt)
        finally:
            shutil.rmtree(staging, ignore_errors=True)
    raise AssertionError("unreachable")


def clear_partial(directory: Path) -> None:
    """Remove leftovers of an interrupted run; complete files are only ever renamed into place."""
    if directory.is_dir():
        for path in list(directory.rglob(".tmp-*")):
            shutil.rmtree(path, ignore_errors=True)
        for path in list(directory.rglob("*.tmp")):
            path.unlink(missing_ok=True)


def write_record(directory: Path, metadata: dict, entries: dict[str, dict], limit: int | None = None) -> None:
    record = dict(metadata, created=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), files=entries)
    if limit:
        record["limit_files"] = limit  # a testing preparation; --check rejects it
    temporary = directory / (RECORD + ".tmp")
    temporary.write_text(json.dumps(record, indent=1, sort_keys=True) + "\n")
    os.replace(temporary, directory / RECORD)


def check_record(directory: Path, deep: bool) -> list[str]:
    path = directory / RECORD
    if not path.is_file():
        return [f"{directory}: missing completion record (not prepared)"]
    record = json.loads(path.read_text())
    errors = [f"{directory}: prepared with --limit-files {record['limit_files']} (testing only)"] \
        if record.get("limit_files") else []
    known = set(record["files"]) | {RECORD, ".progress.jsonl"}
    extra = sorted(str(p.relative_to(directory)) for p in directory.rglob("*")
                   if p.is_file() and str(p.relative_to(directory)) not in known)
    if extra:
        errors.append(f"{directory}: {len(extra)} unrecorded files, e.g. {extra[:3]}")
    for name, entry in record["files"].items():
        target = directory / name
        if not target.is_file():
            errors.append(f"{target}: missing")
        elif target.stat().st_size != entry["size"]:
            errors.append(f"{target}: size {target.stat().st_size}, expected {entry['size']}")
        elif deep and sha256_file(target) != entry["sha256"]:
            errors.append(f"{target}: sha256 mismatch")
    return errors


def write_jsonl(path: Path, rows: list[dict]) -> dict:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    os.replace(temporary, path)
    return {"size": path.stat().st_size, "sha256": sha256_file(path), "rows": len(rows)}


def parallel(function, items, workers: int, label: str) -> list:
    results, failures = [], []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(function, item): item for item in items}
        for count, future in enumerate(as_completed(futures), 1):
            try:
                results.append(future.result())
            except Exception as error:
                failures.append(f"{futures[future]}: {type(error).__name__}: {error}")
                log(f"{label}: FAILED {failures[-1]}")
            if count % 25 == 0 or count == len(futures):
                log(f"{label}: {count}/{len(futures)}")
    if failures:
        raise RuntimeError(f"{label}: {len(failures)} failures; first: {failures[0]}")
    return results


# --------------------------------------------------------------------------- context


class Context:
    def __init__(self, args: argparse.Namespace):
        self.sources = json.loads(SOURCES.read_text())
        self.artifacts_root = args.artifacts_root.resolve()
        self.data_root = args.data_root.resolve()
        self.workers = args.workers
        self.limit = args.limit_files
        self.download_root = args.download_root.resolve()

    def model_dir(self) -> Path:
        return self.artifacts_root / self.sources["base_model"]["path"]

    def dataset_dirs(self, spec: dict) -> list[Path]:
        base = self.data_root / spec["name"]
        if spec.get("kind") == "dclm":
            return [base / "sample", base / f"{spec['slice_column']}_ge{spec['slice_min']}"]
        return [base]

    def expected_dirs(self) -> list[Path]:
        dirs = [self.model_dir()]
        for spec in self.sources["datasets"]:
            dirs += self.dataset_dirs(spec)
        return dirs + [self.data_root / "dev_evals"]

    def limited(self, files: list) -> list:
        return files[: self.limit] if self.limit else files

    def scratch(self, name: str) -> Path:
        path = self.download_root / name
        path.mkdir(parents=True, exist_ok=True)
        return path


def output_name(spec: dict, path: str) -> str:
    prefix = spec.get("strip_prefix", "")
    return path[len(prefix):] if prefix and path.startswith(prefix) else path


# --------------------------------------------------------------------------- base model and mirrors


def prepare_model(ctx: Context) -> None:
    spec = ctx.sources["base_model"]
    directory = ctx.model_dir()
    clear_partial(directory)
    files = remote_files(spec["repo"], spec["revision"], spec["include"], repo_type=spec["repo_type"])
    log(f"base model {spec['repo']}@{spec['revision'][:12]}: {len(files)} files")

    def one(remote):
        digest = fetch(spec["repo"], spec["revision"], remote, directory / remote["path"], spec["repo_type"])
        return remote["path"], {"size": remote["size"], "sha256": digest}

    entries = dict(parallel(one, files, ctx.workers, "model"))
    write_record(directory, {"repo": spec["repo"], "revision": spec["revision"]}, entries)


def prepare_mirror(ctx: Context, spec: dict) -> None:
    directory = ctx.data_root / spec["name"]
    clear_partial(directory)
    files = ctx.limited(remote_files(spec["repo"], spec["revision"], spec["include"]))
    total = sum(f["size"] for f in files)
    log(f"{spec['name']}: {len(files)} files, {total / 1e9:.1f} GB")

    def one(remote):
        name = output_name(spec, remote["path"])
        digest = fetch(spec["repo"], spec["revision"], remote, directory / name)
        return name, {"size": remote["size"], "sha256": digest, "source": remote["path"]}

    entries = dict(parallel(one, files, ctx.workers, spec["name"]))
    write_record(directory, {"repo": spec["repo"], "revision": spec["revision"], "name": spec["name"]}, entries,
                 ctx.limit)


# --------------------------------------------------------------------------- DCLM-EDU


def filter_parquet(source: Path, destination: Path, column: str, minimum: int) -> tuple[int, int]:
    """Stream rows with column >= minimum into a zstd Parquet file with the same schema."""
    import pyarrow as pa
    import pyarrow.compute as pc
    import pyarrow.parquet as pq

    parquet = pq.ParquetFile(source)
    temporary = destination.with_name(destination.name + ".tmp")
    rows_in = rows_out = 0
    pending, pending_rows = [], 0
    with pq.ParquetWriter(temporary, parquet.schema_arrow, compression="zstd") as writer:
        for batch in parquet.iter_batches(batch_size=65536):
            rows_in += batch.num_rows
            kept = batch.filter(pc.greater_equal(batch.column(column), minimum))
            if kept.num_rows:
                pending.append(kept)
                pending_rows += kept.num_rows
                rows_out += kept.num_rows
            if pending_rows >= 100_000:
                writer.write_table(pa.Table.from_batches(pending))
                pending, pending_rows = [], 0
        if pending:
            writer.write_table(pa.Table.from_batches(pending))
    os.replace(temporary, destination)
    return rows_in, rows_out


def prepare_dclm(ctx: Context, spec: dict) -> None:
    sample_dir, slice_dir = ctx.dataset_dirs(spec)
    files = remote_files(spec["repo"], spec["revision"], spec["include"])
    names = [f["path"] for f in files]
    sample = set(random.Random(spec["sample_seed"]).sample(names, spec["sample_shards"]))
    if ctx.limit:  # testing aid: a few sampled and a few unsampled shards
        files = [f for f in files if f["path"] in sample][: ctx.limit] + \
                [f for f in files if f["path"] not in sample][: ctx.limit]
        sample &= {f["path"] for f in files}
    log(f"{spec['name']}: {len(files)} shards ({len(sample)} kept unfiltered), "
        f"{sum(f['size'] for f in files) / 1e12:.2f} TB to stream")
    sample_dir.mkdir(parents=True, exist_ok=True)
    slice_dir.mkdir(parents=True, exist_ok=True)
    staging = ctx.scratch("dclm-edu")
    for directory in (sample_dir, slice_dir, staging):
        clear_partial(directory)
    progress_path = slice_dir / ".progress.jsonl"
    done = {}
    if progress_path.is_file():
        for line in progress_path.read_text().splitlines():
            entry = json.loads(line)
            done[entry["source"]] = entry
    progress_lock = threading.Lock()

    def one(remote):
        name = Path(remote["path"]).name
        output = slice_dir / name
        keep = remote["path"] in sample
        previous = done.get(remote["path"])
        if previous and output.is_file() and output.stat().st_size == previous["size"] and \
                (not keep or (sample_dir / name).is_file()):
            return previous
        source = (sample_dir if keep else staging) / name
        source_sha = fetch(spec["repo"], spec["revision"], remote, source)
        rows_in, rows_out = filter_parquet(source, output, spec["slice_column"], spec["slice_min"])
        if not keep:
            source.unlink()
        entry = {"source": remote["path"], "name": name, "source_sha256": source_sha, "rows_in": rows_in,
                 "rows": rows_out, "size": output.stat().st_size, "sha256": sha256_file(output), "sampled": keep}
        with progress_lock, progress_path.open("a") as stream:
            stream.write(json.dumps(entry) + "\n")
        return entry

    results = parallel(one, files, ctx.workers, spec["name"])
    by_name = {r["name"]: r for r in results}
    write_record(slice_dir, {"repo": spec["repo"], "revision": spec["revision"], "name": spec["name"],
                             "derivation": f"rows with {spec['slice_column']} >= {spec['slice_min']} from every shard",
                             "rows_in": sum(r["rows_in"] for r in results), "rows": sum(r["rows"] for r in results)},
                 {n: {"size": r["size"], "sha256": r["sha256"], "rows": r["rows"], "source": r["source"]}
                  for n, r in by_name.items()}, ctx.limit)
    write_record(sample_dir, {"repo": spec["repo"], "revision": spec["revision"], "name": spec["name"],
                              "derivation": f"{spec['sample_shards']} shards sampled with seed {spec['sample_seed']}, unfiltered"},
                 {r["name"]: {"size": (sample_dir / r["name"]).stat().st_size, "sha256": r["source_sha256"],
                              "rows": r["rows_in"], "source": r["source"]}
                  for r in results if r["sampled"]}, ctx.limit)


# --------------------------------------------------------------------------- OpenCodeReasoning-2


def read_arrow_column(path: Path, column: str) -> list:
    import pyarrow as pa

    with pa.memory_map(str(path)) as source:
        try:
            table = pa.ipc.open_stream(source).read_all()
        except pa.ArrowInvalid:
            table = pa.ipc.open_file(source).read_all()
    return table.column(column).to_pylist()


def codeforces_question(row: dict) -> str | None:
    """Verbatim port of the OpenCodeReasoning-2 dataset card's get_question for Codeforces."""
    if not row["description"]:
        return None
    question = row["description"]
    if row["input_format"]:
        question += "\n\nInput\n\n" + row["input_format"]
    if row["output_format"]:
        question += "\n\nOutput\n\n" + row["output_format"]
    if row["examples"]:
        question += "\n\nExamples"
        for example in row["examples"]:
            if "input" in example:
                question += "\n\nInput\n\n" + example["input"]
            if "output" in example:
                question += "\n\nOutput\n\n" + example["output"]
    if row["note"]:
        question += "\n\nNote\n\n" + row["note"]
    return question


def load_question_sources(ctx: Context, sources: dict) -> dict[tuple[str, str], list]:
    """(dataset, split) -> questions in the row order the dataset card's load_dataset calls produce."""
    import pyarrow.parquet as pq

    questions = {}
    for name, spec in sources.items():
        directory = ctx.scratch("ocr2-sources") / name.replace("/", "__")
        files = remote_files(spec["repo"], spec["revision"], spec["include"])
        parallel(lambda r: fetch(spec["repo"], spec["revision"], r, directory / r["path"]), files,
                 ctx.workers, f"ocr2 source {name}")
        by_split = {}
        for remote in files:
            path = remote["path"]
            if name == "taco":
                split = path.split("/")[0]  # TACO.py reads train/*.arrow then test/*.arrow
            elif name == "apps":
                split = Path(path).stem  # apps.py yields every line of train.jsonl / test.jsonl
            else:
                split = Path(path).name.split("-")[0]  # data/<split>-NNNNN-of-NNNNN*.parquet
            by_split.setdefault(split, []).append(directory / path)
        for split, paths in by_split.items():
            values = []
            for path in sorted(paths):
                if name == "taco":
                    values += read_arrow_column(path, "question")
                elif name == "apps":
                    with path.open("rb") as stream:
                        values += [json.loads(line.strip().decode("utf-8"))["question"] for line in stream if line.strip()]
                elif name == "code_contests":
                    values += [d or None for d in pq.read_table(path, columns=["description"]).column(0).to_pylist()]
                else:
                    columns = ["description", "input_format", "output_format", "examples", "note"]
                    values += [codeforces_question(r) for r in pq.read_table(path, columns=columns).to_pylist()]
            questions[(name, split)] = values
            log(f"ocr2 source {name}/{split}: {len(values)} questions")
    return questions


_WORD = re.compile(r"[a-z][a-z_]{4,}")


def overlap(question: str, generation: str) -> float | None:
    """Share of the question's distinctive words that the reasoning trace repeats."""
    words = set(_WORD.findall(question.lower()[:4000]))
    if len(words) < 8:
        return None
    return len(words & set(_WORD.findall(generation.lower()[:8000]))) / len(words)


def prepare_ocr2(ctx: Context, spec: dict) -> None:
    import pyarrow as pa
    import pyarrow.parquet as pq

    directory = ctx.data_root / spec["name"]
    clear_partial(directory)
    questions = load_question_sources(ctx, spec["question_sources"])
    files = remote_files(spec["repo"], spec["revision"], spec["include"])
    if ctx.limit:
        files = [f for f in files if "/python/" in f["path"]][: ctx.limit] + \
                [f for f in files if "/cpp/" in f["path"]][: ctx.limit]
    staging = ctx.scratch("ocr2")
    rng = random.Random(0)
    samples, samples_lock = [], threading.Lock()

    def one(remote):
        name = output_name(spec, remote["path"])
        output = directory / name
        source = staging / Path(remote["path"]).name
        fetch(spec["repo"], spec["revision"], remote, source)
        table = pq.read_table(source)
        rows = table.select(["dataset", "split", "index", "question"]).to_pylist()
        filled = []
        for row in rows:
            if row["question"] != "-":
                raise ValueError(f"{name}: expected the '-' placeholder, found a question")
            values = questions[(row["dataset"], row["split"])]
            index = int(row["index"])
            if not 0 <= index < len(values) or not values[index]:
                raise ValueError(f"{name}: no question for {row['dataset']}/{row['split']}[{index}]")
            filled.append(values[index])
        position = table.schema.get_field_index("question")
        field = table.schema.field(position)
        table = table.set_column(position, field, pa.array(filled, type=field.type))
        output.parent.mkdir(parents=True, exist_ok=True)
        temporary = output.with_name(output.name + ".tmp")
        pq.write_table(table, temporary, compression="zstd")
        os.replace(temporary, output)
        generations = table.column("r1_generation")
        picks = rng.sample(range(len(rows)), min(40, len(rows)))
        with samples_lock:
            samples.extend((rows[i]["dataset"], filled[i], generations[i].as_py() or "") for i in picks)
        source.unlink()
        return name, {"size": output.stat().st_size, "sha256": sha256_file(output), "rows": len(rows),
                      "source": remote["path"]}

    entries = dict(parallel(one, files, ctx.workers, spec["name"]))
    # Misaligned indices would silently pair every trace with an unrelated problem. A trace
    # should share more of its own question's vocabulary than of a neighbouring row's: on
    # real shards that holds for ~97% of rows, and for ~49% (chance) with an off-by-one fill.
    report = {}
    for dataset in sorted({s[0] for s in samples}):
        pairs = [(q, g) for d, q, g in samples if d == dataset]
        others = [g for _, g in pairs][1:] + [g for _, g in pairs][:1]
        scored = [(overlap(q, g), overlap(q, other)) for (q, g), other in zip(pairs, others)]
        scored = [(a, s) for a, s in scored if a is not None and s is not None]
        if len(scored) < 10:
            continue
        report[dataset] = {"pairs": len(scored), "win_rate": round(sum(a > s for a, s in scored) / len(scored), 3),
                           "aligned_median": round(statistics.median(a for a, _ in scored), 3),
                           "shuffled_median": round(statistics.median(s for _, s in scored), 3)}
        log(f"ocr2 alignment {dataset}: {report[dataset]}")
        if report[dataset]["win_rate"] < 0.8:
            raise ValueError(f"OpenCodeReasoning-2 questions for {dataset} do not match their reasoning traces: "
                             f"{report[dataset]}")
    write_record(directory, {"repo": spec["repo"], "revision": spec["revision"], "name": spec["name"],
                             "derivation": "question column filled from the dataset card's source datasets",
                             "question_sources": spec["question_sources"], "alignment": report}, entries, ctx.limit)


# --------------------------------------------------------------------------- evaluation data


def hf_file(ctx: Context, repo: str, revision: str, path: str) -> Path:
    remote = remote_files(repo, revision, [path])[0]
    destination = ctx.scratch("eval-sources") / repo.replace("/", "__") / revision[:12] / path
    fetch(repo, revision, remote, destination)
    return destination


def read_rows(path: Path) -> list[dict]:
    if path.suffix == ".parquet":
        import pyarrow.parquet as pq

        return pq.read_table(path).to_pylist()
    with path.open() as stream:
        return [json.loads(line) for line in stream if line.strip()]


def last_boxed(text: str) -> str | None:
    start = max(text.rfind("\\boxed{"), text.rfind("\\fbox{"))
    if start < 0:
        return None
    index = text.index("{", start)
    depth = 0
    for position in range(index, len(text)):
        depth += {"{": 1, "}": -1}.get(text[position], 0)
        if depth == 0:
            return text[index + 1:position]
    return None


def squash(text: str) -> str:
    return " ".join(text.split())


def math_suite(ctx: Context, specs: dict, seed: int) -> list[dict]:
    spec = specs["math"]
    excluded = {squash(r["problem"]) for r in read_rows(
        hf_file(ctx, spec["exclude_repo"], spec["exclude_revision"], spec["exclude_file"]))}
    rows = []
    for remote in remote_files(spec["repo"], spec["revision"], spec["include"]):
        rows += read_rows(hf_file(ctx, spec["repo"], spec["revision"], remote["path"]))
    candidates = [r for r in rows if squash(r["problem"]) not in excluded and last_boxed(r["solution"])]
    log(f"math dev: {len(rows)} MATH test problems, {len(candidates)} outside MATH-500")
    picked = random.Random(f"{seed}:math").sample(candidates, spec["sample"])
    items = [{"id": f"math-{i}", "source": "math", "subject": r["type"], "level": r["level"],
              "problem": r["problem"], "answer": last_boxed(r["solution"])} for i, r in enumerate(picked)]
    aime = specs["aime"]
    for row in read_rows(hf_file(ctx, aime["repo"], aime["revision"], aime["file"])):
        year = row["url"].rsplit("/", 2)[-2][:4]
        if year in aime["years"]:
            items.append({"id": f"aime-{year}-{row['id']}", "source": "aime", "problem": row["problem"],
                          "answer": str(row["answer"]).strip(), "url": row["url"]})
    return items


def science_suite(ctx: Context, specs: dict, seed: int) -> list[dict]:
    spec = specs["supergpqa"]
    rows = [r for r in read_rows(hf_file(ctx, spec["repo"], spec["revision"], spec["file"]))
            if r["discipline"] == spec["discipline"]]
    picked = random.Random(f"{seed}:science").sample(rows, spec["sample"])
    return [{"id": r["uuid"], "source": "supergpqa", "question": r["question"], "options": list(r["options"]),
             "answer": r["answer_letter"], "field": r["field"], "subfield": r["subfield"],
             "difficulty": r["difficulty"]} for r in picked]


def decode_tests(value: str) -> list:
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        # LiveCodeBench stores private tests as base64(zlib(pickle(json))). This runs on
        # the operator host against the pinned upstream file; agents receive plain JSON.
        return json.loads(pickle.loads(zlib.decompress(base64.b64decode(value.encode("utf-8")))))


def code_suite(ctx: Context, specs: dict) -> list[dict]:
    spec = specs["livecodebench_v1"]
    items = []
    with hf_file(ctx, spec["repo"], spec["revision"], spec["file"]).open() as stream:
        for line in stream:
            row = json.loads(line)
            tests = decode_tests(row["public_test_cases"]) + decode_tests(row["private_test_cases"])
            kinds = {t["testtype"] for t in tests}
            if len(kinds) != 1:
                raise ValueError(f"{row['question_id']}: mixed test types {kinds}")
            kept = [{"input": t["input"], "output": t["output"]} for t in tests
                    if len(t["input"]) + len(t["output"]) <= spec["max_test_bytes"]][: spec["max_tests"]]
            if not kept:
                continue
            metadata = json.loads(row["metadata"] or "{}")
            items.append({"id": row["question_id"], "source": "livecodebench_v1", "title": row["question_title"],
                          "content": row["question_content"], "platform": row["platform"],
                          "contest_date": row["contest_date"], "difficulty": row["difficulty"],
                          "starter_code": row["starter_code"], "testtype": kinds.pop(),
                          "fn_name": metadata.get("func_name"), "tests": kept})
    return items


def prepare_dev_evals(ctx: Context) -> None:
    specs = ctx.sources["dev_evals"]
    directory = ctx.data_root / "dev_evals"
    clear_partial(directory)
    entries = {}
    entries["reasoning/math.jsonl"] = write_jsonl(directory / "reasoning/math.jsonl",
                                                  math_suite(ctx, specs["math"], specs["seed"]))
    entries["reasoning/science.jsonl"] = write_jsonl(directory / "reasoning/science.jsonl",
                                                     science_suite(ctx, specs["science"], specs["seed"]))
    entries["reasoning/code.jsonl"] = write_jsonl(directory / "reasoning/code.jsonl", code_suite(ctx, specs["code"]))
    for name, entry in entries.items():
        log(f"dev evals {name}: {entry['rows']} rows")
    write_record(directory, {"name": "dev_evals", "sources": specs}, entries)


# --------------------------------------------------------------------------- data README


def prepare_readme(ctx: Context) -> None:
    lines = ["# Provided training data", "",
             "Read-only copies of pinned Hugging Face datasets. Files keep their upstream formats",
             "(Parquet or JSON Lines) and columns. Each directory's `.midtrain-for-rl.json` lists its files",
             "with sizes, SHA-256 digests and, where computed, row counts.", ""]
    summary = []
    for spec in ctx.sources["datasets"]:
        for directory in ctx.dataset_dirs(spec):
            record = json.loads((directory / RECORD).read_text())
            size = sum(e["size"] for e in record["files"].values())
            rows = sum(e.get("rows", 0) for e in record["files"].values()) or None
            relative = directory.relative_to(ctx.data_root)
            summary.append({"path": str(relative), "repo": spec["repo"], "revision": spec["revision"],
                            "license": spec["license"], "files": len(record["files"]), "bytes": size, "rows": rows})
            lines += [f"## `{relative}/`", "",
                      f"- Source: [{spec['repo']}](https://huggingface.co/datasets/{spec['repo']}) at `{spec['revision']}`",
                      f"- License: {spec['license'] or 'see the source repository card'}",
                      f"- {len(record['files'])} files, {size / 1e9:.1f} GB" + (f", {rows:,} rows" if rows else ""),
                      f"- {record.get('derivation') or spec['description']}", ""]
            if record.get("derivation"):
                lines += [spec["description"], ""]
    lines += ["## `dev_evals/`", "", "Development evaluation sets; see `/app/evals/README.md`.", ""]
    (ctx.data_root / "README.md").write_text("\n".join(lines))
    (ctx.data_root / "SOURCES.json").write_text(json.dumps(summary, indent=1) + "\n")
    log(f"wrote {ctx.data_root / 'README.md'}")


# --------------------------------------------------------------------------- main


def check(ctx: Context, deep: bool) -> int:
    errors = []
    for directory in ctx.expected_dirs():
        errors += check_record(directory, deep)
    errors += check_dev_evals(ctx.data_root / 'dev_evals', ctx.sources['dev_evals'])
    for name in ("README.md", "SOURCES.json"):
        if not (ctx.data_root / name).is_file():
            errors.append(f"{ctx.data_root / name}: missing")
    for error in errors:
        print("FAIL", error, file=sys.stderr)
    if not errors:
        print(f"midtrain-for-rl inputs validated ({'deep' if deep else 'sizes'}): "
              f"{len(ctx.expected_dirs())} directories")
    return 1 if errors else 0


def check_dev_evals(directory: Path, expected: dict) -> list[str]:
    """Require the development sets to match the pinned dev_evals sources in data_sources.json."""
    try:
        record = json.loads((directory / RECORD).read_text())
        if record.get('sources') != expected:
            return [f'{directory}: development sources differ from data_sources.json; prepare --only dev-evals,readme again']
        path = directory / 'reasoning/math.jsonl'
        if sha256_file(path) != record['files']['reasoning/math.jsonl']['sha256']:
            return [f'{directory}: development math hash mismatch; prepare --only dev-evals,readme again']
        rows = read_rows(path)
        if any(row.get('source') not in expected['math'] for row in rows):
            return [f'{directory}: unsupported development math source']
    except (OSError, KeyError, ValueError) as error:
        return [f'{directory}: invalid development data ({error})']
    return []


def selected(ctx: Context, only: str | None) -> tuple[set, set]:
    names = {spec["name"] for spec in ctx.sources["datasets"]}
    if not only:
        return set(STEPS), names
    steps, datasets = set(), set()
    for token in only.split(","):
        if token in STEPS:
            steps.add(token)
            if token == "corpora":
                datasets |= names
        elif token in names:
            steps.add("corpora")
            datasets.add(token)
        else:
            raise SystemExit(f"--only: unknown step or dataset {token!r}; steps: {STEPS}; datasets: {sorted(names)}")
    return steps, datasets


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    environ = os.environ.get
    parser.add_argument("--artifacts-root", type=Path, default=Path(environ("MIDTRAIN_ARTIFACTS_ROOT", TASK_ROOT / "artifacts")))
    parser.add_argument("--data-root", type=Path, default=Path(environ("MIDTRAIN_DATA_ROOT", TASK_ROOT / "data")))
    parser.add_argument("--download-root", type=Path, default=Path(environ("MIDTRAIN_DOWNLOAD_ROOT", TASK_ROOT / ".downloads")))
    parser.add_argument("--only", help=f"comma-separated steps {STEPS} or dataset names")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--limit-files", type=int, help="testing aid: at most N files per source (never for real preparation)")
    parser.add_argument("--check", action="store_true", help="validate completion records offline")
    parser.add_argument("--deep", action="store_true", help="with --check: recompute every SHA-256")
    parser.add_argument("--dry-run", action="store_true", help="list pinned remote inventories and sizes only")
    args = parser.parse_args()
    ctx = Context(args)
    if args.check:
        return check(ctx, args.deep)
    steps, datasets = selected(ctx, args.only)
    if args.dry_run:
        total = 0
        for spec in ctx.sources["datasets"]:
            if spec["name"] in datasets:
                files = remote_files(spec["repo"], spec["revision"], spec["include"])
                size = sum(f["size"] for f in files)
                total += size
                print(f"{spec['name']:45s} {len(files):6d} files {size / 1e9:10.1f} GB")
        print(f"{'total':45s} {'':6s}       {total / 1e9:10.1f} GB")
        return 0
    started = time.time()
    if "model" in steps:
        prepare_model(ctx)
    for spec in ctx.sources["datasets"]:
        if "corpora" in steps and spec["name"] in datasets:
            {"dclm": prepare_dclm, "ocr2": prepare_ocr2}.get(spec.get("kind"), prepare_mirror)(ctx, spec)
    if "dev-evals" in steps:
        prepare_dev_evals(ctx)
    if "readme" in steps:
        prepare_readme(ctx)
    log(f"done in {(time.time() - started) / 60:.1f} min")
    return 0


if __name__ == "__main__":
    sys.exit(main())
