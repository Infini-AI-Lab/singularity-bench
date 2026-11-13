"""Freeze market anchors and native/o200k tokenizer ratios without model weights.

Only tokenizer.json at the checkpoint lock's immutable revision is downloaded.
The 60 public AA prompt texts are fetched and hash-checked, never redistributed
as a task corpus. Requires the pinned packages in market-requirements.txt.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import importlib.metadata
import io
import json
from pathlib import Path
import sys
import urllib.request
import xml.etree.ElementTree as ET
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "_kit"))
import checkpoints  # noqa: E402

URL = "https://artificialanalysis.ai/downloads/methodology/performance-prompts.xlsx"
CORPUS_SHA = "ec0fb729507077c9764722e50ba74dfacd35d8bbea4d5bb290366fa6a34d7398"
CROSSWALK = {
    "gpt-oss-120b": "gpt-oss-120b", "gpt-oss-20b": "gpt-oss-20b",
    "llama-3.3-70b-instruct": "llama-3-3-instruct-70b",
    "kimi-k3": "kimi-k3", "kimi-k2.7-code": "kimi-k2-7-code",
    "hermes-4-405b": "hermes-4-llama-3-1-405b",
    "llama-4-scout": "llama-4-scout", "llama-4-maverick": "llama-4-maverick",
    "qwen3.8-27b": "qwen3-8-27b", "qwen3.6-27b": "qwen3-6-27b",
    "qwen3.6-35b-a3b": "qwen3-6-35b-a3b",
    "qwen3.5-35b-a3b": "qwen3-5-35b-a3b-non-reasoning",
    "qwen3.5-397b-a17b": "qwen3-5-397b-a17b",
    "qwen3.5-122b-a10b": "qwen3-5-122b-a10b",
    "qwen3-next-80b-a3b-instruct": "qwen3-next-80b-a3b-instruct",
    "qwen3.5-9b": "qwen3-5-9b", "command-a": "command-a",
    "gemma-4-31b-it": "gemma-4-31b", "gemma-4-26b-a4b-it": "gemma-4-26b-a4b-non-reasoning",
    "nemotron-3-ultra-550b-a55b": "nvidia-nemotron-3-ultra-550b-a55b",
    "nemotron-3-super-120b-a12b": "nvidia-nemotron-3-super-120b-a12b",
    "nemotron-3-nano-30b-a3b": "nvidia-nemotron-3-nano-30b-a3b",
    "nemotron-3.5-lightning": "nemotron-3-5-lightning",
    "granite-4.2-8b": "granite-4-2-8b",
    "devstral-2512": "devstral-2", "mistral-small-2603": "mistral-small-4",
    "ministral-14b-2512": "ministral-3-14b",
    "ministral-8b-2512": "ministral-3-8b", "ministral-3b-2512": "ministral-3-3b",
}
VERSIONS = {"huggingface_hub": "1.27.0", "tokenizers": "0.23.1", "tiktoken": "0.9.0"}


def texts(data: bytes) -> list[str]:
    if hashlib.sha256(data).hexdigest() != CORPUS_SHA:
        raise ValueError("AA workbook changed; review a new pin explicitly")
    ns = {"m": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
    with zipfile.ZipFile(io.BytesIO(data)) as z:
        strings = ["".join(e.itertext()) for e in ET.fromstring(z.read("xl/sharedStrings.xml"))]
        sheet = ET.fromstring(z.read("xl/worksheets/sheet1.xml"))
        cells = sheet.findall("m:sheetData/m:row/m:c", ns)
        values = [strings[int(c.find("m:v", ns).text)] for c in cells if c.get("t") == "s"]
    if values[0] != "Prompt_Text" or len(values) != 61:
        raise ValueError("unexpected workbook schema")
    return values[1:]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default=str(ROOT / "_kit/market-anchors.json"))
    parser.add_argument("--workbook", type=Path)
    parser.add_argument("--cache", type=Path, default=Path.home() / ".cache/cluster-serverless-market")
    args = parser.parse_args()
    for package, required in VERSIONS.items():
        if importlib.metadata.version(package) != required:
            raise ValueError(f"requires {package}=={required}")
    from huggingface_hub import hf_hub_download
    from tokenizers import Tokenizer
    import tiktoken
    args.cache.mkdir(parents=True, exist_ok=True)
    workbook = args.workbook or args.cache / f"{CORPUS_SHA}.xlsx"
    if not workbook.exists():
        with urllib.request.urlopen(URL, timeout=60) as response:
            data = response.read()
        texts(data)  # Validate before caching.
        workbook.write_bytes(data)
    samples = texts(workbook.read_bytes())
    encoder = tiktoken.get_encoding("o200k_base")
    standardized = [len(encoder.encode(t, disallowed_special=())) for t in samples]
    snap_path = ROOT / "_dev/snapshots/openrouter-2026-09-08.json.gz"
    with gzip.open(snap_path) as f:
        snap = json.load(f)
    aa = {m["slug"]: m for m in snap["artificial_analysis"]["models"]}
    source = json.loads((ROOT / "_kit/models-openrouter-2026-09-08.json").read_text())
    models = {m["openrouter_id"].split("/", 1)[1]: m for m in source["models"]}
    anchors, unavailable = {}, {}
    for mid, slug in sorted(CROSSWALK.items()):
        model, observed = models[mid], aa[slug]
        locked = checkpoints.entry(model["checkpoint"])
        try:
            expected = locked["files"]["tokenizer.json"]
            filename = Path(hf_hub_download(model["checkpoint"], "tokenizer.json",
                revision=locked["revision"], cache_dir=str(args.cache / "huggingface")))
            if checkpoints.file_hash(filename, expected) != expected.get("sha256", expected.get("git_blob_sha1")):
                raise ValueError("tokenizer does not match checkpoint lock")
            tokenizer = Tokenizer.from_file(str(filename))
            native = [len(tokenizer.encode(t, add_special_tokens=False).ids) for t in samples]
            ratios = sorted(n / s for n, s in zip(native, standardized))
            anchors[mid] = {
                "aa_slug": slug, "aa_name": observed["name"], "checkpoint": model["checkpoint"],
                "checkpoint_revision": locked["revision"],
                "median_ttft_ms": 1000 * observed["median_time_to_first_token_seconds"],
                "median_output_standardized_tokens_per_second": observed["median_output_tokens_per_second"],
                "native_per_standardized_token": sum(native) / sum(standardized),
                "native_counts": native, "standardized_counts": standardized,
                "token_ratio_min": ratios[0], "token_ratio_max": ratios[-1],
                "tokenizer_sha256": hashlib.sha256(filename.read_bytes()).hexdigest(),
                "qualification": "market anchor; mode/quantization are not a hardware service guarantee",
            }
            print(f"{mid}: ratio {anchors[mid]['native_per_standardized_token']:.5f}", file=sys.stderr)
        except Exception as exc:
            # A missing/gated tokenizer is explicitly unmatched, never silently rho=1.
            unavailable[mid] = {"aa_slug": slug, "reason": type(exc).__name__}
            print(f"{mid}: unavailable ({type(exc).__name__})", file=sys.stderr)
    result = {"schema_version": 1, "source": snap["artificial_analysis"]["source"],
              "snapshot_sha256": hashlib.sha256(snap_path.read_bytes()).hexdigest(),
              "retrieved_utc": snap["artificial_analysis"]["retrieved_utc"],
              "measurement": snap["artificial_analysis"]["measurement"],
              "tokenizer_sample": {"url": URL, "sha256": CORPUS_SHA, "documents": len(samples),
                                   "extraction": "sheet1 shared-string cells after Prompt_Text header"},
              "dependencies": VERSIONS, "models": anchors, "unavailable": unavailable}
    Path(args.out).write_text(json.dumps(result, sort_keys=True, indent=1) + "\n")
    print(f"{len(anchors)} matched anchors; {len(unavailable)} unavailable", file=sys.stderr)


if __name__ == "__main__":
    main()
