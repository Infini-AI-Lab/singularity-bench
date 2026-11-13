"""Immutable checkpoint revisions and local content verification.

The lock is metadata, not model weights. Staging verifies every selected file;
launch checks the resulting receipt and file metadata on the read-only mount.
"""
from __future__ import annotations

import hashlib
from functools import lru_cache
import json
from pathlib import Path
import re

LOCK = Path(__file__).with_name("checkpoints.lock.json")
RECEIPT = ".cluster-checkpoint.json"


@lru_cache(maxsize=8)
def entries(path: Path = LOCK) -> dict:
    return json.loads(path.read_text())["checkpoints"]


def entry(checkpoint: str, path: Path = LOCK) -> dict:
    value = entries(path).get(checkpoint)
    if not value or not re.fullmatch(r"[0-9a-f]{40}", value.get("revision", "")):
        raise ValueError(f"checkpoint has no immutable revision: {checkpoint}")
    if not value.get("files") or "config.json" not in value["files"]:
        raise ValueError(f"checkpoint lock is incomplete: {checkpoint}")
    return value


def digest(value: dict) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def serving_weights_gib(checkpoint: str) -> float:
    """Bytes in the root Transformers checkpoint, excluding alternate layouts.

    GPT-OSS also publishes original/*.safetensors containing the same weights.
    Those are staging bytes, not a second copy resident in a vLLM engine. The
    task always serves the root checkpoint in its pinned native weight format.
    Runtime repacking/activations must be budgeted separately and measured.
    """
    files = entry(checkpoint)['files']
    weights = [v for k, v in files.items() if '/' not in k and k.endswith('.safetensors')]
    if not weights:
        weights = [v for k, v in files.items() if '/' not in k and k.endswith('.bin')]
    if not weights:
        raise ValueError(f'no root serving checkpoint: {checkpoint}')
    return sum(v['size'] for v in weights)/2**30


def check_file_set(root: Path, locked: dict) -> bool:
    # Stale weight formats, adapters or configs must not change what the engine loads.
    suffixes = {".json", ".safetensors", ".bin", ".pt", ".pth", ".txt", ".model", ".py", ".tiktoken", ".jinja"}
    return all(str(p.relative_to(root)) in locked["files"]
               for p in root.rglob("*") if p.is_file() and p.suffix in suffixes
               and ".cache" not in p.relative_to(root).parts and p.name != RECEIPT)


def verify(root: Path, checkpoint: str, locked: dict, *, full: bool = False) -> bool:
    """Full hashes at staging; exact stat/lock receipt at launch. Never accept partial shards."""
    try:
        if not check_file_set(root, locked):
            return False
        receipt = json.loads((root / RECEIPT).read_text())
        if receipt["checkpoint"] != checkpoint or receipt["lock_sha256"] != digest(locked):
            return False
        for name, expected in locked["files"].items():
            path = root / name
            info = path.stat()
            if not path.is_file() or info.st_size != expected["size"]:
                return False
            if full:
                if file_hash(path, expected) != expected.get("sha256", expected.get("git_blob_sha1")):
                    return False
            elif receipt["files"].get(name) != [info.st_size, info.st_mtime_ns]:
                return False
        return True
    except (OSError, ValueError, KeyError):
        return False


def file_hash(path: Path, expected: dict) -> str:
    if "sha256" in expected:
        h = hashlib.sha256()
    else:
        h = hashlib.sha1(b"blob " + str(expected["size"]).encode() + b"\0")
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(4 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def certify(root: Path, checkpoint: str, locked: dict) -> None:
    if not check_file_set(root, locked):
        raise ValueError("unlocked serving files in checkpoint directory; stage into a clean destination")
    files = {}
    for name, expected in locked["files"].items():
        path = root / name
        info = path.stat()
        if info.st_size != expected["size"] or file_hash(path, expected) != expected.get("sha256", expected.get("git_blob_sha1")):
            raise ValueError(f"checkpoint file does not match pinned content: {checkpoint}/{name}")
        files[name] = [info.st_size, info.st_mtime_ns]
    value = {"checkpoint": checkpoint, "revision": locked["revision"],
             "lock_sha256": digest(locked), "files": files}
    temp = root / (RECEIPT + ".tmp")
    temp.write_text(json.dumps(value, sort_keys=True, indent=1) + "\n")
    temp.replace(root / RECEIPT)
