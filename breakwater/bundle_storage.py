"""Non-executable, digest-bound bundles. Never mount these into a submission."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import stat


class SafetyError(RuntimeError):
    pass


def encode(value) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def digest_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def private_directory(path: Path) -> Path:
    path = Path(path)
    if any(parent.is_symlink() for parent in (path, *path.parents)):
        raise SafetyError("Trusted storage cannot contain a symlink")
    path.mkdir(parents=True, exist_ok=True, mode=0o700)
    path.chmod(0o700)
    return path


def atomic_json(path: Path, value) -> None:
    temporary = path.with_name(path.name + ".pending")
    fd = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(encode(value) + b"\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def safe_file(root: Path, relative: str) -> Path:
    name = PurePosixPath(relative)
    if not relative or name.is_absolute() or ".." in name.parts or "\\" in relative:
        raise SafetyError("Invalid bundle-relative filename")
    path = root
    for part in name.parts:
        path = path / part
        if path.is_symlink():
            raise SafetyError("Bundle contains a symlink")
    if not stat.S_ISREG(path.lstat().st_mode):
        raise SafetyError("Bundle file is not regular")
    return path


def seal_bundle(root: Path, *, identity: dict, protocol: str, complete: bool,
                expected_ids: list[str] | None = None, details: dict | None = None) -> str:
    root = private_directory(root)
    files = {}
    for path in sorted(root.rglob("*")):
        if path == root / "manifest.json":
            raise SafetyError("Bundle is already sealed")
        if path.is_symlink():
            raise SafetyError("Cannot seal symlink")
        if path.is_dir():
            continue
        relative = path.relative_to(root).as_posix()
        checked = safe_file(root, relative)
        files[relative] = {"bytes": checked.stat().st_size, "sha256": digest_file(checked)}
        checked.chmod(0o400)
    manifest = {"schema_version": 1, "identity": identity, "protocol": protocol,
                "complete": complete, "files": files, "expected_ids": expected_ids,
                "details": details or {}}
    atomic_json(root / "manifest.json", manifest)
    (root / "manifest.json").chmod(0o400)
    return digest_file(root / "manifest.json")


def verify_bundle(root: Path, expected_digest: str, *, identity: dict | None = None,
                  protocol: str | None = None) -> dict:
    root = Path(root)
    manifest_path = safe_file(root, "manifest.json")
    if digest_file(manifest_path) != expected_digest:
        raise SafetyError("Manifest differs from trusted handoff digest")
    manifest = strict_json(manifest_path.read_bytes())
    if manifest.get("schema_version") != 1 or manifest.get("complete") is not True:
        raise SafetyError("Incomplete or unsupported bundle")
    if identity is not None and manifest.get("identity") != identity:
        raise SafetyError("Bundle belongs to another trial/attempt/step")
    if protocol is not None and manifest.get("protocol") != protocol:
        raise SafetyError("Unexpected output protocol")
    if not isinstance(manifest.get("files"), dict):
        raise SafetyError("Missing file inventory")
    for relative, entry in manifest["files"].items():
        path = safe_file(root, relative)
        if path.stat().st_size != entry["bytes"] or digest_file(path) != entry["sha256"]:
            raise SafetyError(f"Output integrity failure: {relative}")
    entries = list(root.rglob("*"))
    if any(p.is_symlink() for p in entries):
        raise SafetyError("Undeclared symlink in bundle")
    actual = {p.relative_to(root).as_posix() for p in entries if not p.is_dir()}
    if actual != set(manifest["files"]) | {"manifest.json"}:
        raise SafetyError("Undeclared files in bundle")
    return manifest


def generation_rows(root: Path, manifest: dict) -> dict[str, list[int]]:
    if "raw/predictions.jsonl" not in manifest["files"]:
        raise SafetyError("Missing captured generation predictions")
    expected = manifest.get("expected_ids")
    if not isinstance(expected, list) or any(type(x) is not str for x in expected) or len(set(expected)) != len(expected):
        raise SafetyError("Invalid expected example ID inventory")
    rows = {}
    with safe_file(root, "raw/predictions.jsonl").open("rb") as stream:
        while line := stream.readline(1024 * 1024 + 1):
            if len(line) > 1024 * 1024:
                raise SafetyError("Oversized prediction record")
            row = strict_json(line)
            example_id, tokens = row.get("example_id"), row.get("generated_token_ids")
            if type(example_id) is not str or example_id in rows or example_id not in expected:
                raise SafetyError("Duplicate or unexpected example ID")
            if not isinstance(tokens, list) or not tokens or any(type(x) is not int for x in tokens):
                raise SafetyError("Invalid generated token IDs")
            rows[example_id] = tokens
    if set(rows) != set(expected):
        raise SafetyError("Missing generation predictions")
    return rows


def strict_json(raw):
    def unique(pairs):
        value = {}
        for key, item in pairs:
            if key in value:
                raise SafetyError("Duplicate JSON key")
            value[key] = item
        return value
    def invalid(value):
        raise SafetyError(f"Non-finite JSON constant: {value}")
    return json.loads(raw, object_pairs_hook=unique, parse_constant=invalid)
