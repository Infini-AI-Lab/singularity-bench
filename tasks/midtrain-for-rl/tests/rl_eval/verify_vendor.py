"""Verify the copied Slime snapshot before building or running it."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent

def verify(root=None):
    root = ROOT if root is None else Path(root)
    manifest = json.loads((root / "vendor_manifest.json").read_text())
    vendor = root / "vendor/slime"
    actual = {}
    for path in vendor.rglob("*"):
        relative = path.relative_to(vendor)
        # pip install --no-deps -e vendor/slime generates this distribution's
        # metadata in the source tree. It is not part of the pinned snapshot.
        if relative.parts[0] == "slime.egg-info" or "__pycache__" in relative.parts or path.suffix == ".pyc":
            continue
        if path.is_file():
            actual[str(path.relative_to(root))] = hashlib.sha256(path.read_bytes()).hexdigest()
    expected = manifest["sha256"]
    missing = sorted(expected.keys() - actual.keys())
    unexpected = sorted(actual.keys() - expected.keys())
    changed = sorted(name for name in expected.keys() & actual.keys() if expected[name] != actual[name])
    if missing or unexpected or changed:
        raise RuntimeError("Copied Slime source differs from vendor_manifest.json: "
                           + json.dumps(dict(missing=missing, unexpected=unexpected, changed=changed)))
    if hashlib.sha256((root / 'checkpoint_check.py').read_bytes()).hexdigest() != manifest['checkpoint_validator']['sha256']:
        raise RuntimeError('Bundled checkpoint validator differs from its source hash')
    return len(actual)

if __name__ == "__main__":
    print(f"Verified {verify()} Slime files")
