"""Install the audited native release, with pinned checksums for both fleet architectures."""
import hashlib
from pathlib import Path
import platform
import shutil
import urllib.request

VERSION = "2.1.284"
RELEASES = {
    "x86_64": ("linux-x64", "5cd90aabd83f8a15136c35aa37bb1d92b348993573316643dc3fe4e04afbf88f"),
    "aarch64": ("linux-arm64", "3dd0f96d7ada463152d20300186f6cfc6ab94b57e218f49e3ac86db42ac695a6"),
}

if __name__ == "__main__":
    arch, expected = RELEASES[platform.machine()]
    target = Path("/opt/author-cli/claude")
    target.parent.mkdir(parents=True, exist_ok=True)
    url = f"https://downloads.claude.ai/claude-code-releases/{VERSION}/{arch}/claude"
    with urllib.request.urlopen(url, timeout=120) as response, target.open("wb") as output:
        shutil.copyfileobj(response, output)
    with target.open("rb") as binary:
        actual = hashlib.file_digest(binary, "sha256").hexdigest()
    if actual != expected:
        target.unlink()
        raise SystemExit("Claude binary checksum mismatch")
    target.chmod(0o750)
    Path("/opt/author-cli/version").write_text(VERSION + "\n")
