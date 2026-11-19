#!/usr/bin/env python3
"""Install the pinned agent CLI into a host directory for the Apptainer author sandbox.

    python3 -I install_cli.py --dest DIR [--agent claude-code|codex]

The versions and per-architecture SHA-256 checksums are the Docker author image's (author/install_claude.py,
author/install_codex.py). DIR is bound read-only into the author sandbox only, and records what it holds in
``harness`` and ``version``.

Codex's executor (``codex exec-server``) is the same binary, run beside the CLI in the author sandbox by
codex_forward.py, which forwards every command to the broker.
"""
import argparse
import hashlib
import importlib.util
from pathlib import Path
import platform
import shutil
import tarfile
import tempfile
import urllib.request

AUTHOR = Path(__file__).resolve().parents[1] / 'author'


def pinned(name):
    spec = importlib.util.spec_from_file_location('pinned_' + name, AUTHOR / f'install_{name}.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def download(url, path, expected, what):
    with urllib.request.urlopen(url, timeout=300) as response, path.open('wb') as output:
        shutil.copyfileobj(response, output)
    with path.open('rb') as stream:
        if hashlib.file_digest(stream, 'sha256').hexdigest() != expected:
            path.unlink()
            raise SystemExit(what + ' checksum mismatch')


def install_claude(dest):
    release = pinned('claude')
    arch, expected = release.RELEASES[platform.machine()]
    partial = dest / 'claude.partial'
    download(f'https://downloads.claude.ai/claude-code-releases/{release.VERSION}/{arch}/claude', partial, expected,
             'Claude Code')
    partial.chmod(0o755)
    partial.replace(dest / 'claude')
    return release.VERSION, f'Claude Code {release.VERSION} ({arch}) at {dest / "claude"}'


def install_codex(dest):
    release = pinned('codex')
    url, expected, name = release.cli_release(platform.machine())
    with tempfile.TemporaryDirectory(prefix='codex-cli-') as scratch:
        archive = Path(scratch) / 'cli.tar.gz'
        download(url, archive, expected, 'Codex')
        with tarfile.open(archive) as tar:
            member = tar.getmember(name)
            if not member.isfile():
                raise SystemExit(f'{name} is not a regular file in the Codex release')
            partial = dest / 'codex.partial'
            with tar.extractfile(member) as source, partial.open('wb') as output:
                shutil.copyfileobj(source, output)
    partial.chmod(0o755)
    partial.replace(dest / 'codex')
    return release.VERSION, f'Codex {release.VERSION} ({name}) at {dest / "codex"}'


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--dest', type=Path, required=True)
    parser.add_argument('--agent', choices=('claude-code', 'codex'), default='claude-code')
    args = parser.parse_args()
    args.dest.mkdir(parents=True, exist_ok=True)
    version, summary = (install_codex if args.agent == 'codex' else install_claude)(args.dest)
    (args.dest / 'version').write_text(version + '\n')
    (args.dest / 'harness').write_text(args.agent + '\n')
    print('installed ' + summary)


if __name__ == '__main__':
    main()
