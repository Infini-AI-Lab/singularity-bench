#!/usr/bin/env python3
"""Install the pinned Codex release that runs the verifier's judge, for the Apptainer verifier sandbox.

    python3 -I install_judge.py --dest DIR [--tarball codex-VERSION-linux-ARCH.tgz]

The version and SHA-256 checksums are the Docker verifier image's (Dockerfile.verifier): the npm package tarball
is checked first, then the static ``codex`` binary and its bundled ``bwrap``, which land in DIR as ``codex`` and
``codex-resources/bwrap``. As in the image build, the installed binary must report the shell and multi-agent tools
off under the judge's settings. DIR is bound read-only into the verifier sandbox only (``run.py --judge-dir DIR``).
"""
import argparse
import hashlib
from pathlib import Path
import platform
import re
import shutil
import subprocess
import tarfile
import tempfile
import urllib.request

VERSION = '0.164.0'
# machine: (npm package suffix, tarball sha256, vendor triple, codex sha256, bwrap sha256)
RELEASES = {
    'x86_64': ('linux-x64', '62659fefd9a8efa21a496402fde3c11c025bbeee735a582968f8ffd14cb5f79c',
               'x86_64-unknown-linux-musl', 'b93f6811d3f653f0fb83be4050d857fb9abbb111dfa2ce160563327daff7f306',
               '3a81c19ad10a70436529c58e77898e2a447abe80167f813fc20ec34bd9eafdeb'),
    'aarch64': ('linux-arm64', '048a62731ce28aa000508efcc032302e58488ca079970dcae06d28df39ae5ce0',
                'aarch64-unknown-linux-musl', '8813c8d8a4cea2eeba271b72ab22f49628b184b24ab18dbef06b4ce41e9a8d39',
                '0d784834f1674ff02aee25b42f2cbcfd27c5305863806cd6475d0eac8e33c378'),
}


def sha256(path):
    with open(path, 'rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def extract(archive, member, destination, expected):
    """Copy one named regular file out of the checked tarball and check its own digest."""
    info = archive.getmember(member)
    if not info.isreg():
        raise SystemExit(f'{member} is not a regular file in the Codex package')
    partial = destination.with_name(destination.name + '.partial')
    with archive.extractfile(info) as source, partial.open('wb') as target:
        shutil.copyfileobj(source, target)
    if sha256(partial) != expected:
        partial.unlink()
        raise SystemExit(f'checksum mismatch for {member}')
    partial.chmod(0o755)
    partial.replace(destination)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--dest', type=Path, required=True)
    parser.add_argument('--tarball', type=Path, help='an already downloaded npm package tarball for this machine')
    args = parser.parse_args()
    suffix, tarball_sha, triple, codex_sha, bwrap_sha = RELEASES[platform.machine()]
    with tempfile.TemporaryDirectory(prefix='breakwater-judge-') as scratch:
        tarball = Path(scratch) / 'codex.tgz'
        if args.tarball:
            shutil.copyfile(args.tarball, tarball)
        else:
            url = f'https://registry.npmjs.org/@openai/codex/-/codex-{VERSION}-{suffix}.tgz'
            with urllib.request.urlopen(url, timeout=300) as response, tarball.open('wb') as output:
                shutil.copyfileobj(response, output)
        if sha256(tarball) != tarball_sha:
            raise SystemExit('Codex package checksum mismatch')
        (args.dest / 'codex-resources').mkdir(parents=True, exist_ok=True)
        with tarfile.open(tarball, 'r:gz') as archive:
            extract(archive, f'package/vendor/{triple}/bin/codex', args.dest / 'codex', codex_sha)
            extract(archive, f'package/vendor/{triple}/codex-resources/bwrap', args.dest / 'codex-resources' / 'bwrap',
                    bwrap_sha)
    with tempfile.TemporaryDirectory(prefix='breakwater-judge-home-') as home:
        features = subprocess.run([str(args.dest / 'codex'), '-c', 'features.shell_tool=false', '-c',
                                   'features.multi_agent=false', 'features', 'list'], capture_output=True, text=True,
                                  check=True, timeout=120, env={'PATH': '/usr/bin:/bin', 'CODEX_HOME': home}).stdout
    for name in ('shell_tool', 'multi_agent'):
        if not re.search(rf'^{name}\s.*\sfalse$', features, re.MULTILINE):
            raise SystemExit(f'the installed Codex does not report {name} off')
    (args.dest / 'version').write_text(VERSION + '\n')
    print(f'installed Codex {VERSION} ({triple}) at {args.dest / "codex"}')


if __name__ == '__main__':
    main()
