"""Install pinned native Codex and build its private worker-only executor."""
import hashlib
import os
from pathlib import Path
import platform
import shutil
import subprocess
import tarfile
import tempfile
import urllib.request

VERSION = '0.158.0'
SOURCE_SHA = 'e0c6f2492a1afad6ace9101b01a85deb58f81c52ee25af351d3b7478425a3558'
CLI_SHA = {
    'x86_64': 'af9f5aa6e6662accf9d707cef0d9ca083880a173a9c2b6c22947edb7830e5778',
    'aarch64': '91e3fafe5ff845a8f685a05d04919146fb30fdc9245af1a42a2f7f239959e90c',
}
RUST_SHA = {
    'x86_64': '88f28fa9af20594179f85d6df67078dfd6fa93e2f6da5e1e9b0ac4997988ca4f',
    'aarch64': '9a7a2c336b4787f1b72f6bab7c35d5b7af2fd03cbd39b4fc721466a70d402a7d',
}


def download(url, path, digest):
    with urllib.request.urlopen(url, timeout=120) as response, path.open('wb') as output:
        shutil.copyfileobj(response, output)
    if hashlib.sha256(path.read_bytes()).hexdigest() != digest:
        raise SystemExit('Download digest mismatch: ' + path.name)


def extract(archive, directory, components=None):
    with tarfile.open(archive) as tar:
        members = None if components is None else (member for member in tar
            if len(member.name.split('/')) <= 2 or member.name.split('/')[1] in components)
        tar.extractall(directory, members=members, filter='data')


arch = platform.machine()
if arch not in CLI_SHA:
    raise SystemExit('Unsupported Codex author architecture')
cli_version = VERSION
cli_digest = CLI_SHA[arch]
root = Path('/opt/author-cli')
root.mkdir(mode=0o750)
with tempfile.TemporaryDirectory(prefix='codex-build-') as tmp:
    work = Path(tmp)
    cli_name = f'codex-{arch}-unknown-linux-musl'
    archive = work / 'cli.tar.gz'
    download(f'https://github.com/openai/codex/releases/download/rust-v{cli_version}/{cli_name}.tar.gz', archive, cli_digest)
    extract(archive, work / 'native')
    shutil.copyfile(work / 'native' / cli_name, root / 'codex')
    (root / 'codex').chmod(0o750)

    source = work / 'source.tar.gz'
    download(f'https://codeload.github.com/openai/codex/tar.gz/refs/tags/rust-v{VERSION}', source, SOURCE_SHA)
    extract(source, work)
    cargo_root = work / f'codex-rust-v{VERSION}' / 'codex-rs'
    lock = cargo_root / 'Cargo.lock'
    if hashlib.sha256(lock.read_bytes()).hexdigest() != '5c3fd1fe59676117cd7fc9c55daafee90d95d0afedd13b9e1a71e10cad029a81':
        raise SystemExit('Unexpected upstream Cargo lock')
    # The release tag updates workspace versions but leaves their lock entries at
    # 0.0.0. Normalize only those entries; no external dependency changes.
    lock.write_text(lock.read_text().replace('version = "0.0.0"', f'version = "{VERSION}"'))
    if hashlib.sha256(lock.read_bytes()).hexdigest() != 'e93a06df3e346a7de763d5715715eaf51675a83a632c14e22ca3e5651c1fc74d':
        raise SystemExit('Unexpected normalized Cargo lock')
    target = cargo_root / 'exec-server/src/bin/breakwater-worker.rs'
    target.parent.mkdir(exist_ok=True)
    shutil.copyfile('/opt/breakwater-runtime/codex-worker.rs', target)

    rust_name = f'rust-1.97.1-{arch}-unknown-linux-gnu'
    rust_archive = work / 'rust.tar.xz'
    download(f'https://static.rust-lang.org/dist/{rust_name}.tar.xz', rust_archive, RUST_SHA[arch])
    # The distribution also contains many GB of docs/tools we never install.
    # Keep its installer metadata, rustc, Cargo and this target's standard library.
    components = ['rustc', 'cargo', 'rust-std-' + arch + '-unknown-linux-gnu']
    extract(rust_archive, work, components)
    prefix = work / 'rust'
    subprocess.run(['sh', str(work / rust_name / 'install.sh'), '--disable-ldconfig',
                    '--components=' + ','.join(components),
                    '--prefix=' + str(prefix)], check=True)
    env = dict(os.environ, PATH=str(prefix / 'bin') + ':/usr/local/bin:/usr/bin:/bin',
               CARGO_HOME=str(work / 'cargo-home'), CARGO_TARGET_DIR=str(work / 'target'))
    subprocess.run([str(prefix / 'bin/cargo'), 'build', '--locked', '--profile', 'dev-small',
                    '-j', '8', '-p', 'codex-exec-server', '--bin', 'breakwater-worker'],
                   cwd=cargo_root, env=env, check=True)
    shutil.copyfile(work / 'target/dev-small/breakwater-worker', root / 'codex-worker')
(root / 'version').write_text(VERSION + '\n')
# Docker sets the worker entrypoint's final ownership/setuid bit after chown.
(root / 'codex-worker').chmod(0o750)
