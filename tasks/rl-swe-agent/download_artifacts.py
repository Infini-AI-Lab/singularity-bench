#!/usr/bin/env python3
"""Download the pinned dataset snapshot; optionally the GLM-4.7-Flash weights and pinned source checkouts.

The snapshot (tasks.parquet plus the dataset card and its extraction script, which is never executed) goes to
artifacts/dataset_snapshot/. Source checkouts go to .deps/<name> at the commits in sources.lock.json: harbor
(the sandbox worker and the evaluator's trial runner), harbor-datasets (SWE-bench Verified tasks) and swe-bench
(the image recipe generator used by arm64_tasks.py plan). Nothing is built or started.
"""
import argparse
import hashlib
import json
import os
import subprocess
import tempfile
from pathlib import Path
from urllib.request import urlopen

ROOT = Path(__file__).resolve().parent
LOCK = json.loads((ROOT / 'sources.lock.json').read_text())
SNAPSHOT = ROOT / 'artifacts' / 'dataset_snapshot'
SOURCES = ('harbor', 'harbor-datasets', 'swe-bench')

def download(url, dest, checksum):
    dest = Path(dest)
    if dest.exists():
        if hashlib.sha256(dest.read_bytes()).hexdigest() != checksum:
            raise ValueError(f'Existing file has wrong checksum: {dest}')
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(dir=dest.parent, prefix='.download-')
    try:
        with os.fdopen(fd, 'wb') as stream, urlopen(url, timeout=120) as response:
            while block := response.read(1024 * 1024):
                stream.write(block)
        if hashlib.sha256(Path(name).read_bytes()).hexdigest() != checksum:
            raise ValueError(f'Download checksum mismatch: {url}')
        os.replace(name, dest)
    finally:
        Path(name).unlink(missing_ok=True)

def checkout(name, deps):
    spec = LOCK[name]
    dst = deps / name
    if not dst.exists():
        dst.mkdir(parents=True)
        subprocess.run(['git', 'init', str(dst)], check=True)
        subprocess.run(['git', '-C', str(dst), 'remote', 'add', 'origin', spec['url']], check=True)
    remote = subprocess.check_output(['git', '-C', str(dst), 'remote', 'get-url', 'origin'], text=True).strip()
    if remote != spec['url']:
        raise ValueError(f'Unexpected origin: {dst}')
    if subprocess.check_output(['git', '-C', str(dst), 'status', '--porcelain']).strip():
        raise ValueError(f'Dirty source checkout; preserve it: {dst}')
    subprocess.run(['git', '-C', str(dst), 'fetch', '--depth=1', 'origin', spec['commit']], check=True)
    subprocess.run(['git', '-C', str(dst), 'checkout', '--detach', spec['commit']], check=True)

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--sources', nargs='+', choices=SOURCES, default=[], help='Pinned source checkouts to fetch')
    p.add_argument('--deps', type=Path, default=ROOT / '.deps')
    p.add_argument('--model', type=Path, help='Download the pinned GLM-4.7-Flash weights (about 59 GiB) here')
    a = p.parse_args()
    for name, item in LOCK['dataset']['files'].items():
        download(item['url'], SNAPSHOT / name, item['sha256'])
    for name in a.sources:
        checkout(name, a.deps)
    if a.model:
        from huggingface_hub import snapshot_download
        spec = LOCK['model']
        snapshot_download(spec['id'], revision=spec['revision'], local_dir=a.model)
        (a.model / 'model-source.json').write_text(json.dumps(spec, indent=2)+'\n')
    print('Requested downloads verified; nothing was built or started.')

if __name__ == '__main__':
    main()
