#!/usr/bin/env python3
"""Download the pinned OpenThoughts-Agent-v1-RL snapshot and prepare the fixed 655/73 train/dev split.

The snapshot files go to artifacts/dataset_snapshot/ and the prepared split to data/: train.jsonl, dev.jsonl,
smoke.jsonl (eight of the training tasks), manifest.json and the 728 extracted task trees. Both are checked
against the SHA-256 values in sources.lock.json and data_manifest.json; no task code is executed. Existing
inputs that fail the checks are left untouched unless --replace is given.
"""
import argparse
import shutil

from dataset_contract import digest, validate
from download_artifacts import LOCK, ROOT, SNAPSHOT, download
from prepare_data import prepare


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check', action='store_true', help='Validate data/ and the snapshot only; no downloads')
    parser.add_argument('--replace', action='store_true',
                        help='Delete existing data/ or snapshot files that fail the checks and prepare them again')
    args = parser.parse_args()
    data, files = ROOT / 'data', LOCK['dataset']['files']
    if args.check:
        validate(data)
        for name, item in files.items():
            if not (SNAPSHOT / name).is_file() or digest(SNAPSHOT / name) != item['sha256']:
                raise ValueError(f'Pinned snapshot file is missing or differs: {SNAPSHOT / name}')
        print('Verified the 728-task release: 655 train / 73 dev / 8 training-smoke tasks')
        return
    for name, item in files.items():
        path = SNAPSHOT / name
        if path.exists() and digest(path) != item['sha256']:
            if not args.replace:
                raise ValueError(f'{path} differs from the pinned snapshot; use --replace to download it again')
            path.unlink()
        download(item['url'], path, item['sha256'])
    try:
        validate(data)
    except (OSError, ValueError, KeyError):
        if data.exists():
            if not args.replace:
                raise ValueError(f'{data} differs from the fixed split; use --replace to prepare it again')
            shutil.rmtree(data)
    else:
        print(f'{data} already holds the fixed 655/73 split')
        return
    prepare(SNAPSHOT / 'tasks.parquet', data)
    validate(data)
    print(f'Prepared 655 train / 73 dev tasks in {data}')


if __name__ == '__main__':
    main()
