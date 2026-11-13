"""Validate the pinned 728-task corpus and its fixed 655/73 train/dev split."""
import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def validate(directory, *, deep=True):
    directory = Path(directory).resolve()
    lock = json.loads((ROOT / 'sources.lock.json').read_text())['dataset']
    expected = json.loads((ROOT / 'data_manifest.json').read_text())
    manifest = json.loads((directory / 'manifest.json').read_text())
    if (manifest.get('dataset'), manifest.get('revision'), manifest.get('original_rows')) != (
            lock['repo_id'], lock['revision'], 728):
        raise ValueError('Expected the pinned 728-task release of OpenThoughts-Agent-v1-RL')
    if manifest.get('counts') != {'train': 655, 'dev': 73, 'smoke': 8}:
        raise ValueError('Expected exactly 655 train, 73 dev and 8 training-smoke tasks')
    if manifest.get('source_sha256') != lock['files']['tasks.parquet']['sha256']:
        raise ValueError('Wrong dataset provenance')
    for name, value in expected['prepared_file_sha256'].items():
        if digest(directory / name) != value:
            raise ValueError(f'Prepared data differs from the fixed split: {name}')
    tasks = {row['instance_id']: row for row in manifest['tasks']}
    if len(tasks) != 728 or {p.name for p in (directory / 'tasks').iterdir() if p.is_dir()} != set(tasks):
        raise ValueError('Extracted task inventory must contain exactly the pinned 728 tasks')
    split_ids = {}
    for split in ('train', 'dev', 'smoke'):
        rows = [json.loads(line) for line in (directory / f'{split}.jsonl').read_text().splitlines() if line.strip()]
        ids = [row['metadata']['instance_id'] for row in rows]
        if len(ids) != manifest['counts'][split] or len(set(ids)) != len(ids):
            raise ValueError(f'Invalid {split} task IDs/count')
        split_ids[split] = set(ids)
        for row in rows:
            meta = row['metadata']
            task = tasks[meta['instance_id']]
            if (meta.get('dataset_revision') != lock['revision'] or
                    meta.get('split') != ('train' if split == 'smoke' else split) or
                    meta.get('archive_sha256') != task['archive_sha256'] or
                    row['prompt'] != (directory / 'tasks' / task['instance_id'] / 'instruction.md').read_text()):
                raise ValueError(f'{split} row provenance or instruction mismatch')
    if split_ids['train'] & split_ids['dev'] or not split_ids['smoke'] <= split_ids['train']:
        raise ValueError('Train/dev overlap or non-training smoke task')
    if deep:
        for ident, task in tasks.items():
            tree = directory / 'tasks' / ident
            actual = {str(p.relative_to(tree)) for p in tree.rglob('*') if p.is_file()}
            if actual != set(task['files']):
                raise ValueError(f'Changed task inventory: {ident}')
            for name, value in task['files'].items():
                if digest(tree / name) != value:
                    raise ValueError(f'Changed task file: {ident}/{name}')
    return manifest


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, default=ROOT / 'data')
    args = parser.parse_args()
    manifest = validate(args.data)
    print(json.dumps({'dataset': manifest['dataset'], 'revision': manifest['revision'],
                      'counts': manifest['counts'], 'verified': True}, indent=2))
