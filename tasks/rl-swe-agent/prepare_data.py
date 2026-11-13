#!/usr/bin/env python3
"""Validate and prepare the pinned RL corpus without executing any task code."""
import argparse
import hashlib
import io
import json
import math
import os
import re
import shutil
import tarfile
import tempfile
import tomllib
from pathlib import Path, PurePosixPath

ROOT = Path(__file__).resolve().parent
LOCK = json.loads((ROOT / 'sources.lock.json').read_text())['dataset']
REQUIRED = ('instruction.md', 'task.toml', 'environment/Dockerfile', 'tests/test.sh')

def sha(data):
    return hashlib.sha256(data).hexdigest()

def extract_archive(blob, destination, max_bytes=100 * 1024 * 1024):
    """Strict extraction: reject unsafe/colliding paths and special files, preserve exec bits."""
    destination = Path(destination)
    seen, total, inventory = set(), 0, {}
    with tarfile.open(fileobj=io.BytesIO(blob), mode='r:*') as archive:
        members = archive.getmembers()
        if len(members) > 10000:
            raise ValueError('Too many archive entries')
        for member in members:
            path = PurePosixPath(member.name)
            if path.is_absolute() or '..' in path.parts or '\\' in member.name:
                raise ValueError(f'Unsafe archive path: {member.name}')
            if not path.parts or path.as_posix() in seen:
                raise ValueError(f'Duplicate/empty archive path: {member.name}')
            seen.add(path.as_posix())
            if not (member.isfile() or member.isdir()):
                raise ValueError(f'Unsupported archive entry: {member.name}')
            total += member.size
            if total > max_bytes:
                raise ValueError('Archive exceeds extracted size limit')
        destination.mkdir(parents=True, exist_ok=False)
        try:
            for member in members:
                target = destination / member.name
                if member.isdir():
                    target.mkdir(parents=True, exist_ok=True)
                    continue
                target.parent.mkdir(parents=True, exist_ok=True)
                with archive.extractfile(member) as stream:
                    payload = stream.read()
                if len(payload) != member.size:
                    raise ValueError('Short archive member')
                target.write_bytes(payload)
                target.chmod(0o755 if member.mode & 0o111 else 0o644)
                inventory[member.name] = sha(payload)
            if any(not (destination / name).is_file() for name in REQUIRED):
                raise ValueError('Task missing required Harbor files')
            tomllib.loads((destination / 'task.toml').read_text())
            if not (destination / 'instruction.md').read_text().strip():
                raise ValueError('Empty instruction')
        except BaseException:
            shutil.rmtree(destination)
            raise
    return inventory

def prepare(source, output, dev_fraction=0.1, seed=42, smoke_size=8):
    import pyarrow.parquet as pq
    if not 0 < dev_fraction < 1 or smoke_size < 1:
        raise ValueError('Invalid split/smoke size')
    if output.exists():
        raise FileExistsError(f'Refusing to replace prepared data: {output}')
    checksum = sha(source.read_bytes())
    if checksum != LOCK['files']['tasks.parquet']['sha256']:
        raise ValueError('Dataset does not match pinned checksum')
    table = pq.read_table(source)
    if table.num_rows != LOCK['row_count'] or table.column_names != ['path','task_binary']:
        raise ValueError('Unexpected dataset shape')
    output.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix='.prepare-', dir=output.parent))
    try:
        records, tasks, groups, ids = [], [], {}, set()
        for index, row in enumerate(table.to_pylist()):
            instance = row['path']
            if not isinstance(instance, str) or not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]{0,127}', instance) or instance in ids:
                raise ValueError(f'Unsafe or duplicate task ID: {instance!r}')
            ids.add(instance)
            blob = row['task_binary']
            if not isinstance(blob, bytes):
                raise ValueError('Task archive is not binary')
            inventory = extract_archive(blob, stage / 'tasks' / instance)
            prompt = (stage / 'tasks' / instance / 'instruction.md').read_text()
            # Conservatively keep identical normalized instructions together even if environments differ.
            group = sha(' '.join(prompt.split()).encode())
            groups.setdefault(group, []).append(instance)
            task_hash = sha(json.dumps(inventory, sort_keys=True).encode())
            metadata = dict(instance_id=instance, agent_name='mini-swe-agent', source_row=index,
                            dataset_revision=LOCK['revision'], archive_sha256=sha(blob), task_sha256=task_hash)
            records.append(dict(prompt=prompt, metadata=metadata))
            tasks.append(dict(**metadata, instruction_group=group, files=inventory))
        ranked = sorted(groups, key=lambda g: sha(f'{seed}:{g}'.encode()))
        ndev = min(len(ranked)-1, max(1, math.ceil(len(ranked)*dev_fraction)))
        dev_ids = {i for g in ranked[:ndev] for i in groups[g]}
        train = [r for r in records if r['metadata']['instance_id'] not in dev_ids]
        dev = [r for r in records if r['metadata']['instance_id'] in dev_ids]
        for row in train:row['metadata']['split']='train'
        for row in dev:row['metadata']['split']='dev'
        smoke = sorted(train, key=lambda r: sha(f"{seed}:smoke:{r['metadata']['instance_id']}".encode()))[:smoke_size]
        for name, rows in [('train', train), ('dev', dev), ('smoke', smoke)]:
            (stage / f'{name}.jsonl').write_text(''.join(json.dumps(r, ensure_ascii=False)+'\n' for r in rows))
        manifest = dict(dataset=LOCK['repo_id'], revision=LOCK['revision'], source_sha256=checksum,
                        original_rows=len(records), seed=seed, dev_fraction=dev_fraction,
                        counts=dict(train=len(train), dev=len(dev), smoke=len(smoke)),
                        split_method='normalized-instruction groups; SHA256(seed:group) order',
                        train_ids=[r['metadata']['instance_id'] for r in train],
                        dev_ids=[r['metadata']['instance_id'] for r in dev],
                        smoke_ids=[r['metadata']['instance_id'] for r in smoke],
                        tasks=tasks, validation=dict(schema='local TOML syntax only',
                        semantic_duplicates='pending', evaluation_overlap='pending',
                        docker_images='not built', verifier_execution='not run'))
        (stage / 'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
        os.rename(stage, output)
        return manifest
    finally:
        if stage.exists():
            shutil.rmtree(stage)

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source', type=Path, default=ROOT/'artifacts/dataset_snapshot/tasks.parquet')
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--dev-fraction', type=float, default=0.1)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--smoke-size', type=int, default=8)
    a=p.parse_args()
    m=prepare(a.source,a.output,a.dev_fraction,a.seed,a.smoke_size)
    print(json.dumps(dict(output=str(a.output), counts=m['counts'], validation=m['validation']),indent=2))

if __name__ == '__main__':
    main()
