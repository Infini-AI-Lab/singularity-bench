#!/usr/bin/env python3
"""Download and validate the pinned GLM-5.3-Flash checkpoint on the host, before offline runs.

The checkpoint goes to $MODEL_HOST_DIR/GLM-5.3-Flash and is checked file by file against
model_manifest.json (pinned revision, sizes and hashes). Existing good files are reused.
--check validates without downloading. The script also checks that the agent-visible
copies under environment/ match their authoritative copies.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess

ROOT = Path(__file__).resolve().parent
MANIFEST = json.loads((ROOT / 'model_manifest.json').read_text())
# Agent-visible copy under environment/ -> the task's own copy, which the launcher and the
# grader use (the prompt, the traffic corpus).
COPIES = {'environment/instruction.md': 'instruction.md',
          'environment/agentx/code.txt': 'tests/agentx/code.txt',
          'environment/agentx/sessions.jsonl.gz': 'tests/agentx/sessions.jsonl.gz'}
EVALUATION_SETS = {'tests/hidden_eval.json': 500, 'tests/canaries.json': 60}


def model_directory():
    parent = os.environ.get('MODEL_HOST_DIR')
    if not parent:
        raise ValueError('Set MODEL_HOST_DIR to the directory that holds (or will hold) GLM-5.3-Flash')
    return Path(parent).expanduser().resolve() / 'GLM-5.3-Flash'


def invalid_files(directory, manifest=MANIFEST):
    invalid = []
    for name, entry in manifest['files'].items():
        path = directory / name
        if not path.is_file() or path.stat().st_size != entry['size']:
            invalid.append(name)
            continue
        if 'sha256' in entry:
            digest = hashlib.sha256()
            expected = entry['sha256']
        else:
            digest = hashlib.sha1(f"blob {entry['size']}\0".encode())
            expected = entry['git_blob_sha1']
        with path.open('rb') as stream:
            while chunk := stream.read(8 * 1024 * 1024):
                digest.update(chunk)
        if digest.hexdigest() != expected:
            invalid.append(name)
    return invalid


def check_task_files():
    for copy, original in COPIES.items():
        if (ROOT / copy).read_bytes() != (ROOT / original).read_bytes():
            raise ValueError(f'Agent-visible {copy} differs from {original}')
    for name, count in EVALUATION_SETS.items():
        if len(json.loads((ROOT / name).read_text())) != count:
            raise ValueError(f'Wrong example count: {name}')


def prepare(check=False):
    check_task_files()
    directory = model_directory()
    if directory.is_relative_to(ROOT):
        raise ValueError('Keep checkpoints outside the task source/build context')
    bad = invalid_files(directory)
    if bad and check:
        raise ValueError('Missing or corrupt checkpoint files: ' + ', '.join(bad))
    if bad:
        if not shutil.which('hf'):
            raise ValueError('Host hf CLI required; install huggingface_hub before preparation')
        directory.mkdir(parents=True, exist_ok=True)
        # Existing good shards are reused. Force only invalid files: HF's local cache
        # metadata alone cannot detect a file modified after an earlier download.
        required = sum(MANIFEST['files'][name]['size'] for name in bad)
        if shutil.disk_usage(directory).free < required + 2 * 1024**3:
            raise ValueError(f'Need {required / 1024**3 + 2:.1f} GiB free for missing/corrupt checkpoint files')
        subprocess.run(['hf', 'download', MANIFEST['repository'], *bad,
                        '--revision', MANIFEST['revision'], '--local-dir', str(directory),
                        '--force-download', '--max-workers', os.environ.get('MODEL_DOWNLOAD_WORKERS', '4')],
                       env=dict(os.environ, HF_HUB_OFFLINE='0'), check=True)
        bad = invalid_files(directory)
        if bad:
            raise ValueError('Checkpoint verification failed: ' + ', '.join(bad))
    config = json.loads((directory / 'config.json').read_text())
    index = json.loads((directory / 'model.safetensors.index.json').read_text())
    if config.get('model_type') != MANIFEST['model_type']:
        raise ValueError('Wrong model architecture')
    shards = set(index['weight_map'].values())
    if shards != {name for name in MANIFEST['files'] if name.endswith('.safetensors')}:
        raise ValueError('Checkpoint index does not match the pinned shard set')
    print(f"Validated {len(shards)} checkpoint shards at {directory}; revision {MANIFEST['revision']}")
    print('Task files validated; the evaluation data ships with the task.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check', action='store_true', help='validate only; no downloads or builds')
    args = parser.parse_args()
    prepare(args.check)
