"""Prepare pinned AMC-83 avg@4, sandboxed HumanEvalPlus and GPQA Diamond inputs."""
import argparse
import csv
import json
import math
from pathlib import Path
import random
import re

from prepare_data import digest, normalize

ROOT = Path(__file__).resolve().parent


def amc_rows(rows):
    """Keep all 83 modified integer-answer problems, using globally unique URLs."""
    rows = list(rows)
    if len(rows) != 83 or len({row['url'] for row in rows}) != 83:
        raise ValueError('AMC requires exactly 83 distinct problems')
    result = []
    for row in rows:
        answer = float(row['answer'])
        if not math.isfinite(answer) or not answer.is_integer():
            raise ValueError('AMC requires finite integer answers')
        item = normalize('math', {'problem': row['problem'], 'answer': str(int(answer))})
        item['metadata']['source_id'] = row['url']
        result.append(item)
    if len({item['id'] for item in result}) != 83:
        raise ValueError('AMC contains duplicate problem text')
    return result


def humanevalplus_rows(rows):
    """Preserve the published expanded test program; never execute it here."""
    rows = list(rows)
    if len(rows) != 164 or {row['task_id'] for row in rows} != {f'HumanEval/{i}' for i in range(164)}:
        raise ValueError('HumanEvalPlus requires all 164 distinct tasks')
    result = []
    for row in sorted(rows, key=lambda row: int(row['task_id'].split('/')[1])):
        if not re.fullmatch(r'[A-Za-z_][A-Za-z_0-9]*', row['entry_point']):
            raise ValueError('Invalid HumanEvalPlus entry point')
        if any(not isinstance(row[key], str) or not row[key].strip() for key in ('prompt', 'test')):
            raise ValueError('Incomplete HumanEvalPlus task')
        # The reference implementation must never appear in a generation prompt.
        prompt = ('<|begin_of_text|>Complete the Python function below. Preserve its signature. '
                  'Put your complete implementation, including necessary imports, in a ```python``` block.'
                  '\n\nUser: ' + row['prompt'] + '\nAssistant:')
        result.append({'id': row['task_id'], 'prompt': prompt,
                       'label': {'testtype': 'humanevalplus', 'entry_point': row['entry_point'],
                                 'test': row['test']},
                       'metadata': {'domain': 'code', 'source_id': row['task_id']}})
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--cache-dir')
    args = parser.parse_args()
    from huggingface_hub import hf_hub_download
    import pyarrow.parquet as pq
    sources = json.loads((ROOT / 'eval_sources.json').read_text())
    args.output.mkdir(parents=True, exist_ok=False)
    seed = sources['seed']
    manifest = {'sources': sources, 'datasets': {}, 'seed': seed}

    def download(spec):
        return hf_hub_download(spec['repo'], spec['file'], revision=spec['revision'],
                               repo_type='dataset', cache_dir=args.cache_dir)

    def save(name, rows, extra=None):
        spec = sources[name]
        if len(rows) != spec['items']:
            raise ValueError(f'{name}: require {spec["items"]} items, found {len(rows)}')
        for row in rows:
            row['metadata']['benchmark'] = name
        path = args.output / f'{name}.jsonl'
        path.write_text(''.join(json.dumps(row, ensure_ascii=False) + '\n' for row in rows))
        manifest['datasets'][name] = {'items': len(rows), 'samples': spec['samples'],
                                     'sha256': digest(path), **(extra or {})}
        print(f'{name}: {len(rows)} items x {spec["samples"]} responses', flush=True)

    save('amc', amc_rows(pq.read_table(download(sources['amc'])).to_pylist()))
    with Path(download(sources['humanevalplus'])).open() as stream:
        save('humanevalplus', humanevalplus_rows(json.loads(line) for line in stream if line.strip()),
             {'metric': 'fraction passing the complete published expanded test program in the code sandbox'})

    gpqa = []
    with Path(download(sources['gpqa_diamond'])).open(newline='') as stream:
        for i, row in enumerate(csv.DictReader(stream)):
            options = [row['Correct Answer']] + [row[f'Incorrect Answer {j}'] for j in range(1, 4)]
            permutation = list(range(4))
            random.Random(f'{seed}:gpqa:{i}').shuffle(permutation)
            question = row['Question'] + '\n' + '\n'.join(f'{chr(65+j)}: {options[k]}' for j, k in enumerate(permutation))
            item = normalize('science', {'input': question, 'expected_answer': chr(65 + permutation.index(0))})
            gpqa.append(item)
    save('gpqa_diamond', gpqa)

    (args.output / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')


if __name__ == '__main__':
    main()
