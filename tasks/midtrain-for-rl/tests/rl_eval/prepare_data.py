"""Prepare one fixed, pinned 19K/7K/17K RL mixture on the operator's node.

Run once, before any submission is graded; every submission uses the same mixture.
No model-based difficulty filter.
The base tokenizer defines the <=2K input filter for every submission.
"""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import random
import re

ROOT = Path(__file__).resolve().parent


def digest(path):
    checksum = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            checksum.update(block)
    return checksum.hexdigest()


def metadata(row):
    value = row.get('metadata') or {}
    return json.loads(value) if isinstance(value, str) else value


def normalize_tests(raw, fn_name=None):
    tests = json.loads(raw) if isinstance(raw, str) else raw
    if isinstance(tests, dict):
        inputs, outputs = tests['inputs'], tests['outputs']
        if len(inputs) != len(outputs):
            raise ValueError('code inputs and outputs differ in length')
        fn_name = tests.get('fn_name') or fn_name
        tests = [{'input': i, 'output': o, 'testtype': 'functional' if fn_name else 'stdin'}
                 for i, o in zip(inputs, outputs)]
    if not isinstance(tests, list) or not tests:
        raise ValueError('code problem has no tests')
    kinds = {t.get('testtype', 'functional' if fn_name else 'stdin') for t in tests}
    if len(kinds) != 1 or not kinds <= {'stdin', 'functional'}:
        raise ValueError(f'unsupported code test types: {kinds}')
    kind = kinds.pop()
    if kind == 'functional' and not fn_name:
        raise ValueError('functional tests need a function name')
    if fn_name and not re.fullmatch(r'[A-Za-z_][A-Za-z_0-9]*', fn_name):
        raise ValueError('invalid function name')
    result = {'testtype': kind, 'tests': []}
    if fn_name:
        result['fn_name'] = fn_name
    for test in tests:
        if 'input' not in test or 'output' not in test:
            raise ValueError('incomplete code test')
        if kind == 'stdin' and not all(isinstance(test[k], str) for k in ('input', 'output')):
            raise ValueError('stdin tests require strings')
        result['tests'].append({'input': test['input'], 'output': test['output']})
    return result


def normalize(domain, row):
    meta = {'domain': domain}
    if domain == 'math':
        problem, label = row['problem'], str(row['answer']).strip()
        if not label:
            raise ValueError('empty math answer')
        instruction = 'Solve the math problem. Put your final answer in \\boxed{}.'
    elif domain == 'code':
        problem = row['problem']
        fn_name = metadata(row).get('func_name')
        label = normalize_tests(row['tests'], fn_name)
        starter = row.get('starter_code')
        if starter:
            problem += '\n\nStarter code:\n' + starter
        instruction = 'Solve the coding problem in Python. Put your complete solution in a ```python``` block.'
    elif domain == 'science':
        problem = row['input']
        if not isinstance(problem, str) or not problem.strip():
            raise ValueError('science requires a nonempty input question')
        # OpenScienceReasoning-2 publishes reference labels separately from output.
        # Never infer a label from the teacher's reasoning or final response.
        answer = row.get('expected_answer')
        if not isinstance(answer, str):
            raise ValueError('science has no explicit expected_answer')
        answer = answer.strip()
        letter = re.fullmatch(r'\\(?:text|mathrm)\{\s*([A-Z])\s*\}|([A-Z])', answer)
        if not letter:
            raise ValueError('science expected_answer is not an option letter')
        label = letter.group(1) or letter.group(2)
        options = re.findall(r'^\s*([A-Z])\s*[:.)]\s*.+', problem, re.MULTILINE)
        if len(options) < 2 or len(options) != len(set(options)) or label not in options:
            raise ValueError('science options do not match expected_answer')
        meta['valid_letters'] = options
        instruction = 'Answer the science question. Put the final option letter in \\boxed{}.'
    else:
        raise ValueError(domain)
    if not isinstance(problem, str) or not problem.strip():
        raise ValueError('empty problem')
    # Base-model format, independent of any submitted tokenizer or chat template.
    prompt = '<|begin_of_text|>' + instruction + '\n\nUser: ' + problem.strip() + '\nAssistant:'
    key = hashlib.sha256(problem.strip().encode()).hexdigest()
    return {'id': domain + '-' + key, 'prompt': prompt, 'label': label, 'metadata': meta}


def reservoir(rows, count, seed, tokenizer, max_input_tokens, domain):
    rng = random.Random(seed)
    selected, seen, stats = [], set(), Counter()
    for source_id, row in rows:
        stats['scanned'] += 1
        try:
            item = normalize(domain, row)
        except (KeyError, ValueError, TypeError):
            stats['invalid'] += 1
            continue
        if item['id'] in seen:
            stats['duplicate'] += 1
            continue
        seen.add(item['id'])
        if len(tokenizer.encode(item['prompt'], add_special_tokens=False)) > max_input_tokens:
            stats['over_length'] += 1
            continue
        item['metadata']['source_id'] = source_id
        stats['eligible'] += 1
        if len(selected) < count:
            selected.append(item)
        else:
            index = rng.randrange(stats['eligible'])
            if index < count:
                selected[index] = item
    if len(selected) != count:
        raise ValueError(f'{domain}: need {count} distinct eligible prompts; found {len(selected)}; {dict(stats)}')
    return selected, dict(stats)


def upstream_rows(domain, source, cache_dir):
    from datasets import load_dataset
    if domain == 'math':
        from huggingface_hub import hf_hub_download
        path = hf_hub_download(source['repo'], source['files'][0], revision=source['revision'],
                               repo_type='dataset', cache_dir=cache_dir)
        rows = json.loads(Path(path).read_text())
        for i, row in enumerate(rows):
            yield f'deepscaler:{i}', row
    elif domain == 'code':
        # Explicit train files only: never include Codeforces or LCB test splits.
        from huggingface_hub import HfApi
        files = HfApi().list_repo_files(source['repo'], revision=source['revision'], repo_type='dataset')
        for config in source['configs']:
            urls = [f'https://huggingface.co/datasets/{source["repo"]}/resolve/{source["revision"]}/{p}'
                    for p in sorted(files) if p.startswith(config + '/train-') and p.endswith('.parquet')]
            if not urls:
                raise ValueError(f'missing DeepCoder train files: {config}')
            rows = load_dataset('parquet', data_files={'train': urls}, split='train',
                                streaming=True, cache_dir=cache_dir)
            for i, row in enumerate(rows):
                yield f'{config}:train:{i}', row
    else:
        rows = load_dataset(source['repo'], revision=source['revision'], split=source['split'],
                            streaming=True, cache_dir=cache_dir, columns=['input', 'expected_answer'])
        # A deterministic shuffled 100K-row initial pool, shared by all submissions.
        rows = rows.shuffle(seed=20261001, buffer_size=10000).take(source['pool_size'])
        for i, row in enumerate(rows):
            yield f'openscience:pool:{i}', row


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base', type=Path, required=True, help='operator-owned Llama-3.2-1B base tokenizer')
    parser.add_argument('--output', type=Path, required=True, help='fresh private RL data directory')
    parser.add_argument('--cache-dir')
    parser.add_argument('--local-sources', type=Path, help='offline fixtures: math.jsonl, code.jsonl, science.jsonl')
    args = parser.parse_args()
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.base, local_files_only=True, trust_remote_code=False)
    recipe = json.loads((ROOT / 'recipe.json').read_text())
    sources = json.loads((ROOT / 'sources.json').read_text())
    args.output.mkdir(parents=True, exist_ok=False)
    combined, stats = [], {}
    for domain, count in recipe['counts'].items():
        print(f'Preparing {domain}: requesting {count} prompts', flush=True)
        if args.local_sources:
            def local_rows(domain=domain):
                with (args.local_sources / f'{domain}.jsonl').open() as stream:
                    for i, line in enumerate(stream):
                        if line.strip():
                            yield f'local:{i}', json.loads(line)
            rows = local_rows()
        else:
            rows = upstream_rows(domain, sources[domain], args.cache_dir)
        selected, stats[domain] = reservoir(rows, count, f'{recipe["seed"]}:{domain}', tokenizer,
                                            recipe['max_input_tokens'], domain)
        combined.extend(selected)
        print(f'{domain}: {stats[domain]}', flush=True)
    random.Random(recipe['seed']).shuffle(combined)
    pending = args.output / 'train.jsonl.tmp'
    pending.write_text(''.join(json.dumps(row, ensure_ascii=False) + '\n' for row in combined))
    pending.rename(args.output / 'train.jsonl')
    report = {'sources': sources, 'counts': recipe['counts'], 'seed': recipe['seed'], 'stats': stats,
              'tokenizer': str(args.base.resolve()), 'max_input_tokens': recipe['max_input_tokens'],
              'selection': 'uniform reservoir over distinct length-filtered eligible prompts; no difficulty filter',
              'sha256': digest(args.output / 'train.jsonl')}
    (args.output / 'manifest.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
