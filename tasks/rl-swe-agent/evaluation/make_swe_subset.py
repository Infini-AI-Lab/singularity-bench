"""Draw the fixed SWE-bench Verified evaluation subset (swe-subset.json) from the tasks not excluded.

  python3 make_swe_subset.py SWE_BENCHMARK [--size 100] [--seed 0]

Stratified by repository x upstream difficulty (task.toml [metadata] difficulty): each cell gets its
proportional share of the size, rounded by largest remainder, and its tasks are a seeded sample. The
subset keeps the full set's repository and difficulty mix; it is not chosen by any model's results.
"""
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import random
import tomllib

HERE = Path(__file__).resolve().parent


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('swe_benchmark', type=Path)
    p.add_argument('--size', type=int, default=100)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--swe-exclusions', type=Path, default=HERE / 'swe-exclusions.json')
    p.add_argument('--output', type=Path, default=HERE / 'swe-subset.json')
    a = p.parse_args()
    exclusions = a.swe_exclusions.read_bytes()
    excluded = set(json.loads(exclusions)['tasks'])
    cells = defaultdict(list)
    for config in sorted(a.swe_benchmark.glob('*/task.toml')):
        name = config.parent.name
        if name not in excluded:
            difficulty = tomllib.loads(config.read_text())['metadata']['difficulty']
            cells[(name.split('__')[0], difficulty)].append(name)
    eligible = sum(map(len, cells.values()))
    quota = {cell: len(names) * a.size / eligible for cell, names in cells.items()}
    count = {cell: int(q) for cell, q in quota.items()}
    for cell in sorted(quota, key=lambda c: (count[c] - quota[c], c))[:a.size - sum(count.values())]:
        count[cell] += 1
    rng = random.Random(a.seed)
    tasks = sorted(name for cell in sorted(cells) for name in rng.sample(sorted(cells[cell]), count[cell]))
    strata = {f'{repo} | {difficulty}': f'{count[(repo, difficulty)]}/{len(cells[(repo, difficulty)])}'
              for repo, difficulty in sorted(cells)}
    a.output.write_text(json.dumps({
        'method': 'stratified by repository x difficulty, proportional allocation (largest remainder), seeded sample',
        'seed': a.seed, 'size': len(tasks), 'eligible': eligible,
        'swe_exclusions_sha256': hashlib.sha256(exclusions).hexdigest(),
        'strata': strata, 'tasks': tasks}, indent=2) + '\n')
    print(f'{len(tasks)} of {eligible} eligible tasks -> {a.output}')


if __name__ == '__main__':
    main()
