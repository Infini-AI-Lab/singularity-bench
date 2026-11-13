#!/usr/bin/env python3
"""Protected coding-agent-harness grid: run every (task, driver) cell through cell.py, then score.

    python3 breakwater/grid.py run    --harness-run R --harness-authority SHA --images images.json \
                                    --run-root /trusted/grid/TAG --gpus 0,1 [--profile P] [--drivers a,b] [--wall-sec N]
    python3 breakwater/grid.py report --harness-run R --harness-authority SHA --run-root /trusted/grid/TAG [--profile P]

images.json: {"observer": ..., "gateway": ..., "cells": {"<task>": {"cell": ..., "verifier": ...}}}

The harness score is the mean over drivers of the mean over tasks. A cell is SCORED (judge PASS: its recomputed reward), ZERO (judge FAIL or
INCONCLUSIVE, an ineligible verification, or a rule violation: the framework's failure
reward) or VOID (no score: oracle unavailable, driver endpoint rejected, or an
infrastructure failure). Any void leaves the harness score incomplete, never zero.
Only a root submission whose own protected verification passed is graded.
"""
import argparse
import concurrent.futures
import json
import os
from pathlib import Path
import queue
import subprocess
import sys

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE / 'downstream')]
from contract import read_json, require
from registry import require_task

TASK = HERE.parent / 'tasks' / 'coding-agent-harness'


def selection(profile, drivers):
    config = read_json(TASK / 'eval' / 'config.json')
    profile = profile or config.get('default_task_profile') or 'compact5-v1'
    tasks = [path.split('/', 1)[1] for path in config['task_profiles'][profile]]
    for task in tasks:
        require_task(task)
    names = drivers.split(',') if drivers else sorted(config['drivers'])
    require(set(names) <= set(config['drivers']), 'Unknown driver')
    return profile, tasks, names


def attempts(root, task, driver):
    folder = Path(root) / f'{task}__{driver}'
    return sorted(folder.glob('attempt-*'), key=lambda path: int(path.name.split('-')[1])) if folder.is_dir() else []


def classify(attempt):
    reports = attempt / 'reports'
    if (reports / 'rule-violation.json').is_file():
        return 'ZERO', 0.0, 'rule violation'
    if (reports / 'cell-void.json').is_file():
        return 'VOID', None, read_json(reports / 'cell-void.json').get('reason')
    if (reports / 'judge.json').is_file():
        verdict = read_json(reports / 'judge.json')['verdict']
        reward = read_json(reports / 'reward.json')['reward']
        return ('SCORED' if verdict == 'PASS' else 'ZERO'), float(reward), 'judge ' + verdict
    if (reports / 'verification-error.json').is_file():
        return 'ZERO', 0.0, 'ineligible: ' + read_json(reports / 'verification-error.json').get('error', '')[:200]
    if (reports / 'pipeline-error.json').is_file():
        return 'VOID', None, read_json(reports / 'pipeline-error.json').get('error', '')[:200]
    return 'MISSING', None, 'no completed attempt'


def run(args):
    profile, tasks, drivers = selection(args.profile, args.drivers)
    images = read_json(args.images)
    gpus = queue.Queue()
    for index in args.gpus.split(','):
        gpus.put(index.strip())
    root = Path(args.run_root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    cells = [(task, driver) for task in tasks for driver in drivers]   # task-major

    def one(cell):
        task, driver = cell
        previous = attempts(root, task, driver)
        if previous and classify(previous[-1])[0] in ('SCORED', 'ZERO'):
            return task, driver, 'skipped (complete)'
        number = int(previous[-1].name.split('-')[1]) + 1 if previous else 1
        target = root / f'{task}__{driver}' / f'attempt-{number}'
        target.parent.mkdir(exist_ok=True)
        gpu = gpus.get()
        try:
            command = [sys.executable, str(HERE / 'cell.py'), '--task', task, '--driver', driver,
                       '--harness-run', str(args.harness_run), '--harness-authority', args.harness_authority,
                       '--cell-image', images['cells'][task]['cell'], '--verifier-image', images['cells'][task]['verifier'],
                       '--observer-image', images['observer'], '--gateway-image', images['gateway'],
                       '--run-dir', str(target), '--gpus', f'device={gpu}']
            if args.wall_sec:
                command += ['--wall-sec', str(args.wall_sec)]
            log = root / f'{task}__{driver}' / f'attempt-{number}.log'
            with log.open('wb') as stream:
                code = subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT).returncode
            return task, driver, f'exit {code}'
        finally:
            gpus.put(gpu)

    with concurrent.futures.ThreadPoolExecutor(max_workers=len(args.gpus.split(','))) as pool:
        for task, driver, status in pool.map(one, cells):
            print(f'[grid] {task} x {driver}: {status}', flush=True)
    return report(args)


def report(args):
    profile, tasks, drivers = selection(args.profile, args.drivers)
    root_run = Path(args.harness_run).resolve()
    import cell as cellmod
    launch, harness, _ = cellmod.root_submission(root_run, args.harness_authority)
    rows, by_driver, incomplete = [], {}, False
    for task in tasks:
        for driver in drivers:
            history = attempts(args.run_root, task, driver)
            status, reward, reason = classify(history[-1]) if history else ('MISSING', None, 'never run')
            if history and status != 'MISSING':
                manifest = read_json(history[-1] / 'authority' / 'manifest.json') if \
                    (history[-1] / 'authority' / 'manifest.json').is_file() else {}
                cell = manifest.get('cell') or {}
                if cell and cell.get('root_authority_sha256') != args.harness_authority:
                    status, reward, reason = 'VOID', None, 'cell graded a different root submission'
            if status in ('VOID', 'MISSING'):
                incomplete = True
            rows.append({'task': task, 'driver': driver, 'status': status, 'reward': reward, 'reason': reason,
                         'attempts': len(history)})
            by_driver.setdefault(driver, []).append(reward)
    per_driver = {driver: (sum(values) / len(values) if all(v is not None for v in values) else None)
                  for driver, values in by_driver.items()}
    score = None if incomplete else sum(per_driver.values()) / len(per_driver)
    result = {'profile': profile, 'root_run_id': launch['identity']['run_id'],
              'root_submission_sha256': launch['submission_sha256'], 'harness_files': len(harness),
              'score': score, 'status': 'incomplete' if incomplete else 'complete',
              'per_driver_mean': per_driver, 'cells': rows}
    output = Path(args.run_root) / 'grid-report.json'
    output.write_text(json.dumps(result, indent=1, sort_keys=True))
    for row in rows:
        shown = '-' if row['reward'] is None else f"{row['reward']:.3f}"
        print(f"  {row['task']:28} {row['driver']:8} {row['status']:8} {shown:>6}  {row['reason']}")
    print(f"MEAN {'incomplete' if score is None else f'{score:.4f}'}  ({output})")
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('command', choices=('run', 'report'))
    parser.add_argument('--harness-run', type=Path, required=True)
    parser.add_argument('--harness-authority', required=True)
    parser.add_argument('--run-root', type=Path, required=True)
    parser.add_argument('--profile')
    parser.add_argument('--drivers')
    parser.add_argument('--images', type=Path)
    parser.add_argument('--gpus', default='0')
    parser.add_argument('--wall-sec', type=float)
    args = parser.parse_args()
    if args.command == 'run':
        require(args.images, '--images is required to run cells')
        return run(args)
    return report(args)


if __name__ == '__main__':
    raise SystemExit(main())
