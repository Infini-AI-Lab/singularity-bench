#!/usr/bin/env python3
"""Protected coding-agent-harness grid: run every (task, driver) cell through cell.py, then score.

    python3 tasks/coding-agent-harness/grid/grid.py run    --harness-run R --harness-authority SHA --images images.json \
                                    --run-root /trusted/grid/TAG --gpus 0,1 [--profile P] [--drivers a,b] [--wall-sec N]
                                    [--tasks a,b] [--backend apptainer --judge-dir JUDGE_DIR [--scratch DIR]]
    python3 tasks/coding-agent-harness/grid/grid.py report --harness-run R --harness-authority SHA --run-root /trusted/grid/TAG [--profile P]

images.json: {"observer": ..., "gateway": ..., "cells": {"<task>": {"cell": ..., "verifier": ...}}} (Docker), or
{"cells": {"<task>": {"sif": CELL.sif}}} (Apptainer: each cell image built with images/apptainer.def), whose cells run
through apptainer_cell.py.

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
FRAMEWORK = HERE.parents[2] / 'breakwater'
sys.path[:0] = [str(HERE), str(HERE / 'downstream'), str(FRAMEWORK)]
from contract import read_json, require
from registry import require_task

TASK = HERE.parent


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
    """The cell's attempt directories, oldest first (each attempt-N.log beside them is not one)."""
    folder = Path(root) / f'{task}__{driver}'
    found = [path for path in folder.glob('attempt-*') if path.is_dir() and path.name[len('attempt-'):].isdigit()] \
        if folder.is_dir() else []
    return sorted(found, key=lambda path: int(path.name[len('attempt-'):]))


def classify(attempt):
    reports = attempt / 'reports'
    if (reports / 'rule-violation.json').is_file():
        return 'ZERO', 0.0, 'rule violation'
    for name in ('void.json', 'cell-void.json'):  # cell-void.json: cells run before the generic verifier
        if (reports / name).is_file():
            return 'VOID', None, read_json(reports / name).get('reason')
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
    if args.tasks:  # run only these cells; the report still covers the whole profile
        chosen = args.tasks.split(',')
        require(set(chosen) <= set(tasks), 'A task outside the profile')
        tasks = [task for task in tasks if task in chosen]
    images = read_json(args.images)
    keys = ('sif',) if args.backend == 'apptainer' else ('cell', 'verifier')
    missing = [task for task in tasks if not all(images.get('cells', {}).get(task, {}).get(key) for key in keys)]
    require(not missing, f"{args.images} lists no {' or '.join(keys)} image for: {', '.join(missing)}")
    require(args.backend == 'apptainer' or (images.get('observer') and images.get('gateway')),
            f'{args.images} lists no observer or gateway image')
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
            if args.backend == 'apptainer':
                command = [sys.executable, '-I', str(HERE / 'apptainer_cell.py'), '--task', task, '--driver', driver,
                           '--harness-run', str(args.harness_run), '--harness-authority', args.harness_authority,
                           '--image', images['cells'][task]['sif'], '--judge-dir', str(args.judge_dir),
                           '--run-dir', str(target), '--gpus', gpu,
                           *(['--scratch', str(args.scratch)] if args.scratch else [])]
            else:
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
                cell = manifest.get('task_metadata') or manifest.get('cell') or {}
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
    parser.add_argument('--tasks', help='run only these tasks of the profile (comma-separated)')
    parser.add_argument('--backend', choices=('docker', 'apptainer'), default='docker')
    parser.add_argument('--judge-dir', type=Path, help='the judge runner (Apptainer: install_judge.py)')
    parser.add_argument('--scratch', type=Path, help="local disk for each sandbox's /tmp (Apptainer)")
    args = parser.parse_args()
    if args.command == 'run':
        require(args.images, '--images is required to run cells')
        require(args.backend == 'docker' or args.judge_dir, 'The Apptainer grid needs --judge-dir')
        return run(args)
    return report(args)


if __name__ == '__main__':
    raise SystemExit(main())
