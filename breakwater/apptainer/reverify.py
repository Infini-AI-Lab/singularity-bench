#!/usr/bin/env python3
"""Run the verifier stage again on a finished Apptainer run.

    python3 breakwater/apptainer/reverify.py --run-dir RUN_DIR --task tasks/<name> --image TASK.sif \
      --asset-root ASSET_ROOT [--bindings bindings.json] --judge-dir JUDGE_DIR [--judge-key-env VAR] [--scratch DIR]

Nothing the run sealed changes: the evidence, submission, agent logs and evaluation outputs are read as they are. A
fresh verifier sandbox runs breakwater/verifier.py against a new authority, RUN_DIR/authority-reverify-N: the run's
sealed launch record with the new verifier sandbox, naming the authority it re-verifies (`reverify_of`). Its reports
go to RUN_DIR/reports-reverify-N; the earlier reports stay. Use it when verification itself failed (the judge's
provider, a verifier error) or after a verifier update, with the run's task contract, image and judge.
"""
import argparse
import json
import os
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parent)]
import run as launcher  # noqa: E402  (breakwater/apptainer/run.py)
from contract import CREDENTIAL_ENV, FRAMEWORK_REVISION, atomic_json, digest, read_json, require  # noqa: E402
import spec  # noqa: E402

SEALS = ('external_sha256', 'evaluation_sha256', 'submission_sha256', 'local_log_sha256', 'image_sha256')


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--task', type=Path, required=True, help='the task directory the run used')
    parser.add_argument('--image', type=Path, required=True, help='the task runtime SIF the run used')
    parser.add_argument('--asset-root', type=Path, required=True)
    parser.add_argument('--bindings', type=Path, help="moved asset sources, relative to --asset-root (default: the task's)")
    parser.add_argument('--judge-dir', type=Path, required=True, help='host directory holding the judge runner')
    parser.add_argument('--judge-key-env', default='FIREWORKS_API_KEY', help='variable holding the judge key')
    parser.add_argument('--judge-concurrency', type=int, choices=range(1, 65), metavar='1-64',
                        help='judge calls in flight (default 8), within the provider\'s rate limit')
    parser.add_argument('--scratch', type=Path, help="host directory for the sandbox's private /tmp, on local disk")
    args = parser.parse_args()
    held = {name: os.environ.pop(name) for name in {*CREDENTIAL_ENV, args.judge_key_env} if name in os.environ}
    require(held.get(args.judge_key_env), 'Missing the judge key')
    require((args.judge_dir / 'codex').is_file(), 'Missing the judge runner: install it with install_judge.py --dest DIR')
    run, task = Path(os.path.realpath(args.run_dir)), args.task.resolve()
    config = spec.load(task)
    previous_path = run / 'authority' / 'manifest.json'
    previous = read_json(previous_path)
    require(previous.get('backend') == 'apptainer' and previous.get('identity', {}).get('task') == config['task'],
            'Not an Apptainer run of this task')
    require(previous.get('framework_revision') == FRAMEWORK_REVISION,
            'This run was launched by another Breakwater revision: re-verify it with that revision')
    # The verifier's rule: a key the run's authority does not record holds its default on both sides.
    sealed = previous.get('pipeline') or {}
    require(all(sealed.get(key, spec.DEFAULTS.get(key)) == config.get(key, spec.DEFAULTS.get(key))
                for key in spec.SEALED_PIPELINE), "The task's contract differs from the one the run was launched with")
    require(launcher.file_sha256(args.image) == previous['image_sha256'], 'Use the image the run used')
    bindings = spec.bindings(config, args.asset_root, run, args.bindings)
    scratch = Path(os.path.realpath(args.scratch or launcher.SCRATCH))
    require(scratch.is_dir(), 'The scratch directory does not exist: ' + str(scratch))
    number = 1
    while any((run / f'{name}-reverify-{number}').exists() for name in ('authority', 'reports')):
        number += 1
    authority, reports = run / f'authority-reverify-{number}', run / f'reports-reverify-{number}'
    for folder in (authority, reports):
        folder.mkdir(mode=0o700)
    # The task's contract as it resolves now (its sealed pipeline was checked against the run's above).
    contract = run / f'contract-reverify-{number}.json'
    atomic_json(contract, config)
    contract.chmod(0o444)
    # What the run sealed beside its stages: whether it was a smoke run (the task scores it with its smoke settings)
    # and the digest of the worker's home it snapshotted, which the verifier checks against the snapshot itself.
    home = run / 'submission-home' if previous.get('home_sha256') else None
    require(bool(home) == bool(config['snapshot_worker_home']) and (home is None or home.is_dir()),
            "The run's home snapshot does not match the task contract")
    carried = {key: previous[key] for key in ('smoke', 'home_sha256', 'recording', 'task_metadata', 'task_plan_sha256',
                                              'host_hooks') if key in previous}
    code, authority_sha = launcher.verify(
        run=run, task=task, config=config, contract=contract, image=args.image,
        image_env=launcher.image_environment(args.image),
        bindings=bindings, judge_dir=args.judge_dir, judge_key=held[args.judge_key_env], scratch=scratch,
        identity=previous['identity'], stages={key: value for key, value in previous['stages'].items() if key != 'verifier'},
        seals={key: previous[key] for key in SEALS}, authority=authority, reports=reports,
        logs=run / f'verifier-reverify-{number}', extra_roots=(run / 'authority', run / 'reports'), home=home,
        extra={'reverify_of': digest(previous_path), **carried}, judge_concurrency=args.judge_concurrency)
    print(json.dumps({'task': config['task'], 'run_id': previous['identity']['run_id'], 'reverify_of': digest(previous_path),
                      **({'smoke': True} if carried.get('smoke') else {}),
                      'authority_sha256': authority_sha, 'verifier_exit_code': code, 'reports': str(reports)}))
    return code


if __name__ == '__main__':
    sys.exit(main())
