"""Trusted evaluation coordinator in a fresh container; never receives external logs or judge keys.

It runs the task's trusted code (tests/breakwater_task.py, ``evaluate(run)``; task_api.Evaluation) with the task's
private inputs. Submitted code never runs here: under Docker it runs as the worker account, whose processes a private
parent directory keeps away from the tests, labels and raw outputs; under Apptainer it runs in a separate sandbox that
has none of them.
"""
import os
from pathlib import Path
import stat
import sys

sys.path[:0] = ['/opt/breakwater', '/tests', '/opt/breakwater-drivers']
from contract import atomic_json, read_json, require  # noqa: E402
import task_api  # noqa: E402

RAW = Path('/evaluation/raw')
AUTHOR_GID, SHARED_GID = 1100, 1200


def worker_groups():
    """The workspace group plus the device-node groups, exactly as authoring grants them."""
    groups = {SHARED_GID}
    for pattern in ('nvidia*', 'nvidia-caps/*', 'dri/*', 'infiniband/*'):
        for device in Path('/dev').glob(pattern):
            info = device.stat()
            if stat.S_ISCHR(info.st_mode) and info.st_gid not in (0, AUTHOR_GID):
                groups.add(info.st_gid)
    return sorted(groups)


def main():
    from workers import apptainer
    require((apptainer() or os.geteuid() == 0) and os.environ.get('BREAKWATER_STAGE') == 'intermediate',
            'Trusted evaluation coordinator only')
    config = read_json(task_api.CONTRACT)
    RAW.mkdir()
    if apptainer():
        # Submitted code runs in a separate sandbox that has none of the private paths, so neither root-only
        # permissions nor a worker account are needed; the launcher binds the tests and private inputs directly.
        require(not Path('/tests').is_symlink(), 'Evaluation image has an unexpected private-data path: /tests')
        evaluate(config)
        return
    # A private parent directory, not file permissions supplied by the agent, protects
    # trusted tests, labels, raw outputs and the coordinator from submitted processes.
    Path('/private').chmod(0o700)
    Path('/evaluation').chmod(0o700)
    path = Path('/tests')  # the task's tests are mounted under the private parent
    if path.is_symlink():
        path.unlink()
    if path.is_dir() and not any(path.iterdir()):
        path.rmdir()
    require(not path.exists(), 'Evaluation image has an unexpected private-data path: /tests')
    path.symlink_to('/private/tests', target_is_directory=True)
    runtime = Path('/run/breakwater')
    runtime.mkdir(exist_ok=True, parents=True)
    env = task_api.worker_environment(config)
    if os.environ.get('LD_LIBRARY_PATH'):
        env['LD_LIBRARY_PATH'] = os.environ['LD_LIBRARY_PATH']
    (runtime / 'worker.env').write_bytes(b'\0'.join((key + '=' + value).encode() for key, value in env.items()))
    (runtime / 'worker-groups').write_text(' '.join(map(str, worker_groups())) + '\n')
    (runtime / 'finished').write_text('Fresh intermediate container; no author processes exist here.\n')
    # The coordinator already runs under the observer's `strace -f` (observed.py barrier), so
    # every worker it starts is traced from fork. A nested run_worker capture barrier would
    # SIGSTOP an already-traced process that the observer never attaches to (it only attaches
    # to untraced stopped processes) and would hang until the evaluation deadline. Workers here
    # are therefore observed by inheritance, and run_worker drops privileges without a barrier.
    Path('/tmp/worker').mkdir(mode=0o700, exist_ok=True)
    os.chown('/tmp/worker', 1101, 1200)
    evaluate(config)


def evaluate(config):
    task_api.evaluate(config)
    atomic_json(Path('/evaluation/complete.json'), {'complete': True})
    # Publish readable, immutable raw files only after the entire evaluation has ended.
    for directory, dirs, files in os.walk('/evaluation'):
        for name in files:
            (Path(directory) / name).chmod(0o444)
        Path(directory).chmod(0o555)


if __name__ == '__main__':
    main()
