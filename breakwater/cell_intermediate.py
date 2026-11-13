"""Trusted evaluation coordinator for ONE downstream grid cell (root, fresh container).

The fresh container starts from the pristine task image. Only the sealed workspace
overlay crosses from the cell's authoring container -- files the harness changed or
added inside the task's declared workspace, plus deletions -- never /tmp, caches,
site-packages, background processes or the harness itself. Private tests, expectations
and the oracle portfolio sit under the root-only /private parent. Submitted kernels and
vLLM code execute only as the worker through run_worker.py (each worker behind its own
external-observer barrier) using fixed public runners; raw outputs are written by root.
"""
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

# Trusted code lives under the root-only /private parent; the worker sees only /opt/breakwater-worker.
sys.path[:0] = ['/private/breakwater', '/private/breakwater/downstream', '/opt/breakwater-worker']
from contract import atomic_json, read_json, require
from registry import KERNELS, PRS, WORKDIRS, require_task

RAW = Path('/evaluation/raw')
SUBMISSION = Path('/submission')
DROP = ('BREAKWATER_', 'FIREWORKS', 'OPENAI', 'ANTHROPIC', 'UPSTREAM', 'HARBOR_', 'GATEWAY_')


def apply_overlay(workdir):
    """Reconstruct the final workspace: pristine image workspace + sealed changes."""
    changes = read_json(SUBMISSION / 'changes.json')
    require(set(changes) == {'workdir', 'added', 'modified', 'deleted', 'executable'}, 'Malformed overlay manifest')
    require(changes['workdir'] == workdir, 'Overlay belongs to another workspace')
    root = Path(workdir)
    for relative in changes['deleted']:
        path = root / relative
        require('..' not in Path(relative).parts and not Path(relative).is_absolute(), 'Unsafe overlay path')
        if path.is_symlink() or path.is_file():
            path.unlink()
    for relative in changes['added'] + changes['modified']:
        require('..' not in Path(relative).parts and not Path(relative).is_absolute(), 'Unsafe overlay path')
        source = SUBMISSION / 'files' / relative
        require(source.is_file() and not source.is_symlink(), 'Overlay file missing from the sealed submission')
        target = root / relative
        require(not any(path.is_symlink() for path in [target, *target.parents] if path.is_relative_to(root)),
                'Overlay target crosses a link')
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists():
            target.unlink()
        shutil.copyfile(source, target)
        target.chmod(0o755 if relative in changes['executable'] else 0o644)
    return changes


def worker_environment(workdir):
    env = {key: value for key, value in os.environ.items() if not key.startswith(DROP)}
    env.update(HOME='/tmp/worker', TMPDIR='/tmp/worker', BREAKWATER_WORKDIR=workdir, PYTHONDONTWRITEBYTECODE='1',
               PYTHONUNBUFFERED='1', TRITON_CACHE_DIR='/tmp/worker/.triton',
               TORCH_EXTENSIONS_DIR='/tmp/worker/torch_extensions', HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1')
    if Path('/opt/breakwater-hf-cache').is_dir():
        env['HF_HOME'] = '/opt/breakwater-hf-cache'
    env['PATH'] = ':'.join(part for part in env.get('PATH', '/usr/local/bin:/usr/bin:/bin').split(':')
                           if part and not part.startswith(('/root', '/home/')))
    return env


def run_pr(task, workdir):
    import importlib
    adapter = importlib.import_module(PRS[task]['adapter'])
    cases, private = adapter.prepare_cases()
    exchange = Path('/run/breakwater-cell/pr')
    exchange.mkdir(parents=True, mode=0o755)
    (exchange / 'cases.json').write_text(json.dumps(cases))
    (exchange / 'cases.json').chmod(0o644)
    output = exchange / 'out'
    output.mkdir(mode=0o755)
    os.chown(output, 1101, 1200)
    from workers import kill_workers
    with (RAW / 'worker.stdout').open('wb') as out, (RAW / 'worker.stderr').open('wb') as err:
        try:
            result = subprocess.run(['python3', '-I', '-B', '/opt/breakwater-runtime/run_worker.py', 'python3', '-I', '-B',
                                     f"/opt/breakwater-worker/{PRS[task]['adapter']}_worker.py",
                                     str(exchange / 'cases.json'), str(output / 'predictions.jsonl')],
                                    stdin=subprocess.DEVNULL, stdout=out, stderr=err, timeout=3000)
        finally:
            kill_workers()  # every submitted descendant, including detached ones
    predictions = output / 'predictions.jsonl'
    require(predictions.is_file() and not predictions.is_symlink(), 'Public PR runner produced no raw rows')
    shutil.copyfile(predictions, RAW / 'predictions.jsonl')
    atomic_json(RAW / 'cases.json', cases)
    atomic_json(RAW / 'private.json', private)
    atomic_json(RAW / 'execution.json', {'exit_status': result.returncode})


def main():
    require(os.geteuid() == 0 and os.environ.get('BREAKWATER_STAGE') == 'intermediate', 'Trusted cell coordinator only')
    cell = read_json('/private/cell/cell.json')
    task = require_task(cell['task'])
    workdir = WORKDIRS[task]
    require(cell['workdir'] == workdir, 'Cell workspace differs from the trusted registry')
    RAW.mkdir(parents=True)
    Path('/private').chmod(0o700)
    Path('/evaluation').chmod(0o700)
    alias = Path('/tests')
    if alias.is_dir() and not alias.is_symlink() and not any(alias.iterdir()):
        alias.rmdir()
    require(not alias.exists() or alias.is_symlink(), 'Cell image has an unexpected /tests path')
    if alias.is_symlink():
        alias.unlink()
    alias.symlink_to('/private/tests', target_is_directory=True)
    changes = apply_overlay(workdir)
    runtime = Path('/run/breakwater')
    runtime.mkdir(parents=True, exist_ok=True)
    env = worker_environment(workdir)
    (runtime / 'worker.env').write_bytes(b'\0'.join((key + '=' + value).encode() for key, value in env.items()))
    (runtime / 'worker-groups').write_text('1200')
    (runtime / 'finished').write_text('Fresh cell evaluation container; no harness processes exist here.\n')
    # The coordinator already runs under the observer's `strace -f` (observed.py barrier), so
    # every worker it starts is traced from fork. A nested run_worker capture barrier would
    # SIGSTOP an already-traced process that the observer never attaches to (it only attaches
    # to untraced stopped processes) and would hang until the evaluation deadline. Workers here
    # are therefore observed by inheritance, and run_worker drops privileges without a barrier.
    Path('/tmp/worker').mkdir(mode=0o700, exist_ok=True)
    os.chown('/tmp/worker', 1101, 1200)
    Path('/run/breakwater-cell').mkdir(mode=0o755, exist_ok=True)
    if task in KERNELS:
        from kernel_parent import collect
        os.environ['TMPDIR'] = '/private/parent-tmp'
        Path('/private/parent-tmp').mkdir(mode=0o700, exist_ok=True)
        collect(task, Path('/private/tests'), RAW, Path('/run/breakwater-cell/kernel'))
    else:
        run_pr(task, workdir)
    atomic_json(Path('/evaluation/complete.json'), {'complete': True, 'kind': 'harness_cell', 'task': task,
                                                    'overlay': {key: len(value) for key, value in changes.items()
                                                                if isinstance(value, list)}})
    for directory, dirs, files in os.walk('/evaluation'):
        for name in files:
            (Path(directory) / name).chmod(0o444)
        Path(directory).chmod(0o555)


if __name__ == '__main__':
    main()
