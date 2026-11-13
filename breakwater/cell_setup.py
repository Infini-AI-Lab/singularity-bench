"""Trusted root setup inside a downstream cell's authoring container, before the harness runs.

Grants the worker its workspace, writes the minimal worker environment consumed by
run_worker.py and enables the external-observer barrier. No provider key exists in this
container: the only model route is the internal driver gateway named in BREAKWATER_DRIVER_BASE.
"""
import argparse
import os
from pathlib import Path
import stat

WORKER, SHARED = 1101, 1200
DROP = ('BREAKWATER_STAGE', 'FIREWORKS', 'OPENAI', 'ANTHROPIC', 'UPSTREAM', 'HARBOR_', 'GATEWAY_')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--workdir', required=True)
    args = parser.parse_args()
    if os.geteuid() != 0:
        raise SystemExit('cell setup must run as root')
    workdir = Path(args.workdir)
    if not workdir.is_dir() or workdir.is_symlink():
        raise SystemExit('Cell workspace is missing')
    # Grant the declared workspace only; never follow links out of it.
    for directory, dirs, files in os.walk(workdir, followlinks=False):
        for path in [Path(directory), *(Path(directory) / name for name in files + dirs)]:
            info = os.lstat(path)
            if stat.S_ISLNK(info.st_mode):
                continue
            os.chown(path, WORKER, SHARED, follow_symlinks=False)
            path.chmod((info.st_mode & 0o7777) | (0o770 if stat.S_ISDIR(info.st_mode) else 0o660))
    env = {key: value for key, value in os.environ.items() if not key.startswith(DROP)}
    env.update(HOME='/tmp/worker', TMPDIR='/tmp/worker', BREAKWATER_WORKDIR=str(workdir), PYTHONDONTWRITEBYTECODE='1',
               PYTHONUNBUFFERED='1', TRITON_CACHE_DIR='/tmp/worker/.triton',
               TORCH_EXTENSIONS_DIR='/tmp/worker/torch_extensions', HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1',
               HARNESS_RECORD='/run/breakwater-cell/record/eval_record.json')
    if Path('/opt/breakwater-hf-cache').is_dir():
        env['HF_HOME'] = '/opt/breakwater-hf-cache'
    env['PATH'] = ':'.join(part for part in env.get('PATH', '/usr/local/bin:/usr/bin:/bin').split(':')
                           if part and not part.startswith(('/root', '/home/')))
    runtime = Path('/run/breakwater')
    runtime.mkdir(parents=True, exist_ok=True)
    (runtime / 'worker.env').write_bytes(b'\0'.join((key + '=' + value).encode() for key, value in env.items()))
    (runtime / 'worker-groups').write_text(str(SHARED))
    (runtime / 'finished').write_text('Downstream cell: no author CLI is installed in this container.\n')
    (runtime / 'external-capture-enabled').touch()
    for folder in (Path('/tmp/worker'), Path('/run/breakwater-cell/record')):
        folder.mkdir(parents=True, exist_ok=True)
        os.chown(folder, WORKER, SHARED)
        folder.chmod(0o770)


if __name__ == '__main__':
    main()
