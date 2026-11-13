"""Root-side worker process and file control inside an intermediate container (needs CAP_KILL).

Only submitted code runs as UID 1101 in an intermediate container, so "every process
with UID 1101" is exactly the set a trusted coordinator must stop between scenarios,
cells and before it finishes. Signalling by UID also reaches detached grandchildren
that left the original process group or session.

The file helpers run only while no worker process exists: they install fresh
worker-owned copies of sealed trees and remove what the worker left in shared
temporary directories, and they never follow a link the submission recorded.
"""
import os
from pathlib import Path, PurePosixPath
import shutil
import signal
import stat
import time

WORKER_UID = 1101
SHARED_GID = 1200
SEAL_MANIFEST = 'manifest.json'


class InvalidSubmission(ValueError):
    """The sealed submission breaks the task's submission contract (scored, never run)."""


def worker_pids(uid=WORKER_UID):
    found = []
    for status in Path('/proc').glob('[0-9]*/status'):
        try:
            lines = status.read_text().splitlines()
        except (FileNotFoundError, ProcessLookupError, PermissionError):
            continue
        uids = next((line.split()[1:] for line in lines if line.startswith('Uid:')), [])
        state = next((line for line in lines if line.startswith('State:')), '')
        if str(uid) in uids and 'Z' not in state:
            found.append(int(status.parent.name))
    return found


def kill_workers(grace=0.0, timeout=60.0, uid=WORKER_UID):
    """Optionally SIGTERM and wait ``grace`` seconds, then SIGKILL until none remain."""
    stages = ([(signal.SIGTERM, grace)] if grace > 0 else []) + [(signal.SIGKILL, timeout)]
    for sig, wait in stages:
        deadline = time.monotonic() + wait
        while time.monotonic() < deadline:
            pids = worker_pids(uid)
            if not pids:
                return
            for pid in pids:
                try:
                    os.kill(pid, sig)
                except ProcessLookupError:
                    pass
            time.sleep(.1)
    if worker_pids(uid):
        raise RuntimeError('Submitted worker processes survived teardown')


def clear(target):
    """Empty ``target`` (created if missing) without following any link inside it."""
    target = Path(target)
    if target.is_symlink():
        target.unlink()
    target.mkdir(parents=True, exist_ok=True)
    for child in list(target.iterdir()):
        if child.is_dir() and not child.is_symlink():
            shutil.rmtree(child)  # fd-based on Linux: never follows links
        else:
            child.unlink()


def real_directory(root, relative):
    """``root/relative`` where every component below ``root`` is a real directory (never a link)."""
    path = Path(root)
    for part in PurePosixPath(relative).parts:
        if part in ('', '.', '..'):
            raise InvalidSubmission('Unsafe path in the sealed submission')
        path = path / part
        if not stat.S_ISDIR(path.lstat().st_mode):
            raise InvalidSubmission('A sealed submission path passes through a link or file')
    return path


def install_tree(pairs, allowed=(), uid=WORKER_UID, gid=SHARED_GID):
    """Replace each target with a fresh copy of its sealed source, owned by the worker.

    ``pairs`` is a list of (sealed source, target). Regular files and directories are
    copied (the host's seal manifest at the source root is not part of the submission).
    Symbolic links the host recorded in that manifest are recreated with their recorded
    text, never followed. Every link must resolve inside the installed targets or the
    read-only ``allowed`` roots; anything else is an invalid submission.
    """
    import json
    links, created = [], []
    for source, target in pairs:
        source, target = Path(source), Path(target)
        clear(target)
        for directory, dirs, files in os.walk(source):
            relative = Path(directory).relative_to(source)
            for name in dirs:
                (target / relative / name).mkdir()
                created.append(target / relative / name)
            for name in files:
                if relative == Path('.') and name == SEAL_MANIFEST:
                    continue
                path = Path(directory) / name
                if path.is_symlink() or not path.is_file():
                    raise InvalidSubmission('Sealed submission contains a non-regular file')
                shutil.copyfile(path, target / relative / name)
                (target / relative / name).chmod(0o775 if path.stat().st_mode & 0o111 else 0o664)
                created.append(target / relative / name)
        manifest = source / SEAL_MANIFEST
        recorded = json.loads(manifest.read_text()).get('symlinks', {}) if manifest.is_file() else {}
        for relative, text in sorted(recorded.items()):
            path = PurePosixPath(relative) if isinstance(relative, str) else None
            if (path is None or path.is_absolute() or not path.parts or any(part in ('', '.', '..') for part in path.parts)
                    or not isinstance(text, str) or not text or '\0' in text):
                raise InvalidSubmission('Invalid recorded symbolic link')
            parent = real_directory(target, str(path.parent)) if path.parent != PurePosixPath('.') else target
            link = parent / path.name
            if link.is_symlink() or link.exists():
                raise InvalidSubmission('A recorded symbolic link duplicates a submitted path')
            os.symlink(text, link)
            links.append(link)
    roots = [Path(target) for _, target in pairs] + [Path(path) for path in allowed]
    for link in links:
        resolved = Path(os.path.realpath(link))
        if not any(resolved == root or resolved.is_relative_to(root) for root in roots):
            raise InvalidSubmission(f'Symbolic link {link} resolves outside the submission: {os.readlink(link)}')
    for _, target in pairs:
        target = Path(target)
        os.chown(target, uid, gid, follow_symlinks=False)
        target.chmod(0o2775)
    for path in created:
        os.chown(path, uid, gid, follow_symlinks=False)
        if path.is_dir():
            path.chmod(0o2775)
    for link in links:
        os.chown(link, uid, gid, follow_symlinks=False)
    return {'files': sum(1 for path in created if not path.is_dir()), 'symlinks': len(links)}


def wipe_worker_files(roots=('/tmp', '/var/tmp', '/dev/shm', '/run/lock', '/dev/mqueue'), uid=WORKER_UID):
    """Remove every entry the worker owns under shared temporary roots; no worker may be running."""
    removed = 0
    for top in roots:
        top = Path(top)
        if top.is_symlink() or not top.is_dir():
            continue
        for directory, dirs, files in os.walk(top, followlinks=False):
            for name in list(dirs) + files:
                path = Path(directory) / name
                try:
                    info = path.lstat()
                except FileNotFoundError:
                    continue
                if info.st_uid != uid:
                    continue
                if stat.S_ISDIR(info.st_mode):
                    shutil.rmtree(path)
                    dirs.remove(name)
                else:
                    path.unlink()
                removed += 1
    return removed
