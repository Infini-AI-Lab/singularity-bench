"""Hourly source inventories and a single final processed-dataset archive.

All reads are relative to the pinned target root, with no-follow directory FDs.
The worker cannot redirect a source symlink into the observer's host filesystem.
Dataset selection is a submission claim, not proof of training provenance.
"""
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import stat
import time
import uuid


SOURCE_SUFFIXES = {
    '.py', '.sh', '.bash', '.c', '.cc', '.cpp', '.cxx', '.h', '.hpp', '.cu', '.cuh',
    '.rs', '.go', '.js', '.jsx', '.ts', '.tsx', '.java', '.scala', '.jl', '.r',
    '.toml', '.yaml', '.yml', '.json', '.ini', '.cfg', '.conf', '.md', '.patch',
    '.diff', '.cmake', '.mk', '.sql', '.proto', '.ipynb',
}
SKIP_DIRS = {
    '.git', '.cache', '__pycache__', 'node_modules', '.venv', 'venv', '.triton',
    'torch_extensions', 'checkpoints', 'intermediate_checkpoints',
}
MAX_SOURCE_BYTES = 16 * 1024 * 1024


def absolute_parts(path):
    value = PurePosixPath(path)
    if not value.is_absolute() or '..' in value.parts or str(value) == '/':
        raise ValueError('Expected a non-root absolute container path: ' + str(path))
    return value.parts[1:]


def open_target(root_fd, path, *, directory=False):
    parts = absolute_parts(path)
    fd = os.dup(root_fd)
    try:
        for number, part in enumerate(parts):
            flags = os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK
            if number < len(parts) - 1 or directory:
                flags |= os.O_DIRECTORY
            child = os.open(part, flags, dir_fd=fd)
            os.close(fd)
            fd = child
        return fd
    except BaseException:
        os.close(fd)
        raise


def atomic_json(path, value):
    temporary = path.with_name(path.name + '.tmp')
    with temporary.open('w') as stream:
        json.dump(value, stream, sort_keys=True)
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def source_file(name):
    return (Path(name).suffix.lower() in SOURCE_SUFFIXES
            or '.' not in name or name.startswith(('Dockerfile', 'Makefile', 'requirements'))
            or name in {'.gitignore', '.dockerignore', 'CMakeLists.txt'})


class Snapshots:
    def __init__(self, root_fd, output, roots, *, excludes=(), immutable_roots=(), interval_sec=3600,
                 dataset_manifest=None, dataset_roots=(), clock=time.monotonic):
        if not 0 < interval_sec < float('inf'):
            raise ValueError('Snapshot interval must be finite and positive')
        self.root_fd = root_fd
        self.output = Path(output)
        self.roots = list(roots)
        self.dataset_roots = list(dataset_roots) or self.roots
        self.immutable_roots = list(immutable_roots)
        self.excludes = [*excludes, *self.immutable_roots]
        for path in [*self.roots, *self.excludes, *self.dataset_roots]:
            absolute_parts(path)
        if dataset_manifest:
            absolute_parts(dataset_manifest)
        self.dataset_manifest = dataset_manifest
        self.interval_sec = interval_sec
        self.clock = clock
        self.deadline = clock() + interval_sec
        self.sequence = 0
        self.output.mkdir(mode=0o700, parents=True, exist_ok=True)
        (self.output / 'blobs').mkdir(mode=0o700, exist_ok=True)

    def excluded(self, path):
        return any(path == root or path.startswith(root.rstrip('/') + '/')
                   for root in self.excludes)

    def due(self):
        return self.clock() >= self.deadline and not (self.output / 'final.json').exists()

    def store_file(self, fd, *, max_bytes=None):
        before = os.fstat(fd)
        if not stat.S_ISREG(before.st_mode):
            raise ValueError('Not a regular file')
        if max_bytes is not None and before.st_size > max_bytes:
            raise ValueError('Source exceeds source-size policy')
        temporary = self.output / ('blob-' + uuid.uuid4().hex + '.tmp')
        digest = hashlib.sha256()
        size = 0
        try:
            with temporary.open('xb') as output:
                while data := os.read(fd, 1024 * 1024):
                    size += len(data)
                    if max_bytes is not None and size > max_bytes:
                        raise ValueError('Source exceeds source-size policy')
                    if max_bytes is not None and b'\0' in data:
                        raise ValueError('Binary file excluded from source snapshots')
                    digest.update(data)
                    output.write(data)
                output.flush()
                os.fsync(output.fileno())
            after = os.fstat(fd)
            if (before.st_size, before.st_mtime_ns, before.st_ctime_ns) != (
                    after.st_size, after.st_mtime_ns, after.st_ctime_ns) or size != after.st_size:
                raise ValueError('File changed during capture')
            sha = digest.hexdigest()
            try:
                os.link(temporary, self.output / 'blobs' / sha)
            except FileExistsError:
                pass
            return {'path': 'blobs/' + sha, 'sha256': sha, 'bytes': size,
                    'mode': stat.S_IMODE(after.st_mode)}
        finally:
            temporary.unlink(missing_ok=True)

    def capture(self, *, final=False):
        final_path = self.output / 'final.json'
        if final_path.exists():
            return json.loads(final_path.read_text())
        files, omitted = {}, []
        for root in self.roots:
            if self.excluded(root):
                continue
            try:
                root_fd = open_target(self.root_fd, root, directory=True)
            except FileNotFoundError:
                omitted.append({'path': root, 'reason': 'source root absent'})
                continue
            try:
                for parent, dirs, names, directory_fd in os.fwalk('.', dir_fd=root_fd, follow_symlinks=False):
                    prefix = root.rstrip('/') + ('' if parent == '.' else '/' + parent.removeprefix('./'))
                    for name in list(dirs):
                        path = prefix + '/' + name
                        try:
                            info = os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
                            if stat.S_ISLNK(info.st_mode):
                                files[path] = {'symlink': os.readlink(name, dir_fd=directory_fd)}
                                dirs.remove(name)
                        except OSError:
                            if final:
                                raise
                            dirs.remove(name)
                    dirs[:] = [name for name in dirs if name not in SKIP_DIRS
                               and not self.excluded(prefix + '/' + name)]
                    for name in names:
                        path = prefix + '/' + name
                        try:
                            info = os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
                        except FileNotFoundError:
                            if final:
                                raise
                            continue
                        if stat.S_ISLNK(info.st_mode):
                            try:
                                files[path] = {'symlink': os.readlink(name, dir_fd=directory_fd)}
                            except OSError:
                                if final:
                                    raise
                            continue
                        if not stat.S_ISREG(info.st_mode):
                            omitted.append({'path': path, 'reason': 'Special file excluded from source snapshot'})
                            continue
                        if self.excluded(path) or not source_file(name):
                            continue
                        try:
                            fd = os.open(name, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=directory_fd)
                            try:
                                files[path] = self.store_file(fd, max_bytes=MAX_SOURCE_BYTES)
                            finally:
                                os.close(fd)
                        except (OSError, ValueError) as error:
                            # A mutable hourly inventory can race a rename. Final
                            # capture is after drain; unexpected failures are fatal.
                            reason = str(error)
                            policy = 'source-size policy' in reason or 'Binary file excluded' in reason
                            if final and not policy:
                                raise
                            omitted.append({'path': path, 'reason': reason})
            finally:
                os.close(root_fd)
        report = {'schema_version': 1, 'sequence': self.sequence, 'final': final,
                  'interval_sec': self.interval_sec, 'files': files, 'omitted': omitted}
        if final and self.dataset_manifest:
            report['training_dataset'] = self.capture_dataset()
        atomic_json(final_path if final else self.output / f'{self.sequence:06d}.json', report)
        self.sequence += 1
        self.deadline = self.clock() + self.interval_sec
        return report

    def capture_dataset(self):
        fd = open_target(self.root_fd, self.dataset_manifest)
        try:
            data = os.read(fd, 1024 * 1024 + 1)
        finally:
            os.close(fd)
        if len(data) > 1024 * 1024:
            raise ValueError('Training dataset manifest exceeds 1 MiB')
        manifest = json.loads(data)
        if (not isinstance(manifest, dict) or manifest.get('schema_version') != 1
                or not isinstance(manifest.get('files'), list)):
            raise ValueError('Training dataset manifest requires schema_version=1 and files')
        if not manifest['files'] and (manifest.get('mode') not in {'provided_inputs', 'streaming'}
                                     or not isinstance(manifest.get('reason'), str) or not manifest['reason'].strip()):
            raise ValueError('An empty dataset requires an explicit streaming/provided-inputs explanation')
        records = {}
        for item in manifest['files']:
            path = item.get('path') if isinstance(item, dict) else None
            if not isinstance(path, str):
                raise ValueError('Dataset entry needs an absolute file path')
            absolute_parts(path)
            if (not any(path.startswith(root.rstrip('/') + '/') for root in self.dataset_roots)
                    or any(path == root or path.startswith(root.rstrip('/') + '/')
                           for root in self.immutable_roots)):
                raise ValueError('Dataset must be agent-produced, inside a dataset root and outside immutable mounts')
            if path in records:
                raise ValueError('Duplicate training dataset path')
            fd = open_target(self.root_fd, path)
            try:
                record = self.store_file(fd)
            finally:
                os.close(fd)
            if item.get('sha256') and item['sha256'] != record['sha256']:
                raise ValueError('Training dataset digest mismatch')
            records[path] = record
        return {'manifest_path': self.dataset_manifest, 'selection': manifest,
                'selection_trust': 'agent-declared; verify provenance against code and terminal evidence',
                'files': records}
