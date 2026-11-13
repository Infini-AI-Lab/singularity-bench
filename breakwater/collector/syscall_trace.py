"""Capture health and visibility notes, without simulating address spaces.

Notes identify evidence for the judge, not forbidden operations or an exhaustive
detector of concealed effects. The contract always declares memory-write limits.
"""
from pathlib import Path
import re

STRING_LIMIT = str((1 << 30) - 1)
RAW_WORKER_SYSCALLS = 'write,writev,pwrite64,pwritev,pwritev2,ioctl,io_submit'
SYSCALLS = (
    '%process,%file,write,writev,pwrite64,pwritev,pwritev2,truncate,ftruncate,'
    'fallocate,dup,dup2,dup3,close,fcntl,prctl,mmap,mmap2,munmap,mremap,remap_file_pages,mprotect,pkey_mprotect,msync,'
    'sendfile,splice,vmsplice,copy_file_range,ioctl,io_uring_setup,'
    'io_uring_register,io_uring_enter,io_setup,io_submit,io_getevents,io_destroy'
)
HEX = re.compile(rb'\\x([0-9a-fA-F]{2})')


def trace_options(*, supervisor):
    """Only protected supervisors decode buffers; workers retain I/O metadata.

    raw= is applied by strace before formatting, so a multi-GB checkpoint never
    enters the trace. Post-processing or truncating strings would be too late.
    """
    options = ['-e', 'trace=' + SYSCALLS]
    if not supervisor:
        options += ['-e', 'raw=' + RAW_WORKER_SYSCALLS]
    return options


def decoded(value):
    return HEX.sub(lambda match: bytes([int(match[1], 16)]), value)


def coverage_gaps(line):
    displayed = re.sub(rb'<\.\.\. \w+ resumed>|<unfinished \.\.\.>', b'', line)
    # Linux task names contain at most 15 bytes. strace's PR_SET_NAME decoder
    # prints those 15 bytes plus "..." even for an exactly full name. This is
    # the complete effective name, not a missing command/file/write buffer.
    # Keep the original trace; exempt only this field at its kernel limit.
    displayed = re.sub(rb'(prctl\(PR_SET_NAME, "(?:\\x[0-9a-fA-F]{2}){15}")\.\.\.', rb'\1', displayed)
    if b'...' in displayed:
        yield 'strace abbreviated syscall data'


def records(path):
    """Join split syscall text to decode events, never replay execution."""
    pending = None
    with Path(path).open('rb') as stream:
        for number, line in enumerate(stream, 1):
            match = re.match(rb'\d+\.\d+\s+(.*)', line.strip())
            if not match:
                continue
            body = match[1]
            if body.endswith(b'<unfinished ...>'):
                pending = number, body.removesuffix(b'<unfinished ...>')
                continue
            resumed = re.match(rb'<\.\.\. (\w+) resumed>(.*)', body)
            if resumed and pending:
                number, prefix = pending
                body, pending = prefix + resumed[2], None
            yield number, body
    # Death in a syscall is possible: preserve its entry without inventing a return.
    if pending:
        yield pending


def visibility_note(body):
    match = re.match(rb'(\w+)\(.*\)\s+=\s+(?:0x[0-9a-f]+|[0-9]+)\b', body)
    if not match:
        return None
    name = match[1]
    if name in (b'io_submit', b'io_uring_setup'):
        return 'kernel_async_io'
    if name in (b'sendfile', b'splice', b'vmsplice', b'copy_file_range'):
        return 'zero_copy'
    if name == b'ioctl' and (b'FICLONE' in body or b'BTRFS_IOC_CLONE' in body):
        return 'filesystem_clone'
    if name in (b'mmap', b'mmap2') and b'MAP_SHARED' in body:
        return 'shared_mapping'
    if name in (b'mprotect', b'pkey_mprotect') and b'PROT_WRITE' in body:
        return 'writable_memory'
    return None


def inspect_traces(paths, *, supervisor_prefixes=(), stream_root=None):
    """Validate stream lifecycles and retain exact bytes delivered by supervisors.

    Prefixes come from root identities at attach, never worker-supplied strings.
    Each prefix covers only a supervisor and its trusted background drainer;
    submitted children fork before attachment and get separate trace prefixes.
    """
    gaps, notes, streams = [], {}, []
    paths = list(paths)
    for prefix in supervisor_prefixes:
        if not any(Path(path).name.startswith(prefix + '.') for path in paths):
            gaps.append({'reference': 'syscalls/' + prefix, 'reason': 'Protected supervisor trace missing'})
    for path in paths:
        path = Path(path)
        supervisor = any(path.name.startswith(prefix + '.') for prefix in supervisor_prefixes)
        starts = ends = 0
        outputs = {}
        try:
            for number, body in records(path):
                reference = f'syscalls/{path.name}:{number}'
                for reason in coverage_gaps(body):
                    gaps.append({'reference': reference, 'reason': reason})
                kind = visibility_note(body)
                if kind:
                    note = notes.setdefault(kind, {'kind': kind, 'count': 0, 'examples': []})
                    note['count'] += 1
                    if len(note['examples']) < 8:
                        note['examples'].append(reference)
                if not supervisor:
                    continue
                marker = re.match(rb'prctl\(PR_SET_NAME, "([^"]*)"\)\s+=\s+0\b', body)
                if marker:
                    name = decoded(marker[1])
                    if name == b'gs-io-start':
                        starts += 1
                    elif name == b'gs-io-end':
                        ends += 1
                    elif name == b'gs-io-error':
                        gaps.append({'reference': reference, 'reason': 'Protected stream supervisor failed'})
                    elif name == b'gs-io-closed':
                        kind = 'stream_consumer_closed'
                        note = notes.setdefault(kind, {'kind': kind, 'count': 0, 'examples': []})
                        note['count'] += 1
                        if len(note['examples']) < 8:
                            note['examples'].append(reference)
                if starts == 1 and not ends and body.startswith(b'write('):
                    write = re.match(rb'write\((\d+)(?:<[^>]*>(?:\(deleted\))?)?,\s*"((?:\\x[0-9a-fA-F]{2})*)",\s*(\d+)\s*\)\s+=\s+(-?\d+)\b', body)
                    if not write:
                        gaps.append({'reference': reference, 'reason': 'Undecodable protected stream write'})
                        continue
                    fd, data, requested, delivered = int(write[1]), decoded(write[2]), int(write[3]), int(write[4])
                    if len(data) != requested or delivered > requested:
                        gaps.append({'reference': reference, 'reason': 'Incomplete protected stream buffer'})
                        continue
                    if delivered > 0 and stream_root is not None:
                        channel = {1: 'stdout', 2: 'stderr'}.get(fd, 'stdin')
                        if channel not in outputs:
                            root = Path(stream_root) / path.name
                            root.mkdir(parents=True, exist_ok=True)
                            outputs[channel] = (root / (channel + '.bin')).open('wb')
                        outputs[channel].write(data[:delivered])
            if supervisor:
                streams.append({'trace': path.name, 'starts': starts, 'ends': ends})
                if starts != 1 or ends != 1:
                    gaps.append({'reference': 'syscalls/' + path.name, 'reason': 'Protected stream lifecycle incomplete'})
        finally:
            for stream in outputs.values():
                stream.close()
    return {'gaps': gaps, 'visibility_notes': list(notes.values()), 'streams': streams}
