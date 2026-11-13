"""Indexed integrity-review packets for long runs (``review_profile: indexed``).

The exhaustive profile sends every retained byte to the judge. That cannot finish for
multi-hour authoring or grid cells: syscall traces and cumulative model transcripts
grow to gigabytes. This profile changes only HOW evidence is presented, never what is
retained or mechanically checked:

* Raw syscall traces are replaced by a security-event index derived by trusted code:
  every exec, every write-capable/creating open, every rename/unlink/link/mkdir/chmod/
  chown/truncate/mount/prctl/signal/io_uring/shared-mapping syscall, failed writes, and
  every path probe outside common system roots. Other syscalls are counted per process.
  Each index names its raw file, byte size and SHA-256; the raw bytes stay sealed.
* Model journals are shown in journal order as conversation DELTAS: each request shows
  only the messages/input items it adds to the previous request of the same lane and
  system prompt (the earlier items are named by count). Responses are shown in full.
* The submitted workspace is shown as a diff against the trusted pre-agent inventory:
  unchanged files are proven by SHA-256 and listed by count; changed and added files
  are shown in full; deletions are listed.
* Verified byte-identical copies (agent-local native logs) are reviewed once.
* Tensor payloads are represented by dtype/shape/stride/SHA-256 metadata.
* Other data payloads (a NUL byte in the first 8 KB, a tensor/array/checkpoint suffix, or
  a path in the task's ``review_metadata_paths``) are shown by SHA-256, size and a short
  head. A zip archive (torch checkpoint, .npz) also lists its members, and every pickle is
  shown in full: a .pkl file, a pickle member of an archive, and the pickle streams a
  legacy checkpoint starts with. Unpickling can execute code.
* Raw evaluation outputs that score_workload.py recomputes row by row are summarized
  (SHA-256, size, row count and a bounded head); the sealed bytes are the score's input.
Everything else is reviewed byte-exhaustively, exactly as in the default profile.
"""
from fnmatch import fnmatchcase
import hashlib
import json
from pathlib import Path
import re

from contract import digest, read_json

PACKET = 64000
SNIFF = 8192
HEAD = 1024
DATA_SUFFIXES = ('.pt', '.pth', '.bin', '.ckpt', '.npy', '.npz', '.distcp', '.parquet', '.arrow')
PICKLE_SUFFIXES = ('.pkl', '.pickle')
# Code is never reduced to metadata by a review_metadata_paths entry.
CODE_SUFFIXES = ('.py', '.sh', '.bash')
PICKLE_HEADERS = (b'\x80\x02', b'\x80\x03', b'\x80\x04', b'\x80\x05')
PICKLE_BOUND = 64 * 1024**2
# Evaluation outputs (relative to the sealed evaluation root) that score_workload.py recomputes.
RECOMPUTED = {
    'nano': ('inputs/ppl.input', 'raw/logprobs.parquet', 'raw/generation.stdout'),
    'midtrain': ('raw/rl/evaluation/*.generations.jsonl',),
    'serving': ('raw/traffic-pass-*.json', 'raw/accuracy.generations.json', 'raw/traffic_records.json'),
    'cluster': ('raw/cluster/records-*.jsonl',),
}
SYSTEM_ROOTS = ('/usr/', '/lib/', '/lib64/', '/bin/', '/sbin/', '/etc/', '/proc/', '/sys/', '/dev/',
                '/opt/venv/', '/usr/local/lib/', '/opt/conda/')
MUTATING = {'rename', 'renameat', 'renameat2', 'unlink', 'unlinkat', 'mkdir', 'mkdirat', 'rmdir', 'symlink',
            'symlinkat', 'link', 'linkat', 'chmod', 'fchmod', 'fchmodat', 'fchmodat2', 'chown', 'fchown', 'lchown',
            'fchownat', 'truncate', 'ftruncate', 'mknod', 'mknodat', 'mount', 'umount2', 'pivot_root', 'chroot',
            'prctl', 'kill', 'tgkill', 'tkill', 'execve', 'execveat', 'io_uring_setup', 'io_uring_register',
            'io_uring_enter', 'io_setup', 'io_submit', 'sendfile', 'splice', 'vmsplice', 'copy_file_range',
            'utimensat', 'setxattr', 'lsetxattr', 'fsetxattr', 'removexattr', 'chdir', 'fchdir', 'acct',
            'swapon', 'swapoff', 'quotactl', 'name_to_handle_at', 'open_by_handle_at', 'fanotify_mark',
            'inotify_add_watch'}
OPEN_WRITE = re.compile(r'O_(?:WRONLY|RDWR|CREAT|TRUNC|APPEND|TMPFILE)')
CALL = re.compile(r'^(?:\[pid\s+\d+\]\s+)?(\w+)\((.*)$', re.S)
QUOTED = re.compile(r'"((?:[^"\\]|\\.)*)"')


def chunks(reference, text):
    if not text:
        yield f'{reference}[empty]', ''
        return
    for offset in range(0, len(text), PACKET):
        yield f'{reference}[chars {offset}:{offset + PACKET}]', text[offset:offset + PACKET]


def raw_chunks(reference, path):
    with Path(path).open('rb') as stream:
        yield from stream_chunks(reference, stream)


def stream_chunks(reference, stream):
    offset = 0
    while chunk := stream.read(PACKET):
        yield f'{reference}[bytes {offset}:{offset + len(chunk)}]', chunk.decode('utf-8', errors='backslashreplace')
        offset += len(chunk)
    if offset == 0:
        yield f'{reference}[empty]', ''


def first_path(arguments):
    match = QUOTED.search(arguments)
    return match.group(1) if match else ''


def syscall_parser():
    """The collector's own trace parser, loaded by path (collector/ has its own contract module)."""
    import importlib.util
    spec = importlib.util.spec_from_file_location('breakwater_syscall_trace', Path(__file__).resolve().parent / 'collector'
                                                  / 'syscall_trace.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def index_trace(path):
    """Security-event index of one strace -ff file; never executes or replays anything."""
    parser = syscall_parser()
    decoded, records = parser.decoded, parser.records
    events, counts, probes, failed = [], {}, {}, []
    for number, body in records(path):
        text = decoded(body).decode('utf-8', errors='backslashreplace')
        match = CALL.match(text)
        if not match:
            if text.startswith(('+++', '---')):
                events.append(f'L{number}: {text[:4000]}')
            continue
        name, arguments = match.group(1), match.group(2)
        counts[name] = counts.get(name, 0) + 1
        error = re.search(r'\)\s+=\s+-1\s+(\w+)', text)
        if name in ('open', 'openat', 'openat2', 'creat'):
            target = first_path(arguments)
            if name == 'creat' or OPEN_WRITE.search(arguments):
                events.append(f'L{number}: {text[:8000]}')
            elif not target.startswith(SYSTEM_ROOTS):
                probes[target] = probes.get(target, 0) + 1
            continue
        if name in MUTATING or (name in ('mmap', 'mmap2') and 'MAP_SHARED' in arguments and '</' in arguments):
            events.append(f'L{number}: {text[:8000]}')
            continue
        if name in ('stat', 'lstat', 'newfstatat', 'statx', 'access', 'faccessat', 'faccessat2', 'readlink',
                    'readlinkat', 'getdents64', 'statfs', 'getxattr', 'lgetxattr', 'listxattr'):
            target = first_path(arguments)
            if target and not target.startswith(SYSTEM_ROOTS):
                probes[target] = probes.get(target, 0) + 1
            continue
        if error and name in ('write', 'writev', 'pwrite64', 'pwritev', 'pwritev2', 'ioctl', 'fcntl'):
            failed.append(f'L{number}: {text[:2000]}')
    header = {'raw_file': str(Path(path).name), 'bytes': Path(path).stat().st_size, 'sha256': digest(path),
              'note': 'Index derived by trusted code; raw syscall bytes are retained, sealed and hash-bound.'}
    lines = ['SYSCALL SECURITY INDEX ' + json.dumps(header, sort_keys=True),
             'SYSCALL COUNTS ' + json.dumps(dict(sorted(counts.items())), sort_keys=True),
             f'EVENTS ({len(events)}): exec/write-capable opens/mutations/signals/prctl/mount/io_uring/shared maps']
    lines += events
    lines.append(f'FAILED WRITE/IOCTL/FCNTL ({len(failed)})')
    lines += failed
    lines.append(f'PATH PROBES OUTSIDE SYSTEM ROOTS ({len(probes)} distinct): path -> count')
    lines += [f'{target} -> {count}' for target, count in sorted(probes.items())]
    return '\n'.join(lines) + '\n'


def conversation_items(payload):
    """-> (key, items) for Messages, Responses and chat-completions request bodies."""
    if not isinstance(payload, dict):
        return None, None
    for field in ('messages', 'input'):
        if isinstance(payload.get(field), list):
            context = {key: payload.get(key) for key in ('system', 'instructions', 'tools', 'model')}
            key = hashlib.sha256(json.dumps(context, sort_keys=True, default=str).encode()).hexdigest()
            return key, payload[field]
    return None, None


def index_journal(folder, label):
    """Journal-ordered model I/O: request deltas and complete responses, all hash-bound."""
    folder = Path(folder)
    journal = folder / 'requests.jsonl'
    previous, seen_context, lines = {}, set(), []
    lines.append(f'MODEL JOURNAL INDEX {label}: journal sha256 {digest(journal)}; every blob below is '
                 'cited by its SHA-256; unchanged earlier conversation items are named by count.')
    for raw in journal.read_bytes().splitlines():
        row = json.loads(raw)
        output = row.get('output') if isinstance(row.get('output'), dict) else None
        meta = {key: row.get(key) for key in ('seq', 'event', 'call_id', 'lane', 'kind', 'route', 'status', 'outcome',
                                               'tokens', 'reason', 'error_type', 'faulted', 'model', 'alias', 'utc')
                if row.get(key) is not None}
        if output is None:
            lines.append('EVENT ' + json.dumps(meta, sort_keys=True))
            continue
        blob = folder / output['path']
        data = blob.read_bytes()
        if row.get('event') == 'model_request':
            try:
                payload = json.loads(data)
            except ValueError:
                payload = None
            key, items = conversation_items(payload)
            lane = str(row.get('lane') or row.get('route') or 'default')
            if items is None:
                lines.append('REQUEST ' + json.dumps(meta, sort_keys=True) + f' blob {output["sha256"]} (full):')
                lines.append(data.decode('utf-8', errors='backslashreplace'))
                continue
            context = {k: payload.get(k) for k in payload if k not in ('messages', 'input')}
            context_digest = hashlib.sha256(json.dumps(context, sort_keys=True, default=str).encode()).hexdigest()
            earlier = previous.get((lane, key), [])
            shared = 0
            while shared < min(len(earlier), len(items)) and earlier[shared] == items[shared]:
                shared += 1
            previous[(lane, key)] = items
            lines.append('REQUEST ' + json.dumps(meta, sort_keys=True) + f' blob {output["sha256"]}: '
                         f'{len(items)} items; first {shared} identical to the previous request of this lane/context; '
                         f'context sha256 {context_digest}')
            if context_digest not in seen_context:
                seen_context.add(context_digest)
                lines.append('NEW REQUEST CONTEXT (system/instructions/tools/options): '
                             + json.dumps(context, sort_keys=True, default=str))
            for index in range(shared, len(items)):
                lines.append(f'ITEM {index}: ' + json.dumps(items[index], sort_keys=True, default=str))
        else:
            lines.append('RESPONSE ' + json.dumps(meta, sort_keys=True) + f' blob {output["sha256"]}:')
            lines.append(data.decode('utf-8', errors='backslashreplace'))
    return '\n'.join(lines) + '\n'


def index_fleet_journal(path, windows):
    """Cluster fleet inference: agent-stage requests in full; graded traffic as one line each."""
    from cluster_fleet import phase_of
    lines = [f'FLEET INFERENCE JOURNAL {Path(path).name} sha256 {digest(path)} (worker-recorded before inference '
             'and before forwarding; prompts are token IDs; no tokenizer is available to the verifier)']
    texts = {}
    with Path(path).open('rb') as stream:
        for raw in stream:
            row = json.loads(raw)
            event, phase = row.get('event'), phase_of(int(row.get('time_ns', 0)), windows)
            if event == 'worker_started':
                lines.append('WORKER ' + json.dumps({k: row.get(k) for k in ('run_id', 'node', 'worker', 'model',
                                                                             'model_info', 'utc')}, sort_keys=True))
            elif event == 'request':
                request = row.get('request') or {}
                prompt = request.get('prompt') if isinstance(request, dict) else None
                summary = {'seq': row.get('seq'), 'phase': phase, 'audit_id': row.get('audit_id'),
                           'model': request.get('model') if isinstance(request, dict) else None,
                           'prompt_tokens': len(prompt) if isinstance(prompt, list) else None,
                           'max_tokens': request.get('max_tokens') if isinstance(request, dict) else None,
                           'routing': request.get('routing') if isinstance(request, dict) else None}
                lines.append('REQUEST ' + json.dumps(summary, sort_keys=True, default=str))
                if phase == 'agent':
                    lines.append('AGENT-STAGE REQUEST BODY ' + json.dumps(request, sort_keys=True, default=str))
            elif event == 'response':
                frame = row.get('frame') or {}
                for choice in frame.get('choices', []) if isinstance(frame, dict) else []:
                    if isinstance(choice, dict) and isinstance(choice.get('text'), str):
                        texts.setdefault(row.get('audit_id'), []).append(choice['text'])
            elif event == 'request_finished':
                text = ''.join(texts.pop(row.get('audit_id'), []))
                lines.append('FINISHED ' + json.dumps({k: row.get(k) for k in ('seq', 'audit_id', 'outcome')},
                                                      sort_keys=True, default=str)
                             + (f' phase={phase} response_text={json.dumps(text)}' if phase == 'agent'
                                else f' phase={phase} response_chars={len(text)}'))
            elif event in ('upstream_error', 'upstream_status', 'worker_stopped'):
                lines.append(event.upper() + ' ' + json.dumps({k: v for k, v in row.items()
                                                               if k not in ('bytes_base64',)}, sort_keys=True, default=str))
    for audit_id, parts in texts.items():
        lines.append(f'UNFINISHED {audit_id} response_text={json.dumps("".join(parts))}')
    return '\n'.join(lines) + '\n'


def covered(name, paths):
    return any(name == item or name.startswith(item.rstrip('/') + '/') for item in paths)


def workspace_view(root, baseline, label, metadata_paths=()):
    """Diff of the sealed workspace against the trusted pre-agent inventory, with its recorded links."""
    root = Path(root)
    from contract import inventory
    final = inventory(root, ('manifest.json',))
    unchanged = sorted(name for name in final if baseline.get(name) == final[name])
    changed = sorted(name for name in final if name in baseline and baseline[name] != final[name])
    added = sorted(name for name in final if name not in baseline)
    deleted = sorted(name for name in baseline if name not in final)
    links = read_json(root / 'manifest.json').get('symlinks', {}) if (root / 'manifest.json').is_file() else {}
    summary = {'unchanged_files': len(unchanged),
               'unchanged_inventory_sha256': hashlib.sha256(json.dumps({n: final[n] for n in unchanged},
                                                                       sort_keys=True).encode()).hexdigest(),
               'changed': changed, 'added': added, 'deleted': deleted,
               'symbolic_links_not_sealed_as_files': links}
    yield from chunks(f'{label}[diff-summary]', 'WORKSPACE DIFF AGAINST TRUSTED PRE-AGENT INVENTORY '
                      + json.dumps(summary, sort_keys=True))
    for name in changed + added:
        yield from file_packets(f'{label}/{name}', root / name, metadata=covered(name, metadata_paths))


class _Bounded:
    """A read-only view that refuses to read past ``limit`` bytes (pickle scanning)."""

    def __init__(self, stream, limit):
        self.stream, self.limit = stream, limit

    def read(self, size=-1):
        if size < 0 or self.stream.tell() + size > self.limit:
            raise ValueError('pickle stream exceeds the review bound')
        return self.stream.read(size)

    def readline(self):
        line = self.stream.readline(max(0, self.limit - self.stream.tell()) + 1)
        if self.stream.tell() > self.limit:
            raise ValueError('pickle stream exceeds the review bound')
        return line

    def tell(self):
        return self.stream.tell()


def leading_pickles(path, bound=PICKLE_BOUND):
    """Bytes of the consecutive pickle streams a file starts with (legacy torch saves), or None
    when one cannot be delimited within ``bound``. Opcodes are only parsed, never executed."""
    import pickletools
    with Path(path).open('rb') as stream:
        end = 0
        while True:
            stream.seek(end)
            if stream.read(2) not in PICKLE_HEADERS:
                return end
            stream.seek(end)
            try:
                for _ in pickletools.genops(_Bounded(stream, end + bound)):
                    pass
            except Exception:
                return None
            end = stream.tell()


def file_packets(reference, path, metadata=False):
    """One retained file: tensor headers, data payloads as metadata, pickles and text in full."""
    path = Path(path)
    if path.suffix == '.safetensors':
        from tensor_inventory import inventory as tensors
        yield reference, json.dumps({'sha256': digest(path), 'tensor_header': tensors(path)})
        return
    if path.suffix in PICKLE_SUFFIXES:
        yield from raw_chunks(reference, path)
        return
    with path.open('rb') as stream:
        head = stream.read(SNIFF)
    if not ((metadata and path.suffix not in CODE_SUFFIXES) or b'\0' in head or path.suffix in DATA_SUFFIXES):
        yield from raw_chunks(reference, path)
        return
    summary = {'kind': 'data_payload', 'sha256': digest(path), 'bytes': path.stat().st_size,
               'head': head[:HEAD].decode('utf-8', errors='backslashreplace'),
               'note': 'Shown by metadata; the sealed bytes are retained and hash-bound.'}
    import zipfile
    if not zipfile.is_zipfile(path):
        prefix = leading_pickles(path) if head[:2] in PICKLE_HEADERS else 0
        if prefix is None:  # a pickle that cannot be delimited is reviewed in full, never as metadata
            yield from raw_chunks(reference, path)
            return
        if prefix:
            summary['leading_pickle_bytes'] = prefix
        yield reference + '[metadata]', json.dumps(summary, sort_keys=True)
        if prefix:
            import io
            with path.open('rb') as stream:
                yield from stream_chunks(reference + '[pickle]', io.BytesIO(stream.read(prefix)))
        return
    try:
        with zipfile.ZipFile(path) as archive:
            members = archive.infolist()
            summary['zip_members'] = [{'name': item.filename, 'bytes': item.file_size,
                                       'compressed_bytes': item.compress_size, 'crc32': f'{item.CRC:08x}'}
                                      for item in members]
            yield from chunks(reference + '[metadata]', json.dumps(summary, sort_keys=True))
            for item in members:
                if item.is_dir() or not item.filename.endswith(PICKLE_SUFFIXES):
                    continue
                try:
                    with archive.open(item) as stream:
                        yield from stream_chunks(f'{reference}::{item.filename}', stream)
                except (zipfile.BadZipFile, NotImplementedError, OSError, RuntimeError) as error:
                    yield (f'{reference}::{item.filename}[unreadable]',
                           f'This pickle member could not be decompressed for review: {type(error).__name__}: {error}')
    except (zipfile.BadZipFile, OSError) as error:
        summary['zip_error'] = f'{type(error).__name__}: {error}'
        yield reference + '[metadata]', json.dumps(summary, sort_keys=True)


def row_count(path):
    """Rows of a recomputed output: Parquet rows, a JSON list or its 'records', or newline-ended lines."""
    path = Path(path)
    if path.suffix == '.parquet':
        try:
            import pyarrow.parquet as parquet
            return parquet.ParquetFile(path).metadata.num_rows
        except Exception:
            return None
    with path.open('rb') as stream:
        if b'\0' in stream.read(SNIFF):
            return None
    if path.suffix == '.json':
        try:
            value = json.loads(path.read_bytes())
        except ValueError:
            return None
        records = value.get('records') if isinstance(value, dict) else value
        return len(records) if isinstance(records, list) else None
    lines = 0
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024**2), b''):
            lines += chunk.count(b'\n')
    return lines


def recomputed_summary(reference, path, relative):
    with Path(path).open('rb') as stream:
        head = stream.read(4096)
    return reference + '[recomputed-output]', json.dumps({
        'kind': 'mechanically_recomputed_output', 'file': relative, 'sha256': digest(path),
        'bytes': Path(path).stat().st_size, 'rows': row_count(path),
        'head': head.decode('utf-8', errors='backslashreplace'),
        'note': 'score_workload.py recomputed the deterministic score from every row of this sealed file; '
                'the mechanical report and DETERMINISTIC SCORE carry the result.'}, sort_keys=True)


def tensor_tree(path):
    """downstream tensor_files payloads: metadata and hashes, never expanded floats."""
    return path.suffix == '.bin' and (path.parent / 'tree.json').is_file()


def packets(roots, mechanical=None, launch=None, config=None):
    mechanical = mechanical or {}
    config = config or {}
    recomputed = RECOMPUTED.get(config.get('kind'), ())
    metadata_paths = config.get('review_metadata_paths') or ()
    external = Path(roots['external'])
    windows = None
    if (external / 'fleet-inference.json').is_file():
        windows = read_json(external / 'fleet-inference.json').get('windows')
    journals = {folder.name for folder in external.iterdir()
                if folder.is_dir() and (folder / 'requests.jsonl').is_file()}
    identical = (mechanical.get('native_agreement') or {}).get('status') == 'PASS'
    baseline = None
    if (external / 'workspace-baseline.json').is_file():
        baseline = read_json(external / 'workspace-baseline.json').get('files')
    for label, root in roots.items():
        root = Path(root)
        if label == 'submission' and baseline is not None:
            yield from workspace_view(root, baseline, label, metadata_paths)
            continue
        for path in sorted(root.rglob('*')):
            if not path.is_file():
                continue
            if path.is_symlink():
                raise ValueError('Symlink in audit inputs')
            relative = path.relative_to(root).as_posix()
            reference = f'{label}/{relative}'
            parts = Path(relative).parts
            if label == 'external':
                if relative == 'environment-baseline.json':
                    value = read_json(path)
                    yield reference, json.dumps({'kind': 'mechanically_verified_runtime_inventory', 'sha256': digest(path),
                                                 'files': len(value['files']), 'specification': value['specification']})
                    continue
                if relative in ('workspace-baseline.json', 'cell-workspace-baseline.json', 'cell-workspace-final.json'):
                    value = read_json(path)
                    files = value.get('files') if isinstance(value.get('files'), dict) else {}
                    summary = {'kind': 'trusted_workspace_inventory', 'file': relative, 'sha256': digest(path),
                               'files': len(files) if files else value.get('files'),
                               **{key: item for key, item in value.items() if key not in ('files',)}}
                    yield reference, json.dumps(summary, sort_keys=True, default=str)
                    continue
                if len(parts) >= 3 and parts[1] == 'syscalls' and parts[2].startswith('exec-') and not parts[2].endswith('.stderr'):
                    yield from chunks(reference + '[index]', index_trace(path))
                    continue
                if parts[0] in journals:
                    if relative == f'{parts[0]}/requests.jsonl':
                        yield from chunks(reference + '[conversation-index]', index_journal(root / parts[0], parts[0]))
                    elif len(parts) >= 2 and parts[1] == 'blobs':
                        continue  # every blob is shown, by digest, inside its journal's index
                    else:
                        yield from raw_chunks(reference, path)
                    continue
                if parts[0] == 'fleet-inference' and path.suffix == '.jsonl':
                    yield from chunks(reference + '[index]', index_fleet_journal(path, windows))
                    continue
            if label == 'agent-local' and identical and relative in ('native.jsonl', 'native.stdout', 'native.stderr'):
                yield reference, ('Byte-identical to external/agent-native.* by mechanical reconciliation '
                                  f'(sha256 {digest(path)}); reviewed once in the external copy.')
                continue
            if label == 'evaluation' and any(fnmatchcase(relative, pattern) for pattern in recomputed):
                yield recomputed_summary(reference, path, relative)
                continue
            if tensor_tree(path):
                yield reference, json.dumps({'kind': 'tensor_payload', 'sha256': digest(path), 'bytes': path.stat().st_size,
                                             'tree': (path.parent / 'tree.json').name})
                continue
            yield from file_packets(reference, path, metadata=label == 'submission' and covered(relative, metadata_paths))
