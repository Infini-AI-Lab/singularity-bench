"""Indexed integrity-review packets for long runs (``review_profile: indexed``).

The exhaustive profile sends every retained byte to the judge. That cannot finish for
multi-hour authoring: syscall traces and cumulative model transcripts
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
* The agent CLI's native stream is shown line by line, except that each run of its content-free progress events
  (Claude Code reports thinking as one event per token, with counters and identifiers but no text) is one line with
  the run's count, line range, counters and SHA-256.
* Each command's output under the Apptainer brokers (commands/, evaluation-commands/) is headed by what the broker
  journaled for that command: its text (the first 2,000 characters), working directory and exit status.
* The agent CLI's own copy of its conversation (Claude Code's session file in agent-local) is summarized by digest,
  size and entry counts when the authoritative records it repeats are present: the author gateway's journal (every
  model request and response) and the stream the host captured.
* A sealed directory's manifest lists every file, except that each model journal's blobs are summarized by count,
  bytes and the SHA-256 of their entries: each blob is cited by digest in its journal's index, and every manifest is
  verified mechanically.
* Tensor payloads are represented by dtype/shape/stride/SHA-256 metadata.
* Other data payloads (a NUL byte in the first 8 KB, a tensor/array/checkpoint suffix, or
  a path in the task's ``review_metadata_paths``) are shown by SHA-256, size and a short
  head. A zip archive (torch checkpoint, .npz) also lists its members, and every pickle is
  shown in full: a .pkl file, a pickle member of an archive, and the pickle streams a
  legacy checkpoint starts with. Unpickling can execute code.
* Raw evaluation outputs that the task's scorer recomputes row by row ([breakwater.scoring] recomputed) are summarized
  (SHA-256, size, row count and a bounded head); the sealed bytes are the score's input.
Everything else is reviewed byte-exhaustively, exactly as in the default profile.
"""
from fnmatch import fnmatchcase
import hashlib
import json
from pathlib import Path
import re
import sys

from contract import digest, read_json

PACKET = 64000
SMALL = 8000  # packets shorter than this are bundled with their neighbors (tiny streams, digest files, small JSON)
SNIFF = 8192
HEAD = 1024
DATA_SUFFIXES = ('.pt', '.pth', '.bin', '.ckpt', '.npy', '.npz', '.distcp', '.parquet', '.arrow')
PICKLE_SUFFIXES = ('.pkl', '.pickle')
# Code is never reduced to metadata by a review_metadata_paths entry.
CODE_SUFFIXES = ('.py', '.sh', '.bash')
PICKLE_HEADERS = (b'\x80\x02', b'\x80\x03', b'\x80\x04', b'\x80\x05')
PICKLE_BOUND = 64 * 1024**2
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
# Distinct events of one call in one directory beyond which the trace index gives a count and examples.
ROLLUP = 50
# Calls whose lines the trace index counts together only when identical (an exec also when its arguments differ only
# in temp-file names, below).
VERBATIM = {'execve', 'execveat', 'kill', 'tgkill', 'tkill', 'prctl', 'mount', 'umount2', 'pivot_root', 'chroot',
            'ptrace', 'acct', 'swapon', 'swapoff'}
# strace -v -xx: every string is hex-escaped, so brackets and quotes here are structure, never data.
EXECVE = re.compile(rb'execve\((?P<file>"[^"]*"(?:\.\.\.)?), (?P<argv>\[[^\]]*\]), (?P<envp>\[[^\]]*\])(?P<rest>.*)$', re.S)
ENTRY = re.compile(rb'"((?:\\x[0-9a-fA-F]{2})*)"')
# A file named by Python's tempfile (Triton writes each kernel's PTX and object to one): repeats of an exec that differ
# only in such names are counted together. The program itself is always compared exactly.
TEMPFILE = re.compile(r'(/tmp/tmp)[A-Za-z0-9_]{6,}')
DURATION = re.compile(r'\s*<\d+\.\d+>\s*$')
NUMBERS = re.compile(r'0x[0-9a-fA-F]+|\d+')


def bundled(items, limit=PACKET, small=SMALL):
    """(reference, text, members) review packets, in order: consecutive small packets share one, up to a packet's
    size, each headed by its reference; any other packet is reviewed alone."""
    group, size = [], 0
    for reference, text in items:
        part = len(reference) + len(text) + 9
        if group and (len(text) >= small or size + part > limit):
            yield bundle(group)
            group, size = [], 0
        if len(text) >= small:
            yield reference, text, [reference]
            continue
        group.append((reference, text))
        size += part
    if group:
        yield bundle(group)


def bundle(group):
    references = [reference for reference, _ in group]
    if len(group) == 1:
        return (*group[0], references)
    return (f'bundle of {len(group)} small files, {references[0]} to {references[-1]}',
            '\n'.join(f'=== {reference} ===\n{text}' for reference, text in group), references)


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


def busy_trees(paths):
    """Map each path to the deepest directory that holds (at any depth) more than ROLLUP of the paths, if any; a
    directory left with ROLLUP or fewer paths after deeper ones took theirs maps nothing."""
    below = {}
    for path in paths:
        folder = path.rpartition('/')[0]
        while folder:
            below[folder] = below.get(folder, 0) + 1
            folder = folder.rpartition('/')[0]
    owner = {}
    for path in paths:
        folder = path.rpartition('/')[0]
        while folder and below.get(folder, 0) <= ROLLUP:
            folder = folder.rpartition('/')[0]
        if folder:
            owner[path] = folder
    sizes = {}
    for folder in owner.values():
        sizes[folder] = sizes.get(folder, 0) + 1
    return {path: folder for path, folder in owner.items() if sizes[folder] > ROLLUP}


def environment(raw, decoded):
    """An execve environment (strace -v -xx) as its list of strings."""
    return [decoded(item).decode('utf-8', errors='backslashreplace') for item in ENTRY.findall(raw)]


def environment_lines(environments):
    """env#1 in full, later environments as what they add, change (+NAME=value) or remove (-NAME) from it."""
    lines, first = [], None
    for items, key in sorted(environments.items(), key=lambda item: item[1]):
        if first is None:
            first = dict(item.partition('=')[::2] for item in items)
            lines.append(f'env#{key}: ' + json.dumps(list(items)))
            continue
        current = dict(item.partition('=')[::2] for item in items)
        changes = [f'+{name}={value}' for name, value in current.items() if first.get(name) != value]
        changes += [f'-{name}' for name in first if name not in current]
        lines.append(f'env#{key}: env#1 with ' + (json.dumps(changes) if changes else 'no changes'))
    return lines


def normalized(text):
    """An event without its variable numbers (descriptors, sizes, addresses, results), for counting repeats; quoted
    strings (paths, names, arguments) stay exact."""
    parts = re.split(r'("(?:[^"\\]|\\.)*")', text)
    return ''.join(part if index % 2 else NUMBERS.sub('#', part) for index, part in enumerate(parts))


def index_trace(path):
    """Security-event index of one strace file; never executes or replays anything.

    Every distinct event is kept. An event repeated with only different descriptors, sizes or addresses appears once
    with its count; more than ROLLUP distinct events of one call in one directory (compiler caches, for example)
    appear as one line with their count and examples; each distinct execve environment is printed once and referred
    to as env#N; a program looked up along PATH and not found appears once with the directories tried. Calls in
    VERBATIM are counted together only when identical, an exec also when its arguments differ only in temp-file
    names."""
    parser = syscall_parser()
    decoded, records = parser.decoded, parser.records
    groups, counts, probes, failed, environments, missing = {}, {}, {}, {}, {}, {}

    def plain(raw):
        return decoded(raw).decode('utf-8', errors='backslashreplace')

    def record(bucket, key, number, text, name='', target=''):
        entry = bucket.setdefault(key, {'first': number, 'count': 0, 'text': text, 'name': name, 'path': target})
        entry['count'] += 1

    def line(entry):
        return f'L{entry["first"]}' + (f' [{entry["count"]} times]' if entry['count'] > 1 else '') + f': {entry["text"]}'

    for number, body in records(path):
        text = DURATION.sub('', plain(body))
        match = CALL.match(text)
        if not match:
            if text.startswith(('+++', '---')):
                record(groups, ('', normalized(text)), number, text[:4000])
            continue
        name, arguments = match.group(1), match.group(2)
        counts[name] = counts.get(name, 0) + 1
        error = re.search(r'\)\s+=\s+-1\s+(\w+)', text)
        if name == 'execve':
            if error and error.group(1) == 'ENOENT':
                folder, _, program = first_path(arguments).rpartition('/')
                entry = missing.setdefault(program, {'first': number, 'count': 0, 'dirs': []})
                entry['count'] += 1
                if folder not in entry['dirs']:
                    entry['dirs'].append(folder)
                continue
            parts = EXECVE.match(body)
            identity = (name, text)
            if parts:
                key = environments.setdefault(tuple(environment(parts['envp'], decoded)), len(environments) + 1)
                program, argv = plain(parts['file']), plain(parts['argv'])
                rest = DURATION.sub('', plain(parts['rest']))
                text = f'execve({program}, {argv}, env#{key}{rest}'
                identity = (name, program, TEMPFILE.sub(r'\1~', argv), key, rest)
            record(groups, identity, number, text[:8000], name)
            continue
        if name in ('open', 'openat', 'openat2', 'creat'):
            target = first_path(arguments)
            if name == 'creat' or OPEN_WRITE.search(arguments):
                record(groups, (name, normalized(text)), number, text[:8000], name, target)
            elif not target.startswith(SYSTEM_ROOTS):
                probes[target] = probes.get(target, 0) + 1
            continue
        if name in MUTATING or (name in ('mmap', 'mmap2') and 'MAP_SHARED' in arguments and '</' in arguments):
            if name in VERBATIM:
                record(groups, (name, text), number, text[:8000], name)
            else:
                record(groups, (name, normalized(text)), number, text[:8000], name, first_path(arguments))
            continue
        if name in ('stat', 'lstat', 'newfstatat', 'statx', 'access', 'faccessat', 'faccessat2', 'readlink',
                    'readlinkat', 'getdents64', 'statfs', 'getxattr', 'lgetxattr', 'listxattr'):
            target = first_path(arguments)
            if target and not target.startswith(SYSTEM_ROOTS):
                probes[target] = probes.get(target, 0) + 1
            continue
        if error and name in ('write', 'writev', 'pwrite64', 'pwritev', 'pwritev2', 'ioctl', 'fcntl'):
            record(failed, (name, normalized(text)), number, text[:2000], name)
    emitted, calls = [], {}
    for entry in groups.values():
        if entry['path']:
            calls.setdefault(entry['name'], []).append(entry)
        else:
            emitted.append((entry['first'], line(entry)))
    for name, entries in calls.items():
        owner, trees = busy_trees([entry['path'] for entry in entries]), {}
        for entry in entries:
            if entry['path'] in owner:
                trees.setdefault(owner[entry['path']], []).append(entry)
            else:
                emitted.append((entry['first'], line(entry)))
        for folder, members in trees.items():
            members.sort(key=lambda entry: entry['first'])
            examples = '; '.join(line(entry) for entry in members[:3] + members[-1:])
            emitted.append((members[0]['first'], f'L{members[0]["first"]} [{sum(entry["count"] for entry in members)} '
                            f'times on {len(members)} distinct paths under {folder}/]: {name}, e.g. {examples}'))
    for program, entry in missing.items():
        emitted.append((entry['first'], f'L{entry["first"]} [{entry["count"]} times]: execve of "{program}" failed '
                        f'with ENOENT, tried in {", ".join(entry["dirs"]) or "the working directory"}'))
    emitted.sort(key=lambda item: item[0])
    probe_lines, owner, trees = [], busy_trees(list(probes)), {}
    for target, count in sorted(probes.items()):
        if target in owner:
            trees.setdefault(owner[target], []).append((target, count))
        else:
            probe_lines.append(f'{target} -> {count}')
    for folder, members in sorted(trees.items()):
        probe_lines.append(f'{folder}/ -> {sum(count for _, count in members)} probes of {len(members)} distinct '
                           f'paths under it, e.g. {", ".join(target for target, _ in members[:3])}')
    header = {'raw_file': str(Path(path).name), 'bytes': Path(path).stat().st_size, 'sha256': digest(path),
              'note': 'Index derived by trusted code; raw syscall bytes are retained, sealed and hash-bound.'}
    total = sum(entry['count'] for entry in groups.values()) + sum(entry['count'] for entry in missing.values())
    lines = ['SYSCALL SECURITY INDEX ' + json.dumps(header, sort_keys=True),
             'SYSCALL COUNTS ' + json.dumps(dict(sorted(counts.items())), sort_keys=True),
             f'EVENTS ({total} in {len(emitted)} lines): exec/write-capable opens/mutations/signals/prctl/mount/'
             'io_uring/shared maps. Repeats are counted: exec, signal, prctl and mount lines only when identical (exec '
             'arguments may differ in temp-file names), other events when they differ only in descriptors, sizes or '
             'addresses; busy directories are summarized.']
    lines += [text for _, text in emitted]
    lines.append(f'FAILED WRITE/IOCTL/FCNTL ({sum(entry["count"] for entry in failed.values())})')
    lines += [line(entry) for entry in sorted(failed.values(), key=lambda entry: entry['first'])]
    lines.append(f'PATH PROBES OUTSIDE SYSTEM ROOTS ({len(probes)} distinct): path -> count')
    lines += probe_lines
    lines.append(f'EXECVE ENVIRONMENTS ({len(environments)} distinct, referred to above as env#N)')
    lines += environment_lines(environments)
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


def reassembled(raw, request=None):
    """A streamed model response (server-sent events) as the message it delivered, or None if it is not a stream
    this knows. Anthropic Messages events are joined block by block (signatures and redacted thinking given by
    length, events of other kinds counted); an OpenAI Responses stream ends with the whole response, whose echo of
    the request's instructions and tools is named rather than repeated."""
    events = []
    for line in raw.decode('utf-8', errors='backslashreplace').splitlines():
        if line.startswith('data:') and line[5:].strip() not in ('', '[DONE]'):
            try:
                events.append(json.loads(line[5:]))
            except ValueError:
                return None
    if not events or not all(isinstance(event, dict) for event in events):
        return None
    for event in events:
        if event.get('type') in ('response.completed', 'response.incomplete', 'response.failed'):
            response = event.get('response')
            if isinstance(response, dict) and isinstance(request, dict):
                response = {key: 'as in the request' if key in ('instructions', 'tools') and value
                            and value == request.get(key) else value for key, value in response.items()}
            return {'stream': 'responses', 'events': len(events), 'response': response}
    message, blocks, other, complete = {}, {}, {}, False
    for event in events:
        kind, delta = event.get('type'), event.get('delta') if isinstance(event.get('delta'), dict) else {}
        if kind == 'message_start':
            message = {key: value for key, value in (event.get('message') or {}).items() if key != 'content'}
        elif kind == 'content_block_start':
            blocks[event.get('index')] = dict(event.get('content_block') or {})
        elif kind == 'content_block_delta':
            block = blocks.setdefault(event.get('index'), {})
            fields = {'text': 'text', 'thinking': 'thinking', 'partial_json': 'input_json', 'signature': 'signature'}
            for field, value in delta.items():
                if field in fields and isinstance(value, str):
                    block[fields[field]] = block.get(fields[field], '') + value
                elif field != 'type':
                    block.setdefault('other_deltas', []).append({field: value})
        elif kind == 'message_delta':
            message.update(delta)
            if isinstance(event.get('usage'), dict):
                message['usage'] = {**message.get('usage', {}), **event['usage']}
        elif kind == 'message_stop':
            complete = True
        elif kind == 'error':
            message['error'] = event.get('error')
        elif kind != 'content_block_stop':
            other[str(kind)] = other.get(str(kind), 0) + 1
    if not message and not blocks:
        return None
    content = []
    for index in sorted(blocks, key=lambda key: (key is None, key if isinstance(key, int) else 0)):
        block = blocks[index]
        if 'input_json' in block:
            text = block.pop('input_json')
            try:
                block['input'] = json.loads(text or '{}')
            except ValueError:
                block['input_partial'] = text
        for field in ('signature', 'data'):  # opaque: a thinking signature, redacted thinking
            if isinstance(block.get(field), str):
                block[field] = f'({len(block[field])} chars)'
        content.append(block)
    return {'stream': 'messages', 'events': len(events), 'complete': complete, **message, 'content': content,
            **({'other_events': other} if other else {})}


def index_journal(folder, label):
    """Journal-ordered model I/O: request deltas and complete responses, all hash-bound. A response, whose chunks the
    journal records one by one, is shown once its call ends: a stream reassembled into the message it delivered,
    anything else as received."""
    folder = Path(folder)
    journal = folder / 'requests.jsonl'
    previous, seen_context, lines, streams, requests = {}, set(), [], {}, {}

    def flush(call_id):
        chunks, request = streams.pop(call_id, None), requests.pop(call_id, None)
        if not chunks:
            return
        data = b''.join(chunks)
        message = reassembled(data, request)
        header = (f'RESPONSE call {call_id}: {len(chunks)} chunks, {len(data)} bytes, sha256 of their concatenation '
                  f'{hashlib.sha256(data).hexdigest()}')
        if message is None:
            lines.extend([header + ' (as received):', data.decode('utf-8', errors='backslashreplace')])
        else:
            lines.extend([header + ' (streamed events reassembled):',
                          json.dumps(message, ensure_ascii=False, sort_keys=True)])
    lines.append(f'MODEL JOURNAL INDEX {label}: journal sha256 {digest(journal)}; every request blob below is cited '
                 'by its SHA-256 and each response by that of its chunks joined; unchanged earlier conversation items '
                 'are named by count.')
    for raw in journal.read_bytes().splitlines():
        row = json.loads(raw)
        output = row.get('output') if isinstance(row.get('output'), dict) else None
        meta = {key: row.get(key) for key in ('seq', 'event', 'call_id', 'lane', 'kind', 'route', 'status', 'outcome',
                                               'tokens', 'reason', 'error_type', 'faulted', 'model', 'alias', 'utc')
                if row.get(key) is not None}
        if output is None:
            if row.get('event') == 'model_response_end':
                flush(row.get('call_id'))
            lines.append('EVENT ' + json.dumps(meta, sort_keys=True))
            continue
        blob = folder / output['path']
        data = blob.read_bytes()
        if row.get('event') == 'model_response_chunk':
            streams.setdefault(row.get('call_id'), []).append(data)
            continue
        if row.get('event') == 'model_request':
            try:
                payload = json.loads(data)
            except ValueError:
                payload = None
            key, items = conversation_items(payload)
            if isinstance(payload, dict):  # what a Responses stream echoes back
                requests[row.get('call_id')] = {name: payload.get(name) for name in ('instructions', 'tools')}
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
    for call_id in list(streams):  # a call the journal never ended
        flush(call_id)
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
        'note': "the task's scorer recomputed the deterministic score from every row of this sealed file; "
                'the mechanical report and DETERMINISTIC SCORE carry the result.'}, sort_keys=True)


# A progress event of the agent CLI's stream that carries no text: Claude Code's per-token thinking ticks.
PROGRESS_SUBTYPES = ('thinking_tokens',)
IDENTIFIER = re.compile(r'[0-9A-Za-z_.:-]{1,80}')


def progress_event(line):
    """The event, if a line of the agent CLI's stream is a progress event without text: type "system" with a subtype
    in PROGRESS_SUBTYPES, and otherwise only numbers, booleans, nulls and short identifiers."""
    try:
        value = json.loads(line)
    except ValueError:
        return None
    if not (isinstance(value, dict) and value.get('type') == 'system' and value.get('subtype') in PROGRESS_SUBTYPES):
        return None
    for key, item in value.items():
        if key not in ('type', 'subtype') and not (item is None or isinstance(item, (bool, int, float))
                                                    or (isinstance(item, str) and IDENTIFIER.fullmatch(item))):
            return None
    return value


def index_transcript(path):
    """The agent CLI's native stream as L<number>: <line>, each run of consecutive progress events (progress_event) of
    one subtype as a single line."""
    path = Path(path)
    header = {'raw_file': path.name, 'bytes': path.stat().st_size, 'sha256': digest(path),
              'note': 'Every line as it is, except runs of progress events without text, each shown as one line with '
                      'its count and the SHA-256 of its bytes; the raw stream is retained, sealed and hash-bound.'}
    lines, run = ['AGENT CLI STREAM ' + json.dumps(header, sort_keys=True)], []

    def flush():
        if run:
            (first, event, _), (last, final, _) = run[0], run[-1]
            fields = sorted({key for _, value, _ in run for key in value} - {'type', 'subtype'})
            counters = {key: [event[key], final[key]] for key in fields
                        if isinstance(event.get(key), (int, float)) and isinstance(final.get(key), (int, float))}
            lines.append(f'L{first}-L{last} [{len(run)} lines]: {event["subtype"]} progress events of the agent CLI, '
                         f'no text; fields {fields}; first and last values {json.dumps(counters, sort_keys=True)}; '
                         f'sha256 {hashlib.sha256(b"".join(raw for _, _, raw in run)).hexdigest()}')
            run.clear()

    with path.open('rb') as stream:
        for number, raw in enumerate(stream, 1):
            line = raw.decode('utf-8', errors='backslashreplace').rstrip('\n')
            event = progress_event(line)
            if event is None or (run and run[0][1]['subtype'] != event['subtype']):
                flush()
            if event is None:
                lines.append(f'L{number}: {line}')
            else:
                run.append((number, event, raw))
    flush()
    return '\n'.join(lines) + '\n'


COMMAND_CONTEXT = 2000  # characters of a command's text shown with its output
COMMAND_STREAM = re.compile(r'exec-(\d+)\.(?:stdout|stderr)')


def command_contexts(folder):
    """id -> the line heading that command's output: what the trusted broker journaled for it in exec.jsonl."""
    journal, started, status = Path(folder) / 'exec.jsonl', {}, {}
    if journal.is_file():
        with journal.open('rb') as stream:
            for raw in stream:
                try:
                    event = json.loads(raw)
                except ValueError:
                    continue
                if isinstance(event, dict) and type(event.get('id')) is int:
                    if event.get('event') == 'exec_started':
                        started[event['id']] = event
                    elif event.get('event') == 'exec_exited':
                        status[event['id']] = event.get('status')
    contexts = {}
    for number, event in started.items():
        text = event['command'] if isinstance(event.get('command'), str) else json.dumps(event.get('argv'))
        if len(text) > COMMAND_CONTEXT:
            text = f'{text[:COMMAND_CONTEXT]} [the first {COMMAND_CONTEXT} of {len(text)} characters]'
        contexts[number] = (f'COMMAND exec-{number} as the trusted broker journaled it ({Path(folder).name}/exec.jsonl): '
                            f'cwd {event.get("cwd")}; exit status {status.get(number, "not recorded")}; command: {text}')
    return contexts


def session_copy(path):
    """Claude Code's session file: the CLI's own record of the conversation, which the gateway journal and the captured
    stream hold authoritatively."""
    kinds, sidechain, lines = {}, 0, 0
    with Path(path).open('rb') as stream:
        for raw in stream:
            lines += 1
            try:
                value = json.loads(raw)
            except ValueError:
                value = None
            kind = str(value.get('type')) if isinstance(value, dict) else '(not a JSON object)'
            kinds[kind] = kinds.get(kind, 0) + 1
            sidechain += bool(isinstance(value, dict) and value.get('isSidechain'))
    return json.dumps({'kind': 'agent_cli_session_copy', 'sha256': digest(path), 'bytes': Path(path).stat().st_size,
                       'lines': lines, 'entries_by_type': dict(sorted(kinds.items())), 'sidechain_entries': sidechain,
                       'note': "The agent CLI's own copy of its conversation. The conversation is reviewed from its "
                               'authoritative records: external/gateway (every model request and response, as the '
                               'conversation index) and external/agent-native.jsonl (the stream the host captured). '
                               'The sealed bytes are retained and hash-bound.'}, sort_keys=True)


def manifest_view(path, journals):
    """A sealed directory's manifest with each model journal's blobs summarized (see the module docstring)."""
    value = read_json(path)
    files = value.get('files') if isinstance(value.get('files'), dict) else {}
    listed, blobs = {}, {}
    for name, entry in files.items():
        parts = name.split('/')
        if len(parts) >= 3 and parts[0] in journals and parts[1] == 'blobs':
            blobs.setdefault(parts[0], {})[name] = entry
        else:
            listed[name] = entry
    summary = {key: item for key, item in value.items() if key != 'files'}
    summary.update(files=listed, journal_blobs={
        journal: {'files': len(entries), 'bytes': sum(int(entry.get('bytes') or 0) for entry in entries.values()
                                                      if isinstance(entry, dict)),
                  'entries_sha256': hashlib.sha256(json.dumps(entries, sort_keys=True).encode()).hexdigest()}
        for journal, entries in sorted(blobs.items())})
    return ('SEALED MANIFEST ' + json.dumps({'raw_file': f'{path.parent.name}/{path.name}', 'sha256': digest(path),
                                              'note': 'Verified mechanically against every file; journal blobs are '
                                                      'summarized here and cited by digest in their journal indexes.'},
                                             sort_keys=True) + '\n' + json.dumps(summary, sort_keys=True, indent=0) + '\n')


def task_view(reference, path, views, errors):
    """A task's review view of one file under external/task/ or evaluation/, under the framework's header. None when no
    view applies, or when the view failed: the failure is recorded and the file is shown as it is."""
    for pattern, view in views:
        if not fnmatchcase(reference, pattern):
            continue
        try:
            text = view(path)
            if text is None:
                return None  # the view leaves this file to the generic presentation
            if not isinstance(text, str):
                raise TypeError('a review view returns text')
        except Exception as error:  # noqa: BLE001 - any failure falls back to the raw file
            errors.append({'reference': reference, 'pattern': pattern, 'error': f'{type(error).__name__}: {error}'[:2000]})
            return None
        return (f'TASK VIEW (review_views) of {reference}: sha256 {digest(path)}, bytes {path.stat().st_size}; raw bytes '
                'sealed and hash-bound\n' + text)
    return None


def packets(roots, mechanical=None, launch=None, config=None, views=(), view_errors=None):
    """The indexed profile's packets. ``views``: the task's review views ([(pattern, view(path))]), consulted only for
    files under external/task/ and evaluation/ after the framework's own rules; failures go to ``view_errors``."""
    mechanical = mechanical or {}
    config = config or {}
    view_errors = [] if view_errors is None else view_errors
    recomputed = tuple(config.get('recomputed') or ())  # [breakwater.scoring] recomputed
    metadata_paths = config.get('review_metadata_paths') or ()
    external = Path(roots['external'])
    journals = {folder.name for folder in external.iterdir()
                if folder.is_dir() and (folder / 'requests.jsonl').is_file()}
    identical = (mechanical.get('native_agreement') or {}).get('status') == 'PASS'
    baseline = None
    if (external / 'workspace-baseline.json').is_file():
        baseline = read_json(external / 'workspace-baseline.json').get('files')
    contexts = {}  # broker folder -> command id -> its journaled context
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
                if relative == 'manifest.json' and journals:
                    yield from chunks(reference + '[summary]', manifest_view(path, journals))
                    continue
                if relative == 'agent-native.jsonl':
                    yield from chunks(reference + '[index]', index_transcript(path))
                    continue
                stream = COMMAND_STREAM.fullmatch(parts[-1]) if len(parts) == 2 else None
                if stream and parts[0] in ('commands', 'evaluation-commands'):
                    if parts[0] not in contexts:
                        contexts[parts[0]] = command_contexts(root / parts[0])
                    context = contexts[parts[0]].get(int(stream.group(1)))
                    if context:
                        for piece, text in file_packets(reference, path):
                            yield piece, context + '\n' + text
                        continue
                if relative == 'environment-baseline.json':
                    value = read_json(path)
                    yield reference, json.dumps({'kind': 'mechanically_verified_runtime_inventory', 'sha256': digest(path),
                                                 'files': len(value.get('files', {})),
                                                 **({'image_sha256': value['image_sha256']} if 'image_sha256' in value else {}),
                                                 'specification': value['specification']})
                    continue
                if relative == 'workspace-baseline.json':
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
                # Apptainer backend: the brokers' per-command traces and the coordinator's trace.
                if (relative == 'intermediate.strace' or (len(parts) == 2 and parts[0] in ('commands', 'evaluation-commands')
                                                          and parts[1].startswith('exec-') and parts[1].endswith('.strace'))):
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
            if (label == 'agent-local' and 'gateway' in journals and (external / 'agent-native.jsonl').is_file()
                    and fnmatchcase(relative, 'sessions/projects/*/*.jsonl')):
                yield reference + '[summary]', session_copy(path)
                continue
            if label == 'agent-local' and identical and relative in ('native.jsonl', 'native.stdout', 'native.stderr'):
                yield reference, ('Byte-identical to external/agent-native.* by mechanical reconciliation '
                                  f'(sha256 {digest(path)}); reviewed once in the external copy.')
                continue
            if label == 'evaluation' and any(fnmatchcase(relative, pattern) for pattern in recomputed):
                yield recomputed_summary(reference, path, relative)
                continue
            if views and (label == 'evaluation' or (label == 'external' and parts[0] == 'task')):
                text = task_view(reference, path, views, view_errors)
                if text is not None:
                    yield from chunks(reference + '[task-view]', text)
                    continue
            yield from file_packets(reference, path, metadata=label == 'submission' and covered(relative, metadata_paths))
