"""Trusted root coordinator for the coding-agent-harness validity battery (intermediate stage).

The scenario book, expected files and checks stay in this root process and under the
root-only ``/private`` parent. For each scenario a fresh worker child (UID 1101, via
``run_worker.py`` with its own external-observer barrier) imports the sealed harness and
calls ``run()``. Every model call crosses a pipe to this process, where the task's own
``ScriptedClient`` answers it and records it. After the child and all of its descendants
are gone, this process checks the scenario workspace without following links. The
authoritative raw record (calls, tools, faults, checks, exit) is written by root into the
root-only evaluation directory and recomputed by the final verifier.
"""
import json
import os
from pathlib import Path
import select
import shutil
import signal
import stat
import subprocess
import sys
import time

WORKER_UID, SHARED_GID = 1101, 1200
PER_SCENARIO_TIMEOUT = 300   # per-scenario hang catcher, as in tests/smoke/run_smoke.py
HARD_CALL_CAP = 400          # runaway-loop catcher, as in tests/smoke/run_smoke.py
MAX_LINE = 64 * 1024**2
MAX_READ = 16 * 1024**2
WORK_ROOT = Path('/eval-work')
# A harness is Python source plus the small text files it needs, never a way to carry another agent
# in: a submission carrying an executable, an archive, a symbolic link or more than 10 MiB is not run.
MAX_SUBMISSION_BYTES = 10 * 1024 * 1024
FORBIDDEN_MAGIC = (
    (b'\x7fELF', 'an ELF executable'), (b'MZ', 'a Windows executable'),
    (b'\xcf\xfa\xed\xfe', 'a Mach-O executable'), (b'\xfe\xed\xfa\xcf', 'a Mach-O executable'),
    (b'\xca\xfe\xba\xbe', 'a Mach-O or Java binary'), (b'PK\x03\x04', 'a zip archive'),
    (b'\x1f\x8b', 'a gzip archive'), (b'\xfd7zXZ\x00', 'an xz archive'), (b'BZh', 'a bzip2 archive'),
    (b'7z\xbc\xaf\x27\x1c', 'a 7z archive'), (b'\x28\xb5\x2f\xfd', 'a zstd archive'))


def static_run_binding(module):
    """Does agent.py bind ``run`` at module scope? Parsed by root, never executed.

    The child's ``imported``/``has_run`` status is a claim made inside a process the
    submission controls. This corroboration needs a module-level ``def``/``class``,
    assignment, or ``from ... import ... [as] run`` binding (including inside top-level
    if/try/with/for/while/match blocks). Every spec-conforming harness has one; a forged
    status line from a module without one cannot make it gradeable.
    """
    import ast
    try:
        tree = ast.parse(Path(module).read_bytes(), filename=str(module))
    except (OSError, SyntaxError, ValueError):
        return False

    def targets(node):
        if isinstance(node, ast.Name):
            yield node.id
        elif isinstance(node, (ast.Tuple, ast.List)):
            for item in node.elts:
                yield from targets(item)
        elif isinstance(node, ast.Starred):
            yield from targets(node.value)

    def binds(statements):
        for node in statements:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) and node.name == 'run':
                return True
            if isinstance(node, ast.Assign) and any('run' in targets(t) for t in node.targets):
                return True
            if isinstance(node, (ast.AnnAssign, ast.AugAssign)) and 'run' in targets(node.target):
                return True
            if isinstance(node, ast.ImportFrom) and any((alias.asname or alias.name) == 'run' for alias in node.names):
                return True
            nested = []
            for field in ('body', 'orelse', 'finalbody'):
                nested += getattr(node, field, []) if isinstance(node, (ast.If, ast.Try, ast.With, ast.For, ast.While,
                                                                         ast.AsyncWith, ast.AsyncFor)) else []
            if isinstance(node, ast.Try):
                for handler in node.handlers:
                    nested += handler.body
            if hasattr(ast, 'TryStar') and isinstance(node, ast.TryStar):
                nested += node.body + node.orelse + node.finalbody
                for handler in node.handlers:
                    nested += handler.body
            if hasattr(ast, 'Match') and isinstance(node, ast.Match):
                for case in node.cases:
                    nested += case.body
            if nested and binds(nested):
                return True
        return False

    return binds(tree.body)


def declared_tools(calls):
    """ScriptedClient.declared_tools, tolerant of malformed tool schemas (never crashes root)."""
    names = set()
    for call in calls:
        for tool in call.get('tools') or []:
            function = tool.get('function', tool) if isinstance(tool, dict) else {}
            if isinstance(function, dict) and function:
                names.add(str(function.get('name', '')))
    return sorted(names)


def kill_workers(timeout=60):
    """No submitted descendant survives into the next scenario or the next stage."""
    from workers import kill_workers as kill
    kill(timeout=timeout)


def safe_path(root, relative):
    """Resolve a scenario path without following any link component."""
    root = Path(root)
    parts = Path(relative).parts
    if Path(relative).is_absolute() or not parts or any(part in ('', '.', '..') for part in parts):
        return None
    path = root
    for part in parts:
        path = path / part
        try:
            info = os.lstat(path)
        except OSError:
            return None
        if stat.S_ISLNK(info.st_mode):
            return None
    return path


def safe_text(root, relative):
    path = safe_path(root, relative)
    if path is None:
        return None
    try:
        descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
    except OSError:
        return None
    with os.fdopen(descriptor, 'rb') as stream:
        if not stat.S_ISREG(os.fstat(stream.fileno()).st_mode):
            return None
        data = stream.read(MAX_READ + 1)
    return None if len(data) > MAX_READ else data.decode('utf-8', errors='replace')


def check(scenario, workspace, record):
    """The smoke checks of tests/smoke/run_smoke.py (_check), performed by root without following links."""
    expect = scenario.get('expect') or {}
    checks = {}
    for relative, wanted in (expect.get('file_is') or {}).items():
        checks[f'file_is:{relative}'] = safe_text(workspace, relative) == wanted
    for relative in expect.get('file_exists') or []:
        path = safe_path(workspace, relative)
        checks[f'exists:{relative}'] = path is not None and (path.is_file() or path.is_dir())
    if expect.get('file_contains'):
        relative, needle = expect['file_contains']
        text = safe_text(workspace, relative)
        checks[f'contains:{relative}'] = text is not None and needle in text
    if expect.get('max_model_calls') is not None:
        checks['within_budget'] = int(record.get('n_calls', 10 ** 6)) <= int(expect['max_model_calls'])
    return checks


def prepare(scenario, work):
    workspace = work / 'ws'
    workspace.mkdir(parents=True)
    for relative, body in (scenario.get('files') or {}).items():
        if Path(relative).is_absolute() or '..' in Path(relative).parts:
            raise ValueError('Unsafe trusted scenario path')
        path = workspace / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(body)
    for directory, dirs, files in os.walk(workspace):
        for path in [Path(directory), *(Path(directory) / name for name in files)]:
            os.chown(path, WORKER_UID, SHARED_GID)
            path.chmod(0o2775 if path.is_dir() else 0o664)
    task = work / 'task.txt'
    task.write_text(scenario['task'])
    task.chmod(0o644)
    work.chmod(0o711)
    return workspace, task


def child_command(module, workspace, task, seconds):
    """The worker launch: protected run_worker.py barrier, then the fixed public runner."""
    return ['python3', '-I', '-B', '/opt/breakwater-runtime/run_worker.py', 'python3', '-I', '-B',
            '/opt/breakwater/harness_child.py', '--module', str(module), '--workspace', str(workspace),
            '--task-file', str(task), '--seconds', str(seconds)]


def run_scenario(index, scenario, module, client_class, out):
    scenario_out = out / f'{index:02d}-{scenario["id"]}'
    scenario_out.mkdir(mode=0o700)
    work = WORK_ROOT / f'{index:02d}-{scenario["id"]}'
    workspace, task = prepare(scenario, work)
    client = client_class(scenario['script'], scenario.get('faults'))
    record = {'id': scenario['id'], 'raised': None, 'n_calls': 0, 'tools': [], 'imported': False,
              'has_run': False, 'child_status': [], 'protocol_errors': []}
    command = child_command(module, workspace, task, float(scenario.get('seconds', 240)))
    calls = 0
    with (scenario_out / 'child.stderr').open('wb') as errors:
        process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=errors,
                                   start_new_session=True)
        deadline = time.monotonic() + PER_SCENARIO_TIMEOUT
        buffer, done = b'', False
        try:
            while not done:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    record['fatal'] = f'run() did not return within {PER_SCENARIO_TIMEOUT}s'
                    record['timed_out'] = True
                    break
                ready, _, _ = select.select([process.stdout], [], [], min(remaining, 1.0))
                if not ready:
                    if process.poll() is not None:
                        break
                    continue
                chunk = os.read(process.stdout.fileno(), 1 << 20)
                if not chunk:
                    break
                buffer += chunk
                if len(buffer) > MAX_LINE and b'\n' not in buffer:
                    record['protocol_errors'].append('oversized model-channel message')
                    break
                while b'\n' in buffer:
                    line, buffer = buffer.split(b'\n', 1)
                    try:
                        message = json.loads(line)
                        operation = message['op']
                    except (ValueError, KeyError, TypeError):
                        record['protocol_errors'].append('malformed model-channel message')
                        continue
                    if operation == 'status':
                        record['child_status'].append({key: message.get(key) for key in ('imported', 'has_run', 'fatal')})
                        if len(record['child_status']) > 1:
                            # The fixed runner sends exactly one status; more means the channel was forged.
                            record['protocol_errors'].append('duplicate status claim')
                            record['imported'] = record['has_run'] = False
                            record['fatal'] = 'model channel carried more than one status claim'
                        else:
                            record['imported'] = bool(message.get('imported'))
                            record['has_run'] = bool(message.get('has_run'))
                            if message.get('fatal'):
                                record['fatal'] = str(message['fatal'])[:300]
                    elif operation == 'complete':
                        calls += 1
                        if calls > HARD_CALL_CAP:
                            reply = {'ok': False, 'error_type': 'RuntimeError',
                                     'message': f'smoke call cap reached ({HARD_CALL_CAP})'}
                        else:
                            try:
                                kw = message.get('kw') if isinstance(message.get('kw'), dict) else {}
                                response = client.complete(message.get('messages'), message.get('tools'), **kw)
                                reply = {'ok': True, 'response': response}
                            except Exception as error:  # noqa: BLE001 - scripted faults and malformed calls
                                reply = {'ok': False, 'error_type': type(error).__name__, 'message': str(error)[:500]}
                        try:
                            process.stdin.write(json.dumps(reply).encode() + b'\n')
                            process.stdin.flush()
                        except (BrokenPipeError, OSError):
                            record['protocol_errors'].append('child closed its reply channel')
                    elif operation == 'done':
                        record['raised'] = message.get('raised')
                        done = True
                        break
                    else:
                        record['protocol_errors'].append('unknown model-channel operation')
        finally:
            try:
                process.stdin.close()
            except OSError:
                pass
            if not done and process.poll() is None:
                process.kill()
            try:
                exit_status = process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                process.kill()
                exit_status = process.wait()
            kill_workers()
    record['child_exit'] = exit_status
    if not record['imported'] and 'fatal' not in record:
        record['fatal'] = 'no result from the harness process; see child.stderr'
    # Trusted observations, not claims from the child.
    record['n_calls'] = len(client.calls)
    record['parent_counted_calls'] = calls
    record['tools'] = declared_tools(client.calls)
    record['acted_on_prose'] = bool(client.acted_on_prose)
    record['checks'] = check(scenario, workspace, record)
    record['reports'] = scenario.get('reports')
    with (scenario_out / 'calls.jsonl').open('w') as stream:
        for call in client.calls:
            stream.write(json.dumps(call, default=str, allow_nan=False) + '\n')
    (scenario_out / 'record.json').write_text(json.dumps(record, sort_keys=True, default=str))
    shutil.rmtree(work, ignore_errors=True)
    return record


def recorded_links(harness_dir):
    """Links the seal recorded under the harness directory (the sealed copy holds no links)."""
    harness_dir = Path(harness_dir)
    manifest = harness_dir.parent / 'manifest.json'
    if not manifest.is_file():
        return []
    prefix = harness_dir.name + '/'
    return sorted(name for name in json.loads(manifest.read_text()).get('symlinks') or {} if name.startswith(prefix))


def payload_problems(harness_dir, links=()):
    """Why the submitted harness directory is more than a harness, read by root without following links."""
    root = Path(harness_dir)
    problems, total = [f'{name}: a symlink' for name in links], 0
    for path in sorted(root.rglob('*')):
        relative = path.relative_to(root).as_posix()
        mode = path.lstat().st_mode
        if stat.S_ISLNK(mode):
            problems.append(f'{relative}: a symlink')
            continue
        if not stat.S_ISREG(mode):
            continue
        total += path.lstat().st_size
        descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC)
        with os.fdopen(descriptor, 'rb') as stream:
            head = stream.read(262)
        kind = next((label for magic, label in FORBIDDEN_MAGIC if head.startswith(magic)), None)
        if kind is None and head[257:262] == b'ustar':
            kind = 'a tar archive'
        if kind:
            problems.append(f'{relative}: {kind}')
    if total > MAX_SUBMISSION_BYTES:
        problems.append(f'{total} bytes in total, over the {MAX_SUBMISSION_BYTES}-byte limit for source')
    return problems


def run_battery(module, smoke, out):
    """-> (rows, summary). ``smoke`` is the trusted /private/tests/smoke directory."""
    sys.path.insert(0, str(smoke))
    from scripted import ScriptedClient
    import run_smoke
    scenarios = json.loads((Path(smoke) / 'scenarios.json').read_text())
    out = Path(out)
    out.mkdir(parents=True, mode=0o700)
    WORK_ROOT.mkdir(mode=0o711, exist_ok=True)
    rows = []
    problems = (payload_problems(Path(module).parent, recorded_links(Path(module).parent))
                if Path(module).is_file() else [])
    if not Path(module).is_file():
        summary = {'gradeable': 0.0, 'valid': 0.0, 'error_no_submission': 1.0}
    elif problems:
        # Not run: the harness carries an executable, an archive, a link or bulk.
        summary = {'gradeable': 0.0, 'valid': 0.0, 'error_invalid_payload': 1.0}
    else:
        for index, scenario in enumerate(scenarios):
            rows.append(run_scenario(index, scenario, Path(module), ScriptedClient, out))
        summary = run_smoke.summarise(rows)
    static = Path(module).is_file() and static_run_binding(module)
    # The reward: the task's own structural gradeability, corroborated by root's parse.
    reward = 1.0 if summary.get('gradeable') == 1.0 and static else 0.0
    (out / 'battery.json').write_text(json.dumps({'scenario_ids': [row['id'] for row in scenarios]
                                                  if isinstance(scenarios, list) else [],
                                                  'rows': rows, 'summary': summary, 'static_run_binding': bool(static),
                                                  'payload_problems': problems, 'reward': reward},
                                                 sort_keys=True, default=str))
    return rows, {**summary, 'static_run_binding': float(bool(static)), 'reward': reward}
