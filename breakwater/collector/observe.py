"""External Linux observer; runs in its own PID/filesystem/network domain.

Trusted launchers stop workers and stream supervisors before submitted execution.
The observer attaches from the daemon's host PID namespace, retaining configured
syscalls and protected stdio writes. No observer path is mounted in the target.
Recording failures and visibility limitations have separate protocol semantics.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import time
import sys
import traceback
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, str(Path(__file__).resolve().parent))
from syscall_trace import STRING_LIMIT, trace_options
from syscall_trace import inspect_traces
from contract import CONTRACT, PROTOCOL
from snapshots import Snapshots, atomic_json
from process_scope import pin, belongs

parser = argparse.ArgumentParser()
parser.add_argument("--identity", required=True)
parser.add_argument("--pid", type=int, required=True)
parser.add_argument("--uid", type=int, required=True)
parser.add_argument("--gid", type=int, required=True)
parser.add_argument('--source-roots', default='[]')
parser.add_argument('--source-excludes', default='[]')
parser.add_argument('--immutable-roots', default='[]')
parser.add_argument('--snapshot-interval-sec', type=float, default=3600)
parser.add_argument('--dataset-manifest')
parser.add_argument('--dataset-roots', default='[]')
args = parser.parse_args()
identity = json.loads(args.identity)
errors_seen = []
root = Path("/capture")
root.mkdir(mode=0o700, exist_ok=True)
syscalls = root / "syscalls"
syscalls.mkdir(mode=0o700, exist_ok=True)
# Pin the namespace object, not just its recyclable inode label. Main may stop
# before sidecar collection finishes; a later container must never be observed
# as this trial merely because Linux reused the old namespace number.
namespace_fd = os.open(f"/proc/{args.pid}/ns/pid", os.O_RDONLY | os.O_CLOEXEC)
namespace = os.readlink(f"/proc/self/fd/{namespace_fd}")
target_cgroups = pin(args.pid)
children = {}
supervisor_prefixes = []
supervisor_pids = set()
seq = 0
previous = "0" * 64
stopping = False
target_root_fd = os.open(f'/proc/{args.pid}/root', os.O_RDONLY | os.O_DIRECTORY)
snapshots = Snapshots(target_root_fd, root / 'source-snapshots', json.loads(args.source_roots),
    excludes=json.loads(args.source_excludes), interval_sec=args.snapshot_interval_sec,
    immutable_roots=json.loads(args.immutable_roots),
    dataset_manifest=args.dataset_manifest, dataset_roots=json.loads(args.dataset_roots))
snapshot_pool = ThreadPoolExecutor(max_workers=1)
snapshot_task = snapshot_pool.submit(snapshots.capture) if snapshots.roots else None
final_sources = False


def event(kind, **details):
    global seq, previous
    value = {"schema_version": 1, "seq": seq, "utc": datetime.now(timezone.utc).isoformat(),
             "monotonic_ns": time.monotonic_ns(), "source": "external-linux-syscalls",
             "trust_level": "externally_observed", "event": kind, "previous_sha256": previous,
             **identity, 'phase': 'validation' if (root / 'validation-phase').exists() else 'agent',
             'tool_id': 'linux-syscall', 'call_id': details.get('pid'), 'arguments': None,
             'cwd': None, 'user': None, 'started_at': None, 'ended_at': None,
             'exit_status': None, 'output': None, **details}
    previous = hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()
    value["sha256"] = previous
    with (root / "observer.jsonl").open("a") as stream:
        stream.write(json.dumps(value) + "\n")
        stream.flush()
        # Full lines reach the external file immediately. Batch durability
        # flushes while indexing; orderly finalization fsyncs the whole ledger.
        if seq % 256 == 0:
            os.fsync(stream.fileno())
    seq += 1


def stop(*_):
    global stopping
    stopping = True


signal.signal(signal.SIGTERM, stop)
signal.signal(signal.SIGINT, stop)
event("collector_ready", pid_namespace=namespace, cgroups=target_cgroups)
(root / 'contract.json').write_text(json.dumps(CONTRACT))
(root / "ready.json").write_text(json.dumps({"schema_version": 2, "protocol": PROTOCOL, "ready": True}))
try:
    while not stopping:
        if snapshot_task is not None and snapshot_task.done():
            report = snapshot_task.result()  # Recorder errors must not be silent.
            event('source_snapshot', sequence=report['sequence'], final=report['final'])
            if report['final']:
                atomic_json(root / 'sources-finalized.json', {'complete': True})
            snapshot_task = None
        if snapshot_task is None and not final_sources:
            if (root / 'finalize-sources').exists():
                final_sources = True
                snapshot_task = snapshot_pool.submit(snapshots.capture, final=True)
            elif snapshots.roots and snapshots.due():
                snapshot_task = snapshot_pool.submit(snapshots.capture)
        for path in Path("/proc").glob("[0-9]*/status"):
            try:
                pid = int(path.parent.name)
                # A protected host/NVIDIA process outside this container can deny
                # namespace-link access. It is outside the recorder's authority;
                # never inspect it or mistake that denial for a target capture gap.
                if pid in children or not belongs(pid, target_cgroups):
                    continue
                # The pinned, container-specific cgroup is authoritative here,
                # including nested PID namespaces. Namespace links can become
                # unreadable during setuid/GPU-helper transitions; inspecting
                # them is unnecessary once cgroup membership is established.
                fields = dict(line.split(":", 1) for line in path.read_text().splitlines() if ":" in line)
                if not {'State', 'TracerPid', 'Name', 'Uid', 'PPid'}.issubset(fields):
                    if not path.exists():
                        continue
                    raise RuntimeError('Incomplete live target process status')
                if fields["State"].strip()[0] != "T" or int(fields["TracerPid"]) != 0:
                    continue
                (root / f'initial-maps-{pid}.txt').write_bytes((path.parent / 'maps').read_bytes())
                is_supervisor = fields['Name'].strip() == 'gs-io-ready' and set(fields['Uid'].split()) == {'0'}
                if is_supervisor:
                    supervisor_prefixes.append(f'exec-{pid}')
                    supervisor_pids.add(pid)
                elif fields['Uid'].split()[1] == '0' and int(fields['PPid']) not in supervisor_pids:
                    raise RuntimeError('Worker observation barrier lacks a protected stream supervisor; rebuild runtime')
                command = ["strace", "-f", "-ff", "-ttt", "-T", "-v", "-s", STRING_LIMIT, "-xx", "-y", "--decode-pids=pidns",
                           *trace_options(supervisor=is_supervisor),
                           "-o", str(syscalls / f"exec-{pid}"), "-p", str(pid)]
                errors = (syscalls / f"exec-{pid}.stderr").open("wb")
                tracer = subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                                          stderr=errors)
                children[pid] = tracer, errors
                deadline = time.monotonic() + 10
                while time.monotonic() < deadline:
                    status = path.read_text()
                    tracer_pid = next(line for line in status.splitlines() if line.startswith("TracerPid:"))
                    if int(tracer_pid.split()[1]) != 0:
                        break
                    if tracer.poll() is not None:
                        raise RuntimeError("Could not attach before execution")
                    time.sleep(.005)
                else:
                    raise RuntimeError("Observer attach deadline exceeded")
                event("exec_attached", pid=pid, trace_prefix=f"syscalls/exec-{pid}",
                      protected_stream_supervisor=is_supervisor, parent_pid=int(fields['PPid']),
                      uids=fields['Uid'].split())
                os.kill(pid, signal.SIGCONT)
            except (FileNotFoundError, ProcessLookupError):
                continue
        time.sleep(.005)
except BaseException as exc:
    reason = f'{type(exc).__name__}: {exc}'
    errors_seen.append({"reason": reason})
    event("collector_error", error=reason)
    atomic_json(root / 'failure.json', {'complete': False, 'error': reason,
        'traceback': traceback.format_exc()[-16000:],
        'snapshot_pending': snapshot_task is not None, 'final_sources_requested': final_sources,
        'protected_supervisors_observed': len(supervisor_prefixes), 'attached_tracers': len(children)})
    if final_sources:
        atomic_json(root / 'sources-finalized.json', {'complete': False, 'error': reason})
    raise
finally:
    # Ordinary collection requests final source/data capture after worker drain,
    # before validation. On early collector failure, retain the last source view
    # but report failure rather than manufacture a successful final archive.
    snapshot_pool.shutdown(wait=True)
    if snapshot_task is not None:
        try:
            snapshot_task.result()
        except BaseException as exc:
            errors_seen.append({'reason': 'Source snapshot failed: ' + str(exc)})
    for tracer, errors in children.values():
        if tracer.poll() is None:
            tracer.send_signal(signal.SIGINT)
            try:
                tracer.wait(timeout=10)
            except subprocess.TimeoutExpired:
                tracer.kill()
                tracer.wait()
                errors_seen.append({"reason": "tracer did not drain"})
                event("capture_gap", reason="tracer did not drain")
        if tracer.returncode not in (0, -signal.SIGINT):
            errors_seen.append({"reason": "strace exited abnormally", "exit_status": tracer.returncode})
        errors.close()
    gaps = list(errors_seen)
    if snapshots.roots and not (snapshots.output / 'final.json').exists():
        gaps.append({'reason': 'Required final source/dataset archive did not complete'})
    for path in sorted(syscalls.iterdir()):
        if path.suffix == ".stderr":
            continue
        with path.open("rb") as stream:
            digest = hashlib.sha256()
            size = 0
            for line in stream:
                digest.update(line)
                size += len(line)
            # One inventory entry per trace, rather than a second JSON event for
            # every syscall. Native timestamps and line references stay intact.
            event('syscall_trace', output={'path': f'syscalls/{path.name}',
                'bytes': size, 'sha256': digest.hexdigest()},
                timestamp_semantics='indexing time; occurrence time in raw syscall records')
    report = inspect_traces([path for path in sorted(syscalls.iterdir()) if path.suffix != '.stderr'],
                            supervisor_prefixes=supervisor_prefixes, stream_root=root / 'streams')
    gaps.extend(report['gaps'])
    if not supervisor_prefixes:
        gaps.append({'reason': 'No protected stream supervisor was observed'})
    gaps = [dict(items) for items in sorted({tuple(sorted(gap.items())) for gap in gaps})]
    event("collector_stopped", complete=not gaps, gaps=gaps)
    with (root / "observer.jsonl").open("rb") as ledger:
        os.fsync(ledger.fileno())
    (root / "coverage.json").write_text(json.dumps({"schema_version": 2, "protocol": PROTOCOL, "complete": not gaps,
        "gaps": gaps, "sources": ["process", "file-syscalls", "terminal-write-bytes"],
        'visibility_notes': report['visibility_notes'], 'streams': report['streams'],
        'contract': CONTRACT,
        "last_event_sha256": previous}))
    for directory, dirs, files in os.walk(root):
        for path in [Path(directory), *(Path(directory) / name for name in files)]:
            os.chown(path, args.uid, args.gid)
            path.chmod(0o700 if path.is_dir() else 0o600)
    os.close(namespace_fd)
    os.close(target_root_fd)
