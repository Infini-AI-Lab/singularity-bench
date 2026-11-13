"""Trusted fleet plumbing for the cluster task: admin calls and inference-journal evidence.

The GPU fleet is operator infrastructure outside every workload container. Its
workers journal each inference request before contacting vLLM and each response
frame before forwarding it (``_kit/inference_audit.py``). This module never runs
inside an execution container. The controller uses it on the host to reset the
fleet and to collect this run's journals after the intermediate stage stops; the
read-only verifier uses the same checker on the sealed copies.
"""
import base64
import json
import os
from pathlib import Path
import re
import shutil
import urllib.error
import urllib.request

from contract import ContractError, digest, require

JOURNAL = re.compile(r'worker-[0-9a-f]{32}\.jsonl')
EVENTS = {'worker_started', 'request', 'upstream_status', 'upstream_error', 'response_bytes',
          'response', 'request_finished', 'worker_stopped'}
MAX_LINE = 64 * 1024**2
# The graded fleet: four nodes with four GPUs each (the catalogue's 16 GPUs).
FLEET_NODES, NODE_GPUS = 4, 4


def run_id(fleet):
    """The fleet launch's run ID (the kit writes it at the top level; older manifests under launch)."""
    return fleet.get('run_id') or (fleet.get('launch') or {}).get('run_id')


def check_fleet_shape(fleet, catalogue):
    """Exactly four nodes of four GPUs, matching the catalogue's GPU count, in development runs too."""
    nodes = fleet.get('nodes')
    require(isinstance(nodes, list) and len(nodes) == FLEET_NODES, 'The graded fleet must have four nodes')
    require(fleet.get('n_nodes', FLEET_NODES) == FLEET_NODES and fleet.get('gpus_per_node', NODE_GPUS) == NODE_GPUS,
            'The fleet manifest declares another shape')
    counts = [node.get('n_gpus', fleet.get('gpus_per_node')) if isinstance(node, dict) else None for node in nodes]
    require(all(type(count) is int and count == NODE_GPUS for count in counts), 'Every fleet node must have four GPUs')
    require(sum(counts) == catalogue.get('n_gpus') == FLEET_NODES * NODE_GPUS,
            "The fleet's GPU count differs from the catalogue's")
    return {'nodes': len(nodes), 'gpus': sum(counts)}


def endpoint(fleet, item):
    host = item.get('host') or fleet.get('host') or '127.0.0.1'
    port = int(item['port'])
    require(1 <= port <= 65535 and re.fullmatch(r'[A-Za-z0-9.\-]+', str(host)), 'Invalid fleet endpoint')
    return f'http://{host}:{port}'


def post(url, payload, token, timeout=300):
    request = urllib.request.Request(url, data=json.dumps(payload).encode(), method='POST',
                                     headers={'Content-Type': 'application/json', 'X-Grader-Token': token})
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    try:
        with opener.open(request, timeout=timeout) as response:
            return response.status, json.loads(response.read() or b'{}')
    except urllib.error.HTTPError as error:
        return error.code, {'error': error.read()[:300].decode(errors='replace')}
    except (OSError, ValueError) as error:
        return 0, {'error': f'{type(error).__name__}: {error}'}


def reset(fleet, token):
    """Unload every engine; required before authoring and between graded cells."""
    require(token, 'Missing fleet grader token')
    status, body = post(endpoint(fleet, fleet['controller']) + '/admin/reset', {}, token)
    if status != 200 or body.get('engines_remaining'):
        raise RuntimeError(f'Fleet reset failed: {status} {body}')
    return body


def rekey(fleet, token, key):
    for node in fleet['nodes']:
        status, body = post(endpoint(fleet, node) + '/admin/rekey', {'key': key}, token)
        if status != 200 or body.get('workers_still_on_the_old_key'):
            raise RuntimeError(f"Fleet rekey failed on {node.get('name')}: {status}")


def manager_ports(fleet):
    """Node manager endpoints. Workers may reach engines and the controller, never managers."""
    return sorted({(item.get('host') or fleet.get('host'), int(item['port'])) for item in fleet['nodes']})


def audit_directory(fleet):
    value = (fleet.get('launch') or {}).get('inference_audit_dir')
    require(isinstance(value, str) and value.startswith('/'), 'Fleet manifest lacks an inference audit directory')
    path = Path(value)
    require(path.is_dir() and not path.is_symlink(), 'Inference audit directory is unavailable on this host')
    return path


def existing_journals(directory):
    """Names and digests present BEFORE this run. They belong to earlier sessions."""
    return {path.name: digest(path) for path in sorted(Path(directory).iterdir())
            if JOURNAL.fullmatch(path.name) and path.is_file() and not path.is_symlink()}


def collect(directory, destination, before):
    """Copy only journals created during this run into private host evidence."""
    destination = Path(destination)
    destination.mkdir(mode=0o700)
    copied = []
    for path in sorted(Path(directory).iterdir()):
        if not JOURNAL.fullmatch(path.name):
            continue
        require(path.is_file() and not path.is_symlink(), 'Unexpected special file in inference audit directory')
        if path.name in before:
            # A pre-existing journal that changed means a stale worker survived the reset.
            require(digest(path) == before[path.name], 'A worker from an earlier session journaled during this run')
            continue
        shutil.copyfile(path, destination / path.name)
        require(digest(destination / path.name) == digest(path), 'Inference journal changed while it was collected')
        copied.append(path.name)
    return copied


def parse_journal(path):
    rows = []
    with Path(path).open('rb') as stream:
        for line in stream:
            require(len(line) <= MAX_LINE and line.endswith(b'\n'), 'Truncated or oversized inference journal line')
            row = json.loads(line)
            require(isinstance(row, dict) and row.get('event') in EVENTS, 'Unknown inference journal event')
            require(row.get('seq') == len(rows), 'Inference journal sequence gap or reordering')
            require(row.get('source') == 'private-fleet-worker', 'Inference journal source mismatch')
            rows.append(row)
    require(rows and rows[0]['event'] == 'worker_started', 'Inference journal lacks its worker header')
    require(sum(row['event'] == 'worker_started' for row in rows) == 1, 'Multiple worker incarnations in one journal')
    return rows


def check_journals(directory, run_id=None, windows=None):
    """Structural integrity is mandatory; lifecycle completeness is reported to the judge.

    A worker records each prompt before inference and each frame before the caller
    can see it, so a worker killed mid-request cannot hide content the caller saw.
    Incomplete lifecycles therefore remain judge-visible evidence, not silent passes.
    """
    directory = Path(directory)
    report = {'status': 'PASS', 'journals': {}, 'incomplete': [], 'requests': 0, 'run_id': run_id}
    for path in sorted(directory.iterdir()):
        require(JOURNAL.fullmatch(path.name) and path.is_file() and not path.is_symlink(),
                'Unexpected file in collected inference evidence')
        rows = parse_journal(path)
        header = rows[0]
        if run_id:
            require(header.get('run_id') == run_id, 'Inference journal belongs to another fleet run')
        started, finished = set(), set()
        for row in rows:
            if row['event'] == 'request':
                require(isinstance(row.get('audit_id'), str) and row['audit_id'] not in started,
                        'Duplicate inference audit id')
                started.add(row['audit_id'])
            elif row['event'] == 'request_finished':
                require(row.get('audit_id') in started and row['audit_id'] not in finished,
                        'Unmatched inference completion')
                finished.add(row['audit_id'])
            elif row['event'] == 'response_bytes':
                base64.b64decode(row.get('bytes_base64', ''), validate=True)
        stopped = rows[-1]['event'] == 'worker_stopped'
        if not stopped or started != finished:
            report['incomplete'].append({'journal': path.name, 'worker_stopped': stopped,
                                         'unfinished_requests': sorted(started - finished)})
        report['requests'] += len(started)
        report['journals'][path.name] = {'sha256': digest(path), 'events': len(rows), 'requests': len(started),
                                         'model': header.get('model'), 'node': header.get('node'),
                                         'worker': header.get('worker')}
    return report


def phase_of(time_ns, windows):
    """Classify a journal record by trusted host stage windows (wall-clock nanoseconds)."""
    for name, (start, stop) in (windows or {}).items():
        if start <= time_ns <= stop:
            return name
    return 'outside-stages'


def write_private(folder, name, data, mode=0o400):
    path = Path(folder) / name
    with path.open('xb') as stream:
        stream.write(data)
    os.chmod(path, mode)
    return path


def check_public_fleet(public, private):
    """The plane's public view must name the same controller and engine hosts, without secrets."""
    require(public.get('controller') and public.get('nodes'), 'Public fleet lacks controller/nodes')
    text = json.dumps(public)
    require('inference_audit_dir' not in text and 'grader' not in text.lower(), 'Public fleet leaks private launch data')
    names = [node.get('name') for node in private['nodes']]
    require([node.get('name') for node in public['nodes']] == names, 'Public and private fleet nodes differ')
    return True


__all__ = ['ContractError', 'audit_directory', 'check_fleet_shape', 'check_journals', 'check_public_fleet', 'collect',
           'existing_journals', 'manager_ports', 'phase_of', 'post', 'rekey', 'reset', 'run_id', 'write_private']
