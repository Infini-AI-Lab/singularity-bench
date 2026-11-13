"""Trusted cluster-serverless grading coordinator (root, fresh intermediate container).

Only the submitted control plane runs as the worker (UID 1101, through run_worker.py
and its external-observer barrier), once per graded cell, from fresh copies of the
sealed workspace at /app and of the sealed worker home at /home/worker (its HOME, as while authoring).
Before each cell every worker-owned file in /tmp, /var/tmp and /dev/shm is removed,
and the plane runs with RLIMIT_NPROC 512 on its own half of the container's cpuset.
Private traces, the graded fleet manifest, the fleet admin token and every per-cell
receipt key stay with root under the 0700 /private parent and never enter the worker's
environment, argv or filesystem view. The trusted load driver runs as root from
/private/tests/kit with a controlled interpreter path and never imports from /app.
Worker egress is limited to loopback plus the fleet controller and engine ports;
manager ports are rejected. Raw records are written by root into the root-only
evaluation directory and are recomputed by the read-only verifier.
"""
import contextlib
import hashlib
import json
import math
import os
from pathlib import Path
import secrets
import signal
import subprocess
import sys
import threading
import time

WORKER_UID, SHARED_GID = 1101, 1200
PLANE_CPUS = 8
PLANE_PROCESSES = 512
LOG_LIMIT = 8 << 20
PORT = 8000
# The submission contract: /app and the home together, at most 8 GiB and 100,000 entries.
MAX_BYTES, MAX_ENTRIES = 8 << 30, 100_000
TESTS = Path('/private/tests')
PRIVATE = Path('/private/cluster')
SUBMISSION = Path('/submission')
HOME_SUBMISSION = Path('/submission-home')
APP = Path('/app')
HOME = Path('/home/worker')  # the authoring HOME, so absolute paths carry over
# Run by root just before run_worker.py drops to the worker: lowering a limit needs no capability.
NPROC_LIMIT = ('import os,resource,sys\n'
               'resource.setrlimit(resource.RLIMIT_NPROC,(int(sys.argv[1]),int(sys.argv[1])))\n'
               'os.execvp(sys.argv[2],sys.argv[2:])')


def read_json(path):
    return json.loads(Path(path).read_text())


def cpu_split():
    allowed = sorted(os.sched_getaffinity(0))
    if len(allowed) < 2 * PLANE_CPUS:
        return None, None
    return set(allowed[:PLANE_CPUS]), set(allowed[PLANE_CPUS:])


@contextlib.contextmanager
def pinned(cpus):
    if not cpus:
        yield
        return
    previous = os.sched_getaffinity(0)
    os.sched_setaffinity(0, cpus)
    try:
        yield
    finally:
        os.sched_setaffinity(0, previous)


def stop_plane(timeout=60):
    """Every process of the launch -- and anything else the worker started -- is gone."""
    sys.path.insert(0, '/opt/breakwater')
    from workers import kill_workers
    kill_workers(grace=5, timeout=timeout)


def firewall(fleet):
    """Worker egress: loopback, the fleet controller and engine ports; never node managers."""
    sys.path.insert(0, '/opt/breakwater')
    import socket
    rules = [['-A', 'OUTPUT', '-m', 'owner', '--uid-owner', str(WORKER_UID), '-d', '127.0.0.0/8', '-j', 'ACCEPT']]
    def addresses(host):
        return sorted({row[4][0] for row in socket.getaddrinfo(host, None, socket.AF_INET, socket.SOCK_STREAM)})
    for node in fleet['nodes']:
        for address in addresses(node.get('host') or fleet['host']):
            rules.append(['-A', 'OUTPUT', '-m', 'owner', '--uid-owner', str(WORKER_UID), '-p', 'tcp',
                          '-d', address, '--dport', str(int(node['port'])), '-j', 'REJECT'])
    for node in fleet['nodes']:
        for address in addresses(node.get('host') or fleet['host']):
            rules.append(['-A', 'OUTPUT', '-m', 'owner', '--uid-owner', str(WORKER_UID), '-p', 'tcp',
                          '-d', address, '--dport', '9101:9299', '-j', 'ACCEPT'])
    controller = fleet['controller']
    for address in addresses(controller.get('host') or fleet['host']):
        rules.append(['-A', 'OUTPUT', '-m', 'owner', '--uid-owner', str(WORKER_UID), '-p', 'tcp',
                      '-d', address, '--dport', str(int(controller['port'])), '-j', 'ACCEPT'])
    rules.append(['-A', 'OUTPUT', '-m', 'owner', '--uid-owner', str(WORKER_UID), '-j', 'REJECT'])
    subprocess.run(['iptables', '-F', 'OUTPUT'], check=True)
    for rule in rules:
        subprocess.run(['iptables', *rule], check=True)
    applied = [' '.join(rule) for rule in rules]
    # IPv6 fails closed: if any non-loopback interface carries an IPv6 address, the worker's
    # IPv6 egress must be rejected (loopback only) or grading does not start.
    interfaces = Path('/proc/net/if_inet6')
    present = interfaces.is_file() and any(line.split()[-1] != 'lo'
                                           for line in interfaces.read_text().splitlines() if line.split())
    if present:
        for rule in (['-F', 'OUTPUT'],
                     ['-A', 'OUTPUT', '-m', 'owner', '--uid-owner', str(WORKER_UID), '-d', '::1', '-j', 'ACCEPT'],
                     ['-A', 'OUTPUT', '-m', 'owner', '--uid-owner', str(WORKER_UID), '-j', 'REJECT']):
            subprocess.run(['ip6tables', *rule], check=True)
            applied.append('ip6tables ' + ' '.join(rule))
    else:
        applied.append('ipv6: no non-loopback IPv6 interface')
    return applied


def submission_problem(roots):
    """Why the sealed submission breaks the 8 GiB / 100,000-entry contract, or None.

    Every root, directory, regular file and recorded symbolic link is an entry; the
    host's seal manifest at each root is not part of the submission.
    """
    entries = size = 0
    for root in roots:
        root = Path(root)
        manifest = root / 'manifest.json'
        links = read_json(manifest).get('symlinks') or {} if manifest.is_file() else {}
        entries += 1 + len(links) - manifest.is_file()
        size -= manifest.stat().st_size if manifest.is_file() else 0
        for directory, dirs, files in os.walk(root):
            entries += len(dirs) + len(files)
            size += sum((Path(directory) / name).lstat().st_size for name in files)
    if size > MAX_BYTES:
        return 'submission exceeds the 8 GiB artifact budget'
    if entries > MAX_ENTRIES:
        return 'submission exceeds 100000 filesystem entries'
    return None


def install_app(catalogue, public_fleet):
    """Fresh worker-owned copies of the sealed /app and home for each cell; graded public inputs.

    Recorded symbolic links are recreated without being followed; one that resolves
    outside /app and /home/worker makes the submission invalid (workers.install_tree).
    """
    sys.path.insert(0, '/opt/breakwater')
    from workers import clear, install_tree
    pairs = [(SUBMISSION, APP)]
    if HOME_SUBMISSION.is_dir():
        pairs.append((HOME_SUBMISSION, HOME))
    else:
        clear(HOME)
        os.chown(HOME, WORKER_UID, SHARED_GID)
        HOME.chmod(0o2775)
    install_tree(pairs, allowed=(HOME,))
    for source, target in ((catalogue, APP / 'catalogue.json'), (public_fleet, APP / 'fleet.json')):
        target.unlink(missing_ok=True)  # also removes a submitted link of that name, never following it
        target.write_bytes(Path(source).read_bytes())
        target.chmod(0o444)


def prepare_cell(catalogue, public_fleet):
    """A fresh filesystem view for one cell. No worker process may be running."""
    sys.path.insert(0, '/opt/breakwater')
    from workers import wipe_worker_files
    wipe_worker_files()
    temporary = Path('/tmp/worker')
    temporary.mkdir(mode=0o700, exist_ok=True)
    os.chown(temporary, WORKER_UID, SHARED_GID)
    temporary.chmod(0o700)
    install_app(catalogue, public_fleet)


def drain(pipe, output):
    remaining = LOG_LIMIT
    while chunk := os.read(pipe.fileno(), 65536):
        if remaining:
            output.write(chunk[:remaining])
            output.flush()
            remaining = max(0, remaining - len(chunk))


def driver_timeout(bound_s, time_scale):
    """The driver's wall-clock limit: the trace-time replay bound on the scaled clock, plus slack."""
    return bound_s / time_scale + 120


def read_records(path):
    """A killed driver can leave one unfinished final journal line (grade.py semantics)."""
    lines = path.read_text().splitlines(keepends=True)
    out = []
    for index, line in enumerate(lines):
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            if index == len(lines) - 1 and not line.endswith('\n'):
                break
            raise
    return out


def run_cells(raw, options):
    """Grade every cell and repeat. Returns the provisional final report (recomputed later)."""
    sys.path.insert(0, str(TESTS / 'kit'))
    sys.path.insert(0, '/opt/breakwater')
    import cluster_fleet
    import reference
    import score as scorer
    import traceunits
    raw = Path(raw)
    raw.mkdir(parents=True, exist_ok=True)
    fleet, public_fleet = read_json(PRIVATE / 'fleet.json'), read_json(PRIVATE / 'fleet.public.json')
    token = (PRIVATE / 'grader-token').read_text().strip()
    cluster_fleet.check_public_fleet(public_fleet, fleet)
    suite = read_json(TESTS / 'suite-graded.json')
    traces = [(name, TESTS / f'trace-graded-{name}.jsonl') for name in scorer.CELLS]
    if suite.get('schema_version') != 4 or [case['name'] for case in suite['cases']] != [name for name, _ in traces]:
        raise ValueError('unexpected suite definition')
    # A target-qualified suite is always graded with its measurement binding; a development
    # (unqualified) suite is graded only when the task contract declares allow_unqualified.
    qualified = set(suite['qualification'].values()) == {'target-qualified'}
    development_only = not qualified
    if development_only and not options.get('allow_unqualified'):
        raise RuntimeError('suite has no hardware preparation; a development run must be declared explicitly')
    fingerprints = {name: hashlib.sha256(path.read_bytes()).hexdigest() for name, path in traces}
    for case in suite['cases']:
        if fingerprints[case['name']] != case['trace_sha256']:
            raise ValueError(f"trace fingerprint mismatch: {case['name']}")
    catalogue = TESTS / 'public' / 'catalogue.json'
    public_catalogue = read_json(catalogue)
    public_hash = hashlib.sha256(json.dumps(public_catalogue, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
    if suite.get('public_catalogue_sha256') and public_hash != suite['public_catalogue_sha256']:
        raise ValueError('public catalogue fingerprint mismatch')
    if set(suite['model_ids']) != {model['id'] for model in public_catalogue['models']}:
        raise ValueError('suite and catalogue disagree')
    shape = cluster_fleet.check_fleet_shape(fleet, public_catalogue)
    cluster_fleet.check_fleet_shape(public_fleet, public_catalogue)
    setup_s, retries = float(suite['setup_s']), int(suite['session_retries'])
    repeats = int(options.get('repeats', 1))
    time_scale = float(options.get('time_scale', 1))
    if not (math.isfinite(time_scale) and time_scale > 0):
        raise ValueError('invalid time scale')
    controller = cluster_fleet.endpoint(fleet, fleet['controller'])
    def nodes():
        import urllib.request
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        with opener.open(controller + '/nodes', timeout=60) as response:
            return json.loads(response.read())
    # Four trays of four B300-named GPUs (GB300 matches) with 16 distinct UUIDs, in development runs too.
    hardware = reference.hardware_identity(nodes())
    binding = None
    if not development_only:
        reference.validate_qualification(str(TESTS), fleet=fleet, nodes=nodes(), runtime=reference.driver_runtime())
        binding = reference.binding(str(TESTS), fleet, nodes(), setup_s=setup_s, retries=retries, time_scale=time_scale)
    rules = firewall(fleet)
    plane_cpus, driver_cpus = cpu_split()
    roots = [SUBMISSION] + ([HOME_SUBMISSION] if HOME_SUBMISSION.is_dir() else [])
    invalid = submission_problem(roots)
    if invalid is None:
        sys.path.insert(0, '/opt/breakwater')
        from workers import InvalidSubmission
        try:
            install_app(catalogue, PRIVATE / 'fleet.public.json')  # the same install each cell repeats
            if not (APP / 'run.sh').is_file():
                invalid = '/app/run.sh is missing'
        except InvalidSubmission as error:
            invalid = str(error)
    labels, cells = [], []
    driver_home = Path('/private/driver-home')
    driver_home.mkdir(mode=0o700, exist_ok=True)
    try:
        for name, trace in traces:
            trace_items = [json.loads(line) for line in trace.read_text().splitlines()]
            for repeat in range(repeats):
                tag = f'{name}-{repeat}'
                labels.append(tag)
                cell = {'label': tag, 'trace': name, 'invalid': invalid}
                if invalid is not None:
                    cells.append(cell)
                    continue
                cluster_fleet.reset(fleet, token)
                # A fresh key for every cell retires development traffic and earlier cells.
                key = secrets.token_hex(32)
                cluster_fleet.rekey(fleet, token, key)
                prepare_cell(catalogue, PRIVATE / 'fleet.public.json')
                marker = secrets.token_hex(16)
                records, stats = raw / f'records-{tag}.jsonl', raw / f'stats-{tag}.json'
                with (raw / f'plane-{tag}.log').open('wb') as log:
                    started = time.monotonic()
                    with pinned(plane_cpus):
                        plane = subprocess.Popen(
                            [sys.executable, '-I', '-B', '-c', NPROC_LIMIT, str(PLANE_PROCESSES),
                             'python3', '-I', '-B', '/opt/breakwater-runtime/run_worker.py', '/usr/bin/env',
                             f'PORT={PORT}', 'CLUSTER_NODES=/app/fleet.json', 'CLUSTER_CATALOGUE=/app/catalogue.json',
                             f"CLUSTER_TIME_SCALE={options.get('time_scale', 1)}", f'CLUSTER_GRADED_CELL_ID={marker}',
                             '/bin/bash', str(APP / 'run.sh')],
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                            start_new_session=True)
                    reader = threading.Thread(target=drain, args=(plane.stdout, log), daemon=True)
                    reader.start()
                    timeout = driver_timeout(traceunits.replay_bound_s(trace_items, setup_s, retries), time_scale)
                    environment = {'PATH': '/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin',
                                   'HOME': str(driver_home), 'LANG': 'C.UTF-8', 'PYTHONNOUSERSITE': '1',
                                   'PYTHONDONTWRITEBYTECODE': '1', 'CLUSTER_RECEIPT_KEY': key,
                                   'CLUSTER_GRADER_TOKEN': token, 'CLUSTER_CELL_STARTED_AT': str(started),
                                   'CLUSTER_SETUP_S': str(setup_s), 'CLUSTER_SESSION_RETRIES': str(retries),
                                   'CLUSTER_TIME_SCALE': str(options.get('time_scale', 1))}
                    try:
                        with pinned(driver_cpus), (raw / f'loadgen-{tag}.stdout').open('wb') as out, \
                                (raw / f'loadgen-{tag}.stderr').open('wb') as err:
                            result = subprocess.run(
                                [sys.executable, '-B', str(TESTS / 'kit' / 'loadgen.py'), '--trace', str(trace),
                                 '--fleet', str(PRIVATE / 'fleet.json'), '--target', f'http://127.0.0.1:{PORT}',
                                 '--out', str(records), '--stats-out', str(stats)],
                                stdin=subprocess.DEVNULL, stdout=out, stderr=err, cwd=str(TESTS / 'kit'),
                                env=environment, timeout=timeout)
                        cell['driver_exit'] = result.returncode
                        if result.returncode:
                            raise RuntimeError(f'load driver failed with exit {result.returncode}')
                    except subprocess.TimeoutExpired as error:
                        raise RuntimeError('driver exceeded the full per-request/retry drain bound') from error
                    finally:
                        stop_plane()
                        try:
                            plane.wait(timeout=10)
                        except subprocess.TimeoutExpired:
                            plane.kill()
                        reader.join(timeout=10)
                cell['records'] = records.name if records.exists() else None
                cell['stats'] = stats.name if stats.exists() else None
                if records.exists():
                    summary = scorer.summarize(read_records(records), trace_items,
                                               read_json(stats) if stats.exists() else None)
                    cell['provisional_utility'] = summary['utility']
                    cell['provisional_gates_pass'] = scorer.gates_pass(summary)
                cells.append(cell)
    finally:
        cluster_fleet.reset(fleet, token)
    after = None
    if binding is not None:
        after = reference.binding(str(TESTS), fleet, nodes(), setup_s=setup_s, retries=retries, time_scale=time_scale)
        if after != binding:
            raise RuntimeError('fleet/runtime identity changed during grading')
    measurement = {'labels': labels, 'cells': cells, 'invalid': invalid, 'development_only': development_only,
                   'measurement_binding': binding, 'trace_sha256': fingerprints, 'repeats': repeats,
                   'setup_s': setup_s, 'session_retries': retries, 'score_version': scorer.SCORE_VERSION,
                   'model_ids': suite['model_ids'], 'worker_egress': rules, 'fleet_shape': shape,
                   'hardware_identity': hardware, 'plane_process_limit': PLANE_PROCESSES,
                   'cpu_isolation': {'plane': sorted(plane_cpus), 'driver': len(driver_cpus)} if plane_cpus else None,
                   'fleet_sha256': hashlib.sha256((PRIVATE / 'fleet.json').read_bytes()).hexdigest(),
                   'public_fleet_sha256': hashlib.sha256((PRIVATE / 'fleet.public.json').read_bytes()).hexdigest()}
    (raw / 'measurement.json').write_text(json.dumps(measurement, sort_keys=True))
    return measurement
