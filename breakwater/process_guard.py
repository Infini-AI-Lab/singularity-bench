"""Trusted observer-side PID/cgroup drain attestation. Never executes target code."""
import argparse
import json
import os
from pathlib import Path, PurePosixPath
import time


def cgroups(pid):
    result = []
    for line in Path(f'/proc/{pid}/cgroup').read_text().splitlines():
        hierarchy, controllers, path = line.split(':', 2)
        parts = PurePosixPath(path).parts
        if not path.startswith('/') or '..' in parts:
            raise ValueError('Observer needs the host cgroup namespace')
        result.append({'hierarchy': hierarchy, 'controllers': controllers, 'path': path})
    return result


def capture(pid, container_id, identity):
    groups = [row for row in cgroups(pid) if container_id in row['path']]
    # Never accidentally inspect/kill arbitrary host processes or accept a pseudo-root.
    if not groups or not any(container_id in row['path'] for row in groups):
        raise ValueError('Cannot pin this container to a specific host cgroup')
    return {'container_id': container_id, 'identity': identity,
            'pid_namespace': os.readlink(f'/proc/{pid}/ns/pid'), 'cgroups': groups}


def remaining(boundary):
    found = []
    for path in Path('/proc').glob('[0-9]*/cgroup'):
        pid = int(path.parent.name)
        try:
            groups = cgroups(pid)
        except (FileNotFoundError, ProcessLookupError):
            continue
        same_cgroup = any(current['hierarchy'] == original['hierarchy'] and
                          (current['path'] == original['path'] or current['path'].startswith(original['path'].rstrip('/') + '/'))
                          for current in groups for original in boundary['cgroups'])
        if same_cgroup:
            found.append(pid)
            continue
        try:
            namespace = os.readlink(path.parent / 'ns/pid')
        except (FileNotFoundError, ProcessLookupError, PermissionError):
            # Cgroups already prove this is outside the target's process tree.
            # Unrelated protected host processes need not expose namespace links.
            continue
        if namespace == boundary['pid_namespace']:
            found.append(pid)
    return sorted(set(found))


def drained(boundary, timeout=10):
    deadline = time.monotonic() + timeout
    first_empty = False
    while time.monotonic() < deadline:
        pids = remaining(boundary)
        if not pids:
            if first_empty:
                return {**boundary, 'status': 'PASS', 'remaining_pids': [], 'checked_monotonic_ns': time.monotonic_ns()}
            first_empty = True
        else:
            first_empty = False
        time.sleep(.1)
    return {**boundary, 'status': 'FAIL', 'remaining_pids': remaining(boundary),
            'checked_monotonic_ns': time.monotonic_ns()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=('capture', 'drain'))
    parser.add_argument('--pid', type=int)
    parser.add_argument('--container-id')
    parser.add_argument('--identity')
    parser.add_argument('--boundary', type=Path, default=Path('/capture/process-boundary.json'))
    parser.add_argument('--output', type=Path, default=Path('/capture/process-drain.json'))
    args = parser.parse_args()
    if args.mode == 'capture':
        value = capture(args.pid, args.container_id, json.loads(args.identity))
        args.boundary.write_text(json.dumps(value, sort_keys=True) + '\n')
    else:
        value = drained(json.loads(args.boundary.read_text()))
        args.output.write_text(json.dumps(value, sort_keys=True) + '\n')
        if value['status'] != 'PASS':
            raise SystemExit(2)


if __name__ == '__main__':
    main()
