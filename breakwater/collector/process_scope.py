"""Scope /proc enumeration before opening namespace links of unrelated processes."""
from pathlib import Path
import re


def cgroups(pid, proc=Path('/proc')):
    rows = []
    for line in (proc / str(pid) / 'cgroup').read_text().splitlines():
        hierarchy, controllers, path = line.split(':', 2)
        rows.append({'hierarchy': hierarchy, 'controllers': controllers, 'path': path})
    return rows


def pin(pid):
    rows = [row for row in cgroups(pid) if re.search(r'(?<![a-f0-9])[a-f0-9]{64}(?![a-f0-9])', row['path'])]
    if not rows:
        raise ValueError('Cannot scope the recorder to this container cgroup')
    return rows


def belongs(pid, boundary):
    return any(current['hierarchy'] == original['hierarchy'] and
               (current['path'] == original['path'] or current['path'].startswith(original['path'].rstrip('/') + '/'))
               for current in cgroups(pid) for original in boundary)
