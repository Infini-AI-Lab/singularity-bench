"""A run's endpoints: the hosts and ports its sandboxes may reach beyond the framework's own services (SPEC.md,
[[breakwater.endpoints]]).

Host-side only. A launcher resolves the task's static endpoints with the operator's overrides (``--endpoint
NAME=HOST:PORT``), checks the health of those that declare a health path, and gives each role its endpoints: the agent's
commands, the coordinator and the submitted code. Each backend applies them: Apptainer relays each named port into the
sandboxes; Docker hands the agent's to the author image's bootstrap in the authoring plan.
"""
import ipaddress
import urllib.request

from contract import require

ROLES = ('agent', 'coordinator', 'submission')


def overrides(values):
    """``--endpoint NAME=HOST:PORT`` values as {name: (host, port)}."""
    result = {}
    for value in values or ():
        name, _, address = str(value).partition('=')
        host, _, port = address.rpartition(':')
        require(name and host and port.isdigit() and 1 <= int(port) <= 65535 and name not in result,
                '--endpoint takes NAME=HOST:PORT, once per endpoint')
        result[name] = (host, int(port))
    return result


def static_endpoints(config, values=()):
    """The task's [[breakwater.endpoints]] with the operator's overrides: [{name, host, ports, port_map, reach,
    health}]. ``host`` is the real host, ``ports`` the ports the sandboxes use, ``port_map`` the real host's port for
    a sandbox port where they differ (None when they agree)."""
    chosen = overrides(values)
    unknown = set(chosen) - {row['name'] for row in config['endpoints']}
    require(not unknown, 'The task declares no endpoint named ' + ', '.join(sorted(unknown)))
    rows = []
    for item in config['endpoints']:
        host, _, port = item['service'].rpartition(':')
        host, port = chosen.get(item['name'], (host, int(port)))
        rows.append({'name': item['name'], 'host': host, 'ports': [item['port']],
                     'port_map': {item['port']: port} if port != item['port'] else None,
                     'reach': list(item['reach']), 'health': item['health']})
    return rows


def check_health(endpoints, timeout=10):
    """Every endpoint that declares a health path answers HTTP 200 there, at its real host and port, before authoring."""
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    for row in endpoints:
        if not row.get('health'):
            continue
        port = (row.get('port_map') or {}).get(row['ports'][0], row['ports'][0])
        url = f"http://{row['host']}:{port}{row['health']}"
        try:
            with opener.open(url, timeout=timeout) as response:
                healthy = response.status == 200
        except OSError:
            healthy = False
        require(healthy, f"The {row['name']} endpoint does not answer at {url}: start it first")


def by_role(endpoints):
    """{role: [{name, host, ports, port_map}]}: each role's endpoints, merged by name (the union of their ports; one name
    keeps one host)."""
    merged = {role: {} for role in ROLES}
    for row in endpoints:
        for role in row['reach']:
            item = merged[role].setdefault(row['name'], {'name': row['name'], 'host': row['host'], 'ports': [],
                                                         'port_map': {}})
            require(item['host'] == row['host'], f"The endpoint {row['name']} names two hosts")
            item['ports'] += [port for port in row['ports'] if port not in item['ports']]
            item['port_map'].update(row.get('port_map') or {})
    return {role: [{**item, 'port_map': item['port_map'] or None} for item in rows.values()]
            for role, rows in merged.items()}


def rejects(roles):
    """Ports the coordinator reaches on a host the agent's commands also reach, but the agent may not: refused before
    any accept (a manager port inside an engine port range, say): [{host, port}]."""
    agent = {}
    for row in roles['agent']:
        agent.setdefault(row['host'], set()).update(row['ports'])
    return [{'host': row['host'], 'port': port} for row in roles['coordinator'] if row['host'] in agent
            for port in sorted(set(row['ports']) - agent[row['host']])]


def launcher_host(host):
    """Whether a real host is the launcher's own (a loopback address or localhost)."""
    if host == 'localhost':
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


def ranges(ports):
    """Ports as iptables destination ports, consecutive ones compressed: ['P' | 'L:H']."""
    result, ordered = [], sorted(set(ports))
    for port in ordered:
        if result and result[-1][1] == port - 1:
            result[-1][1] = port
        else:
            result.append([port, port])
    return [str(low) if low == high else f'{low}:{high}' for low, high in result]
