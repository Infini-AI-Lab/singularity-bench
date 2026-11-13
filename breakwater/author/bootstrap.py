"""Root-only, fail-closed entrypoint for a protected authoring container."""
import ipaddress
import json
import os
from pathlib import Path
import re
import socket
import subprocess
import sys
import urllib.request

sys.path.insert(0, "/opt/breakwater-runtime")
from task_config import profile, write_environment, grant_workspaces, worker_destinations, fleet_addresses

AUTHOR, WORKER, SHARED = 1100, 1101, 1200


def mkdir(path, uid, gid, mode):
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    os.chown(path, uid, gid)
    path.chmod(mode)


def main():
    if os.geteuid() != 0:
        raise SystemExit("authoring bootstrap must run as root")
    task = profile()
    model = os.environ.get("BREAKWATER_AUTHOR_MODEL", "")
    if not model:
        raise SystemExit("BREAKWATER_AUTHOR_MODEL is required; use the protected author launcher")
    harness = Path("/opt/author-cli/harness").read_text().strip()
    if harness not in ("claude-code", "codex"):
        raise SystemExit("Unknown installed author profile")
    from subagents import enabled
    subagents = enabled(os.environ.get('BREAKWATER_SUBAGENTS', '0'), harness)
    # Resolve only during trusted bootstrap. Children need no DNS permission.
    endpoints = [(AUTHOR, "author-gateway", 8001)]
    if task["development_gateway"]:
        endpoints.append((WORKER, "cell-gateway", 8002) if task.get("downstream_full_evaluation") else (WORKER, "gateway", 8000))
    resolved = []
    for uid, host, port in endpoints:
        addresses = sorted({row[4][0] for row in socket.getaddrinfo(host, port, socket.AF_INET, socket.SOCK_STREAM)})
        if not addresses:
            raise SystemExit(f"Cannot resolve {host}")
        resolved.append((uid, host, port, addresses))
        with urllib.request.urlopen(f"http://{host}:{port}/health", timeout=10) as response:
            health = json.load(response)
        if not health.get("ok") or (uid == AUTHOR and health.get("model") != model):
            raise SystemExit(f"Gateway configuration mismatch: {host}")
        if uid == AUTHOR and health.get('protocol') != ('responses' if harness == 'codex' else 'messages'):
            raise SystemExit('Author gateway protocol does not match the installed CLI')
    fleet_hosts, fleet_denied = fleet_addresses(task)
    # rl-swe-agent: the launcher names the host's sandbox worker (ADDRESS:PORT) and the published session-server
    # port range (LOW:HIGH). Agent commands may connect to that one endpoint, and may answer the task sandboxes
    # that connect to their session servers on those ports; nothing else is opened.
    sandbox, session_ports = os.environ.get("BREAKWATER_SANDBOX_WORKER", ""), os.environ.get("BREAKWATER_SESSION_PORTS", "")
    sandbox_endpoint = None
    if sandbox:
        address, port = sandbox.rsplit(":", 1)
        ipaddress.IPv4Address(address)
        if not 1 <= int(port) <= 65535:
            raise SystemExit("Invalid sandbox worker endpoint")
        sandbox_endpoint = (address, str(int(port)))
    if session_ports and not (sandbox_endpoint and re.fullmatch(r"\d{4,5}:\d{4,5}", session_ports)):
        raise SystemExit("Invalid session port range")
    with open("/etc/hosts", "a") as hosts:
        for _, host, _, addresses in resolved:
            hosts.write(f"\n{addresses[0]} {host}\n")
        for host, address in fleet_hosts:
            hosts.write(f"\n{address} {host}\n")
        if sandbox_endpoint:
            hosts.write(f"\n{sandbox_endpoint[0]} sandbox-worker host.docker.internal\n")
    destinations = worker_destinations(task) + ([sandbox_endpoint] if sandbox_endpoint else [])
    # No private-network, DNS or ESTABLISHED accept rule. Local GPU ranks and
    # task servers get only this container, plus explicitly approved fleet ports.
    for executable in ("iptables", "ip6tables"):
        subprocess.run([executable, "-P", "OUTPUT", "DROP"], check=True)
        subprocess.run([executable, "-F", "OUTPUT"], check=True)
        if executable == "iptables":
            subprocess.run([executable, "-A", "OUTPUT", "-d", "127.0.0.11", "-j", "REJECT"], check=True)
            # Manager ports can fall inside the engine port range, so these
            # denies must precede both fleet and local-address accepts.
            for address, port in fleet_denied:
                subprocess.run([executable, '-A', 'OUTPUT', '-m', 'owner', '--uid-owner', str(WORKER),
                                '-p', 'tcp', '-d', address, '--dport', port, '-j', 'REJECT'], check=True)
            for uid in (WORKER, 0):
                for address, ports in destinations:
                    rule = [executable, "-A", "OUTPUT", "-m", "owner", "--uid-owner", str(uid), "-d", address]
                    if ports:
                        rule += ["-p", "tcp", "--dport", ports]
                    subprocess.run(rule + ["-j", "ACCEPT"], check=True)
            for uid, _, port, addresses in resolved:
                for address in addresses:
                    subprocess.run([executable, "-A", "OUTPUT", "-m", "owner", "--uid-owner", str(uid),
                                    "-p", "tcp", "-d", address, "--dport", str(port), "-j", "ACCEPT"], check=True)
            if session_ports:
                # Replies only: a new outbound connection from these source ports is still rejected.
                subprocess.run([executable, "-A", "OUTPUT", "-m", "owner", "--uid-owner", str(WORKER), "-p", "tcp",
                                "--sport", session_ports, "-m", "conntrack", "--ctstate", "ESTABLISHED",
                                "-j", "ACCEPT"], check=True)
        subprocess.run([executable, "-A", "OUTPUT", "-j", "REJECT"], check=True)
    mkdir("/run/breakwater", 0, 0, 0o755)
    write_environment(task)
    grant_workspaces(task)
    mkdir("/run/breakwater/tmp", 0, SHARED, 0o2770)
    mkdir(f"/run/breakwater/tmp/claude-{AUTHOR}", AUTHOR, SHARED, 0o2770)
    # The agent's local log and session state. This directory contains no provider keys.
    mkdir("/logs/agent", AUTHOR, SHARED, 0o2770)
    mkdir("/logs/verifier", WORKER, SHARED, 0o2770)
    if harness == 'codex':
        # Isolated Python does not import from its script directory implicitly.
        sys.path.insert(0, '/opt/breakwater-runtime')
        from codex_config import configure
        configure(model, task["workdir"], subagents=subagents)
    runtime = {"model": model, "version": Path("/opt/author-cli/version").read_text().strip(),
               "endpoints": resolved, "harness": harness, "task": task["name"], "workdir": task["workdir"],
               "subagents": subagents,
               "profile": {"claude-code": "claude-author-v1", 'codex': 'codex-author-v1'}[harness]}
    Path("/run/breakwater/runtime.json").write_text(json.dumps(runtime))
    Path("/run/breakwater/ready").touch()
    # The idle process does not retain bootstrap capabilities or author rights.
    os.setgroups([])
    os.setgid(65534)
    os.setuid(65534)
    os.environ.clear()
    os.execv("/bin/sleep", ["sleep", "infinity"])


if __name__ == "__main__":
    main()
