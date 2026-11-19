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
from task_config import authoring_plan, firewall_plan, grant_workspaces, profile, write_environment

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
    plan, plan_sha256 = authoring_plan(os.environ)
    # Resolve only during trusted bootstrap. Children need no DNS permission. The agent's own gateway, then the
    # framework services the plan names for its commands (the model-access gateway).
    endpoints = [(AUTHOR, "author-gateway", 8001)] + [(WORKER, service["host"], service["port"])
                                                      for service in plan["services"]]
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
    hosts, commands = firewall_plan(task, plan, resolved)
    with open("/etc/hosts", "a") as stream:
        stream.write(hosts)
    for command in commands:
        subprocess.run(command, check=True)
    mkdir("/run/breakwater", 0, 0, 0o755)
    write_environment(task, plan)
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
               "subagents": subagents, "plan_sha256": plan_sha256,
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
