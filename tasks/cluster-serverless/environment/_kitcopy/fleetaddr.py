"""Where a worker lives. One definition, used by every component that talks to one.

On a cluster each worker sits on a named node, and the host-RAM budget belongs to the node rather than
to the fleet. A component that read only `fleet["host"]` would address the wrong machine, and one that
read a fleet-wide RAM figure would let a plane park weights on a node with no room for them.
"""
from __future__ import annotations


def worker_host(fleet: dict, worker: dict) -> str:
    """The hostname to reach this worker on. Per-worker if present, else the fleet default."""
    return worker.get("host") or fleet.get("host") or "127.0.0.1"


def worker_url(fleet: dict, worker: dict, path: str = "") -> str:
    return f"http://{worker_host(fleet, worker)}:{worker['port']}{path}"


def node_of(worker: dict) -> str:
    return worker.get("node") or worker.get("host") or "default"


def node_ram_gib(fleet: dict, node: str) -> float:
    """Host RAM available on ONE node for sleeping weights."""
    for n in fleet.get("nodes", []):
        if n.get("name") == node:
            return float(n.get("host_ram_gib", fleet.get("host_ram_gib", 1200.0)))
    return float(fleet.get("host_ram_gib", 1200.0))


def nodes_of(fleet: dict) -> list[str]:
    seen, out = set(), []
    for w in fleet["workers"]:
        n = node_of(w)
        if n not in seen:
            seen.add(n)
            out.append(n)
    return out
