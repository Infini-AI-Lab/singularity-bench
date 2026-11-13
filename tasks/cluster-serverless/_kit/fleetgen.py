"""Fleet profiles. The graded profile carries a seeded per-node speed perturbation (used by the emulated
fleet) that the development profile does not: a control plane that computes a placement up front and
executes it blind degrades, while one that runs closed-loop on /stats survives."""
from __future__ import annotations
import argparse, json, os, random, sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import physics


import seeds
def build(n_workers: int, seed: str, base_port: int = 9100, host: str = "simfleet",
          perturb: float = 0.0, host_ram_gib: float = 0.0, nodes: list[str] | None = None,
          gpus_per_node: int = 8, port_stride: int = 0, one_host: str = "",
          controller_port: int = 9000) -> dict:
    """`port_stride` and `one_host` together make a MULTI-NODE fleet runnable on ONE machine.

    On a real cluster every node has its own address, so every node's manager can use port 9100 -- the
    address disambiguates them. Put those nodes on one machine, as the one-box emulated fleet does (a
    single container holding the whole emulated cluster), and they all try to bind 9100; without a stride
    that fleet cannot start.

    Node identity is kept either way. The nodes are still named, host RAM is still per node, and a model
    can still be sharded across them -- only the addresses collapse."""
    # PER NODE, not per fleet: physics.HOST_RAM_PER_GPU_GIB (242 GiB) per GPU, enough to keep a good part
    # of the catalogue one wake away, not enough for all of it. On a cluster each machine has its own
    # memory, so a fleet-wide figure would let a plane believe it can park weights on a node that has no
    # room for them.
    nodes = nodes or [host]
    # From physics, not a literal, so the catalogue and the fleet are sized against the same number.
    per_node_ram = host_ram_gib or physics.HOST_RAM_PER_GPU_GIB * gpus_per_node
    # Through seeds.py like every other generation stream, so the whole world descends from
    # CLUSTER_MASTER_SEED. Only the node `speed` perturbation draws from it.
    rng = seeds.rng("fleet", seed)
    return {
        "fleet_id": seed, "seed": seed, "host": host, "n_workers": n_workers,
        "n_nodes": len(nodes),
        # Host RAM available for SLEEPING weights, node-wide. A PUBLISHED CONSTANT, not a reading of
        # whatever machine this runs on: grading must not get easier on a box with more memory. It is set
        # well below what a node of this class carries, so the fleet is not the only thing on the
        # machine, and far below the catalogue, so not everything can be kept asleep at once.
        # the manager refuses to start if the machine cannot actually back it.
        "host_ram_gib": per_node_ram,
        "gpus_per_node": gpus_per_node,
        # THE CONTROLLER is the plane's entry point and the only address it needs to start from:
        # /deploy, /unload, /nodes and /workers all live there. It runs beside the first node's manager.
        "controller": {"host": one_host or nodes[0], "port": controller_port},
        # ONE MANAGER PER NODE. There is no per-GPU worker list: a worker is created by a deploy and
        # named by the controller, so it cannot be enumerated in advance. `port` is the manager's;
        # `port_stride` keeps them distinct when a multi-node fleet is emulated on one box, where the
        # address can no longer disambiguate them.
        "nodes": [{"name": nd, "host": one_host or nd, "port": base_port + port_stride * k,
                   "n_gpus": gpus_per_node, "host_ram_gib": per_node_ram,
                   "speed": round(rng.uniform(1 - perturb, 1 + perturb), 4) if perturb else 1.0}
                  for k, nd in enumerate(nodes)],
    }


def public_view(fleet: dict) -> dict:
    """What the control plane reads.

    The controller's address and the node NAMES -- which is all a plane needs, because `/deploy` takes
    node names and `GET /nodes` reports everything else about a machine from the machine itself. `speed`
    is withheld: it is behaviour, not a declared term, and handing it over would let a plane pre-compute
    around the perturbation instead of observing it.
    """
    return {"fleet_id": fleet["fleet_id"], "host": fleet["host"],
            "n_nodes": fleet["n_nodes"], "gpus_per_node": fleet["gpus_per_node"],
            "controller": dict(fleet["controller"]),
            "nodes": [{"name": n["name"], "host": n["host"], "port": n["port"],
                       "n_gpus": n["n_gpus"], "host_ram_gib": n["host_ram_gib"]}
                      for n in fleet["nodes"]]}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, required=True)
    ap.add_argument("--seed", required=True)
    ap.add_argument("--perturb", type=float, default=0.0)
    ap.add_argument("--host", default="simfleet")
    ap.add_argument("--base-port", type=int, default=9100)
    ap.add_argument("--out", required=True)
    ap.add_argument("--public", required=True)
    ap.add_argument("--host-ram-gib", type=float, default=0.0, help="per NODE, not per fleet")
    ap.add_argument("--nodes", default="", help="comma-separated node hostnames. One entry means a "
                                                "single-machine fleet; several means a cluster.")
    ap.add_argument("--gpus-per-node", type=int, default=8)
    ap.add_argument("--port-stride", type=int, default=0,
                    help="add k*stride to node k's ports, so a multi-node fleet fits on one machine")
    ap.add_argument("--one-host", default="",
                    help="address every worker here, keeping node identity. With --port-stride, this "
                         "is how the agent's emulated cluster runs inside one container.")
    a = ap.parse_args()
    node_list = [n for n in a.nodes.split(",") if n] or None
    f = build(a.workers, a.seed, a.base_port, a.host, a.perturb, a.host_ram_gib,
              node_list, a.gpus_per_node, a.port_stride, a.one_host)
    json.dump(f, open(a.out, "w"), indent=1)
    json.dump(public_view(f), open(a.public, "w"), indent=1)
    print(f"{a.workers} workers over {f['n_nodes']} node(s), seed {a.seed}, "
          f"perturb ±{a.perturb:.0%}, {f['host_ram_gib']:.0f} GiB host RAM per node")
