"""Bring a fleet up across preallocated nodes, from a launch host, given a nodelist.

The task is launched from a launch host after the nodes are already reserved -- this does not allocate
anything, it only starts one worker supervisor per node and waits for the fleet to answer.

    python3 _kit/cluster_launch.py up \\
        --nodes node001,node002,...        # or --nodelist-file nodes.txt
        --catalogue catalogue.json --out fleet.json
    python3 _kit/cluster_launch.py down --fleet fleet.json

HOW IT REACHES A NODE is deliberately not assumed. Clusters differ -- ssh, srun, pdsh, a site wrapper --
so the remote-exec is a template:

    CLUSTER_REMOTE_LAUNCH='ssh -o BatchMode=yes {node} {cmd}'      (default)
    CLUSTER_REMOTE_LAUNCH='srun --nodelist={node} --nodes=1 {cmd}'
    CLUSTER_REMOTE_LAUNCH='mycluster-exec {node} -- {cmd}'

`{node}` is the hostname, `{cmd}` the shell command to run there. Anything that can run a command on a
named node works, and nothing here depends on a particular scheduler.

Each node runs `manager.py` for ITS OWN machine, selected by CLUSTER_ONLY_NODE, with its
own host-RAM budget. Host RAM is per machine: a fleet-wide figure would let a control plane believe it
can park sleeping weights on a node that has no room for them.
"""
from __future__ import annotations

import argparse
import json
import os
import shlex
import secrets
from pathlib import Path
import subprocess
import sys
import time
import urllib.error
import urllib.request

DEFAULT_LAUNCH = "ssh -o BatchMode=yes {node} {cmd}"
CONTAINER_NAME = "cluster-fleet"


def container_cmd(a, node: str) -> str:
    """`docker run` for one node's slice of the fleet.

    --network host, not a published port range. Two reasons, and both are requirements rather than
    conveniences: a worker must be reachable at the NODE's address for the control plane to route to it,
    and a vLLM engine sharded ACROSS nodes opens its own NCCL and distributed-init ports, which are
    negotiated at run time and cannot be published ahead of time.
    """
    env = {
        "CLUSTER_NODE_NAME": node,
        "CLUSTER_RUN_ID": a.run_id,
        # Which machine this container is, and whether it also carries the fleet-wide controller. The
        # controller is started once, beside the first node's manager.
        "CLUSTER_ONLY_NODE": node,
        "CLUSTER_RUN_CONTROLLER": "1" if node == a._first_node else "0",
        "CLUSTER_CONTROLLER_PORT": str(a.controller_port),
        "CLUSTER_RECEIPT_KEY": a.receipt_key,
        "CLUSTER_GRADER_TOKEN": a.grader_token,
        "CLUSTER_FLEET_MODE": "real" if a.worker_script == "manager.py" else "sim",
        "CLUSTER_NODES": "/fleet/run/fleet.json",
        "CLUSTER_CATALOGUE": "/fleet/run/catalogue.json",
        "CLUSTER_LOG_DIR": "/fleet/run/logs",
        "CLUSTER_AUDIT_DIR": "/fleet/run/inference-audit",
        "HF_HOME": "/weights",
        "HF_HUB_OFFLINE": "1",
    }
    import physics
    host_ram_gib = a.host_ram_gib or physics.HOST_RAM_PER_GPU_GIB * a.gpus_per_node
    if host_ram_gib <= 0:
        raise ValueError("a positive host RAM limit is required")
    env["CLUSTER_HOST_RAM_GIB"] = str(host_ram_gib)
    if a.time_scale != 1.0:
        env["CLUSTER_TIME_SCALE"] = str(a.time_scale)
    flags = " ".join(f"-e {k}={shlex.quote(v)}" for k, v in env.items())
    gpus = "--gpus all" if a.worker_script == "manager.py" else ""
    # --ipc=host and a raised shm: vLLM's workers talk over shared memory, and the default 64 MB is not
    # enough for tensor parallelism -- the symptom is a hang at engine start, not an error.
    #
    # --pid=host, because placement is OBSERVED rather than derived. The NVIDIA driver reports HOST pids
    # from --query-compute-apps and has no notion of a pid namespace, so in a container with its own
    # namespace nvidia-smi cannot map them and returns an EMPTY process list -- not an error, just
    # silence, which would read as "no engine is using any GPU". Sharing the host namespace also puts the
    # worker's own children on the same pids it sees in that query, and makes /proc/<pid>/stat readable
    # for the pid -> pgid -> engine mapping. Safe here: this container runs trusted fleet code only. The
    # agent's control plane runs in a different container.
    # THE HOST-RAM BUDGET, ENFORCED BY THE KERNEL. A published number alone would leave the budget at
    # whatever the machine happens to have free (`nodeinfo` reads MemAvailable). A cgroup makes it exact,
    # identical everywhere, and independent of the hardware's actual size.
    #
    # --memory-swap equal to --memory disables swap for the container; without it Docker allows twice the
    # limit in swap and an over-committed plane degrades instead of failing.
    #
    # Under cgroup v2 the cap binds (exit 137); CUDA PINNED host memory -- what sleep mode parks the
    # weights in -- is charged against it; and page cache from streaming a checkpoint under a smaller cap
    # is reclaimed rather than fatal, so reading checkpoints does not blow the limit.
    memory_bytes = int(host_ram_gib * 1024**3)
    mem = f"--memory={memory_bytes} --memory-swap={memory_bytes} "
    return (f"docker run --rm --name {CONTAINER_NAME}-{a.run_id} --label cluster.run={a.run_id} -d {gpus} --network host --ipc=host --pid=host "
            f"{mem}--shm-size={a.shm_size} --ulimit nofile=65536:65536 {flags} "
            f"-v {shlex.quote(a.run_dir)}:/fleet/run "
            f"-v {shlex.quote(a.weights)}:/weights:ro "
            f"{shlex.quote(getattr(a, '_image_ids', {}).get(node, a.image))}")


def remote(node: str, cmd: str, background: bool = True) -> subprocess.CompletedProcess:
    tmpl = os.environ.get("CLUSTER_REMOTE_LAUNCH", DEFAULT_LAUNCH)
    # `env` in front, not bare VAR=... assignments: setsid execs its argument directly and would try to
    # run "CLUSTER_NODE_NAME=node001" as a program.
    inner = (f"nohup setsid env {cmd} > /tmp/cluster-worker.log 2>&1 < /dev/null &"
             if background else cmd)
    full = tmpl.replace("{node}", shlex.quote(node)).replace("{cmd}", shlex.quote(inner))
    return subprocess.run(full, shell=True, capture_output=True, text=True, timeout=300)


def healthy(host: str, port: int, timeout: float = 3.0, run_id: str = "") -> bool:
    try:
        with urllib.request.urlopen(f"http://{host}:{port}/health", timeout=timeout) as r:
            return r.status == 200 and (not run_id or json.load(r).get("run_id") == run_id)
    except Exception:                                                 # noqa: BLE001
        return False


def wait_fleet(fleet: dict, seconds: float) -> list[str]:
    """Returns the managers (and the controller) still unreachable when the clock runs out.

    There is no worker list to wait on: a worker is created by a deploy and named by the controller, so
    none exist until the plane asks for one.
    """
    end = time.monotonic() + seconds
    todo = [(n["name"], n.get("host") or fleet["host"], n["port"]) for n in fleet["nodes"]]
    ctl = fleet.get("controller")
    if ctl:
        todo.append(("controller", ctl.get("host") or fleet["host"], ctl["port"]))
    while time.monotonic() < end and todo:
        todo = [t for t in todo if not healthy(t[1], t[2], run_id=fleet.get("run_id", ""))]
        if todo:
            time.sleep(3)
    return [t[0] for t in todo]


def up(a) -> int:
    nodes = [n.strip() for n in (a.nodes.split(",") if a.nodes else
                                 open(a.nodelist_file).read().split()) if n.strip()]
    if not nodes:
        print("no nodes given", file=sys.stderr)
        return 2
    n_workers = len(nodes) * a.gpus_per_node          # GPUs, for the fleet-shape print only
    print(f"[cluster] {len(nodes)} nodes x {a.gpus_per_node} GPUs, one manager each + a controller")

    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import fleetgen
    fleet = fleetgen.build(n_workers, a.seed, a.base_port, nodes[0], a.perturb,
                           a.host_ram_gib, nodes, a.gpus_per_node,
                           port_stride=a.port_stride, controller_port=a.controller_port)
    a.host_ram_gib = fleet["host_ram_gib"]
    if a.resolve and not a.local_hosts:
        # ADDRESSES in `host`, names in `node`. The agent's container is on a docker bridge with the
        # daemon's resolver, which knows nothing about site hostnames; an unresolvable name there is a
        # control plane that cannot reach a fleet that is running perfectly well. `node` keeps the name
        # because that is the identity the host-RAM budget and cross-node placement are keyed on.
        import socket
        addr = {}
        for nd in nodes:
            try:
                addr[nd] = socket.gethostbyname(nd)
            except OSError as exc:
                print(f"[cluster] cannot resolve {nd}: {exc}; fleet was not started", file=sys.stderr)
                return 2
        for nd in fleet["nodes"]:
            nd["host"] = addr[nd["name"]]
            nd["addr"] = addr[nd["name"]]
        fleet["controller"]["host"] = addr[nodes[0]]
        fleet["host"] = addr[nodes[0]]
        print("[cluster] " + ", ".join(f"{k}->{v}" for k, v in addr.items()))
    if a.local_hosts:
        for nd in fleet["nodes"]:
            nd["host"] = "127.0.0.1"
        fleet["controller"]["host"] = "127.0.0.1"
        fleet["host"] = "127.0.0.1"
    a.run_id = secrets.token_hex(16)
    fleet["run_id"] = a.run_id
    fleet["launch"] = {"mode": "container" if a.image else "process",
                       "kit_dir": a.kit_dir or str(Path(__file__).resolve().parent), "python": a.python}
    if Path(a.out).exists():
        raise RuntimeError("fleet output already exists; use a fresh run directory")
    if any(healthy(n.get("host") or fleet["host"], n["port"]) for n in fleet["nodes"]) or healthy(
            fleet["controller"]["host"], fleet["controller"]["port"]):
        raise RuntimeError("fleet endpoints are already occupied; no processes were stopped")
    if a.image:
        a._image_ids = {}
        for node in nodes:
            result = remote(node, f"docker image inspect --format '{{{{.Id}}}}' {shlex.quote(a.image)}", background=False)
            image = result.stdout.strip()
            if result.returncode or not __import__('re').fullmatch(r"sha256:[0-9a-f]{64}", image):
                raise RuntimeError(f"cannot resolve fleet image on {node}; no fleet started")
            a._image_ids[node] = image
        fleet["launch"]["images"] = a._image_ids
    # This is the same shared operator directory that carries fleet.json and
    # catalogue.json to every node. It is never mounted in an agent container.
    # Create it as the operator so root-owned worker journals stay readable
    # after Docker --rm, unload, reset, or an interrupted run.
    audit_dir = (Path(a.run_dir) if a.run_dir else Path(a.out).resolve().parent) / "inference-audit"
    audit_dir = audit_dir.resolve()
    audit_dir.mkdir(mode=0o700, parents=True, exist_ok=True)
    audit_dir.chmod(0o700)
    fleet["launch"]["inference_audit_dir"] = str(audit_dir)
    json.dump(fleet, open(a.out, "w"), indent=1)
    json.dump(fleetgen.public_view(fleet), open(a.public, "w"), indent=1)
    print(f"[cluster] wrote {a.out} ({fleet['host_ram_gib']:.0f} GiB host RAM per node)")

    kit = a.kit_dir or os.path.dirname(os.path.abspath(__file__))
    failed = []
    for node in nodes:
        if a.image:
            a._first_node = nodes[0]
            cmd, bg = container_cmd(a, node), False
        else:
            env = (f"CLUSTER_RUN_ID={a.run_id} CLUSTER_NODE_NAME={shlex.quote(node)} "
                   f"CLUSTER_RECEIPT_KEY={shlex.quote(a.receipt_key)} "
                   f"CLUSTER_GRADER_TOKEN={shlex.quote(a.grader_token)} "
                   f"CLUSTER_WEIGHTS={shlex.quote(a.weights)} "
                   f"HF_HUB_OFFLINE=1 CLUSTER_LOG_DIR=/tmp/fleet-logs "
                   f"CLUSTER_AUDIT_DIR={shlex.quote(str(audit_dir))} "
                   f"CLUSTER_TIME_SCALE={a.time_scale}")
            fp = shlex.quote(a.fleet_path_on_node or a.out)
            cmd = (f"{env} CLUSTER_ONLY_NODE={shlex.quote(node)} {a.python} {kit}/{a.worker_script} "
                   f"--fleet {fp} --catalogue {shlex.quote(a.catalogue)}")
            if node == nodes[0]:
                # The controller is fleet-wide, so it is started ONCE, beside the first node's manager --
                # as its OWN remote() call, never spliced into the manager's with an `&`. remote()
                # captures output, and only the last command in a compound gets the redirect: the other
                # one would inherit the captured pipe, hold it open as a background process, and
                # subprocess.run would wait for an EOF that never comes.
                remote(node, f"{env} {a.python} {kit}/controller.py --fleet {fp} "
                             f"--catalogue {shlex.quote(a.catalogue)} --port {a.controller_port}")
            bg = True
        r = remote(node, cmd, background=bg)
        if r.returncode != 0:
            failed.append((node, r.stderr.strip()[:160]))
        print(f"[cluster] launched on {node}" + ("" if r.returncode == 0 else "  FAILED"))
    for node, err in failed:
        print(f"  {node}: {err}", file=sys.stderr)

    # A node whose launch command already returned non-zero is not going to answer later, so do not wait
    # the full window for it (the usual cause is a fleet image that was never built on that node).
    if failed and len(failed) == len(nodes):
        print(f"[cluster] the launch failed on ALL {len(nodes)} nodes; not waiting.", file=sys.stderr)
        if a.image:
            print(f"  If the message above is about the image, {a.image} is not on these nodes:",
                  file=sys.stderr)
            print("  run  ./bin/cluster.sh stage --nodes ...  once per node set.", file=sys.stderr)
        return 1
    if failed:
        print(f"[cluster] {len(failed)} node(s) failed to launch; waiting for the rest anyway, but a "
              f"partial fleet cannot be graded.", file=sys.stderr)

    want = len(nodes) + 1
    print(f"[cluster] waiting up to {a.wait:.0f}s for {len(nodes)} manager(s) + the controller")
    missing = wait_fleet(fleet, a.wait)
    if missing:
        print(f"[cluster] {len(missing)}/{want} never answered: {missing[:8]}", file=sys.stderr)
        print("  Check /tmp/cluster-worker.log on those nodes. A partial fleet is not a smaller fleet:",
              file=sys.stderr)
        print("  the catalogue was sized for the full one, so grading on it measures the outage.",
              file=sys.stderr)
        return 1
    print(f"[cluster] all {n_workers} workers up")
    return 0


def down(a) -> int:
    """A fleet manifest identifies the exact resources this run owns."""
    if not a.fleet:
        print("down requires --fleet from the run being stopped; node names are not ownership", file=sys.stderr)
        return 2
    fleet = json.loads(Path(a.fleet).read_text())
    run_id = fleet.get("run_id", "")
    import re
    if not re.fullmatch(r"[0-9a-f]{32}", run_id):
        raise ValueError("fleet has no valid run identity; refusing broad cleanup")
    launch = fleet["launch"]
    nodes = launch.get("cleanup_nodes") or list(dict.fromkeys(n["name"] for n in fleet["nodes"]))
    failures = []
    for node in nodes:
        if launch["mode"] == "container":
            name = f"{CONTAINER_NAME}-{run_id}"
            script = """import json, subprocess, sys
name, identity = sys.argv[1:]
check = subprocess.run(['docker', 'inspect', name], capture_output=True, text=True)
if check.returncode:
    if 'No such object' in check.stderr or 'No such container' in check.stderr:
        sys.exit(0)
    sys.exit(check.stderr)
info = json.loads(check.stdout)[0]
if info['Config'].get('Labels', {}).get('cluster.run') != identity:
    sys.exit('container ownership mismatch')
subprocess.run(['docker', 'rm', '-f', info['Id']], check=True)
"""
            cmd = f"python3 -c {shlex.quote(script)} {shlex.quote(name)} {run_id}"
        elif launch["mode"] == "process":
            helper = Path(launch["kit_dir"]) / "owned_processes.py"
            cmd = f"{shlex.quote(launch['python'])} {shlex.quote(str(helper))} {run_id}"
        else:
            raise ValueError("unknown fleet launch mode")
        result = remote(node, cmd, background=False)
        if result.returncode:
            failures.append(node)
            print(f"[cluster] cleanup failed on {node}: {result.stderr[:300]}", file=sys.stderr)
        else:
            print(f"[cluster] stopped owned resources on {node}")
    return 1 if failures else 0


def main() -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    u = sub.add_parser("up")
    u.add_argument("--nodes", default="", help="comma-separated hostnames")
    u.add_argument("--nodelist-file", default="", help="file of hostnames, whitespace separated")
    u.add_argument("--gpus-per-node", type=int, default=4)
    u.add_argument("--catalogue", required=True)
    u.add_argument("--out", default="fleet.json")
    u.add_argument("--public", default="fleet.public.json")
    u.add_argument("--seed", default="cluster-1")
    u.add_argument("--perturb", type=float, default=0.10)
    u.add_argument("--host-ram-gib", type=float, default=0.0, help="per node; 0 = published per-GPU host budget")
    u.add_argument("--base-port", type=int, default=9100)
    u.add_argument("--weights", default="/weights")
    u.add_argument("--python", default="python3")
    u.add_argument("--kit-dir", default="", help="where the worker script lives ON THE NODES")
    u.add_argument("--controller-port", type=int, default=9000,
                   help="the fleet controller: /deploy, /unload, /nodes, /workers")
    u.add_argument("--worker-script", default="manager.py",
                   help="manager.py on GPUs; simmanager.py to bring up an emulated cluster, which is "
                        "how the multi-node path is tested without a cluster")
    u.add_argument("--local-hosts", action="store_true",
                   help="TESTING ONLY: keep node identity but address every worker at 127.0.0.1, so a "
                        "multi-node fleet can be stood up on one machine. Never use for grading -- it "
                        "makes every node's traffic local and hides exactly the cross-node cost the "
                        "cluster version exists to measure.")
    u.add_argument("--port-stride", type=int, default=0,
                   help="add k*stride to node k's ports. 0 on a real cluster, where each node has its "
                        "own address; non-zero to stand several nodes up on one machine for testing")
    u.add_argument("--fleet-path-on-node", default="", help="if the fleet file is at a different path there")
    u.add_argument("--image", default="", help="run the fleet as this docker image on each node "
                                               "(the normal path). Empty runs the worker script "
                                               "directly, which needs vLLM installed on the nodes.")
    u.add_argument("--run-dir", default="", help="directory holding fleet.json + catalogue.json, "
                                                 "readable at the SAME path on every node")
    u.add_argument("--shm-size", default="32g")
    u.add_argument("--time-scale", type=float, default=1.0,
                   help="EMULATOR ONLY: divide every simulated cost by this. For testing the plumbing; "
                        "the real fleet ignores it, so a number from a scaled run is not a score.")
    u.add_argument("--resolve", action="store_true", default=True,
                   help="write node IP addresses into the workers' `host` field (default)")
    u.add_argument("--no-resolve", dest="resolve", action="store_false")
    u.add_argument("--receipt-key", default=os.environ.get("CLUSTER_RECEIPT_KEY", "dev-key"))
    u.add_argument("--grader-token", default=os.environ.get("CLUSTER_GRADER_TOKEN", "dev-token"))
    u.add_argument("--wait", type=float, default=900.0)
    d = sub.add_parser("down")
    d.add_argument("--fleet", default="", help="stop the nodes named in this fleet file")
    d.add_argument("--nodes", default="", help="stop these nodes. Preferred: it does not depend on a "
                                               "fleet file that an interrupted run may never have written")
    a = ap.parse_args()
    if a.cmd == "up" and a.image and not a.run_dir:
        a.run_dir = os.path.dirname(os.path.abspath(a.out))
    return up(a) if a.cmd == "up" else down(a)


if __name__ == "__main__":
    sys.exit(main())
