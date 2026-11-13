"""The fleet controller: the plane's entry point, and the only thing with a fleet-wide view.

It owns PLACEMENT, RENDEZVOUS and AGGREGATION, and deliberately owns no process identity: not a worker's
pid, not a rank's pgid. Those are meaningless off the machine that holds them and go stale on any
restart, so the map from a worker id to a process group lives in that node's manager. What lives here is
`worker_id -> [participating nodes]` plus the placement measured once at deploy.

  POST   /deploy    {model, engine_args, placement:[{node, gpus}]}  -> {worker, url, model, placement}
  POST   /unload    {worker}                                        -> idempotent teardown
  GET    /nodes                                                     -> every machine, measured
  GET    /workers                                                   -> what is live, from cache
  GET    /logs?worker=                                              -> that engine's own last words
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import secrets
import sys

import aiohttp
from aiohttp import web

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import engine_args  # noqa: E402

READY_TIMEOUT_S = float(os.environ.get("CLUSTER_READY_TIMEOUT_S", "1200"))
GRADER_TOKEN = os.environ.get("CLUSTER_GRADER_TOKEN", "")


def _err(status: int, message: str, **extra) -> web.Response:
    return web.json_response({"error": message, **extra}, status=status)


class Fleet:
    def __init__(self, nodes: list[dict], catalogue: dict) -> None:
        # node name -> manager base URL. The only addressing the controller does.
        self.managers = {n["name"]: f"http://{n['host']}:{n['port']}" for n in nodes}
        self.models = {m["id"]: m for m in catalogue.get("models", [])}
        self.workers: dict[str, dict] = {}
        self.seq = 0
        self.lock = asyncio.Lock()
        self.generation = ""
        self.resetting = False
        self.deployments: set[asyncio.Task] = set()
        self.node_sizes = {n["name"]: int(n["n_gpus"]) for n in nodes}

    def new_id(self) -> str:
        self.seq += 1
        return "w" + secrets.token_hex(12)


def fleet(request: web.Request) -> Fleet:
    return request.app["fleet"]


async def _call(sess, method: str, url: str, payload: dict | None = None, timeout: float = 60.0):
    try:
        async with sess.request(method, url, json=payload,
                                headers={"X-Grader-Token": GRADER_TOKEN},
                                timeout=aiohttp.ClientTimeout(total=timeout)) as r:
            body = await r.text()
            return r.status, (json.loads(body) if body.strip().startswith(("{", "[")) else {"raw": body})
    except (aiohttp.ClientError, asyncio.TimeoutError) as e:
        return 0, {"error": f"{type(e).__name__}: {e}"}


# ---------------------------------------------------------------- deploy

async def handle_deploy(request: web.Request) -> web.Response:
    f = fleet(request)
    if f.resetting:
        return _err(503, "fleet reset in progress")
    task = asyncio.current_task()
    f.deployments.add(task)
    try:
        return await _deploy(request)
    finally:
        f.deployments.discard(task)


async def _deploy(request: web.Request) -> web.Response:
    f = fleet(request)
    body = await request.json()
    model = body.get("model")
    spec = f.models.get(model)
    if spec is None:
        return _err(404, f"{model!r} is not in the catalogue")

    placement = body.get("placement") or []
    if not placement:
        return _err(400, "placement is required: a list of {node, gpus}, one entry per node. "
                         "One node is a one-element list.")
    unknown = [p.get("node") for p in placement if p.get("node") not in f.managers]
    if unknown:
        return _err(400, f"unknown node(s) {unknown}; GET /nodes lists what exists")

    # Parse the RAW arguments, not sanitised ones: `sanitise` DROPS reserved flags, so parsing its output
    # can never report one and a plane that set --quantization would have it silently ignored instead of
    # refused with a reason. Reserved arguments are a 400.
    extra, shape, refused = engine_args.validate(body.get("engine_args"))
    if refused:
        return _err(400, "; ".join(refused))
    # tp comes back 0 when the plane did not ask for one. The model's own floor is the right default --
    # the fewest GPUs its weights fit across -- and 0 would reach vLLM as --tensor-parallel-size 0.
    tp = shape.get("tp") or int(spec.get("min_tp") or 1)
    pp = max(1, int(shape.get("pp") or 1))
    if len({p["node"] for p in placement}) != len(placement):
        return _err(400, "placement must contain each node only once")
    if tp * pp % len(placement):
        return _err(400, "tp * pp must be divisible by the number of nodes")
    local_size = tp * pp // len(placement)
    for p in placement:
        visible = p.get("gpus") or list(range(f.node_sizes[p["node"]]))
        if (not isinstance(visible, list) or not all(type(c) is int and 0 <= c < f.node_sizes[p["node"]] for c in visible)
                or len(set(visible)) != len(visible) or len(visible) < local_size):
            return _err(400, f"invalid or insufficient visible GPUs on {p['node']}")
        p["gpus"] = visible[:local_size]

    nodes = [p["node"] for p in placement]
    head_node = nodes[0]
    master_addr = f.managers[head_node].split("//", 1)[1].split(":")[0]
    wid = f.new_id()
    sess = request.app["session"]

    # PHASE 1 -- launch every rank CONCURRENTLY. Each blocks at the rendezvous until its peers arrive, so
    # issuing these one at a time deadlocks the group on the first node.
    async def launch(i: int, p: dict, master_port: int = 0):
        payload = {"worker_id": wid, "model": model, "checkpoint": spec.get("local_dir")
                   or spec["checkpoint"], "engine_args": list(extra), "gpus": p.get("gpus") or [],
                   "nnodes": len(placement), "node_rank": i, "master_addr": master_addr,
                   "is_head": i == 0, "tp": tp, "pp": pp,
                   "master_port": master_port, "generation": f.generation,
                   "context_len": spec.get("context_len"),
                   # The head needs to reach its own ranks. Lifecycle operations apply to the whole
                   # engine -- one /admin/sleep on the head covers every machine it spans -- so the head
                   # has to be able to carry the transition to its peers. Passed as addresses rather
                   # than resolved in the manager because the real launcher runs one process per
                   # machine: there is no shared memory to look siblings up in.
                   "peers": [{"node": q["node"], "manager": f.managers[q["node"]]}
                             for q in placement]}
        # Real managers may wait for an earlier co-located engine to finish
        # startup profiling before spawning this rank.
        return p["node"], await _call(sess, "POST", f"{f.managers[p['node']]}/engines", payload,
                                    timeout=READY_TIMEOUT_S + 30)

    # The head's manager allocates a distinct rendezvous port. Launch returns at
    # spawn, so peers can join while the head waits for the collective.
    head = await launch(0, placement[0])
    results = [head]
    if head[1][0] == 200:
        master_port = head[1][1].get("master_port", 0)
        results += await asyncio.gather(*(launch(i, p, master_port)
                                        for i, p in enumerate(placement) if i))
    failed = [(n, r) for n, (st, r) in results if st != 200]
    if failed:
        # ASYMMETRIC FAILURE. If one node's launch fails, the ranks that did start sit at the rendezvous
        # indefinitely -- they have nobody to fail against. Tear the whole group down or they hold GPU
        # memory on every surviving node for the rest of the run.
        await _teardown(sess, f, wid, nodes)
        # PROPAGATE the managers' status when they agree on one: 507 means "no room here, try another
        # node", 500 means "this is broken, routing around it will not help". Collapsing everything into
        # 500 would hide the distinction a plane can act on.
        codes = {st for _, (st, _) in results if st != 200}
        st_out = codes.pop() if len(codes) == 1 else 500
        if not 400 <= st_out <= 599:
            st_out = 502
        return _err(st_out, f"{model}: launch failed on {[n for n, _ in failed]}",
                    detail={n: r for n, r in failed})

    # PHASE 2 -- readiness, then measure. Only rank 0 serves HTTP; peers are --headless and have no
    # endpoint at all. The manager verifies the head's authenticated deployment
    # identity after the collective has formed and every rank has loaded.
    st, ready = await _call(sess, "GET", f"{f.managers[head_node]}/engines/{wid}/ready",
                            timeout=READY_TIMEOUT_S + 30)
    if st != 200 or not ready.get("ready"):
        await _teardown(sess, f, wid, nodes)
        return _err(500, f"{model} did not become ready on {head_node}",
                    detail=ready.get("reason"), log_tail=ready.get("log_tail", []))

    # Every rank belongs to readiness. A peer may legitimately finish more
    # than the ordinary 60-second RPC timeout after the head (for example on
    # heterogeneous simulated trays), so this wait uses the full readiness timeout.
    settled = await asyncio.gather(*(_call(sess, "POST", f"{f.managers[n]}/engines/{wid}/settled", {},
                                          timeout=READY_TIMEOUT_S + 30)
                                     for n in nodes))
    if any(st != 200 for st, _ in settled):
        await _teardown(sess, f, wid, nodes)
        return _err(500, f"{model}: could not release startup locks",
                    detail={n: {"status": st, "response": response}
                            for n, (st, response) in zip(nodes, settled) if st != 200})

    st, wk = await _call(sess, "POST", f"{f.managers[head_node]}/engines/{wid}/worker", {})
    if st != 200:
        await _teardown(sess, f, wid, nodes)
        return _err(500, f"{model}: could not start the worker on {head_node}", detail=wk)

    # Placement is MEASURED, once, and cached. The device set does not change for a running engine, so
    # GET /workers never has to fan out; live memory is a GET /nodes concern.
    async def snap(node: str):
        st, r = await _call(sess, "GET", f"{f.managers[node]}/engines/{wid}/placement")
        return {"node": node, "gpus": r.get("gpus", []) if st == 200 else []}

    measured = await asyncio.gather(*(snap(n) for n in nodes))
    if any(not row["gpus"] for row in measured):
        await _teardown(sess, f, wid, nodes)
        return _err(503, f"{model}: GPU placement could not be measured on every rank",
                    placement=measured)
    async with f.lock:
        f.workers[wid] = {"worker": wid, "model": model, "url": wk["url"], "nodes": nodes,
                          "head": head_node, "requested": placement, "placement": list(measured),
                          "engine_args": list(extra)}
    return web.json_response(f.workers[wid])


# ---------------------------------------------------------------- unload

async def _teardown(sess, f: Fleet, wid: str, nodes: list[str]) -> list[dict]:
    """Stop serving, then kill every rank -- concurrently, and idempotently on each node."""
    out = await asyncio.gather(*(_call(sess, "DELETE", f"{f.managers[n]}/engines/{wid}")
                                 for n in nodes if n in f.managers))
    return [r for _, r in out]


async def handle_unload(request: web.Request) -> web.Response:
    """IDEMPOTENT by design: cleaning up after a crash is exactly when the plane calls this, and a
    404 here would leave it no way to reclaim the card."""
    f = fleet(request)
    wid = (await request.json()).get("worker", "")
    rec = f.workers.pop(wid, None)
    nodes = rec["nodes"] if rec else list(f.managers)
    detail = await _teardown(request.app["session"], f, wid, nodes)
    return web.json_response({"worker": wid, "status": "unloaded",
                              "was_known": rec is not None, "nodes": detail})


# ---------------------------------------------------------------- views

async def handle_reset(request: web.Request) -> web.Response:
    """Retire in-flight deployments and reset controller and managers together."""
    if not GRADER_TOKEN or request.headers.get("X-Grader-Token") != GRADER_TOKEN:
        return _err(403, "grader token required")
    f = fleet(request)
    f.resetting = True
    pending = list(f.deployments)
    for task in pending:
        task.cancel()
    await asyncio.gather(*pending, return_exceptions=True)
    f.generation = secrets.token_hex(24)
    results = await asyncio.gather(*(_call(request.app["session"], "POST", base + "/admin/reset",
                                          {"generation": f.generation}, timeout=300)
                                     for base in f.managers.values()))
    f.workers.clear()
    if any(st != 200 or body.get("engines_remaining") for st, body in results):
        return _err(503, "fleet reset failed", nodes=results)
    f.resetting = False
    return web.json_response({"engines_remaining": 0, "nodes_reset": len(results)})

async def handle_nodes(request: web.Request) -> web.Response:
    """Every machine, as its driver and kernel report it. One call per node, whatever else is running."""
    f = fleet(request)
    sess = request.app["session"]
    names = list(f.managers)
    out = await asyncio.gather(*(_call(sess, "GET", f"{f.managers[n]}/stats") for n in names))
    nodes = []
    for name, (st, r) in zip(names, out):
        nodes.append(r if st == 200 else {"node": name, "unreachable": True, "detail": r})
    return web.json_response(nodes)


async def handle_workers(request: web.Request) -> web.Response:
    """What is live, served from the placement measured at deploy -- no fan-out.

    Reports what was REQUESTED alongside what was MEASURED. The gap between them is how a plane discovers
    it stacked two engines on one card: with no `gpus` list, every engine on a node defaults to the same
    first cards.
    """
    f = fleet(request)
    names = list(f.managers)
    replies = await asyncio.gather(*(_call(request.app["session"], "GET", f.managers[n] + "/engines")
                                     for n in names))
    states = {n: {e["worker"]: e["state"] for e in body} if st == 200 else None
              for n, (st, body) in zip(names, replies)}
    out = []
    for wid, rec in list(f.workers.items()):
        observed = {n: states[n].get(wid, "gone") if states[n] is not None else "unreachable"
                    for n in rec["nodes"]}
        out.append(rec | {"live": all(s in {"up", "awake", "asleep", "sleeping", "waking"}
                                      for s in observed.values()), "node_states": observed})
    return web.json_response(out)


async def handle_logs(request: web.Request) -> web.Response:
    """Every participating node's log for this engine, not just the head's.

    TWO reasons this is on the controller and not on the worker.

    First, the worker may be dead -- which is exactly when its log is wanted. The log lives with the
    manager that launched the process, survives the engine, and survives the unload; asking a
    process for the reason it died does not work.

    Second, a multi-node engine has a log on EVERY node it spans, and the interesting one is usually not
    the head's. When a peer rank hits CUDA OOM, the head's log shows a rendezvous that never completed --
    a symptom -- while the cause sits on the other machine. Returning only the head would hand back the
    least informative half of the story. Only the controller knows which managers took part, so only the
    controller can collect them all.
    """
    f = fleet(request)
    wid = request.query.get("worker", "")
    lines = request.query.get("lines", "80")
    rec = f.workers.get(wid)
    # After an unload the record is gone but the post-mortems are not, so fall back to asking everyone.
    nodes = rec["nodes"] if rec else list(f.managers)
    got = await asyncio.gather(*(_call(request.app["session"], "GET",
                                       f"{f.managers[n]}/logs?worker={wid}&lines={lines}")
                                 for n in nodes if n in f.managers))
    found = [r for (st, r) in got if st == 200]
    if not found:
        return _err(404, f"no log for {wid!r} anywhere in the fleet")
    return web.json_response({"worker": wid, "nodes": found,
                              "head": rec["head"] if rec else None})


def build(f: Fleet) -> web.Application:
    app = web.Application(client_max_size=64 * 1024 * 1024)
    app["fleet"] = f
    app.add_routes([
        web.post("/deploy", handle_deploy),
        web.post("/unload", handle_unload),
        web.post("/admin/reset", handle_reset),
        web.get("/nodes", handle_nodes),
        web.get("/workers", handle_workers),
        web.get("/logs", handle_logs),
        web.get("/health", lambda r: web.json_response({"status": "ok", "run_id": os.environ.get("CLUSTER_RUN_ID", ""),
                                                        "nodes": len(fleet(r).managers)})),
    ])

    async def _session(a):
        # The default connection pool is 100, and a deploy fans out to every node of a placement at
        # once, so an unbounded pool avoids queueing deploys behind each other.
        a["session"] = aiohttp.ClientSession(
            connector=aiohttp.TCPConnector(limit=0, limit_per_host=0))
        yield
        await a["session"].close()
    app.cleanup_ctx.append(_session)
    return app


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fleet", default=os.environ.get("CLUSTER_NODES", "/app/fleet.json"))
    ap.add_argument("--catalogue", default=os.environ.get("CLUSTER_CATALOGUE", "/app/catalogue.json"))
    ap.add_argument("--port", type=int,
                    default=int(os.environ.get("CLUSTER_CONTROLLER_PORT", "9000")))
    a = ap.parse_args()
    fl = json.load(open(a.fleet))
    f = Fleet(fl["nodes"], json.load(open(a.catalogue)))
    print(f"[controller] :{a.port}, {len(f.managers)} node(s), {len(f.models)} model(s)", flush=True)
    web.run_app(build(f), host="0.0.0.0", port=a.port, access_log=None, print=None)


if __name__ == "__main__":
    main()
