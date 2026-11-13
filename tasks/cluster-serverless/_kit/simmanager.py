"""The emulated fleet, presenting the SAME manager surface the real one does.

`controller.py` runs unchanged on top of this: it only ever speaks HTTP to managers, so the emulator has
only to answer the same calls. That lets a plane be exercised with no GPUs.

WHAT IT SIMULATES. The real fleet keeps no books at all -- the plane's plan plays out physically and an
over-packed node meets the OOM killer. This has no GPUs to be physical about, so it keeps a memory model
and answers `GET /stats` from it. "Does it fit" is therefore APPROXIMATE here, which is what a simulator
is; it is a development utility, not the graded fleet.

The main divergence: **the emulator refuses cleanly where the real fleet crashes.** A plan that
oversubscribes gets a tidy 507 here and an OOM-killed engine with a `crashed` state on real GPUs.
"""
from __future__ import annotations

import argparse
import asyncio
from collections import OrderedDict, deque
import hashlib
import json
import math
import os
import signal
import subprocess
import sys
import time
from dataclasses import dataclass, field

import aiohttp
import uvloop
from aiohttp import web

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import engine_args  # noqa: E402
import physics  # noqa: E402
import prompts  # noqa: E402
import receipts  # noqa: E402
import protocol  # noqa: E402
import manager_auth  # noqa: E402
import looplag  # noqa: E402
import kv_layout  # noqa: E402
import sim_fabric  # noqa: E402

TIME_SCALE = float(os.environ.get("CLUSTER_TIME_SCALE", "1"))
# THE POLLING QUANTA SCALE TOO. These are wall-clock periods, and every other duration in this file
# is divided by TIME_SCALE -- leaving them fixed would coarsen the simulation's RESOLUTION by the same
# factor it speeds up (at scale 10 a 20 ms tick would become 200 ms of emulated time) and inflate
# simulated latencies. Dividing keeps emulated resolution constant at the cost of more wakeups.
TICK_S = 0.02 / TIME_SCALE
GPU_MIB = int(physics.GPU_GIB * 1024)
# vLLM's own default for --gpu-memory-utilization, used when the plane does not say.
DEFAULT_UTIL = 0.9
# How long a request will sit in the engine's waiting queue before the fleet gives up on it. The client
# normally abandons first -- this only catches a queue with nobody left waiting on it.
KV_QUEUE_MAX_S = float(os.environ.get("CLUSTER_KV_QUEUE_MAX_S", "1200"))
KV_QUEUE_POLL_S = 0.05 / TIME_SCALE
KEY = os.environ.get("CLUSTER_RECEIPT_KEY", "insecure-dev-key").encode()
GRADER_TOKEN = os.environ.get("CLUSTER_GRADER_TOKEN", "")
VOCAB = ("alpha beta gamma delta epsilon zeta eta theta iota kappa lambda mu nu xi omicron pi rho "
         "sigma tau upsilon phi chi psi omega").split()


def _err(status: int, message: str, **extra) -> web.Response:
    return web.json_response({"error": message, **extra}, status=status)


@dataclass
class SimEngine:
    worker_id: str
    model: str
    weights_gib: float          # this node's SHARD, not the whole model
    gpus: list                  # the cards it was pinned to, or [] for "all visible"
    is_head: bool
    active_gib: float = 0.0     # the ACTIVE part of that shard -- what decode reads per token.
                                # Equal to weights_gib for a dense model; as little as 1/25th of it for
                                # the MoE models, which is most of the large end of this catalogue.
                                # Serving from weights_gib alone would simulate every MoE as a dense
                                # model of its TOTAL size.
    tp: int = 1
    is_moe: bool | None = None
    tensor_parallel: int = 1
    pipeline_parallel: int = 1
    n_nodes: int = 1
    geometry: int = 32*4096
    architecture: dict = field(default_factory=dict)
    prefix_cache: OrderedDict = field(default_factory=OrderedDict)
    prefill_queue: deque = field(default_factory=deque)
    prefill_task: object = None
    kv_bytes_per_rank: float = 0.
    cache_layout: object = None
    kv_blocks: int = 0
    peers: tuple = ()           # [{"node", "manager"}] for every rank of this worker, head included
    extra_args: tuple = ()
    state: str = "launching"    # launching | asleep | awake | crashed | gone
    ready_at: float = 0.0
    worker_port: int = 0
    kv_tokens: float = 0.0
    ram_charged: float = 0.0    # GiB of host memory this engine has been charged so far
    slept: bool = False         # has it slept once? the offload buffer appears then and persists
    active: dict = field(default_factory=dict)
    served: int = 0
    aborted: int = 0
    log: list = field(default_factory=list)
    admin: asyncio.Lock = field(default_factory=asyncio.Lock)
    runner: object = None
    load_task: object = None
    transitions: set = field(default_factory=set)


@dataclass
class SimNode:
    name: str
    host: str
    port: int
    n_gpus: int
    host_ram_gib: float
    speed: float = 1.0
    engines: dict = field(default_factory=dict)
    graveyard: dict = field(default_factory=dict)
    host_ram_used: float = 0.0
    next_port: int = 0
    gpu_lock: dict = field(default_factory=dict)
    generation: str = ""
    timing: object = None
    prefill_locks: dict = field(default_factory=dict)
    prefill_busy: dict = field(default_factory=dict)
    fabric: object = None
    fabric_ledger: object = None


def node(request: web.Request) -> SimNode:
    return request.app["node"]


def _cards(n: SimNode, e: SimEngine) -> list[int]:
    """Which cards this engine occupies.

    Mirrors the real rule rather than inventing one: `gpus` is exactly CUDA_VISIBLE_DEVICES, and the
    engine args decide how many of the visible cards get used. With no list, the visible set is every
    card on the node and vLLM takes the first `tp` of them -- which is why two engines deployed without
    explicit lists collide on cards 0 and 1.
    """
    visible = e.gpus if e.gpus else list(range(n.n_gpus))
    return visible[:max(1, e.tp)]


TRACE_MODEL = os.environ.get("SIM_TRACE_MODEL", "")


def _trace(e, what: str) -> None:
    if TRACE_MODEL and e.model == TRACE_MODEL:
        print(f"[simtrace] {time.monotonic():10.3f} {e.worker_id} {e.model} {what} state={e.state}",
              flush=True)


def _gpu_used_mib(n: SimNode) -> dict[int, int]:
    used: dict[int, int] = {}
    for e in n.engines.values():
        if e.state not in ("awake", "asleep", "launching", "waking", "sleeping"):
            continue
        # Asleep still holds the CUDA context. Awake (or loading) holds exactly what it RESERVED:
        # vLLM takes `--gpu-memory-utilization` of the card up front and manages weights and KV inside
        # it, so more util is more KV and fewer engines per card, which is the trade a real operator
        # makes.
        if e.state == "asleep":
            share = int(physics.RESIDENT_ASLEEP_GIB * 1024)
        else:
            share = int(engine_args.util_of(e.extra_args, DEFAULT_UTIL) * physics.GPU_GIB * 1024)
        for c in _cards(n, e):
            used[c] = used.get(c, 0) + share
    return used


async def _serialise(n: SimNode, cards: list[int], seconds: float) -> None:
    """Loading and waking serialise per GPU, as they do on real hardware."""
    held = []
    try:
        for card in sorted(set(cards)):
            lock = n.gpu_lock.setdefault(card, asyncio.Lock())
            await lock.acquire()
            held.append(lock)
        await asyncio.sleep(max(0.0, seconds))
    finally:
        for lock in reversed(held):
            lock.release()


# ---------------------------------------------------------------- manager surface

async def handle_launch(request: web.Request) -> web.Response:
    n = node(request)
    body = await request.json()
    if body.get("generation", "") != n.generation:
        return _err(409, "deployment belongs to a retired fleet generation")
    if body.get("model") not in request.app["catalogue"]:
        return _err(404, "model is not in the manager catalogue")
    extra, _, problems = engine_args.validate(body.get("engine_args"))
    if problems:
        return _err(400, "; ".join(problems))
    body["engine_args"] = extra
    wid = body["worker_id"]
    if wid in n.engines:
        return _err(409, f"{wid} is already on {n.name}")
    tp = max(1, int(body.get("tp") or 1))
    nnodes = max(1, int(body.get("nnodes") or 1))
    # SIZE THE ENGINE FROM THE CATALOGUE, NOT FROM THE CALLER.
    #
    # A plane that could under-declare `weights_gib` would hold the entire catalogue resident, and you
    # cannot tell real vLLM its weights are smaller than they are either. The declared value is read
    # only to be checked against.
    shard = _weights_of(request, body["model"]) / nnodes
    declared = float(body.get("weights_gib") or 0.0)
    if declared and abs(declared - shard) > 0.05 * max(shard, 1.0):
        e_log_mismatch = (f"declared weights_gib={declared:.1f} but {body['model']} over {nnodes} node(s) "
                          f"is {shard:.1f} GiB/node; charging the real size")
    else:
        e_log_mismatch = ""
    e = SimEngine(worker_id=wid, model=body["model"], weights_gib=shard,
                  active_gib=_active_of(request, body["model"]) / nnodes,
                  is_moe=request.app["catalogue"][body["model"]].get("is_moe"),
                  tensor_parallel=tp, pipeline_parallel=int(body.get('pp') or 1), n_nodes=nnodes,
                  geometry=int(request.app['catalogue'][body['model']].get('architecture',{}).get('hidden_size') or 4096)
                           *int(request.app['catalogue'][body['model']].get('architecture',{}).get('num_hidden_layers') or 32),
                  gpus=list(body.get("gpus") or []), is_head=bool(body.get("is_head", True)),
                  tp=max(1, tp * int(body.get("pp") or 1) // nnodes),
                  extra_args=tuple(body.get("engine_args") or ()),
                  peers=tuple(tuple(sorted(q.items())) for q in (body.get("peers") or ())))
    architecture=request.app['catalogue'][body['model']].get('architecture',{})
    e.architecture = architecture
    e.cache_layout = kv_layout.build(architecture, tp=tp, pp=e.pipeline_parallel,
        max_model_len=int(request.app['catalogue'][body['model']].get('context_len') or 32768),
        extra_args=e.extra_args)
    layers=architecture.get('num_hidden_layers')
    pp=int(body.get('pp') or 1)
    if layers and architecture.get('kv_lora_rank'):
        e.kv_bytes_per_rank=layers/pp*(architecture['kv_lora_rank']+architecture.get('qk_rope_head_dim',64))*2
    elif layers and architecture.get('num_key_value_heads') and architecture.get('head_dim'):
        e.kv_bytes_per_rank=layers/pp*max(1,math.ceil(architecture['num_key_value_heads']/tp))*architecture['head_dim']*4
    if e_log_mismatch:
        e.log = [e_log_mismatch]          # visible in /logs: disagreeing silently would be worse

    # What vLLM will RESERVE on each card: the utilisation asked for, not the weights it happens to
    # hold. A launch that asks for 0.9 takes 90% of the card whether its weights need it or not.
    need = int(engine_args.util_of(body.get("engine_args") or (), DEFAULT_UTIL) * physics.GPU_GIB * 1024)
    used = _gpu_used_mib(n)
    for c in _cards(n, e):
        if used.get(c, 0) + need > GPU_MIB:
            e.log = [f"CUDA out of memory: card {c} has {GPU_MIB - used.get(c, 0)} MiB free, "
                     f"the shard needs {need} MiB"]
            n.graveyard[wid] = {"worker": wid, "model": e.model, "reason": "out of GPU memory",
                                "exit_code": 1, "log_tail": e.log}
            return _err(507, f"{e.model}: card {c} on {n.name} cannot fit this shard",
                        free_mib=GPU_MIB - used.get(c, 0), needed_mib=need,
                        occupants=[{"model": o.model, "state": o.state, "gpus": list(o.gpus),
                                    "util": round(engine_args.util_of(o.extra_args, DEFAULT_UTIL), 2)}
                                   for o in n.engines.values() if c in _cards(n, o)])
    # HOST RAM, THE SCARCITY THIS TASK IS ABOUT. A sleeping engine holds its weights in host RAM, and
    # the catalogue is deliberately oversubscribed against that budget -- deciding what is resident is
    # the problem the plane exists to solve. The budget is checked here, as the kernel enforces it on
    # real nodes, so a plane exercised against the emulator cannot over-commit for free.
    #
    # Every local rank needs process memory. Offload buffers are allocated only
    # on first sleep and then retained in the pinned-memory allocator's cache.
    ram_need = physics.HOST_RAM_PROCESS_GIB * len(_cards(n, e))
    if n.host_ram_used + ram_need > n.host_ram_gib:
        e.log = [f"host RAM exhausted on {n.name}: "
                 f"{n.host_ram_gib - n.host_ram_used:.1f} GiB free, this engine needs {ram_need:.1f}"]
        n.graveyard[wid] = {"worker": wid, "model": e.model, "reason": "out of host RAM",
                            "exit_code": 1, "log_tail": e.log}
        return _err(507, f"{e.model}: {n.name} cannot hold this engine's weights",
                    free_gib=round(n.host_ram_gib - n.host_ram_used, 2),
                    needed_gib=round(ram_need, 2))
    n.engines[wid] = e
    n.host_ram_used += ram_need
    e.ram_charged = ram_need
    if n.fabric:
        try:
            await n.fabric.call('register', worker=wid, node=n.name, cards=_cards(n, e), speed=n.speed)
        except BaseException:
            n.engines.pop(wid, None)
            n.host_ram_used -= ram_need
            raise
    # The cold load, serialised on the first card the engine sits on.
    e.load_task = asyncio.create_task(_load(n, e))
    return web.json_response({"worker": wid, "node": n.name, "pgid": 100000 + len(n.engines),
                              "vllm_port": 0, "state": e.state})


def _weights_of(request: web.Request, model: str) -> float:
    spec = request.app["catalogue"].get(model)
    return float(spec["weights_gib"]) if spec else 1.0


def _active_of(request: web.Request, model: str) -> float:
    """The model's ACTIVE weights -- what a decode step reads. Falls back to the total, which is the
    right answer for a dense model and the conservative one for anything missing the field."""
    spec = request.app["catalogue"].get(model)
    if not spec:
        return 1.0
    return float(spec.get("serving_active_gib") or spec.get("active_gib") or spec["weights_gib"])


async def _load(n: SimNode, e: SimEngine) -> None:
    _trace(e, "load: waiting for card")
    await _serialise(n, _cards(n, e), physics.cold_load_s(e.weights_gib) / n.speed / TIME_SCALE)
    _trace(e, "load: card time done")
    # Process memory was reserved at launch. First-sleep buffers, if allocated,
    # remain charged through waking. Check ownership before completing a load:
    # a teardown during the cold load pops the engine from `n.engines` but does NOT cancel this task,
    # so without this check it would resume and charge host RAM to a worker that no longer exists, with
    # nothing left holding a reference to release it. `ram_charged` makes charge and release symmetric.
    if n.engines.get(e.worker_id) is not e:
        e.state = "gone"
        return
    # ONLY PROCESS OVERHEAD AT LOAD. A vLLM engine holds a fraction of a GiB of shmem after loading,
    # whether or not `--enable-sleep-mode` was passed, and a few GiB of anonymous memory that does not
    # scale with the model. The weights land in host memory on the FIRST SLEEP (see the sleep handler),
    # not at load.
    #
    # Charging the full buffer here would make deploying without sleep mode impossible rather than a
    # different trade. It is the plane's choice: pay a card permanently and keep the host free, or pay
    # the host and get the card back on demand.
    # EVERY engine lands AWAKE, whether or not it can sleep. `--enable-sleep-mode` makes an engine
    # *able* to sleep; it does not put it to sleep, and it costs nothing until the first /admin/sleep:
    # an engine with the flag set but never slept holds the same shmem as one without it, and the
    # offload buffer appears only across the first /sleep call.
    #
    # Landing a flagged engine asleep would fold two separable decisions into the deploy: a model too
    # large to finish load+first-sleep inside the setup window could only be deployed WITHOUT the flag,
    # and would then be stuck awake on its card for the whole run with no way back. That is not a
    # physical limit.
    e.state = "awake"
    e.kv_tokens = _kv_budget(n, e)
    if e.cache_layout and e.kv_blocks < e.cache_layout.startup_blocks():
        e.state = 'crashed'
        e.log.append('KV cache cannot hold the advertised context at this memory utilization')
    e.ready_at = time.monotonic()
    _trace(e, "load: END -> awake")


def _kv_budget(n: SimNode, e: SimEngine) -> float:
    """Tokens of KV the engine's share of the card leaves room for, after its weights.

    FROM THE UTILISATION THE PLANE ASKED FOR. vLLM reserves `--gpu-memory-utilization` of each card and
    the KV cache is whatever is left once the weights are in it, so util is the knob an operator turns
    to trade concurrency against how many models share a card.

    Assigned at EVERY transition into `awake`, not only in wake: sleep is opt-in, so a deploy can land
    awake without ever passing through the wake path, and an engine whose budget stayed at 0.0 would be
    awake with a zero-token KV cache and refuse every request.
    """
    cards = len(_cards(n, e))
    claimed_per_card = engine_args.util_of(e.extra_args, DEFAULT_UTIL) * physics.GPU_GIB
    kv_gib = max(0.0, claimed_per_card - e.weights_gib / cards - physics.GPU_RUNTIME_OVERHEAD_GIB) * cards
    if e.cache_layout:
        available_bytes = kv_gib/cards*2**30
        e.kv_blocks = max(0, int(available_bytes//e.cache_layout.bytes_per_block)-1)
        return e.cache_layout.capacity_tokens(available_bytes)
    if e.kv_bytes_per_rank:
        return kv_gib*2**30/(e.kv_bytes_per_rank*cards)
    return kv_gib / physics.kv_gib_per_token(e.weights_gib)


async def handle_ready(request: web.Request) -> web.Response:
    n = node(request)
    e = n.engines.get(request.match_info["wid"])
    if e is None:
        return _err(404, f"not on {n.name}")
    timeout = float(request.query.get("timeout", 1200))
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        if n.engines.get(e.worker_id) is not e:
            return _err(404, "engine was removed while waiting for readiness")
        if e.state in ("asleep", "awake"):
            return web.json_response({"ready": True, "state": e.state})
        if e.state == "crashed":
            return web.json_response({"ready": False, "reason": "crashed", "log_tail": e.log})
        await asyncio.sleep(0.05)
    return web.json_response({"ready": False, "reason": f"not ready within {timeout:.0f}s"})


async def handle_settled(request: web.Request) -> web.Response:
    # The real collective cannot serve until every rank has loaded. Simulated
    # ranks load independently, so settling must wait for this rank as well.
    response = await handle_ready(request)
    if response.status != 200:
        return response
    value = json.loads(response.text)
    return web.json_response(value, status=200 if value.get("ready") else 503)


async def handle_start_worker(request: web.Request) -> web.Response:
    n = node(request)
    wid = request.match_info["wid"]
    e = n.engines.get(wid)
    if e is None:
        return _err(404, f"{wid} is not on {n.name}")
    if not e.worker_port:
        runner = web.AppRunner(build_worker(n, e), access_log=None, shutdown_timeout=0.1,
                               handler_cancellation=True)
        await runner.setup()
        # Fleets keep the firewall's fixed worker port window (9101-9299), probing for a free port in
        # it. With CLUSTER_SIM_EPHEMERAL_WORKERS=1 (loopback only) the OS picks a free port instead, so
        # several emulated fleets can share one host without exhausting that window.
        ephemeral = os.environ.get('CLUSTER_SIM_EPHEMERAL_WORKERS') == '1'
        if ephemeral and n.host not in ('127.0.0.1', 'localhost'):
            await runner.cleanup()
            return _err(400, 'ephemeral simulator workers require loopback nodes')
        for port in ([0] if ephemeral else range(9101, 9300)):
            site = web.TCPSite(runner, '127.0.0.1' if ephemeral else '0.0.0.0', port)
            try:
                await site.start()
                break
            except OSError:
                continue
        else:
            await runner.cleanup()
            return _err(503, "worker port window is exhausted")
        e.worker_port = site._server.sockets[0].getsockname()[1]
        e.runner = runner
    return web.json_response({"worker": wid, "url": f"http://{n.host}:{e.worker_port}",
                              "port": e.worker_port})


async def handle_rank_state(request: web.Request) -> web.Response:
    """Apply a sleep or wake to this node's rank of a worker, without fanning out again.

    The head calls this on its peers. It is deliberately separate from the engine's own /admin/sleep:
    if peers re-broadcast, a two-node engine would ping-pong forever.
    """
    n = node(request)
    e = n.engines.get(request.match_info["wid"])
    if e is None:
        return _err(404, f"no worker {request.match_info['wid']!r} on {n.name}")
    want = request.match_info["what"]
    ok, detail = await _apply_state(n, e, want)
    if not ok:
        return _err(502, detail.get("error", "refused"), **detail)
    return web.json_response({"worker": e.worker_id, "node": n.name, "state": e.state})


async def _apply_state(n: SimNode, e, want: str) -> tuple[bool, dict]:
    task = asyncio.current_task()
    e.transitions.add(task)
    try:
        return await _transition(n, e, want)
    finally:
        e.transitions.discard(task)


async def _transition(n: SimNode, e, want: str) -> tuple[bool, dict]:
    """Reserve memory before waiting; serialize transitions on every occupied card."""
    async with e.admin:
        if n.engines.get(e.worker_id) is not e:
            return False, {"error": "worker has been unloaded"}
        if want not in {"sleep", "wake"} or e.state not in {"awake", "asleep"}:
            return False, {"error": f"cannot {want} an engine in state {e.state}"}
        target = "awake" if want == "wake" else "asleep"
        if e.state == target:
            return True, {}
        previous = e.state
        if want == "wake":
            need = int(engine_args.util_of(e.extra_args, DEFAULT_UTIL) * GPU_MIB)
            mine = int(physics.RESIDENT_ASLEEP_GIB * 1024)
            used = _gpu_used_mib(n)
            for c in _cards(n, e):
                free = GPU_MIB - (used.get(c, 0) - mine)
                if need > free:
                    return False, {"error": f"{e.model}: card {c} cannot take this engine back",
                                   "free_mib": free, "needed_mib": need}
            # No await between the capacity check and this reservation.
            e.state = "waking"
            seconds = physics.wake_ms(e.weights_gib / len(_cards(n, e))) / 1000.0
        else:
            wanted = physics.host_ram_gib(e.weights_gib, local_ranks=len(_cards(n, e)))
            delta = max(0.0, wanted - e.ram_charged) if not e.slept else 0.0
            if n.host_ram_used + delta > n.host_ram_gib:
                return False, {"error": f"{e.model}: insufficient host RAM to sleep",
                               "free_gib": n.host_ram_gib - n.host_ram_used, "needed_gib": delta}
            n.host_ram_used += delta
            e.ram_charged += delta
            for rec in e.active.values():
                rec["aborted"] = True
            await _stop_prefill(e)
            seconds = physics.sleep_s(e.weights_gib / len(_cards(n, e)), first=not e.slept)
            e.slept = True
            e.state = "sleeping"
        try:
            await _serialise(n, _cards(n, e), seconds / n.speed / TIME_SCALE)
        except BaseException:
            if n.engines.get(e.worker_id) is e:
                e.state = previous
            raise
        if n.engines.get(e.worker_id) is not e:
            return False, {"error": "worker was unloaded during transition"}
        e.state = target
        e.kv_tokens = _kv_budget(n, e) if target == "awake" else 0.0
        if target == 'asleep':
            e.prefix_cache.clear()
        return True, {}


async def _fan_out(request: web.Request, n: SimNode, e, want: str) -> list[tuple[str, int, dict]]:
    """Carry a transition to this worker's other ranks. Real multi-node vLLM is ONE server: the head
    coordinates its workers, so sleeping it sleeps the whole thing (lifecycle operations apply to the
    entire engine). Sleeping only the rank whose URL was called would leave the other ranks awake
    holding their cards while the caller had a 200.

    The node is passed in, not read from `request.app`: this runs on the ENGINE's app, which is a
    different aiohttp application from the manager's and has no "node" key."""
    out = []
    sess = request.app.get("peer_session")
    if sess is None:
        sess = aiohttp.ClientSession()
        request.app["peer_session"] = sess
    for peer in e.peers:
        d = dict(peer)
        if d.get("node") == n.name:
            continue
        url = f"{d['manager']}/engines/{e.worker_id}/rank/{want}"
        try:
            async with sess.post(url, json={}, headers={"X-Grader-Token": GRADER_TOKEN},
                                 timeout=aiohttp.ClientTimeout(total=1800)) as r:
                out.append((d["node"], r.status, await r.json()))
        except Exception as ex:                                           # noqa: BLE001
            out.append((d["node"], 0, {"error": f"{type(ex).__name__}: {ex}"}))
    return out


async def handle_placement(request: web.Request) -> web.Response:
    n = node(request)
    e = n.engines.get(request.match_info["wid"])
    if e is None:
        return _err(404, "unknown")
    used = _gpu_used_mib(n)
    return web.json_response({"node": n.name, "state": e.state,
                              "gpus": [{"index": c, "used_mib": used.get(c, 0)}
                                       for c in _cards(n, e)]})


async def handle_stats(request: web.Request) -> web.Response:
    n = node(request)
    used = _gpu_used_mib(n)
    return web.json_response({
        "node": n.name, "host": n.host, "simulated": True,
        "host_ram": {"total_gib": round(n.host_ram_gib, 2),
                     "used_gib": round(n.host_ram_used, 2),
                     "free_gib": round(n.host_ram_gib - n.host_ram_used, 2)},
        "simulation": (n.timing.snapshot() if n.timing else {"time_scale": TIME_SCALE})
                      | (n.fabric.measurement() if n.fabric else {}),
        "gpus": [{"index": i, "total_mib": GPU_MIB, "used_mib": used.get(i, 0)}
                 for i in range(n.n_gpus)],
        # `gpus` too: without it /stats cannot answer "what is on card 3", which is the one question
        # a plane needs when the fleet refuses a placement it was sure about.
        "engines": [{"worker": e.worker_id, "model": e.model, "state": e.state,
                     "is_head": e.is_head, "exit_code": None, "gpus": list(e.gpus)}
                    for e in n.engines.values()],
    })


async def handle_logs(request: web.Request) -> web.Response:
    n = node(request)
    wid = request.query.get("worker", "")
    e = n.engines.get(wid)
    if e is not None:
        return web.json_response({"worker": wid, "node": n.name, "state": e.state,
                                  "exit_code": None, "lines": e.log})
    if (g := n.graveyard.get(wid)) is not None:
        return web.json_response({**g, "node": n.name, "state": "gone"})
    return _err(404, f"no log for {wid!r} on {n.name}")


async def handle_teardown(request: web.Request) -> web.Response:
    n = node(request)
    wid = request.match_info["wid"]
    e = n.engines.pop(wid, None)
    if e is None:
        return web.json_response({"worker": wid, "node": n.name, "status": "already gone"})
    if n.fabric:
        await n.fabric.call('remove', worker=wid)
    for rec in e.active.values():
        rec["aborted"] = True
    await _stop_prefill(e)
    # Refund only what was actually charged. The charge happens when the load finishes (see `_load`),
    # so an engine torn down mid-load never paid and must not be refunded.
    if e.ram_charged:
        n.host_ram_used = max(0.0, n.host_ram_used - e.ram_charged)
        e.ram_charged = 0.0
    n.graveyard[wid] = {"worker": wid, "model": e.model, "reason": "unloaded by the control plane",
                        "exit_code": None, "log_tail": e.log}
    e.state = "gone"
    pending = list(e.transitions)
    for task in pending:
        task.cancel()
    await asyncio.gather(*pending, return_exceptions=True)
    if e.load_task:
        e.load_task.cancel()
        await asyncio.gather(e.load_task, return_exceptions=True)
    if e.runner:
        await e.runner.cleanup()
    return web.json_response({"worker": wid, "node": n.name, "status": "unloaded",
                              "aborted_requests": len(e.active)})


# ---------------------------------------------------------------- the emulated worker

def _busy_seconds(n, card, now):
    total, started = n.prefill_busy.get(card, (0., None))
    return total + (now-started if started is not None else 0.)


def _decode_rate(n, e, batch, context_tokens=0.):
    # TP divides each token's weight read. PP divides placement, not that serial
    # read's latency; it gets no invented single-stream speedup here.
    ranks = max(1, e.tensor_parallel)
    rate = physics.decode_tok_s(e.weights_gib*e.n_nodes/ranks, batch,
                                e.active_gib*e.n_nodes/ranks or None,
                                is_moe=e.is_moe, context_tokens=context_tokens, architecture=e.architecture)
    rate = batch/(batch/max(1.,rate)+physics.TP_HOP_MS*math.log2(ranks)/1000)
    rate /= physics.cross_node_penalty(e.n_nodes)
    if n.fabric:
        # The shared service clock already charges every rank's co-tenants.
        return rate
    cards = set(_cards(n, e))
    overlap = max((sum(bool(set(_cards(n, other)) & {card})
                       and any(r.get('decoding') and not r['aborted'] for r in other.active.values())
                       for other in n.engines.values()) for card in cards), default=1)
    return rate/max(1, overlap)


async def _prefill_chunk(n, e, duration):
    """One engine forward pass reserves all its GPUs for the measured work."""
    if n.fabric:
        await n.fabric.call('prefill', worker=e.worker_id, seconds=max(0., duration)/TIME_SCALE)
        return
    held = []
    cards = sorted(_cards(n, e))
    try:
        for card in cards:
            lock = n.prefill_locks.setdefault(card, asyncio.Lock())
            await lock.acquire()
            held.append(lock)
        start = time.monotonic()
        for card in cards:
            total, _ = n.prefill_busy.get(card, (0., None))
            n.prefill_busy[card] = (total, start)
        try:
            await asyncio.sleep(max(0., duration)/n.speed/TIME_SCALE)
        finally:
            now = time.monotonic()
            for card in cards:
                n.prefill_busy[card] = (_busy_seconds(n, card, now), None)
    finally:
        for lock in reversed(held):
            lock.release()


async def _prefill_schedule(n, e):
    """Pack token chunks into a common engine iteration, in admission order.

    A real iteration emits completed prefills together; its token budget also
    includes the running decoders. Modelling each request's prefill separately
    would emit first tokens partway through a batch and underpredict burst TTFT.
    """
    def cost(tokens, dispatch=False):
        return physics.prefill_ms(tokens, (e.active_gib or e.weights_gib)*e.n_nodes,
            is_moe=bool(e.is_moe), geometry=e.geometry, tp=e.tensor_parallel,
            architecture=e.architecture, include_dispatch=dispatch)/1000
    initial = True
    try:
        while e.prefill_queue:
            e.prefill_queue = deque(j for j in e.prefill_queue if not j['future'].done())
            if not e.prefill_queue:
                break
            decoding = sum(r.get('decoding', False) and not r['aborted'] for r in e.active.values())
            budget = engine_args.int_of(e.extra_args, '--max-num-batched-tokens', 4096)
            if engine_args.has(e.extra_args, '--no-enable-chunked-prefill'):
                budget = max(budget, max(j['size']-j['done'] for j in e.prefill_queue))
            budget = max(1, budget-decoding)
            scheduled = []
            duration = cost(0, dispatch=True)
            for job in e.prefill_queue:
                take = min(job['size']-job['done'], budget)
                if take <= 0:
                    continue
                duration += cost(job['done']+take)-cost(job['done'])
                scheduled.append((job, take))
                budget -= take
                # An idle vLLM core starts its first arrived prefill while the
                # API process is still forwarding the rest of a burst. Model
                # that first iteration explicitly, instead of letting Python
                # event-loop batching change TTFT with simulator time scale.
                if not budget or initial:
                    break
            await _prefill_chunk(n, e, duration)
            initial = False
            for job, take in scheduled:
                job['done'] += take
                if job['done'] == job['size'] and not job['future'].done():
                    job['future'].set_result(None)
            # Completed prefills can emit their first token before the next
            # scheduling iteration. Existing decoders get one service step.
            await asyncio.sleep(0)
            decoding = sum(r.get('decoding', False) and not r['aborted'] for r in e.active.values())
            if decoding:
                await asyncio.sleep(decoding/max(1., _decode_rate(n, e, decoding))/n.speed/TIME_SCALE)
    except BaseException as exc:
        for job in e.prefill_queue:
            if not job['future'].done():
                job['future'].set_exception(exc)
        e.prefill_queue.clear()
        if isinstance(exc, asyncio.CancelledError):
            raise


async def _prefill(n, e, size, cached=0):
    future = asyncio.get_running_loop().create_future()
    e.prefill_queue.append({'size': size, 'done': cached, 'future': future})
    if e.prefill_task is None or e.prefill_task.done():
        e.prefill_task = asyncio.create_task(_prefill_schedule(n, e))
    try:
        await future
    finally:
        # The scheduler skips a disconnected request at its next iteration.
        future.cancel()


async def _stop_prefill(e):
    if e.prefill_task and not e.prefill_task.done():
        e.prefill_task.cancel()
        await asyncio.gather(e.prefill_task, return_exceptions=True)


def _prefix_keys(tokens, block=16):
    packed = prompts.canon(tokens)
    state = b''
    for offset in range(0, len(tokens)//block*block*4, block*4):
        state = hashlib.sha256(state+packed[offset:offset+block*4]).digest()
        yield state


def _cache_prefix(e, tokens):
    if e.cache_layout:
        if not e.cache_layout.simple_prefix_cache:
            return
        block = e.cache_layout.groups[0].block_size
        free = max(0, e.kv_blocks-sum(r['kv_blocks'] for r in e.active.values()))*block
    else:
        block = engine_args.int_of(e.extra_args, '--block-size', 16)
        free = max(0., e.kv_tokens-sum(r['kv_tokens'] for r in e.active.values()))
    for key in _prefix_keys(tokens, block):
        e.prefix_cache[key] = None
        e.prefix_cache.move_to_end(key)
    while len(e.prefix_cache)*block > free:
        e.prefix_cache.popitem(last=False)

def build_worker(n: SimNode, e: SimEngine) -> web.Application:
    @web.middleware
    async def live(request, handler):
        if n.engines.get(e.worker_id) is not e:
            return _err(410, "worker has been unloaded")
        return await handler(request)

    app = web.Application(client_max_size=64 * 1024 * 1024, middlewares=[live])

    async def close_peers(app):
        if session := app.get("peer_session"):
            await session.close()
    app.on_cleanup.append(close_peers)

    def _sleep_denied():
        if not engine_args.has(e.extra_args, "--enable-sleep-mode"):
            return _err(409, f"{e.model} was deployed without --enable-sleep-mode, so it cannot sleep "
                             f"or wake. Redeploy it with that flag if you want to evict it later.",
                        sleep_supported=False)
        return None

    async def completions(request: web.Request) -> web.StreamResponse:
        body = await request.json()
        if problem := protocol.validate(body):
            return _err(400, problem)
        if body.get("model") not in (None, "", e.model):
            return _err(404, f"{e.worker_id} serves {e.model}")
        out_tok = int(body.get("max_tokens") or 0)
        if out_tok <= 0:
            return _err(400, "max_tokens is required")
        if e.state == "asleep":
            return _err(409, f"{e.model} is asleep on {e.worker_id}", state="asleep")
        if e.state != "awake":
            return _err(503, f"{e.model} is {e.state} on {e.worker_id}", state=e.state)
        # TOKEN IDS, matching the real worker. `in_tok` is an exact token count.
        prompt_ids = [int(x) for x in (body.get("prompt") or [])]
        in_tok = len(prompt_ids)
        # Resolve the engine's output cap before admission, as worker.py does
        # before forwarding to vLLM. The public bucket remains receipt-bound;
        # reserving that larger bucket can strand a request whose actual work
        # fits. Full-output reservation is still a conservative approximation
        # of vLLM's incremental block allocation and preemption.
        wire_max = out_tok
        intended = prompts.read_control(KEY, prompt_ids)
        if intended is not None:
            out_tok = max(1, min(intended, wire_max))
        else:
            import random
            out_tok = random.randint(1, wire_max)
        # A FULL KV CACHE IS A QUEUE, NOT A REFUSAL.
        #
        # vLLM admits a request to its waiting queue and schedules it when the cache has room; the
        # caller sees latency, not an error. Refusing outright would make a burst indistinguishable
        # from an overload: batch jobs submit hundreds of requests at one instant, far above the
        # steady-state concurrency, and work that would fit moments later would come back 503.
        #
        # Waiting is bounded from the other end: the caller abandons at its own deadline and aiohttp
        # cancels this handler, so a queue that never drains costs the request and nothing else. The
        # cap here is a backstop against a wait with no client left to notice.
        waited = 0.0
        sequence_limit = engine_args.int_of(e.extra_args, '--max-num-seqs', 256)
        blocks = e.cache_layout.blocks(in_tok+out_tok) if e.cache_layout else 0
        def cache_full():
            if e.cache_layout:
                return sum(r['kv_blocks'] for r in e.active.values())+blocks > e.kv_blocks
            return sum(r['kv_tokens'] for r in e.active.values())+in_tok+out_tok > max(1., e.kv_tokens)
        while (len(e.active) >= sequence_limit or
               cache_full()):
            if waited >= KV_QUEUE_MAX_S or e.state != "awake":
                return _err(503, f"{e.model} KV cache full on {e.worker_id}", state="kv_full")
            await asyncio.sleep(KV_QUEUE_POLL_S)
            waited += KV_QUEUE_POLL_S * TIME_SCALE

        rid = object()
        resp = web.StreamResponse(status=200, headers={"Content-Type": "text/event-stream",
                                                       "Cache-Control": "no-cache"})
        rec = {"aborted": False, "emitted": 0, "kv_tokens": in_tok + out_tok,
               "kv_blocks": blocks, "decoding": False}
        e.active[rid] = rec
        # The emulator's answer is ids, like the real worker's. `prompts.response` is keyed to the
        # prompt and the model, so a plane that alters either gets an answer that fails verification.
        answer = prompts.response(KEY, e.model, prompt_ids, out_tok)
        cache_tokens = None
        try:
            await resp.prepare(request)
            cache_enabled = (not engine_args.has(e.extra_args, '--no-enable-prefix-caching')
                             and (not e.cache_layout or e.cache_layout.simple_prefix_cache))
            cached = 0
            if cache_enabled:
                block = (e.cache_layout.groups[0].block_size if e.cache_layout else
                         engine_args.int_of(e.extra_args, '--block-size', 16))
                for key in _prefix_keys(prompt_ids, block):
                    if key not in e.prefix_cache: break
                    e.prefix_cache.move_to_end(key)
                    cached += block
            await _prefill(n, e, in_tok, min(cached, max(0, in_tok-1)))
            if rec['aborted']:
                e.aborted += 1
                return resp
            rec["decoding"] = True
            if e.cache_layout:
                rec['kv_blocks'] = e.cache_layout.blocks(in_tok+out_tok, decoding=True)
            if n.fabric:
                await n.fabric.activity(e.worker_id, True)
            # The fitted prefill curve measures time to the FIRST token. Emit
            # it now rather than adding an unrelated 20 ms polling quantum.
            await resp.write(b'data: '+json.dumps({'choices': [{'index': 0, 'text': '',
                              'token_ids': answer[:1]}]}).encode()+b'\n\n')
            rec['emitted'] = 1
            fractional = 0.0
            previous_tick = time.monotonic()
            previous_credit = n.fabric.credit(e.worker_id) if n.fabric else 0.
            previous_busy = {card: _busy_seconds(n, card, previous_tick) for card in _cards(n, e)}
            while rec["emitted"] < out_tok and not rec["aborted"]:
                await asyncio.sleep(TICK_S)
                now = time.monotonic()
                elapsed = now - previous_tick
                previous_tick = now
                # Sleeping is a lower bound. Charging only the requested sleep
                # would reduce throughput whenever the scheduler wakes late, so
                # use elapsed fleet time; lag validation still rejects overload.
                batch = max(1, sum(r["decoding"] and not r["aborted"] for r in e.active.values()))
                rate = _decode_rate(n, e, batch, in_tok+rec['emitted']) / batch
                busy = {card: _busy_seconds(n, card, now) for card in previous_busy}
                prefill_elapsed = max((busy[c]-previous_busy[c] for c in busy), default=0.)
                previous_busy = busy
                if n.fabric:
                    credit = n.fabric.credit(e.worker_id)
                    fractional += rate*max(0., credit-previous_credit)*TIME_SCALE
                    previous_credit = credit
                else:
                    fractional += rate * n.speed * max(0., elapsed-prefill_elapsed) * TIME_SCALE
                n_new = min(int(fractional), out_tok - rec["emitted"])
                if n_new == 0:
                    continue
                fractional -= n_new
                # Every token carries its own leading space, the way a real tokenizer's do, so the
                # concatenation of deltas is exactly what gets hashed.
                chunk_ids = answer[rec["emitted"]:rec["emitted"] + n_new]
                rec["emitted"] += n_new
                # `text` and `token_ids`, the /v1/completions streaming shape the real worker reads.
                await resp.write(b"data: " + json.dumps(
                    {"choices": [{"index": 0, "text": "", "token_ids": chunk_ids}]}).encode() + b"\n\n")
            if rec["aborted"]:
                e.aborted += 1
                return resp
            out_ids = answer[:rec["emitted"]]
            finish = "length" if rec["emitted"] >= out_tok else ""
            # EMIT the finish frame, do not merely sign it: the grader reconstructs finish_reason from
            # the stream, and a signed value that was never sent would fail verification.
            await resp.write(b"data: " + json.dumps(
                {"choices": [{"index": 0, "text": "", "finish_reason": finish}]}).encode() + b"\n\n")
            receipt = receipts.sign(KEY, {
                "model": e.model,
                "prompt_sha256": receipts.sha256_hex(prompts.canon(prompt_ids)),
                "max_tokens": wire_max,
                "text_sha256": receipts.sha256_hex(prompts.canon(out_ids)),
                "finish_reason": finish,
                "request_nonce": body.get("routing", {}).get("request_nonce", ""),
            })
            await resp.write(b"data: " + json.dumps({"cluster_receipt": receipt}).encode() + b"\n\n")
            await resp.write(b"data: [DONE]\n\n")
            e.served += 1
            if cache_enabled:
                cache_tokens = prompt_ids+out_ids
            return resp
        finally:
            e.active.pop(rid, None)
            if n.fabric and not any(r.get('decoding') and not r['aborted'] for r in e.active.values()):
                await n.fabric.activity(e.worker_id, False)
            if cache_tokens is not None and e.state == 'awake':
                _cache_prefix(e, cache_tokens)

    async def sleep_(request: web.Request) -> web.Response:
        if (deny := _sleep_denied()) is not None:
            return deny
        if e.state == "asleep":
            return web.json_response({"status": "already asleep", "aborted_requests": 0})
        n_aborted = len(e.active)
        ok, detail = await _apply_state(n, e, "sleep")
        if not ok:
            # 502, not 507, and the difference is parity rather than taste. A deploy that will not fit
            # is refused before anything starts, which the real manager also does. A SLEEP that will
            # not fit is vLLM's own sleep call failing for want of memory, and the real worker reports
            # exactly that as `502 would not sleep`.
            return _err(502, detail.pop("error", "would not sleep"), **detail)
        peers = await _fan_out(request, n, e, "sleep")
        bad = [(nd, st, d) for nd, st, d in peers if st != 200]
        if bad:
            # Half an engine asleep is not a state the real thing can be in. Put it back.
            await _apply_state(n, e, "wake")
            await _fan_out(request, n, e, "wake")
            return _err(502, f"{e.model} would not sleep: {bad[0][0]} refused",
                        detail={nd: d for nd, _st, d in bad})
        return web.json_response({"status": "asleep", "aborted_requests": n_aborted,
                                  "ranks": 1 + len(peers)})

    async def wake_(request: web.Request) -> web.Response:
        if (deny := _sleep_denied()) is not None:
            return deny
        if e.state == "awake":
            return web.json_response({"status": "already awake"})
        # 507, like a launch that will not fit: the caller can free a card and try again. Ignoring the
        # refusal would put the fleet back in the state this check exists to prevent.
        ok, detail = await _apply_state(n, e, "wake")
        if not ok:
            return _err(507, detail.pop("error", "cannot take this engine back"), **detail)
        peers = await _fan_out(request, n, e, "wake")
        bad = [(nd, st, d) for nd, st, d in peers if st != 200]
        if bad:
            # A half-woken engine is not a state the real thing can be in -- the ranks that did come
            # back are holding cards the collective cannot use. Put them all back to sleep and report
            # the refusal, so the plane frees room and retries rather than believing it has an engine.
            await _apply_state(n, e, "sleep")
            await _fan_out(request, n, e, "sleep")
            nd, st, d = bad[0]
            return _err(507, f"{e.model}: rank on {nd} cannot take this engine back", detail=d)
        return web.json_response({"status": "awake", "ranks": 1 + len(peers)})

    async def rekey(request: web.Request) -> web.Response:
        global KEY
        if not GRADER_TOKEN or request.headers.get("X-Grader-Token") != GRADER_TOKEN:
            return _err(403, "grader token required")
        KEY = ((await request.json()).get("key") or "").encode() or KEY
        return web.json_response({"status": "rekeyed", "worker": e.worker_id})

    app.add_routes([
        web.post("/v1/completions", completions),
        web.get("/v1/models", lambda r: web.json_response(
            {"object": "list", "data": [{"id": e.model, "object": "model"}]})),
        web.post("/admin/sleep", sleep_),
        web.post("/admin/wake", wake_),
        web.post("/admin/rekey", rekey),
        web.get("/stats", lambda r: web.json_response(
            {"worker": e.worker_id, "node": n.name, "model": e.model, "state": e.state,
             "in_flight": len(e.active), "served": e.served, "aborted": e.aborted,
             "sleep_supported": engine_args.has(e.extra_args, "--enable-sleep-mode")})),
        web.get("/health", lambda r: web.json_response({"status": "ok", "worker": e.worker_id})),
    ])
    return app


async def handle_mgr_reset(request: web.Request) -> web.Response:
    """Return this emulated node to empty. Mirrors the real manager's /admin/reset.

    An emulated fleet (`bin/cluster.sh --sim` / `--one-box`) answers the same admin calls as the real
    one, so the grading path's reset and rekey work on both backends.
    """
    if not GRADER_TOKEN or request.headers.get("X-Grader-Token") != GRADER_TOKEN:
        return _err(403, "grader token required")
    n = node(request)
    n.generation = (await request.json()).get("generation", "")
    if n.timing:
        n.timing.reset()
    if n.fabric:
        n.fabric.failure_count = 0
        n.fabric.last_failure = None
    wids = list(n.engines)
    for wid in wids:
        e = n.engines.pop(wid, None)
        if e is None:
            continue
        if n.fabric:
            await n.fabric.call('remove', worker=wid)
        for rec in e.active.values():
            rec["aborted"] = True
        await _stop_prefill(e)
        # Refund exactly what teardown refunds -- only what was actually charged. An engine torn down
        # mid-load never paid, and refunding it would hand the next cell host RAM that never existed.
        if e.ram_charged:
            n.host_ram_used = max(0.0, n.host_ram_used - e.ram_charged)
            e.ram_charged = 0.0
        n.graveyard[wid] = {"worker": wid, "model": e.model, "reason": "reset by the grader",
                            "exit_code": None, "log_tail": e.log}
        e.state = "gone"
        pending = list(e.transitions)
        for task in pending:
            task.cancel()
        await asyncio.gather(*pending, return_exceptions=True)
        if e.load_task:
            e.load_task.cancel()
            await asyncio.gather(e.load_task, return_exceptions=True)
        if e.runner:
            await e.runner.cleanup()
    n.graveyard.clear()
    n.host_ram_used = 0.0
    return web.json_response({"node": n.name, "torn_down": len(wids), "orphans_killed": 0,
                              "engines_remaining": len(n.engines)})


async def handle_mgr_rekey(request: web.Request) -> web.Response:
    """Same call the real manager answers. Its workers are in-process here, so they sign with this."""
    global KEY
    if not GRADER_TOKEN or request.headers.get("X-Grader-Token") != GRADER_TOKEN:
        return _err(403, "grader token required")
    new = (await request.json()).get("key") or ""
    if new:
        KEY = new.encode()
    return web.json_response({"status": "rekeyed", "node": node(request).name,
                              "workers_still_on_the_old_key": []})


def build(n: SimNode, catalogue: dict) -> web.Application:
    app = web.Application(client_max_size=64 * 1024 * 1024,
                          middlewares=[manager_auth.middleware(GRADER_TOKEN)])
    app["node"] = n
    app["catalogue"] = {m["id"]: m for m in catalogue.get("models", [])}
    app["runners"] = []
    async def timing(app):
        async with looplag.Meter(TIME_SCALE) as meter:
            n.timing = meter
            if n.fabric:
                await n.fabric.start()
            yield
        for e in list(n.engines.values()):
            await _stop_prefill(e)
            if e.load_task:
                e.load_task.cancel()
                await asyncio.gather(e.load_task, return_exceptions=True)
            if e.runner:
                await e.runner.cleanup()
        if n.fabric:
            await n.fabric.close()
    app.cleanup_ctx.append(timing)
    async def engines(request):
        return web.json_response([{"worker": e.worker_id, "state": e.state}
                                  for e in n.engines.values()])
    async def fabric(request):
        if n.fabric_ledger is None:
            return _err(404, 'this manager is not the simulated fabric coordinator')
        try:
            return web.json_response(await sim_fabric.apply(n.fabric_ledger, await request.json()))
        except (ValueError, KeyError) as exc:
            return _err(409, str(exc))
    app.add_routes([
        web.post("/engines", handle_launch),
        web.post('/sim/fabric', fabric),
        web.get("/engines", engines),
        web.get("/engines/{wid}/ready", handle_ready),
        web.post("/engines/{wid}/settled", handle_settled),
        web.post("/engines/{wid}/worker", handle_start_worker),
        web.get("/engines/{wid}/placement", handle_placement),
        web.post("/engines/{wid}/rank/{what}", handle_rank_state),
        web.delete("/engines/{wid}", handle_teardown),
        web.get("/stats", handle_stats),
        web.get("/logs", handle_logs),
        web.post("/admin/rekey", handle_mgr_rekey),
        web.post("/admin/reset", handle_mgr_reset),
        web.get("/health", lambda r: web.json_response({"status": "ok", "node": node(r).name, "run_id": os.environ.get("CLUSTER_RUN_ID", "")})),
    ])
    return app


async def main_async(fleet: dict, catalogue: dict, only: str | None) -> None:
    runners = []
    first = fleet['nodes'][0]
    coordinator = sim_fabric.Fabric() if not only or only == first['name'] else None
    fabric_url = f"http://{first.get('host', '127.0.0.1')}:{first['port']}"
    for nd in fleet["nodes"]:
        if only and nd["name"] != only:
            continue
        n = SimNode(name=nd["name"], host=nd.get("host", "127.0.0.1"), port=int(nd["port"]),
                    n_gpus=int(nd.get("n_gpus") or fleet.get("gpus_per_node") or 8),
                    host_ram_gib=float(nd.get("host_ram_gib") or 600.0),
                    speed=float(nd.get("speed") or 1.0))
        n.fabric_ledger = coordinator if nd['name'] == first['name'] else None
        n.fabric = sim_fabric.Client(n, fabric_url, GRADER_TOKEN, TICK_S, local=coordinator)
        runner = web.AppRunner(build(n, catalogue), access_log=None, handler_cancellation=True)
        await runner.setup()
        await web.TCPSite(runner, "0.0.0.0", n.port).start()
        runners.append(runner)
        print(f"[sim] manager {n.name} on :{n.port}, {n.n_gpus} GPU(s)", flush=True)
    print(f"[sim] {len(runners)} manager(s) up", flush=True)
    await asyncio.Event().wait()


def supervise_nodes(fleet: dict, fleet_path: str, catalogue_path: str) -> int:
    """Give each logical node its own event loop, including in one container.

    Only this supervisor's children are signalled. Losing a manager fails the
    fleet instead of silently continuing with fewer simulated resources.
    """
    children = []
    stopping = False

    def stop(signum, frame):
        nonlocal stopping
        stopping = True

    previous = {sig: signal.signal(sig, stop) for sig in (signal.SIGINT, signal.SIGTERM)}
    try:
        for node in fleet['nodes']:
            if stopping:
                break
            child = subprocess.Popen([sys.executable, os.path.abspath(__file__),
                '--fleet', fleet_path, '--catalogue', catalogue_path, '--only-node', node['name']])
            children.append((node['name'], child))
        while not stopping:
            for name, child in children:
                code = child.poll()
                if code is not None:
                    print(f'[sim] manager {name} exited ({code}); stopping fleet', file=sys.stderr, flush=True)
                    return code if code > 0 else 1
            time.sleep(.1)
        return 0
    finally:
        for _, child in children:
            if child.poll() is None:
                child.terminate()
        deadline = time.monotonic()+5
        for _, child in children:
            try:
                child.wait(timeout=max(0., deadline-time.monotonic()))
            except subprocess.TimeoutExpired:
                child.kill()
                child.wait()
        for sig, handler in previous.items():
            signal.signal(sig, handler)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fleet", default=os.environ.get("CLUSTER_NODES", "/app/fleet.json"))
    ap.add_argument("--catalogue", default=os.environ.get("CLUSTER_CATALOGUE", "/app/catalogue.json"))
    ap.add_argument("--only-node", default=os.environ.get("CLUSTER_ONLY_NODE") or None,
                    help="serve just this node; the launcher runs one process per machine")
    ap.add_argument('--single-process', action='store_true',
                    help='diagnostic only: share one event loop across every logical node')
    a = ap.parse_args()
    fleet = json.load(open(a.fleet))
    names = [node['name'] for node in fleet['nodes']]
    if not names or len(set(names)) != len(names):
        ap.error('fleet requires nonempty, distinct node names')
    if a.only_node and a.only_node not in names:
        ap.error('--only-node is absent from the fleet')
    if not a.only_node and not a.single_process and len(names) > 1:
        return supervise_nodes(fleet, a.fleet, a.catalogue)
    asyncio.run(main_async(fleet, json.load(open(a.catalogue)), a.only_node),
                loop_factory=uvloop.new_event_loop)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
