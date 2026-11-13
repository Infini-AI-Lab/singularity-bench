"""One per NODE. The only thing that can launch, kill, watch or measure on this machine.

`nvidia-smi` is per-machine, a multi-node engine has processes on every participating node, and those
processes are children of this one. No per-engine or per-fleet process can reach across a machine
boundary, so each node runs its own manager (`cluster_launch.py` starts one per machine) and the
controller sits above them.

WHAT IT OWNS: process identity. The controller addresses everything by the worker id it minted and never
sees a pid or a pgid, because those are meaningless off this machine and go stale on any restart. The map
from a worker id to the process group serving it lives here and nowhere else.

WHAT IT DOES NOT DO: model memory. There is no `util`, no headroom arithmetic and no 507 computed from
bookkeeping. The plane's plan plays out physically; an over-packed node gets the OOM killer, and the
watchdog below turns that into a `crashed` state with an exit code and a log.
"""
from __future__ import annotations

import argparse
import asyncio
import contextlib
import json
import os
import secrets
import signal
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import aiohttp
from aiohttp import web

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import nodeinfo  # noqa: E402
import engine_args  # noqa: E402
import checkpoints  # noqa: E402
import runtime_compat  # noqa: E402
import manager_auth  # noqa: E402

WEIGHTS = os.environ.get("CLUSTER_WEIGHTS", "/weights")
VLLM_PORT_BASE = int(os.environ.get("CLUSTER_VLLM_PORT_BASE", "21000"))
# MUST stay inside the worker port window (9101-9299) that the agent's and the grader's firewalls admit.
# The plane reaches a worker at the url `/deploy` returns, so a port outside it is unreachable.
WORKER_PORT_LO = int(os.environ.get("CLUSTER_WORKER_PORT_LO", "9101"))
WORKER_PORT_HI = int(os.environ.get("CLUSTER_WORKER_PORT_HI", "9299"))
MAX_MODEL_LEN = int(os.environ.get("CLUSTER_MAX_MODEL_LEN", "32768"))
READY_TIMEOUT_S = float(os.environ.get("CLUSTER_READY_TIMEOUT_S", "1200"))
LOG_DIR = os.environ.get("CLUSTER_LOG_DIR", "/tmp/fleet-logs")
MULTINODE_RDZV_PORT = int(os.environ.get("CLUSTER_RDZV_PORT", "29500"))
GRAVE_KEEP = 64
# THE SIGNING KEY LIVES HERE, not in the workers, because workers do not exist when the grader rotates
# it: they are created by a deploy, which happens inside the cell. The manager holds the current key and
# passes it into each worker's environment at launch, so a rotation before the setup window reaches
# every worker the plane goes on to create.
RECEIPT_KEY = os.environ.get("CLUSTER_RECEIPT_KEY", "insecure-dev-key")
GRADER_TOKEN = os.environ.get("CLUSTER_GRADER_TOKEN", "")


def _err(status: int, message: str, **extra) -> web.Response:
    return web.json_response({"error": message, **extra}, status=status)


@dataclass
class Engine:
    """One vLLM rank on THIS node, plus the worker process if this node is the head."""
    worker_id: str
    model: str
    port: int                                   # vLLM's own port, loopback-only
    is_head: bool
    log_path: str = ""
    proc: object = None
    pgid: int | None = None
    worker_proc: object = None
    worker_port: int = 0
    worker_lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    backend_key: str = field(default_factory=lambda: secrets.token_hex(32))
    audit_model: dict = field(default_factory=dict)
    # Kept because the WORKER needs them: whether the plane asked for --enable-sleep-mode decides
    # whether /admin/sleep is a real operation or a 409, and the worker is a separate process.
    extra_args: tuple = ()
    state: str = "launching"                    # launching | up | crashed | gone
    exit_code: int | None = None
    killing: bool = False
    watcher: object = None
    master_port: int = 0
    settled: asyncio.Event = field(default_factory=asyncio.Event)
    startup_task: object = None


@dataclass
class Manager:
    node: str
    host: str
    port: int
    engines: dict = field(default_factory=dict)          # worker_id -> Engine
    graveyard: dict = field(default_factory=dict)        # worker_id -> post-mortem
    used_ports: set = field(default_factory=set)
    next_port: int = 0
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    # ONE LOCK PER GPU, held across an engine's whole start-up, not just its spawn.
    #
    # vLLM snapshots free GPU memory when the engine core starts and profiles against that snapshot 30-90
    # seconds later. If free memory has GONE UP in between -- because something else on the card released
    # its own -- it refuses to continue:
    #
    #   AssertionError: Error in memory profiling. Initial free memory 71.02 GiB,
    #   current free memory 122.51 GiB. This happens when other processes sharing the same container
    #   release GPU memory
    #
    # Which is exactly what this task asks a plane to do all day: give a card back and put something else
    # on it. `lock` above only spans port allocation and the spawn, so start-ups on a card are serialised
    # here instead, as the emulator does.
    gpu_lock: dict = field(default_factory=dict)
    generation: str = ""
    catalogue: dict = field(default_factory=dict)
    used_master_ports: set = field(default_factory=set)
    used_vllm_ports: set = field(default_factory=set)


def mgr(request: web.Request) -> Manager:
    return request.app["manager"]


# ---------------------------------------------------------------- process control

def _log_tail(path: str, n: int = 40) -> list[str]:
    if not path or not os.path.exists(path):
        return []
    try:
        with open(path, "rb") as f:
            return f.read()[-64_000:].decode(errors="replace").splitlines()[-n:]
    except OSError:
        return []


def _free_port(taken: set[int] | None = None, lo: int = WORKER_PORT_LO,
               hi: int = WORKER_PORT_HI) -> int:
    """A free port INSIDE THE REACHABLE WINDOW.

    Not `bind(port 0)`. The OS would hand back something from the ephemeral range, and the agent's and
    the grader's firewalls only admit WORKER_PORT_LO..HI on a node address -- so a worker on an OS-assigned
    port would be at an address the plane could not talk to. Allocate inside the window and probe. Each
    node is its own machine, so ~200 ports per node is far more than the engines one machine can hold.
    """
    import socket
    taken = taken or set()
    for port in range(lo, hi + 1):
        # Skip what THIS manager has already handed out. The probe releases the socket before returning,
        # so two deploys racing through the fan-out would otherwise both be told the same port is free.
        if port in taken:
            continue
        with socket.socket() as sk:
            sk.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            try:
                sk.bind(("0.0.0.0", port))
            except OSError:
                continue
        return port
    raise RuntimeError(f"no free port in {lo}-{hi}")


def _entomb(m: Manager, e: Engine, why: str) -> None:
    """Keep why an engine is gone, after the engine is gone.

    `unload` drops the record; the post-mortem stays here so a control plane can clean up first and ask
    why the engine died afterwards.
    """
    m.graveyard[e.worker_id] = {"worker": e.worker_id, "model": e.model, "reason": why,
                                "exit_code": e.exit_code, "log_path": e.log_path,
                                "log_tail": _log_tail(e.log_path)}
    while len(m.graveyard) > GRAVE_KEEP:
        m.graveyard.pop(next(iter(m.graveyard)))


async def _kill_proc(proc, grace: float = 30.0) -> None:
    """SIGTERM the whole process group, then SIGKILL what is left.

    The group, not the pid: vLLM's CUDA-holding processes are its children, and signalling only the
    launcher leaves them holding the card. Safe because every launch below uses `start_new_session=True`,
    which puts each engine in its own group -- without that this would signal our own group.
    """
    if proc is None:
        return
    with contextlib.suppress(ProcessLookupError, PermissionError):
        os.killpg(proc.pid, signal.SIGTERM)
    with contextlib.suppress(asyncio.TimeoutError):
        await asyncio.wait_for(proc.wait(), timeout=grace)
    # Children can outlive an already-reaped launcher. Every launch starts a new
    # session, so its original PID is also the process group we own.
    with contextlib.suppress(ProcessLookupError, PermissionError):
        os.killpg(proc.pid, signal.SIGKILL)
    await proc.wait()


async def _watch(m: Manager, e: Engine) -> None:
    """Notice when an engine dies on its own, and leave the plane something to act on.

    vLLM runs as a CHILD, so its death never takes this process down and requests simply come back 502.
    Awaiting the child after startup reaps it, marks the engine `crashed` with its exit code, and keeps
    its log tail for `/logs`.
    """
    proc = e.proc
    if proc is None:
        return
    rc = await proc.wait()
    if e.killing or e.state == "gone":
        return                                   # we killed it; the caller does its own accounting
    await _kill_proc(proc, grace=0.0)
    e.state, e.exit_code = "crashed", rc
    _entomb(m, e, f"the engine exited on its own with code {rc}")
    print(f"[{m.node}] {e.worker_id} ({e.model}) CRASHED, exit {rc}; see {e.log_path}", flush=True)


async def _vllm(sess, port: int, path: str, method: str = "POST", api_key: str = "") -> tuple[int, dict]:
    try:
        async with sess.request(method, f"http://127.0.0.1:{port}{path}",
                                headers={"Authorization": f"Bearer {api_key}"} if api_key else {},
                                timeout=aiohttp.ClientTimeout(total=2)) as r:
            body = await r.text()
            return r.status, (json.loads(body) if body.strip().startswith("{") else {"raw": body[:200]})
    except (aiohttp.ClientError, asyncio.TimeoutError) as exc:
        return 0, {"error": f"{type(exc).__name__}: {exc}"}


async def _backend_ready(sess, e: Engine) -> bool:
    # A free-port probe is not a reservation: another process can bind before
    # vLLM starts. /health alone could accept that unrelated listener. The
    # private API key plus the deployment-specific model alias identify ours.
    status, body = await _vllm(sess, e.port, "/v1/models", "GET", e.backend_key)
    ids = {row.get("id") for row in body.get("data", []) if isinstance(row, dict)}
    return status == 200 and {e.model, e.worker_id} <= ids


# ---------------------------------------------------------------- launching

def _vllm_cmd(e: Engine, body: dict) -> list[str]:
    """The launch command, matching vLLM's documented multi-node contract.

    docs.vllm.ai/en/stable/serving/parallelism_scaling: the head takes `--nnodes N --node-rank 0
    --master-addr <HEAD_IP>`, and every other node takes the same plus `--headless`, because only rank 0
    runs an API server.
    """
    nnodes = int(body.get("nnodes") or 1)
    extra = engine_args.sanitise(body.get("engine_args"))
    # Explicit defaults keep the development scheduler and hybrid-cache sizing
    # independent of vLLM's hardware-dependent default token budget.
    if '--max-num-batched-tokens' not in extra:
        extra += ['--max-num-batched-tokens', '4096']
    if '--max-num-seqs' not in extra:
        extra += ['--max-num-seqs', '256']
    cmd = [sys.executable, "-m", "vllm.entrypoints.cli.main", "serve",
           # local_dir, not the HF id: staging flattens `Org/Name` to `name`, and HF_HUB_OFFLINE=1 turns
           # a wrong path into "not found" rather than a quiet 60 GiB download.
           os.path.join(WEIGHTS, body["checkpoint"]),
           "--served-model-name", e.model, e.worker_id,
           "--port", str(e.port),
           "--host", "127.0.0.1",
           "--dtype", "bfloat16",
           "--max-model-len", str(int(body.get("context_len") or MAX_MODEL_LEN)),
           "--tensor-parallel-size", str(int(body.get("tp") or 1)),
           "--pipeline-parallel-size", str(int(body.get("pp") or 1))]
    if nnodes > 1:
        cmd += ["--nnodes", str(nnodes),
                "--node-rank", str(int(body.get("node_rank") or 0)),
                "--master-addr", str(body["master_addr"]),
                "--master-port", str(body["master_port"]),
                "--distributed-executor-backend", "mp"]
        if not e.is_head:
            cmd.append("--headless")
    # --gpu-memory-utilization and --enable-sleep-mode are the PLANE's to choose and arrive in
    # engine_args. VLLM_SERVER_DEV_MODE stays ours: it is an environment variable, not a flag.
    # Platform-owned instrumentation is identical for every engine.
    # The native KV publisher binds an IPC socket; only its read-only wrapper is public.
    telemetry_args = ['--scheduler-cls', 'telemetry_scheduler.TelemetryScheduler',
                      '--kv-events-config', json.dumps({'enable_kv_cache_events': True,
                          'publisher': 'zmq', 'endpoint': f'ipc://{e.log_path}.kv',
                          'hwm': 20000, 'max_queue_size': 20000})]
    return cmd + runtime_compat.engine_flags(body.get("checkpoint_id", "")) + extra + telemetry_args + ["--no-enable-log-requests"]


async def _hold_cards(m: "Manager", cards: list) -> list:
    """Take every card this engine will sit on, lowest index first so two launches cannot deadlock."""
    locks = [m.gpu_lock.setdefault(int(c), asyncio.Lock()) for c in sorted(set(cards))]
    acquired = []
    try:
        for lk in locks:
            await lk.acquire()
            acquired.append(lk)
        return acquired
    except BaseException:
        _drop_cards(acquired)
        raise


def _drop_cards(locks: list) -> None:
    for lk in locks:
        if lk.locked():
            lk.release()


async def _release_when_settled(m: "Manager", e: "Engine", locks: list, sess) -> None:
    """Hold the cards until this engine has finished profiling -- ready, dead, or out of patience.

    Dead counts: an engine that crashes on its own is releasing GPU memory asynchronously, and a launch
    that overlapped it would fail the assertion above through no fault of the plane. Holding until it is
    reaped makes the next launch see a settled card.
    """
    end = time.monotonic() + READY_TIMEOUT_S
    try:
        while time.monotonic() < end:
            if e.settled.is_set() or e.state == "gone":
                return
            if e.proc is not None and e.proc.returncode is not None:
                return
            if e.is_head and await _backend_ready(sess, e):
                return
            await asyncio.sleep(2.0)
    finally:
        _drop_cards(locks)


async def handle_launch(request: web.Request) -> web.Response:
    """Start one rank on this node. Returns as soon as the process exists, with its pgid.

    It does NOT wait for readiness. Every rank blocks at the rendezvous until its peers arrive, so a
    manager that waited here would make the controller's fan-out serial and deadlock the group on the
    first node. Readiness is a separate call, and only against the head.
    """
    m = mgr(request)
    body = await request.json()
    generation = body.get("generation", "")
    if generation != m.generation:
        return _err(409, "deployment belongs to a retired fleet generation")
    extra, _, problems = engine_args.validate(body.get("engine_args"))
    if problems:
        return _err(400, "; ".join(problems))
    spec = m.catalogue.get(body.get("model"))
    if spec is None:
        return _err(404, "model is not in the manager catalogue")
    if body.get("checkpoint") != (spec.get("local_dir") or spec["checkpoint"]):
        return _err(400, "checkpoint does not match the catalogue model")
    body["engine_args"] = extra
    body["checkpoint_id"] = spec["checkpoint"]
    body["context_len"] = spec["context_len"]
    wid = body["worker_id"]
    if wid in m.engines:
        return _err(409, f"{wid} is already deployed on {m.node}")

    # Take the cards BEFORE the node lock, and keep them until this engine has finished profiling. A
    # launch that has to wait here is waiting for a card to settle, which is the whole point; the node
    # lock below is only about handing out a port and must not be held across that wait.
    cards = [int(g) for g in (body.get("gpus") or [])]
    held = await _hold_cards(m, cards) if cards else []
    async with m.lock:
        if generation != m.generation:
            _drop_cards(held)
            return _err(409, "deployment was cancelled by a fleet reset")
        try:
            locked = checkpoints.entry(spec["checkpoint"])
            if spec.get("checkpoint_revision", locked["revision"]) != locked["revision"]:
                raise ValueError("catalogue revision disagrees with checkpoint lock")
            if not checkpoints.verify(Path(WEIGHTS) / body["checkpoint"], spec["checkpoint"], locked):
                raise ValueError("checkpoint is missing, changed, or not verified against pinned content")
            runtime_compat.needs_config_code(spec["checkpoint"], Path(WEIGHTS)/body["checkpoint"])
        except (OSError, ValueError) as exc:
            _drop_cards(held)
            return _err(503, str(exc))
        try:
            port = _free_port(m.used_vllm_ports, VLLM_PORT_BASE, VLLM_PORT_BASE + 899)
        except RuntimeError as exc:
            _drop_cards(held)
            return _err(503, str(exc))
        m.used_vllm_ports.add(port)
        e = Engine(worker_id=wid, model=body["model"], port=port,
                   is_head=bool(body.get("is_head", True)),
                   audit_model={"checkpoint": spec["checkpoint"],
                                "checkpoint_revision": locked["revision"],
                                "checkpoint_lock_sha256": checkpoints.digest(locked),
                                "tokenizer_checkpoint": spec["checkpoint"],
                                "tokenizer_revision": locked["revision"],
                                "fleet_generation": generation},
                   extra_args=tuple(body.get("engine_args") or ()))
        if int(body.get("nnodes") or 1) > 1 and e.is_head:
            try:
                e.master_port = _free_port(m.used_master_ports, 29501, 39499)
            except RuntimeError as exc:
                m.used_vllm_ports.discard(port)
                _drop_cards(held)
                return _err(503, str(exc))
            m.used_master_ports.add(e.master_port)
            body["master_port"] = e.master_port
        os.makedirs(LOG_DIR, exist_ok=True)
        e.log_path = os.path.join(LOG_DIR, f"vllm-{m.node}-{wid}.log")
        env = os.environ | {
            "VLLM_SERVER_DEV_MODE": "1",       # what exposes /sleep, /wake_up and /is_sleeping
            "VLLM_API_KEY": e.backend_key,
            # Longer than aiohttp's 15 s idle pool lifetime. Burst traffic can
            # otherwise race reuse against the backend's default 5 s close.
            "VLLM_HTTP_TIMEOUT_KEEP_ALIVE": "120",
            "HF_HUB_OFFLINE": "1",
            "VLLM_LOGGING_LEVEL": "WARNING",
            "DYN_FORWARDPASS_METRIC_PORT": str(e.port + 10000),
            "DYN_FPM_WORKER_ID": e.worker_id,
            "CLUSTER_CAPABILITIES_FILE": e.log_path + '.capabilities.json',
            "PYTHONPATH": os.path.dirname(os.path.abspath(__file__)) + os.pathsep + os.environ.get('PYTHONPATH', ''),
        }
        # A retired deployment's metadata must never certify a new process.
        with contextlib.suppress(FileNotFoundError):
            os.unlink(e.log_path + '.capabilities.json')
        gpus = body.get("gpus")
        # Exactly CUDA_VISIBLE_DEVICES. Absent means "set nothing", so every card is visible and the
        # engine args alone decide how many get used.
        if gpus:
            env["CUDA_VISIBLE_DEVICES"] = ",".join(str(g) for g in gpus)
        log = open(e.log_path, "wb")
        try:
            e.proc = await asyncio.create_subprocess_exec(
                *_vllm_cmd(e, body), env=env, stdout=log, stderr=asyncio.subprocess.STDOUT,
                start_new_session=True)          # its own group, so _kill_proc cannot hit us
        except OSError as exc:
            _drop_cards(held)
            m.used_master_ports.discard(e.master_port)
            m.used_vllm_ports.discard(e.port)
            return _err(500, f"could not start vLLM for {wid}: {exc}")
        finally:
            log.close()
        e.pgid = e.proc.pid
        e.watcher = asyncio.create_task(_watch(m, e))
        m.engines[wid] = e
    if held:
        # Released from a background task, not here: this handler must return as soon as the process
        # exists (see the docstring -- a manager that waited would deadlock a multi-node rendezvous),
        # while the cards have to stay held through the profiling window that comes after.
        e.startup_task = asyncio.create_task(_release_when_settled(m, e, held, request.app["session"]))
    return web.json_response({"worker": wid, "node": m.node, "pgid": e.pgid,
                              "master_port": e.master_port,
                              "vllm_port": e.port, "state": e.state})


async def handle_settled(request: web.Request) -> web.Response:
    e = mgr(request).engines.get(request.match_info["wid"])
    if e is None:
        return _err(404, "engine is gone")
    if e.killing or e.state in {"gone", "crashed"} or (e.proc is not None and e.proc.returncode is not None):
        return _err(503, "engine exited before the collective settled")
    # The authenticated controller calls this only after the head has verified
    # the whole collective. Headless peers have no HTTP readiness loop of their
    # own, so this is also their transition from launching to up.
    e.state = "up"
    e.settled.set()
    return web.json_response({"status": "settled"})


async def handle_ready(request: web.Request) -> web.Response:
    """Poll the HEAD's authenticated model list for this deployment's identity.

    Peers run `--headless` and serve no HTTP at all, so there is nothing to ask them. The head cannot
    answer 200 until the collective has formed and every rank has loaded, which is exactly the condition
    the controller needs before it measures placement.
    """
    m = mgr(request)
    wid = request.match_info["wid"]
    e = m.engines.get(wid)
    if e is None:
        return _err(404, f"{wid} is not on {m.node}")
    if not e.is_head:
        return _err(400, f"{wid} on {m.node} is a headless rank and serves no HTTP")
    timeout = float(request.query.get("timeout", READY_TIMEOUT_S))
    end = time.monotonic() + timeout
    sess = request.app["session"]
    while time.monotonic() < end:
        if e.proc is not None and e.proc.returncode is not None:
            return web.json_response({"ready": False, "reason": f"exited with {e.proc.returncode}",
                                      "log_tail": _log_tail(e.log_path)})
        if await _backend_ready(sess, e):
            e.state = "up"
            return web.json_response({"ready": True, "vllm_port": e.port})
        await asyncio.sleep(1.0)
    return web.json_response({"ready": False, "reason": f"not ready within {timeout:.0f}s",
                              "log_tail": _log_tail(e.log_path)})


async def handle_start_worker(request: web.Request) -> web.Response:
    """Start the serving process for an engine whose head is on this node.

    Its own process, not an app hosted here: a bug in one engine's request handling should not be able to
    take down every other engine on the machine.
    """
    m = mgr(request)
    wid = request.match_info["wid"]
    e = m.engines.get(wid)
    if e is None:
        return _err(404, f"{wid} is not on {m.node}")
    if not e.is_head:
        return _err(400, f"{wid} is headless on {m.node}; the worker belongs on the head node")
    async with e.worker_lock:
        if m.engines.get(wid) is not e or e.killing or e.state in {"gone", "crashed"}:
            return _err(409, "engine was retired during worker startup")
        already_running = e.worker_proc is not None and e.worker_proc.returncode is None
        log_path = os.path.join(LOG_DIR, f"worker-{m.node}-{wid}.log")
        if not already_running:
            body = await request.json() if request.can_read_body else {}
            e.worker_port = int(body.get("port") or _free_port(m.used_ports))
            m.used_ports.add(e.worker_port)
            env = os.environ | {"CLUSTER_WORKER_ID": wid, "CLUSTER_WORKER_MODEL": e.model,
                                "CLUSTER_WORKER_PORT": str(e.worker_port),
                                "CLUSTER_VLLM_PORT": str(e.port), "CLUSTER_NODE": m.node,
                                "CLUSTER_BACKEND_KEY": e.backend_key,
                                "CLUSTER_AUDIT_MODEL": json.dumps(e.audit_model),
                                "CLUSTER_ENGINE_ARGS": json.dumps(list(e.extra_args)),
                                "CLUSTER_RECEIPT_KEY": RECEIPT_KEY,
                                "CLUSTER_CAPABILITIES_FILE": e.log_path + '.capabilities.json',
                                "CLUSTER_KV_ENDPOINT": f'ipc://{e.log_path}.kv',
                                "CLUSTER_FPM_ENDPOINT": f'tcp://127.0.0.1:{e.port + 10000}'}
            with open(log_path, "wb") as log:
                e.worker_proc = await asyncio.create_subprocess_exec(
                    sys.executable, os.path.join(os.path.dirname(os.path.abspath(__file__)), "worker.py"),
                    env=env, stdout=log, stderr=asyncio.subprocess.STDOUT, start_new_session=True)
        # A spawned process is not yet an HTTP listener. The controller must
        # not advertise successful deployment until this exact worker answers.
        deadline = time.monotonic() + 30
        while time.monotonic() < deadline:
            if (m.engines.get(wid) is not e or e.killing or e.state in {"gone", "crashed"}
                    or e.worker_proc.returncode is not None):
                return _err(503, "worker exited or engine retired before readiness",
                            log_tail=_log_tail(log_path))
            try:
                async with request.app["session"].get(
                        f"http://127.0.0.1:{e.worker_port}/health",
                        timeout=aiohttp.ClientTimeout(total=1)) as response:
                    if response.status == 200 and (await response.json()).get("worker") == wid:
                        return web.json_response({"worker": wid, "url": f"http://{m.host}:{e.worker_port}",
                                                  "port": e.worker_port, "already_running": already_running})
            except (aiohttp.ClientError, asyncio.TimeoutError, ValueError):
                pass
            await asyncio.sleep(.05)
        return _err(503, "worker HTTP endpoint did not become ready within 30s",
                    log_tail=_log_tail(log_path))


# ---------------------------------------------------------------- measurement

async def handle_placement(request: web.Request) -> web.Response:
    """Which cards this engine ACTUALLY landed on, measured, never derived.

    Deriving it from `--tensor-parallel-size` and a `gpus` list would be a guess about vLLM's behaviour
    that breaks on the first argument the parser does not know -- and it is precisely wrong in the case
    worth catching, where two engines have silently stacked on the same card.
    """
    m = mgr(request)
    wid = request.match_info["wid"]
    e = m.engines.get(wid)
    if e is None:
        return _err(404, f"{wid} is not on {m.node}")
    if e.pgid is None:
        return web.json_response({"node": m.node, "gpus": [], "reason": "no process group"})
    return web.json_response({"node": m.node, "state": e.state, "gpus": nodeinfo.placement_of(e.pgid)})


async def handle_stats(request: web.Request) -> web.Response:
    """This machine, as the driver and the kernel report it."""
    m = mgr(request)
    return web.json_response({
        "node": m.node, "host": m.host,
        "host_ram": nodeinfo.host_ram(),
        "hardware": nodeinfo.hardware(),
        "gpus": nodeinfo.gpus(),
        "engines": [{"worker": e.worker_id, "model": e.model, "state": e.state,
                     "is_head": e.is_head, "exit_code": e.exit_code}
                    for e in m.engines.values()],
    })


async def handle_engines(request: web.Request) -> web.Response:
    return web.json_response([{"worker": e.worker_id,
        "state": "worker_crashed" if e.worker_proc is not None and e.worker_proc.returncode is not None else e.state}
        for e in mgr(request).engines.values()])


async def handle_logs(request: web.Request) -> web.Response:
    """An engine's own last words -- and they outlive the engine.

    Every startup failure leaves its only evidence here: a bad shard count, an NCCL problem, too little
    memory for the KV cache. The graveyard fallback is what makes it readable AFTER an unload, so a plane
    can tidy up first and diagnose second.
    """
    m = mgr(request)
    wid = request.query.get("worker", "")
    n = min(int(request.query.get("lines", "80") or 80), 500)
    e = m.engines.get(wid)
    if e is not None and e.log_path and os.path.exists(e.log_path):
        return web.json_response({"worker": wid, "node": m.node, "state": e.state,
                                  "exit_code": e.exit_code, "log_path": e.log_path,
                                  "lines": _log_tail(e.log_path, n)})
    if (grave := m.graveyard.get(wid)) is not None:
        return web.json_response({**grave, "node": m.node, "state": "gone"})
    return _err(404, f"no log for {wid!r} on {m.node}")


# ---------------------------------------------------------------- teardown

async def handle_teardown(request: web.Request) -> web.Response:
    """Stop serving, then kill the rank. IDEMPOTENT.

    Idempotent because cleaning up after a crash is exactly when the plane calls this, and a 404 there
    would leave it no way to reclaim the card.
    """
    m = mgr(request)
    wid = request.match_info["wid"]
    e = m.engines.pop(wid, None)
    if e is None:
        return web.json_response({"worker": wid, "node": m.node, "status": "already gone"})
    e.killing = True
    m.used_ports.discard(e.worker_port)
    _entomb(m, e, "unloaded by the control plane" if e.state != "crashed"
            else f"the engine exited on its own with code {e.exit_code}, then was unloaded")
    # The worker first: no new request should arrive at an engine that is about to die.
    await _kill_proc(e.worker_proc, grace=10.0)
    await _kill_proc(e.proc)
    if e.startup_task:
        e.startup_task.cancel()
        await asyncio.gather(e.startup_task, return_exceptions=True)
    m.used_master_ports.discard(e.master_port)
    m.used_vllm_ports.discard(e.port)
    e.state = "gone"
    return web.json_response({"worker": wid, "node": m.node, "status": "unloaded"})


async def handle_reset(request: web.Request) -> web.Response:
    """Reset only process groups owned by this manager; retire queued launches."""
    if not GRADER_TOKEN or request.headers.get("X-Grader-Token") != GRADER_TOKEN:
        return _err(403, "grader token required")
    m = mgr(request)
    async with m.lock:
        m.generation = (await request.json()).get("generation", "")
        wids = list(m.engines)
        for wid in wids:
            e = m.engines.pop(wid, None)
            if e is None:
                continue
            e.killing = True
            m.used_ports.discard(e.worker_port)
            _entomb(m, e, "reset by the grader between cells")
            await _kill_proc(e.worker_proc, grace=10.0)
            await _kill_proc(e.proc)
            if e.startup_task:
                e.startup_task.cancel()
                await asyncio.gather(e.startup_task, return_exceptions=True)
            m.used_master_ports.discard(e.master_port)
            m.used_vllm_ports.discard(e.port)
            e.state = "gone"
        orphans = 0
        m.graveyard.clear()
        print(f"[{m.node}] reset: tore down {len(wids)} engine(s), killed {orphans} orphan group(s)",
              flush=True)
        return web.json_response({"node": m.node, "torn_down": len(wids), "orphans_killed": orphans,
                                  "engines_remaining": len(m.engines)})


async def handle_rekey(request: web.Request) -> web.Response:
    """Retire the key the agent may have observed in development, before anything is graded."""
    global RECEIPT_KEY
    if not GRADER_TOKEN or request.headers.get("X-Grader-Token") != GRADER_TOKEN:
        return _err(403, "grader token required")
    new = (await request.json()).get("key") or ""
    if new:
        RECEIPT_KEY = new
    live = [e.worker_id for e in mgr(request).engines.values() if e.worker_proc is not None]
    # Workers already running keep the OLD key -- they were started with it in their environment. The
    # grader rotates before the setup window, when nothing is deployed, so this should always be empty;
    # reporting it means a rotation that came too late is visible rather than silent.
    return web.json_response({"status": "rekeyed", "node": mgr(request).node,
                              "workers_still_on_the_old_key": live})


def build(m: Manager) -> web.Application:
    app = web.Application(client_max_size=64 * 1024 * 1024,
                          middlewares=[manager_auth.middleware(GRADER_TOKEN)])
    app["manager"] = m
    app.add_routes([
        web.post("/engines", handle_launch),
        web.get("/engines", handle_engines),
        web.get("/engines/{wid}/ready", handle_ready),
        web.post("/engines/{wid}/settled", handle_settled),
        web.post("/engines/{wid}/worker", handle_start_worker),
        web.get("/engines/{wid}/placement", handle_placement),
        web.delete("/engines/{wid}", handle_teardown),
        web.get("/stats", handle_stats),
        web.get("/logs", handle_logs),
        web.post("/admin/rekey", handle_rekey),
        web.post("/admin/reset", handle_reset),
        web.get("/health", lambda r: web.json_response({"status": "ok", "node": mgr(r).node, "run_id": os.environ.get("CLUSTER_RUN_ID", "")})),
    ])

    async def _session(a):
        a["session"] = aiohttp.ClientSession()
        yield
        await a["session"].close()
    app.cleanup_ctx.append(_session)
    return app


async def main_async(node: str, host: str, port: int, catalogue: dict) -> None:
    if not GRADER_TOKEN:
        raise RuntimeError("CLUSTER_GRADER_TOKEN is required for manager RPCs")
    m = Manager(node=node, host=host, port=port,
                catalogue={x["id"]: x for x in catalogue["models"]})
    runner = web.AppRunner(build(m), access_log=None)
    await runner.setup()
    await web.TCPSite(runner, "0.0.0.0", port).start()
    print(f"[{node}] manager on :{port}, {len(nodeinfo.gpus())} GPU(s)", flush=True)
    await asyncio.Event().wait()


def main() -> None:
    # Same CLI as `simmanager.py`, so `entrypoint.sh` and `cluster_launch.py` do not have to know which
    # backend they started. The node this process serves comes from the fleet file, selected by name.
    ap = argparse.ArgumentParser()
    ap.add_argument("--fleet", default=os.environ.get("CLUSTER_NODES", "/app/fleet.json"))
    ap.add_argument("--catalogue", default=os.environ.get("CLUSTER_CATALOGUE", "/app/catalogue.json"))
    ap.add_argument("--only-node", default=os.environ.get("CLUSTER_ONLY_NODE") or None)
    a = ap.parse_args()
    fleet = json.load(open(a.fleet))
    name = a.only_node or fleet["nodes"][0]["name"]
    nd = next((n for n in fleet["nodes"] if n["name"] == name), None)
    if nd is None:
        raise SystemExit(f"[manager] {name!r} is not in {a.fleet}")
    asyncio.run(main_async(nd["name"], nd.get("host", "127.0.0.1"), int(nd["port"]),
                           json.load(open(a.catalogue))))


if __name__ == "__main__":
    main()
