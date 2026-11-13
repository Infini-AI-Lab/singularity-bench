"""What a machine actually has, measured -- not modelled.

The fleet keeps no books of its own (no `util` number per engine, no flat charge per resident engine):
the plane's deployment plan plays out physically, so the answer to "what is on this card" is the one
the driver gives.

Two queries, both on the node itself, because `nvidia-smi` is per-machine and has no remote mode.
"""
from __future__ import annotations

import os
import subprocess
import platform
import hashlib
import json
from pathlib import Path
from importlib.metadata import version, PackageNotFoundError
from functools import lru_cache

SMI = os.environ.get("CLUSTER_NVIDIA_SMI", "nvidia-smi")


@lru_cache(maxsize=1)
def runtime() -> dict:
    packages = {}
    for name in ('vllm', 'torch', 'aiohttp', 'ai-dynamo', 'ai-dynamo-runtime', 'msgspec', 'pyzmq'):
        try:
            packages[name] = version(name)
        except PackageNotFoundError:
            packages[name] = None
    root = Path(__file__).parent
    files = sorted(root.glob('*.py')) + sorted(root.glob('*.json'))
    source = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
    entrypoint = root.parent/'entrypoint.sh'
    if entrypoint.is_file():
        source['../entrypoint.sh'] = hashlib.sha256(entrypoint.read_bytes()).hexdigest()
    fingerprint = hashlib.sha256(json.dumps(source,sort_keys=True).encode()).hexdigest()
    return {'architecture': platform.machine(), 'kernel': platform.release(),
            'packages': packages, 'serving_code_sha256': fingerprint}


def hardware() -> dict:
    """Stable hardware and runtime identity; re-read driver/power limits to detect changes."""
    fields = ['uuid', 'name', 'driver_version', 'power.limit', 'clocks.max.sm', 'clocks.max.memory']
    rows = _smi('gpu='+','.join(fields))
    return runtime() | {'devices': [dict(zip(fields, row)) for row in rows if len(row) == len(fields)]}


def _smi(query: str, extra: list[str] | None = None) -> list[list[str]]:
    try:
        out = subprocess.run([SMI, f"--query-{query}", "--format=csv,noheader,nounits",
                              *(extra or [])],
                             capture_output=True, text=True, timeout=30)
    except (OSError, subprocess.SubprocessError):
        return []
    if out.returncode != 0:
        return []
    return [[c.strip() for c in line.split(",")]
            for line in out.stdout.splitlines() if line.strip()]


def gpus() -> list[dict]:
    """Every GPU on this machine, with live memory."""
    rows = _smi("gpu=index,uuid,memory.total,memory.used")
    out = []
    for r in rows:
        if len(r) < 4:
            continue
        try:
            out.append({"index": int(r[0]), "uuid": r[1],
                        "total_mib": int(float(r[2])), "used_mib": int(float(r[3]))})
        except ValueError:
            continue
    return out


def compute_apps() -> list[dict]:
    """Every process holding a CUDA context, and which card it is on.

    Needs the container to run with `--pid=host`. The driver reports HOST pids and has no notion of a pid
    namespace, so from a container with its own namespace this returns an EMPTY list -- silence, not an
    error, which reads as "no engine is using any GPU". `cluster_launch.py` passes the flag; this is the
    reason.
    """
    by_uuid = {g["uuid"]: g["index"] for g in gpus()}
    out = []
    for r in _smi("compute-apps=pid,used_gpu_memory,gpu_uuid"):
        if len(r) < 3:
            continue
        try:
            pid, used = int(r[0]), int(float(r[1]))
        except ValueError:
            continue
        idx = by_uuid.get(r[2])
        if idx is None:
            continue
        out.append({"pid": pid, "used_mib": used, "index": idx})
    return out


def pgid_of(pid: int) -> int | None:
    """The process group a pid belongs to, from /proc.

    Placement is matched by process GROUP, never by the pid we launched: the processes that hold CUDA
    contexts are vLLM's children, not the launcher. The CUDA holder is `VLLM::EngineCore` with a
    different ppid and a pgid inherited from further up -- so a pid match finds nothing, and a pgid match
    finds the whole engine.
    """
    try:
        with open(f"/proc/{pid}/stat", "rb") as f:
            data = f.read()
    except OSError:
        return None
    # comm can contain spaces and parentheses, so split after the LAST ')': fields are then
    # state ppid pgrp ... with pgrp third.
    try:
        rest = data[data.rindex(b")") + 2:].split()
        return int(rest[2])
    except (ValueError, IndexError):
        return None


def placement_of(pgid: int) -> list[dict]:
    """Which cards this engine is ACTUALLY on, and how much of each it holds."""
    by_index: dict[int, int] = {}
    for app in compute_apps():
        if pgid_of(app["pid"]) != pgid:
            continue
        by_index[app["index"]] = by_index.get(app["index"], 0) + app["used_mib"]
    return [{"index": i, "used_mib": m} for i, m in sorted(by_index.items())]


def _cgroup_ram() -> dict | None:
    """The container's OWN memory limit and usage, when it has one.

    This is the number that actually binds: the fleet's nodes run under a cgroup sized to the published
    budget, so the kernel enforces it and every submission meets the same limit on any machine. Reading
    /proc/meminfo instead reports the WHOLE HOST -- on a shared node that includes other tenants' memory,
    and it is not the limit anything is enforced against.

    `memory.current` includes reclaimable page cache, so it overstates what is actually held; subtract
    the file-backed part, which the kernel gives back under pressure rather than OOM-killing (a
    checkpoint streamed under a smaller cap is reclaimed, not fatal).
    """
    try:
        with open("/sys/fs/cgroup/memory.max") as f:
            raw = f.read().strip()
        if raw == "max":
            return None
        total = int(raw) / 1073741824.0
        with open("/sys/fs/cgroup/memory.current") as f:
            cur = int(f.read().strip()) / 1073741824.0
        # `file` INCLUDES `shmem`, and shmem is where vLLM keeps the weights, so subtracting `file`
        # wholesale would hide them. Only the non-shmem part of `file` is page cache the kernel can hand
        # back; shmem cannot be reclaimed with swap off, so it counts as held.
        st = {}
        with open("/sys/fs/cgroup/memory.stat") as f:
            for line in f:
                k, _, v = line.partition(" ")
                st[k] = int(v)
        cache = max(0, st.get("file", 0) - st.get("shmem", 0)) / 1073741824.0
    except (OSError, ValueError):
        return None
    used = max(0.0, cur - cache)
    return {"total_gib": round(total, 2), "used_gib": round(used, 2),
            "free_gib": round(max(0.0, total - used), 2), "source": "cgroup"}


def host_ram() -> dict:
    """Total and available host RAM, in GiB: the enforced cgroup budget if there is one, else the host."""
    cg = _cgroup_ram()
    if cg is not None:
        return cg
    vals: dict[str, int] = {}
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                k, _, v = line.partition(":")
                parts = v.split()
                if parts:
                    try:
                        vals[k] = int(parts[0])            # kB
                    except ValueError:
                        pass
    except OSError:
        return {"total_gib": 0.0, "used_gib": 0.0, "free_gib": 0.0}
    total = vals.get("MemTotal", 0) / 1048576.0
    # MemAvailable, not MemFree: page cache is reclaimable, and MemFree makes a healthy machine look
    # exhausted. This is the number a plane should size a sleeping copy against.
    avail = vals.get("MemAvailable", vals.get("MemFree", 0)) / 1048576.0
    return {"total_gib": round(total, 2), "used_gib": round(total - avail, 2),
            "free_gib": round(avail, 2), "source": "host"}
