"""Capacity-scaled frozen-source model catalogue; default target is four B300 trays.

Catalogue membership/order and demand-class assignments use a frozen selection
heuristic. Runtime memory calibration must not silently reshuffle seeded demand.
GPU reservations and development prefill contracts are explicit hardware
estimates; a target-qualified calibration profile would replace them with
measured configurations.
"""
from __future__ import annotations
import json
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
# The single alias table lives with the stager that creates the directories. Duplicating it here would
# let the catalogue name a path that does not exist.
import physics
import popularity  # noqa: E402
import stage_weights  # noqa: E402
import checkpoints  # noqa: E402

# From the hardware profile, so a different card reshapes the catalogue instead of silently invalidating
# it. See physics.load_profile (CLUSTER_HW_PROFILE).
GPU_GIB = physics.GPU_GIB
# Longest prompt+completion the task will ever ask for. Each model's own limit caps this: a model such as
# `phi-4` stops at 16384 and vLLM refuses to start if asked for more, so the limit is per model rather
# than one global flag.
CONTEXT_LIMIT = 32768
USABLE = 0.9              # Σ util across awake servers on one GPU; above this vLLM starts failing
TARGET_OVERSUBSCRIPTION = 2.5


def selection_cost_gib(weights_gib: float) -> float:
    """Frozen catalogue-ranking cost: a host-buffer heuristic, 1.59 x weights - 2.31 GiB.

    It defines selection and order only, not an engine's RAM charge. It stays
    independent of the process/rank overhead in physics.py, so tuning those
    constants cannot reshuffle the catalogue or its seeded popularity assignments.
    """
    return max(1.0, 1.59 * weights_gib - 2.31)

# REFERENCE TIER LADDER (`_TIERS` below): (tier, weights GiB as shipped, ServeGen class, per-model
# popularity). util is NOT listed: it is derived from the weights and the card's capacity via
# physics.min_util(), because it is a property of the hardware rather than of the model. `build()` does
# not draw models from this ladder -- it selects real checkpoints from the frozen model list -- and
# `summarise()` uses its tier names for reporting.
# PUBLISHED LATENCY TARGETS -- fixed constants, part of the task definition, identical on every machine.
#
# These are absolute, not relative to what the hardware can do, because that is how serving SLOs work
# everywhere they matter. A provider publishes "TTFT under 2 s" and then CHOOSES hardware to meet it; the
# customer never learns which GPU served them. MLPerf Inference fixes its latency constraints per
# benchmark in the spec, identical across every submission and vendor. Aegaeon (SOSP'25), whose setting
# this task models, uses TTFT <= 2 s and TBT <= 20 ms. Hardware-relative targets appear in research
# (AlpaServe, DistServe use an "SLO scale" over single-device latency) but as a sensitivity axis within
# one paper, never as a shared benchmark's definition.
#
# The consequence that matters here: **the test data is a pure function of (seed, catalogue, arrival
# process, these targets)**. No profiling, no measurement, nothing carried between runs. Two runs on
# different node sets replay byte-identical requests.
#
# TTFT is one number for every model because it is dominated by queueing and waking, not by prefill:
# solo TTFT is 11-28 ms, while waking a 32B costs about 1.4 s, so 2000 ms buys one on-demand wake.
# Per-model TPOT targets come from TPOT_BANDS below and, where a market anchor exists, from
# catalogue_profile.py.
# The BASE TTFT target, for the ordinary tier of service.
SLO_TTFT_MS = 2000.0
# Per-tier TPOT table (ms/token) behind `slo_for_tier`, roughly 2x what each size does alone: big models
# are legitimately slower per token, and a flat target would punish them for being big. Catalogue models
# take their TPOT from TPOT_BANDS instead.
SLO_TPOT_MS = {"pico": 5.0, "nano": 8.0, "tiny": 12.0, "small": 20.0, "mid": 30.0,
               "reason": 45.0, "upper": 35.0, "large": 60.0,
               "reasonxl": 90.0, "giant": 80.0, "frontier": 90.0}

# SERVICE TIERS. A provider does not accept an arbitrary latency number per request -- it publishes a
# small set of tiers with committed targets and lets a request choose one, the way OpenAI exposes
# `service_tier` and other providers expose equivalents. A request names its tier; the tier fixes the
# targets. Published, fixed, and identical on every machine, exactly like the base targets. This
# catalogue offers a single tier, `standard`, so every request is held to its model's own TTFT/TPOT pair.
SERVICE_TIERS = {"standard": {"ttft_ms": 2000.0, "tpot_mult": 1.0, "share": 1.0}}


def slo_for_tier(model_tier: str, service_tier: str) -> tuple[float, float]:
    """(TTFT ms, TPOT ms/token) for a model tier served at a given tier of service."""
    st = SERVICE_TIERS[service_tier]
    return st["ttft_ms"], round(SLO_TPOT_MS[model_tier] * st["tpot_mult"], 3)

def _fit(weights_gib: float, gpu_gib: float | None = None) -> int:
    """Fewest GPUs this model's weights fit across on this card -- ROUNDED UP TO A POWER OF TWO.

    The power of two is not tidiness, it is deployability. Tensor parallelism splits attention heads
    across ranks, so `tensor_parallel_size` has to divide the head count; in practice 1/2/4/8 work for
    nearly every model and odd sizes do not (vLLM refuses them with "Total number of attention heads must
    be divisible by tensor parallel size").

    It also lands the big models on clean multi-node shapes: on 4-GPU nodes a model that needs five GPUs
    becomes tp=8, exactly two full machines -- the standard vLLM multi-node configuration -- rather than
    five GPUs split 4+1.
    """
    cap = gpu_gib or physics.GPU_GIB
    per_gpu = (USABLE * cap - physics.KV_BASE_GIB) / (1 + physics.KV_PER_WEIGHT)
    need = max(1, int(-(-weights_gib // per_gpu)))
    n = 1
    while n < need:
        n *= 2
    return n


_TIERS = [
    # (tier, weights GiB as SHIPPED, tp floor is derived, ServeGen class, per-model popularity)
    #
    # Sizes are actual `.safetensors` bytes from the HuggingFace API, in each repo's native dtype --
    # Kimi-K2 and DeepSeek ship FP8, the Qwen models bf16 -- not parameter-count estimates. vLLM keeps a
    # checkpoint in its stored dtype, so file size is weight memory.
    ("frontier", 959.0, "language/m-large",   0.8),   # Kimi-K2, 1T MoE, FP8
    ("giant",    894.0, "language/m-large",   1.0),   # Qwen3-Coder-480B MoE, bf16
    ("reasonxl", 641.0, "reason/deepseek-r1", 1.0),   # DeepSeek-V3.1, FP8
    ("large",    438.0, "language/m-large",   2.4),   # Qwen3-235B MoE
    ("upper",    140.0, "language/m-mid",     2.7),   # Qwen3-Next-80B MoE
    ("reason",    62.0, "reason/deepseek-r1", 1.6),   # R1-Distill-32B, long outputs
    ("mid",       62.0, "language/m-mid",     2.2),
    ("small",     28.0, "language/m-mid",     1.7),
    ("tiny",      16.0, "language/m-small",   1.2),
    ("nano",       8.0, "language/m-small",   1.0),
    ("pico",       3.4, "language/m-small",   1.0),
]
# tp floor is DERIVED from the card, not tabulated: ceil(weights / what one GPU can hold).
TIERS = [(t, w, physics.min_util(w / _fit(w)), _fit(w), c, p) for t, w, c, p in _TIERS]

# Reference checkpoints per tier: (checkpoint, weights GiB). The SIZE BELONGS TO THE CHECKPOINT, not to
# the tier: a tier is a BEHAVIOUR class (output length, latency target), so its members may differ in
# size, and an under-reservation does not waste memory, it fails the load. Sizes are `.safetensors` bytes
# in each repo's native dtype, rounded up where uncertain, because vLLM keeps a checkpoint in its stored
# dtype, so file size is weight memory. The graded catalogue's checkpoints come from the frozen model
# list and checkpoints.lock.json, not from this table.
CHECKPOINTS = {
    "frontier": [("moonshotai/Kimi-K2-Instruct", 959.0)],
    "giant":    [("Qwen/Qwen3-Coder-480B-A35B-Instruct", 894.0)],
    "reasonxl": [("deepseek-ai/DeepSeek-V3.1", 641.0)],
    "large":    [("Qwen/Qwen3-235B-A22B-Instruct-2507", 438.0)],
    "upper":    [("Qwen/Qwen3-Next-80B-A3B-Instruct", 150.0)],
    "reason":   [("deepseek-ai/DeepSeek-R1-Distill-Qwen-32B", 62.0),
                 ("Qwen/QwQ-32B", 62.0),
                 ("deepseek-ai/DeepSeek-R1-Distill-Qwen-14B", 28.0)],
    "mid":      [("Qwen/Qwen3-32B", 62.0), ("Qwen/Qwen2.5-32B-Instruct", 62.0)],
    "small":    [("Qwen/Qwen3-14B", 28.0), ("Qwen/Qwen2.5-14B-Instruct", 28.0),
                 ("microsoft/phi-4", 28.0)],
    "tiny":     [("Qwen/Qwen3-8B", 16.0), ("meta-llama/Llama-3.1-8B-Instruct", 15.0),
                 ("mistralai/Ministral-8B-Instruct-2410", 15.0)],
    "nano":     [("Qwen/Qwen3-4B", 8.0), ("Qwen/Qwen2.5-3B-Instruct", 6.2)],
    "pico":     [("Qwen/Qwen3-1.7B", 3.4), ("Qwen/Qwen3-0.6B", 1.4)],
}

# Reference per-tier model counts for an 8-GPU fleet: a long tail of small models, where a serverless
# fleet's volume actually is, and a few expensive ones. `build()` does not use them; it selects real
# models from the frozen model list and scales the catalogue with the fleet's disk budget.
COUNTS_8GPU = {"frontier": 1, "giant": 2, "reasonxl": 2, "large": 4, "upper": 7,
               "reason": 4, "mid": 11, "small": 15, "tiny": 17, "nano": 18, "pico": 19}


def _context_len(local_dir: str, root: str = "", frozen: int | None = None) -> int:
    """The model's own context limit, capped at CONTEXT_LIMIT.

    `frozen` -- `max_context` from the model list, recorded at fetch time -- WINS, and is the only
    machine-independent source. Reading it off local disk would make the catalogue depend on which
    weights happen to be staged: three checkpoints cap below CONTEXT_LIMIT (`MythoMax-L2-13b` and
    `ReMM-SLERP-L2-13B` at 4096, `L3-8B-Lunaris-v1` at 8192), so generating traces before downloading
    anything would produce a different catalogue, a different `catalogue_sha256`, and therefore different
    traces from the same master seed. The disk read is a fallback for checkpoints with no frozen value
    (gated repos serve no anonymous config.json).
    """
    if frozen:
        return int(min(CONTEXT_LIMIT, int(frozen)))
    root = root or os.environ.get("CLUSTER_WEIGHTS") or os.path.expanduser("~/hf-weights")
    cfg = os.path.join(root, local_dir, "config.json")
    try:
        with open(cfg) as f:
            c = json.load(f)
        n = c.get("max_position_embeddings") or (c.get("text_config") or {}).get("max_position_embeddings")
        return int(min(CONTEXT_LIMIT, int(n))) if n else CONTEXT_LIMIT
    except (OSError, ValueError, TypeError):
        return CONTEXT_LIMIT


# DENSITY FLOOR: the minimum demand per GiB (providers serving a model, per GiB of its selection cost) a
# model needs to be OFFERED at all -- a provider does not list a model it cannot fill. It is 0.0: the
# graded catalogue relies on the demand-per-GiB ordering in build() instead. See the note there.
DENSITY_FLOOR = 0.0

RAM_PER_GPU = physics.HOST_RAM_PER_GPU_GIB    # one definition; fleetgen.py reads the same one
# What a provider can stage. The catalogue is sized against this; GPU and host-RAM pressure follow.
DISK_BUDGET_TIB = float(os.environ.get("CLUSTER_DISK_TIB", "23.0"))
TARGET_RAM_OVERSUB = 1.20  # diagnostic only; catalogue sizing uses awake GPU demand
TARGET_GPU_OVERSUB = 1.45

# TPOT by ACTIVE parameters. Values sit inside the range Artificial Analysis measures across 192 real
# models (p90 4.1, p50 10.8, p10 23.5 ms/token); the band edges are ours, chosen to split this catalogue
# into roughly even groups. Keyed to active rather than total because that is what bounds decode: the
# same 235B model routes 22B per token, and a dense 405B routes all of it.
TPOT_BANDS = ((10.0, 8.0), (30.0, 12.0), (60.0, 18.0), (float("inf"), 25.0))

_MODELS_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "models-openrouter-2026-09-08.json")
_MODELS_CACHE: list | None = None


def _real_models() -> list:
    """The candidate models, most-served first: the frozen model list, derived from the pinned
    OpenRouter snapshot in `_dev/snapshots/`."""
    global _MODELS_CACHE
    if _MODELS_CACHE is None:
        with open(_MODELS_FILE) as f:
            _MODELS_CACHE = json.load(f)["models"]
    return _MODELS_CACHE



def _one_per_release(rows: list) -> list:
    """One model per RELEASE, dropping dated re-snapshots of the same thing.

    OpenRouter lists `kimi-k2` beside `kimi-k2-0905`, `deepseek-r1` beside `deepseek-r1-0528`: byte for
    byte the same weights under two ids. A provider stocks one of those, not both, and for this task
    they are pure cost -- identical size, identical placement problem, 2.6 TiB of staging between them
    and nothing to decide. Which to keep is the more-served one, since that is the one a real host would
    carry. Keeping the most popular also keeps the traffic where the snapshot put it.

    Deliberately narrow: it collapses date and version suffixes only, so genuinely different sizes of a
    family (`qwen3-30b` vs `qwen3-235b`) and different generations (`kimi-k2` vs `kimi-k3`) both stay --
    those are different placement problems and they are what makes the catalogue interesting.
    """
    import re
    def stem(mid: str) -> str:
        return re.sub(r"-(\d{4}|\d{6}|v\d+(\.\d+)?|latest|instruct)$", "", mid)
    best: dict[str, dict] = {}
    for r in rows:
        k = stem(_model_id(r["openrouter_id"]))
        cur = best.get(k)
        if cur is None or float(r["n_providers"]) > float(cur["n_providers"]):
            best[k] = r
    return [r for r in rows if best.get(stem(_model_id(r["openrouter_id"]))) is r]


def _model_id(openrouter_id: str) -> str:
    """`qwen/qwen3-235b-a22b-2507` -> `qwen3-235b-a22b-2507`. Unique already, and readable in a log."""
    return openrouter_id.split("/", 1)[-1]


def _tpot_for(active_gib: float, params_b: float) -> float:
    active_b = params_b if active_gib <= 0 else active_gib * (1024 ** 3) / 1e9 / (1.0 if params_b >= 400 else 2.0)
    for lim, ms in TPOT_BANDS:
        if active_b < lim:
            return ms
    return TPOT_BANDS[-1][1]


def _size_tier(params_b: float) -> str:
    """A label for reporting only -- nothing is priced off it."""
    for lim, name in ((10, "tiny"), (40, "small"), (150, "mid"), (500, "large")):
        if params_b < lim:
            return name
    return "frontier"


# How many models carry the reasoning workload. See `_assign_reason`.
N_REASON_MODELS = 2


def _reasoning_capable(row: dict) -> bool:
    """Can this model serve reasoning traffic at all?

    Either it says so in its name, or it is a frontier model -- by 2026 the large general models are
    hybrid reasoners, which is why a provider does not need a separate reasoning deployment."""
    s = row["openrouter_id"].lower()
    return any(k in s for k in ("think", "reason", "-r1", "/r1", "-r-")) or row["params_b"] >= 200


def _servegen_class(row: dict) -> str:
    """Which ServeGen traffic category this model draws from, before the reasoning pass below.

    Split by size the way ServeGen splits its own language categories."""
    b = row["params_b"]
    return "language/m-small" if b < 40 else "language/m-mid" if b < 200 else "language/m-large"


def _assign_reason(models: list[dict], rows_by_id: dict) -> None:
    """Point the reasoning workload at the reasoning models a provider ACTUALLY SERVES.

    Reading it off the model's name alone would point it at retired models: in the snapshot the only
    models that declare reasoning in their names are `deepseek-r1` (a single provider left) and
    `kimi-k2-thinking` (two), the least economic models in the catalogue. No policy should host them, so
    the class would score zero for every plane, which measures the catalogue and not the plane.

    A provider's reasoning traffic follows the successor, not the retired model, so the class goes to
    the reasoning-capable models with the most demand per GiB. Reasoning answers are 1,392 tokens
    against chat's 156, so this is a real workload, not a relabel."""
    # By DEMAND PER GiB, not by demand. Ranking on provider count alone hands the workload to the two
    # largest models in the catalogue -- 1,026 GiB each -- which is the one profile that least wants an
    # expensive host: reasoning answers are 1,392 tokens against chat's 156, so the traffic is heavy
    # already. A provider serves reasoning on whichever capable model carries the demand for the least
    # memory, which here is a 235B and a 671B rather than two 1T models.
    cap = sorted((m for m in models if _reasoning_capable(rows_by_id[m["id"]])),
                 key=lambda m: -rows_by_id[m["id"]]["n_providers"]
                 / selection_cost_gib(m["weights_gib"]))
    for m in cap[:N_REASON_MODELS]:
        m["servegen_class"] = "reason/deepseek-r1"


def build(n_gpus: int, oversubscription: float = TARGET_GPU_OVERSUB,
          max_gpus_per_model: int = 0, slo_scale: float = 1.0,
          density_floor: float = DENSITY_FLOOR, calibration_profile: dict | None = None,
          *, disk_budget_tib: float | None = None,
          max_models: int | None = None) -> dict:
    """Scale the catalogue to a fleet, holding oversubscription constant.

    A tier whose weights do not fit across `max_gpus_per_model` cards is DROPPED rather than shrunk. It
    has to be: `min_tp` is physics, so a fleet that cannot host a model cannot host it at any setting,
    and leaving it in the catalogue produces a model every policy fails on identically -- a constant
    subtracted from every score, which measures nothing. Defaults to max(4, n_gpus // 8) cards, capped
    at the fleet; the graded catalogue uses 4."""
    # min() with n_gpus, not just the argument: `max_gpus_per_model=2` on a one-GPU fleet must not
    # yield two-GPU models, which are precisely the 'model the fleet cannot host' this parameter exists
    # to prevent.
    if (type(n_gpus) is not int or n_gpus < 1
            or not math.isfinite(oversubscription) or oversubscription <= 1
            or not math.isfinite(slo_scale) or slo_scale <= 0):
        raise ValueError("positive fleet/SLO scale and oversubscription > 1 are required")
    if type(max_gpus_per_model) is not int or max_gpus_per_model < 0:
        raise ValueError("max_gpus_per_model must be a nonnegative integer")
    cap = min(max_gpus_per_model or max(4, n_gpus // 8), n_gpus)
    max_models = max_models if max_models is not None else max(16, 2 * n_gpus)
    if type(max_models) is not int or max_models < 4:
        raise ValueError("max_models must be at least 4")
    catalogue_models = _one_per_release(_real_models())
    fleet_ram = RAM_PER_GPU * n_gpus
    # SIZE THE CATALOGUE BY DISK, not by host RAM.
    #
    # Host RAM is charged only to engines that sleep, so "how much of the catalogue fits in host memory"
    # is the plane's choice rather than a property of the world. Sizing against it would also make the
    # catalogue shrink whenever the measured sleep-buffer multiplier changes, which can take GPU
    # oversubscription below 1.0 and remove the residency decision altogether.
    #
    # Disk is what actually bounds a provider's catalogue: it stages what it can store. The GPU and
    # host-RAM pressures are then OUTPUTS, reported below, and the fleet is sized against them.
    disk_budget_tib = (DISK_BUDGET_TIB * n_gpus / 64.0
                       if disk_budget_tib is None else disk_budget_tib)
    if not math.isfinite(disk_budget_tib) or disk_budget_tib <= 0:
        raise ValueError("disk_budget_tib must be positive")
    disk_cap = disk_budget_tib * 1024.0
    # `oversubscription` is the target ratio of awake-GPU demand (sum of min_util x GPUs) to the fleet's
    # usable GPU capacity: selection stops once it is reached. Host-RAM pressure is reported as an
    # output (`host_ram_oversubscription`).
    target_ram = oversubscription * fleet_ram
    capacity = USABLE * n_gpus            # GPU awake budget, still reported alongside the RAM figure
    dropped = sorted({_size_tier(r["params_b"]) for r in catalogue_models if _fit(r["weights_gib"]) > cap})

    # SELECT, do not DUPLICATE: real models, each appearing once, scaled by taking MORE of them rather
    # than more copies of a few checkpoints. Copies would be independent engines holding identical
    # weights, and on real hardware the second load of a checkpoint is a page-cache hit.
    #
    # ORDERED BY DEMAND PER GiB, not by demand alone.
    #
    # Taking the most-provider PREFIX lets size and demand come apart: a few huge, rarely served models
    # would claim most of the fleet's memory for a few percent of the traffic. That spread in
    # requests-per-GiB is not what a provider's catalogue looks like, because a provider does not offer a
    # model it cannot fill; planes would be asked to solve a catalogue that should never have been offered.
    #
    # Provider count is still the demand signal, and a good one -- it correlates positively with size
    # here (median 4 providers at >=200B against 2 below 50B), so this does not simply delete the large
    # models: a large model with many providers stays, while one served by a single provider drops.
    def _density(r: dict) -> float:
        return r["n_providers"] / selection_cost_gib(r["weights_gib"])

    catalogue_models = sorted((r for r in catalogue_models if _density(r) >= density_floor),
                              key=lambda r: (-_density(r), r["openrouter_id"]))
    # Deterministic round-robin across weight-size bands preserves small models
    # worth colocating and large models worth sharding. A density-only prefix can
    # fill max_models with tiny checkpoints and eliminate residency pressure.
    bands = [[], [], [], []]
    for row in catalogue_models:
        native_gib = round(checkpoints.serving_weights_gib(row["checkpoint"]), 1)
        if _fit(native_gib) <= cap:
            band = sum(native_gib >= edge
                       for edge in (GPU_GIB / 16, GPU_GIB / 4, GPU_GIB))
            bands[band].append(row)
    ordered = []
    while any(bands):
        for band in bands:
            if band:
                ordered.append(band.pop(0))
    catalogue_models = ordered
    models, demanded, ram, disk = [], 0.0, 0.0, 0.0
    with open(os.path.join(os.path.dirname(__file__), 'model-architecture.json')) as source:
        architectures = json.load(source)['models']
    for row in catalogue_models:
        gib, act = round(checkpoints.serving_weights_gib(row['checkpoint']), 1), row["active_gib"]
        fit = _fit(gib)
        if fit > cap:
            continue                      # the fleet cannot host it at any setting; see the docstring
        if disk + gib > disk_cap:
            continue          # this one does not fit the staging budget; a smaller one still might
        tier = _size_tier(row["params_b"])
        architecture = architectures.get(_model_id(row['openrouter_id']))
        if architecture and architecture['checkpoint_revision'] != checkpoints.entry(row['checkpoint'])['revision']:
            raise ValueError('architecture metadata must match the pinned checkpoint')
        is_moe = architecture['is_moe'] if architecture else row['active_params_b'] < row['params_b']
        serving_active = act if is_moe else gib
        if architecture and 'active_params_b' in architecture:
            serving_active = architecture['active_params_b'] * 2e9 / 2**30
        models.append({
            "id": _model_id(row["openrouter_id"]),
            "tier": tier,
            "checkpoint": row["checkpoint"],
            "checkpoint_revision": checkpoints.entry(row["checkpoint"])["revision"],
            "weights_gib": gib,
            "weights_source": "root serving tensors in checkpoints.lock.json; native checkpoint weight format",
            # ACTIVE weights, the number decode speed depends on. Equal to `weights_gib` for a dense
            # model; a fraction of it for MoE, by 9x to 25x across this catalogue. Published by the model
            # itself in its `-a22b` suffix, so it is data rather than an estimate.
            "active_gib": act,
            # Checkpoint-byte rounding and embeddings make active_gib slightly
            # smaller than weights_gib for many dense models. Architecture comes
            # from parameter metadata, never from that numerical inequality.
            "is_moe": is_moe,
            "serving_active_gib": serving_active,
            "architecture": architecture['text_config'] if architecture else {},
            "min_util": physics.min_util(gib / fit),
            "min_tp": fit,
            "tp": fit,
            "gpus": fit,
            "servegen_class": _servegen_class(row),
            "popularity_weight": float(row["n_providers"]),
            "local_dir": stage_weights.local_name(row["checkpoint"]),
            "context_len": _context_len(stage_weights.local_name(row["checkpoint"]),
                                        frozen=row.get("max_context")),
            "slo_ttft_ms": round(SLO_TTFT_MS * slo_scale, 1),
            # Banded by ACTIVE parameters, not by total size. Decode is bandwidth-bound on the weights a
            # token actually routes through, so a 2.4T-A95B model is as fast as a 95B one; a table keyed
            # to total size would hold it to a target 25x looser than it needs while holding a dense 405B
            # model to one it cannot meet.
            "slo_tpot_ms": round(_tpot_for(act, row["params_b"]) * slo_scale, 3),
        })
        ram += physics.host_ram_gib(gib, local_ranks=fit)
        disk += gib
        demanded += models[-1]["min_util"] * fit
        if demanded >= oversubscription * capacity or len(models) >= max_models:
            break
    if demanded < oversubscription * capacity:
        raise ValueError(
            f"catalogue constraints cannot reach {oversubscription:.2f}x awake demand: "
            f"{len(models)} models give {demanded / capacity:.3f}x; increase disk/max_models/"
            "max_gpus_per_model or provide more source models")
    # `expected_share`: the fraction of traffic a model should receive IN EXPECTATION, from the same
    # `popularity_weight` that tracegen samples with. Deriving it from memory footprint instead would
    # invert reality -- traffic proportional to model size, concentrated on the models the fleet can
    # keep the fewest of awake -- and no placement policy can fix a catalogue whose demand does not fit
    # its own hardware.
    by_class: dict[str, list] = {}
    for m in models:
        by_class.setdefault(m["servegen_class"], []).append(m)
    # SHAPE THE CURVE HERE, once, and let tracegen sample the same thing. `n_providers` is the raw
    # signal (how many providers serve this model) and stays for provenance; `popularity_weight` is
    # the Zipf curve laid on that ranking, which is what traffic actually follows (see
    # _kit/popularity.py).
    for m in models:
        m.setdefault("n_providers", m["popularity_weight"])
    shaped = popularity.shaped_weights(models)
    for m in models:
        m["popularity_weight"] = shaped[m["id"]]
        m["expected_share"] = round(shaped[m["id"]], 6)

    _assign_reason(models, {_model_id(r["openrouter_id"]): r for r in catalogue_models})

    result = {
        # THE SERVICE-TIER TABLE IS PUBLISHED, machine-readable, so the composition rule -- ttft from the
        # service tier, tpot from the model's target times the service tier's multiplier -- is not left
        # to prose. A plane resolves its own targets from this and the per-model figures; nothing is
        # resolved for it on the request, because doing the join is part of the job.
        "service_tiers": {name: {"ttft_ms": round(t["ttft_ms"] * slo_scale, 1),
                                 "tpot_mult": t["tpot_mult"]}
                          for name, t in SERVICE_TIERS.items()},
        "n_gpus": n_gpus, "gpu_gib": GPU_GIB, "usable_util_per_gpu": USABLE,
        "fleet_capacity_units": round(capacity, 3),
        "demanded_units": round(demanded, 3),
        "oversubscription": round(demanded / capacity, 3),          # GPU awake units, reported
        "host_ram_gib": round(fleet_ram, 1),
        "host_ram_demanded_gib": round(ram, 1),
        "host_ram_oversubscription": round(ram / fleet_ram, 3),        # the binding constraint
        "dropped_tiers": dropped,
        "schema_version": 2,
        "slo_scope": "model",
        "selection": {"algorithm": "size-band-round-robin-v1",
                      "target_gpu_oversubscription": oversubscription,
                      "disk_budget_tib": disk_budget_tib, "max_models": max_models,
                      "max_gpus_per_model": cap, "checkpoint_gib": round(disk, 3)},
        "models": models,
    }

    if slo_scale != 1.0:
        raise ValueError("the suite uses fixed service contracts, not a global SLO-scale sweep")
    import catalogue_profile
    return catalogue_profile.apply(result, calibration_profile)


# Fields derived from the TRACE, which the agent must not see. A plane that knows each model's share of
# traffic can rank the catalogue by cost-per-request and refuse the tail, and that is the one strategy
# the score exists to discourage. Withholding the forecast does not make the task unfair: a
# real operator learns demand by observing it, which this plane can do too, and the public traces are
# generated from the same process as the private ones so the SHAPE of the workload is discoverable.
TRACE_DERIVED_MODEL_FIELDS = ("expected_share", "popularity_weight", "n_providers", "servegen_class")
TRACE_DERIVED_TOP_FIELDS = ("demanded_units", "host_ram_demanded_gib",
                            "oversubscription", "host_ram_oversubscription")


def redact_for_agent(cat: dict) -> dict:
    """The agent's copy: hardware, SLOs and model shapes, with every trace-derived field removed.

    What stays is what a provider knows before any traffic arrives -- how big each model is, how many
    cards it needs, what latency it is advertised at, and what the fleet has. What goes is anything
    that amounts to being handed the answer sheet for the workload.
    """
    out = {k: v for k, v in cat.items() if k not in TRACE_DERIVED_TOP_FIELDS}
    out["models"] = [{k: v for k, v in m.items() if k not in TRACE_DERIVED_MODEL_FIELDS}
                     for m in cat["models"]]
    return out


def summarise(cat: dict) -> str:
    by = {}
    for m in cat["models"]:
        b = by.setdefault(m["tier"], [0, 0.0, 0.0])
        b[0] += 1; b[1] += m["min_util"] * m["tp"]; b[2] += m["weights_gib"]
    out = [f"{cat['n_gpus']} GPUs | capacity {cat['fleet_capacity_units']:.2f} units | "
           f"demanded {cat['demanded_units']:.2f} | oversubscription {cat['oversubscription']:.2f}x | "
           f"{len(cat['models'])} models",
           f"  {'tier':8} {'n':>3} {'util ea':>8} {'units':>7} {'weights GiB':>12}"]
    # From the models themselves, not from TIERS: a tier's members may differ in size, so the tier's
    # nominal figure and the tier's actual cost are two different numbers.
    for tier, *_ in TIERS:
        if tier in by:
            n, units, w = by[tier]
            out.append(f"  {tier:8} {n:3d} {units/n:8.2f} {units:7.2f} {w:12.0f}")
    out.append(f"  {'TOTAL':8} {len(cat['models']):3d} {'':8} {cat['demanded_units']:7.2f} "
               f"{sum(m['weights_gib'] for m in cat['models']):12.0f}")
    return "\n".join(out)


if __name__ == "__main__":
    import argparse
    from pathlib import Path
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--gpus', type=int, default=16)
    parser.add_argument('--max-gpus-per-model', type=int, default=0)
    parser.add_argument('--disk-budget-tib', type=float)
    parser.add_argument('--max-models', type=int)
    parser.add_argument('--calibration-profile', type=Path)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--public', type=Path)
    args = parser.parse_args()
    profile = json.loads(args.calibration_profile.read_text()) if args.calibration_profile else None
    cat = build(args.gpus, max_gpus_per_model=args.max_gpus_per_model,
                disk_budget_tib=args.disk_budget_tib, max_models=args.max_models, calibration_profile=profile)
    args.out.write_text(json.dumps(cat, indent=1)+'\n')
    if args.public:
        args.public.write_text(json.dumps(redact_for_agent(cat), indent=1)+'\n')
    print(summarise(cat))
