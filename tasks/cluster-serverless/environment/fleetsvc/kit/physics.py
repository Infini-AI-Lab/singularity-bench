"""Surrogate costs for development replay, not target-hardware timing guarantees.

The real grader drives real vLLM at wall-clock speed. These constants affect the
simulator and the generated development catalogue. Capacity constants describe the
intended B300 fleet; they still need inventory checks at deployment.

Prefill and decode coefficients are fitted to pinned vLLM 0.28 engine measurements on
H200 GPUs, with separately labelled B200 measurements from an older vLLM runtime.
Large families, cross-tray communication, hybrid KV layouts and unmeasured families
are approximations. The sleep, wake and cold-load estimates below come from
single-node vLLM measurements; Grace CPU links and target storage can behave
differently.

CLUSTER_HW_PROFILE overrides hardware assumptions for declared sensitivity tests.
A successful accelerated replay establishes timing headroom, not GPU fidelity.
"""
from __future__ import annotations

# Target specification assumptions, not observed usable capacities.
GPU_GIB = 288.0           # B300, spec
# 17 TiB / 72 GPUs from the GB300 rack specification, approximately 242 GiB/GPU.
# Verify the actual host/container allocation; sleeping engines use this budget.
HOST_RAM_PER_GPU_GIB = 242.0
HBM_TBPS = 8.0            # B300, spec
BF16_TFLOPS = 2200.0      # B300 dense, spec
DECODE_EFF = 0.63
PREFILL_EFF = 0.35        # DERIVED: fraction of peak FLOPs a real prefill reaches

# MEASURED with vLLM **sleep level 1** on one GPU node, six models spanning 1.2-62 GiB, every one checked
# for a COHERENT answer before and after the round trip. Level 2 is faster on a stopwatch and does not
# round-trip: it discards the weights and `wake_up` restores only buffers, so the model comes back
# emitting "!!!!".
#
# In-container subprocess launch, which is how the real fleet works -- NOT `docker run` per model, so
# cold load here includes engine init but not image start.
#
#   GiB   cold s   L1 sleep s   L1 wake ms   host RAM GiB
#   1.2     81.5        1.35           69          2.9
#   3.4     81.5        2.60          119          5.6
#   8.0     87.6        5.04          221         16.0
#  16.0     89.7       10.05          390         22.6
#  28.0     93.7       23.29          667         48.5
#  62.0    112.0       38.95         1431         63.0
#
# Least squares over all six. cold and wake fit to within 3%; sleep to 25%, worst at the small end where
# the absolute error is a second.
COLD_LOAD_S_BASE, COLD_LOAD_S_PER_GIB = 81.29, 0.4914
# Sleep has TWO regimes, and the difference is 28x. The FIRST sleep of a server allocates the pinned host
# buffer the weights are copied into; every sleep after that reuses it.
#
#   8B  (16 GiB):  9837 ms, then  347 /  351 /  349 ms
#   32B (62 GiB): 37071 ms, then 1273 / 1276 ms
#
# The second regime lands close to a WAKE of the same model (349 vs 388 ms; 1273 vs 1430 ms), which makes
# sense -- both are one copy of the weights across PCIe, in opposite directions.
#
# Engines deploy awake. Only an explicit first sleep pays pinned-buffer allocation;
# later sleeps reuse PyTorch's cached buffers. Transfer times use weights PER LOCAL
# GPU: the pinned-runtime 14B TP2 sleep/wake is approximately half of TP1.
SLEEP_S_BASE, SLEEP_S_PER_GIB = 0.912, 0.6391                 # first sleep, from the fresh-load table
RESLEEP_MS_BASE, RESLEEP_MS_PER_GIB = 27.6, 20.09             # subsequent, through both measured points
WAKE_MS_BASE, WAKE_MS_PER_GIB = 40.05, 22.40
# Pinned-buffer measurements (H200, vLLM 0.19.1):
#
#     weights 7.49 GiB (Qwen3-4B)  -> 10.16 GiB of shmem
#     weights 15.26 GiB (Qwen3-8B) -> 21.04
#     weights 27.51 GiB (Qwen3-14B)-> 41.79
#
# Least squares over those: shmem = 1.59 x weights - 2.31, residuals <= 0.91 GiB.
# Extrapolation to larger checkpoints is uncertain; it is not a measurement of
# any 931-GiB shard or of Grace host memory.
HOST_RAM_BASE_GIB, HOST_RAM_PER_GIB = -2.31, 1.59
# This is a conservative buffer proxy, not an exact linear law. Dedicated v0.28
# cgroups measured 20.95/41.58/68.38 GiB shmem for 15.26/27.51/61.02 GiB checkpoints.
# PyTorch rounds pinned allocations to powers of two; the 32B buffer is substantially
# smaller than this proxy. Keep process memory separate and never charge less than
# the backed-up weights.
# Per GPU, including contexts/graphs retained outside the sleep pool. v0.28 TP1
# observations: 2161, 2511, 2619 MiB; TP2: 2557/2517 MiB. Round up to 3 GiB.
RESIDENT_ASLEEP_GIB = 3.0
# Per local rank. Dedicated cgroups: 2.67--2.77 GiB TP1 and 5.91 GiB TP2 after
# subtracting the collector. This conservative rounding also covers worker overhead.
HOST_RAM_PROCESS_GIB = 3.0


# Minimum KV reservation on top of the weights, fitted to hand-chosen per-tier reservations: 9.4 GiB for
# a 3.4 GiB model up to 17.4 for a 70 GiB one. It is mostly a floor -- a model needs enough KV to hold a
# few concurrent sequences at the context limit, and that floor barely depends on model size.
KV_BASE_GIB, KV_PER_WEIGHT = 9.0, 0.12


def min_util(weights_per_gpu_gib: float, gpu_gib: float | None = None) -> float:
    """The smallest `--gpu-memory-utilization` that will actually start this model.

    DERIVED from the card size, not tabulated: a table written for one card size would leave every model
    reserving twice what it needs on a card twice as large -- about one model per GPU instead of three or
    four, and a task that is impossible for a reason that has nothing to do with scheduling.
    """
    cap = gpu_gib or GPU_GIB
    need = weights_per_gpu_gib + KV_BASE_GIB + KV_PER_WEIGHT * weights_per_gpu_gib
    return round(min(0.95, need / cap), 3)


# The target's logical nodes are compute trays in ONE GB300 NVL72 rack. All
# 72 GPUs share an NVLink domain; crossing a tray is not an Ethernet boundary:
# https://docs.nvidia.com/dgx/dgxgb200-user-guide/hardware.html
# TP_HOP_MS still charges communication with increasing TP size. No additional
# tray multiplier is assumed by default; a slower fabric (for example a 2.5x
# multiplier) is a declared sensitivity profile via CLUSTER_HW_PROFILE, not
# target-topology evidence. Actual NCCL routing and latency still require the
# launch preflight.
CROSS_NODE_DECODE_PENALTY = 1.0
CROSS_NODE_HOP_MS = 0.4            # one-way routing cost to a worker on another node


def cross_node_penalty(n_nodes_spanned: int) -> float:
    """Decode slowdown for a model sharded over `n_nodes_spanned` machines. 1.0 within one node."""
    if n_nodes_spanned <= 1:
        return 1.0
    return CROSS_NODE_DECODE_PENALTY


# Routed experts have an overhead beyond their active weight bytes. These
# coefficients are fitted to single-engine H200/B200 measurements, with
# separate dense/MoE identities from pinned checkpoint architecture metadata.
MOE_DECODE_EFF = 0.55
# Fitted to engine measurements on Qwen3-8B and Qwen3-30B-A3B (H200 with vLLM
# 0.28, B200 with vLLM 0.19). B300 extrapolation is an explicit sensitivity
# assumption, not a target measurement.
DECODE_OVERHEAD_MS = .6
DENSE_STEP_FLOOR_MS = 2.8
TP_HOP_MS = .35
MOE_STEP_FLOOR_MS = 3.7
DENSE_BATCH_COEF = 0.025
MOE_BATCH_COEF = .30
DECODE_CONTEXT_MS_PER_TOKEN = 1.5e-05
# A*ceil(tokens/4096) + B*(tokens/1000) + C*(tokens/1000)^2, fitted at 256/4096/32768
# and checked at 1024/8192/16384, on the pinned v0.28 H200 measurements; the
# older B200 runtime has substantially different burst behavior. This is
# deliberately a surrogate for the target, tested by sensitivity runs.
PREFILL_CURVES = {'dense': [12.782859955886321, 21.994297359989243, 0.4658341774858858],
 'moe': [33.63347710131034, 9.437520394886947, 0.683969814317477]}
PREFILL_REFERENCE_ACTIVE_GIB = {'dense': 15.26, 'moe': 3.3*2e9/2**30}
PREFILL_REFERENCE_GEOMETRY = {'dense': 32*4096, 'moe': 48*2048}
# Collective coefficients are zero in the target proxy until a fabric profile
# supplies them. H200 PCIe measurements must not become a GB300 tray penalty.
PREFILL_TP_LATENCY_MS = 0.
PREFILL_TP_TOKEN_MS = 0.
PREFILL_FAMILY_PROFILES = {'recurrent_dense': {'active_gib': 51.7,
                     'curve': [48.037990293752074, 78.12009145522183, 0.3787399010484514],
                     'geometry': 327680,
                     'observed_tp': 1},
 'recurrent_moe': {'active_gib': 5.6,
                   'curve': [99.0512353724972, 9.179424030322933, 0.002513316935522652],
                   'geometry': 98304,
                   'observed_tp': 2},
 'window_dense': {'active_gib': 51.1,
                  'curve': [7.166468409732703, 102.26200635792163, 1.1009841292621207],
                  'geometry': 333312,
                  'observed_tp': 1}}
RECURRENT_DECODE_OVERHEAD_MS = 1.8
RECURRENT_BATCH_STEP_MS = .11
GPU_RUNTIME_OVERHEAD_GIB = 1.5


def prefill_ms(tokens: int, active_gib: float, *, is_moe: bool = False,
               geometry: float = 32*4096, tp: int = 1, architecture: dict | None = None,
               include_dispatch: bool = True, chunk_tokens: int = 4096) -> float:
    """Measured family curve with disclosed size/parallelism extrapolation."""
    family = 'moe' if is_moe else 'dense'
    a, b, c = PREFILL_CURVES[family]
    ref_active, ref_geometry = PREFILL_REFERENCE_ACTIVE_GIB[family], PREFILL_REFERENCE_GEOMETRY[family]
    kind = architecture_family(architecture, is_moe)
    specialized = PREFILL_FAMILY_PROFILES.get(kind)
    if specialized:
        a, b, c = specialized['curve']
        ref_active, ref_geometry = specialized['active_gib'], specialized['geometry']
    n = max(0, tokens)/1000
    import math
    iterations = max(1, math.ceil(tokens/chunk_tokens)) if include_dispatch else 0
    collective = 0. if specialized else (PREFILL_TP_LATENCY_MS*iterations + PREFILL_TP_TOKEN_MS*n)*math.log2(max(1, tp))*geometry/(32*4096)
    return a*iterations + (b*n*active_gib/ref_active + c*n*n*geometry/ref_geometry)/max(1, tp) + collective


def architecture_family(architecture, is_moe):
    types = (architecture or {}).get('layer_types', ())
    if 'linear_attention' in types:
        return 'recurrent_moe' if is_moe else 'recurrent_dense'
    if 'sliding_attention' in types or 'chunked_attention' in types:
        return 'window_moe' if is_moe else 'window_dense'
    return 'moe' if is_moe else 'dense'


def decode_tok_s(weights_gib: float, batch: int, active_gib: float | None = None,
                 *, is_moe: bool | None = None, context_tokens: float = 0.,
                 architecture: dict | None = None) -> float:
    """Decode is memory-bound: every token re-reads the weights, and a batch amortises that read.

    ACTIVE weights, not total. A mixture-of-experts model routes each token through a fraction of its
    parameters, so decode re-reads only that fraction while memory still holds every expert. The two
    decouple by 9x to 25x across real MoE models -- Qwen3-235B-A22B reads 22B of 235B, Qwen3.8-2.4T-A95B
    reads 95B of 2400B. Pricing decode on the total would make those models 10-25x slower than they are,
    and since decode speed is what an SLO is written against, it would make their targets unmeetable for
    a reason that does not exist.
    
    `active_gib` defaults to `weights_gib`, which is correct for a dense model and is what every caller
    that has no MoE information should get.

    Single-stream is `bandwidth / weights`; batching multiplies it until compute binds. The compute cap
    matters for small models at large batch, which is exactly the regime a busy tiny model sits in.
    """
    # Callers without an architecture field: only a substantial reduction can
    # suggest routed experts. Generated catalogues always supply is_moe.
    if is_moe is None:
        is_moe = active_gib is not None and active_gib < .8 * weights_gib
    act = max(active_gib, .1) if is_moe and active_gib is not None else weights_gib
    single = HBM_TBPS * 1024 * DECODE_EFF / max(act, 0.1)
    if is_moe:
        single *= MOE_DECODE_EFF
    params_b = act / 2.0                               # bf16: 2 bytes per parameter
    compute_cap = BF16_TFLOPS * 1e12 * PREFILL_EFF / (2 * params_b * 1e9)
    import math
    step_ms = 1000/single + DECODE_OVERHEAD_MS
    step_ms = max(step_ms, MOE_STEP_FLOOR_MS if is_moe else DENSE_STEP_FLOOR_MS)
    recurrent = 'linear_attention' in (architecture or {}).get('layer_types', ())
    if recurrent:
        step_ms += RECURRENT_DECODE_OVERHEAD_MS
        step_ms += RECURRENT_BATCH_STEP_MS*max(0, batch-1)
    else:
        step_ms *= 1 + (MOE_BATCH_COEF if is_moe else DENSE_BATCH_COEF)*math.log2(max(1, batch))
    step_ms += DECODE_CONTEXT_MS_PER_TOKEN*max(0., context_tokens)
    return min(1000*batch/step_ms, compute_cap)


def prefill_tok_s(weights_gib: float, active_gib: float | None = None) -> float:
    """ACTIVE weights, for the same reason as decode: prefill's FLOPs are per routed parameter."""
    params_b = (weights_gib if active_gib is None else max(active_gib, 0.1)) / 2.0
    return BF16_TFLOPS * 1e12 * PREFILL_EFF / (2 * params_b * 1e9)


def kv_gib_per_token(weights_gib: float) -> float:
    """DERIVED, and deliberately crude: KV scales with layers × heads × head_dim, which tracks model size
    closely enough for a development fleet. Real engines report their own cache geometry."""
    return 1.2e-4 * (weights_gib / 16.0) ** 0.5


def sleep_s(weights_gib: float, first: bool = False) -> float:
    """Level-1 sleep for one GPU's weights; first sleep allocates pinned buffers.

    Deployment itself leaves the engine awake. Concurrent local ranks transfer
    their own shards; callers pass the largest per-GPU shard, not the node sum.
    """
    if first:
        return SLEEP_S_BASE + SLEEP_S_PER_GIB * weights_gib
    return (RESLEEP_MS_BASE + RESLEEP_MS_PER_GIB * weights_gib) / 1000.0


def sleep_ms(weights_gib: float, first: bool = False) -> float:
    return sleep_s(weights_gib, first) * 1000.0


# WHICH FUNCTIONS TAKE ACTIVE AND WHICH TAKE TOTAL, since getting this backwards breaks the cost model:
#   active  -- decode_tok_s, prefill_tok_s          (compute and bandwidth per token)
#   total   -- host_ram_gib, cold_load_s, wake_ms, sleep_s, min_util, kv_gib_per_token
#              (every expert must be stored, loaded, moved and reserved for, routed to or not)
def host_ram_gib(weights_gib: float, local_ranks: int = 1) -> float:
    """Host RAM a model holds ONCE IT HAS SLEPT -- from the first sleep until the process exits.

    NOT a cost of being deployed. Measured across three models, the offload buffer does not exist until
    the engine sleeps for the first time, and it does not matter whether sleep mode was enabled:

        loaded, sleep mode OFF ....... 0.14 GiB of shmem   (7.49 GiB model)
        loaded, sleep mode ON ........ 0.14            -- enabled but never slept
        after /sleep level 1 ........ 10.16            -- the buffer appears here
        after /wake_up .............. 10.16            -- and persists

    Charging every deployed engine from the moment of load would make deploying WITHOUT sleep mode
    impossible rather than merely a different trade, and that is not what vLLM does: an engine that
    never sleeps holds its weights on the card and keeps only process overhead on the host, so "pay a
    card permanently instead of host RAM" is a legitimate strategy and the plane may choose it.


    Weights are node-wide. Add process memory for every local GPU rank; the
    shared-memory buffer measurements above exclude that overhead.
    """
    return (HOST_RAM_PROCESS_GIB * max(1, local_ranks)
            + max(weights_gib, HOST_RAM_BASE_GIB + HOST_RAM_PER_GIB * weights_gib))


def wake_ms(weights_gib: float) -> float:
    return WAKE_MS_BASE + WAKE_MS_PER_GIB * weights_gib


def cold_load_s(weights_gib: float) -> float:
    return COLD_LOAD_S_BASE + COLD_LOAD_S_PER_GIB * weights_gib


def load_profile(path: str | None = None) -> dict:
    """Override the hardware profile from JSON. Anything absent retains this file's declared target proxy assumption."""
    import json
    import os
    path = path or os.environ.get("CLUSTER_HW_PROFILE")
    if not path or not os.path.exists(path):
        return {}
    prof = json.load(open(path))
    for k, v in prof.items():
        if k.isupper() and k in globals():
            globals()[k] = v
    return prof


load_profile()


def check_fit() -> list[str]:
    """Sanity-check the fitted constants against the level-1 table they came from."""
    out = []
    for gib, m_sleep, m_wake, m_cold, m_ram in ((1.2, 1.35, 69, 81.5, 2.9), (3.4, 2.60, 119, 81.5, 5.6),
                                                (8.0, 5.04, 221, 87.6, 16.0),
                                                (16.0, 10.05, 390, 89.7, 22.6),
                                                (28.0, 23.29, 667, 93.7, 48.5),
                                                (62.0, 38.95, 1431, 112.0, 63.0)):
        out.append(f"  {gib:5.1f} GiB  sleep {sleep_s(gib, first=True):6.2f} vs {m_sleep:5.1f} s | "
                   f"wake {wake_ms(gib):7.1f} vs {m_wake:4d} ms | "
                   f"cold {cold_load_s(gib):5.1f} vs {m_cold:.1f} s | "
                   f"host {host_ram_gib(gib):5.1f} vs {m_ram:.1f} GiB")
    return out


if __name__ == "__main__":
    print("fitted vs MEASURED (level 1, FIRST sleep):")
    print("\n".join(check_fit()))
    print("\nsubsequent sleeps (fitted vs measured), against a wake of the same model:")
    for gib, meas in ((16.0, 349), (62.0, 1273)):
        print(f"  {gib:5.1f} GiB  resleep {sleep_ms(gib):7.1f} vs {meas:4d} ms | wake {wake_ms(gib):7.1f} ms")
    print("\nderived serving rates (simulator surrogate):")
    print(f"  {'model':10} {'prefill tok/s':>14} {'decode b=1':>11} {'decode b=16':>12} {'decode b=64':>12}")
    for name, gib in (("nano 1.7B", 3.4), ("tiny 4B", 8.0), ("small 8B", 16.0),
                      ("mid 14B", 28.0), ("large 32B", 62.0), ("xl 70B", 140.0)):
        print(f"  {name:10} {prefill_tok_s(gib):14.0f} {decode_tok_s(gib,1):11.0f} "
              f"{decode_tok_s(gib,16):12.0f} {decode_tok_s(gib,64):12.0f}")
