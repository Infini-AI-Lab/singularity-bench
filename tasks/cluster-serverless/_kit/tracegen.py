"""ServeGen request shapes and public/private source partitions.

The suite (workload.py) composes these primitives. Resolved SLOs come from
fixed public model/tier/input tables, including the complete wire prompt length.
"""
from __future__ import annotations

import argparse
import hashlib

import seeds
import json
from pathlib import Path
import os
from functools import lru_cache
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import popularity  # noqa: E402
import prompts  # noqa: E402
import service_contract  # noqa: E402

# SLOs are MODEL-RELATIVE, not flat. Each model has its own TTFT/mean-TPOT pair, resolved into the
# catalogue (catalogue.py, catalogue_profile.py) and read here from the catalogue's tables. A flat
# inter-token budget -- such as Aegaeon's 20 ms, set for their hardware and model mix -- would sit below
# the single-stream decode floor of large models, so every such request would fail no matter what the
# control plane did. A real per-model SLA is written against what that model can do.
# Longest prompt+completion any engine will accept. ServeGen's tail runs to 57k tokens; a request over
# the engine's limit comes back 400 and is not a scheduling decision, so it is clamped here rather than
# counted against a plane that could not have done anything about it. Keep in step with the worker's
# MAX_MODEL_LEN.
CONTEXT_LIMIT = 32768
_ST = __import__("catalogue").SERVICE_TIERS
TIER_NAMES = list(_ST)
TIER_SHARES = [_ST[t]["share"] for t in TIER_NAMES]
CLASS_OF = {"language/m-small": "chat", "language/m-mid": "chat",
            "language/m-large": "chat", "reason/deepseek-r1": "reason"}


# The tier table of the catalogue currently being generated against. Set by `generate`.
#
# THE TRACE AND THE PLANE MUST RESOLVE TARGETS FROM THE SAME PLACE: the catalogue file. Reading
# module-level constants (`catalogue.slo_for_tier`) instead would ignore any SLO scale applied when the
# targets were resolved into the catalogue, so the trace would be scored against unscaled numbers while
# the plane read scaled ones from the file it was handed, silently.
_TIERS: dict = {}


# Per-request patience, as a multiplier on the base slack. Wide enough that the deadline cannot be
# inferred from a handful of observations, narrow enough that it is not a second load knob.
DEADLINE_SLACK_BASE_S = float(os.environ.get("CLUSTER_DEADLINE_SLACK_S", "300.0"))
DEADLINE_JITTER = (0.65, 1.35)


def slo_for_request(model: dict, service_tier: str, input_tokens: int = 0) -> tuple[float, float]:
    return service_contract.resolve(model, _TIERS, service_tier, input_tokens)


def choose_tier(model: dict, lengths: list[int], rng, required: str | None = None) -> str:
    names = service_contract.available(model, _TIERS, lengths)
    if required:
        if required not in names:
            raise service_contract.UnsupportedProduct(f"{model['id']} cannot serve {required} at these input lengths")
        return required
    if not names:
        raise service_contract.UnsupportedProduct(f"{model['id']} has no product for these input lengths")
    return rng.choices(names, weights=[_TIERS[n].get("share", _ST.get(n, {}).get("share", 1.))
                                       for n in names], k=1)[0]


@lru_cache(maxsize=8)
def source_pool(sg_class: str):
    from servegen import Category, ClientPool
    cat_name, pool_name = sg_class.split("/")
    cwd = os.getcwd()
    try:
        os.chdir(SERVEGEN_ROOT)
        return ClientPool(Category(cat_name), pool_name)
    finally:
        os.chdir(cwd)



CACHE = os.environ.get("SERVEGEN_CACHE", "/tmp/servegen-cache")
# ServeGen resolves its corpus as a literal `data/` relative to the process CWD, with no override, so a
# caller has to be standing in the right directory. Rather than force that on everything that imports
# this module, chdir only around pool construction and put it back.
SERVEGEN_ROOT = os.environ.get("SERVEGEN_ROOT") or os.path.expanduser("~/servegen-root")

# The ServeGen commit these traces were generated against. ServeGen ships its data INSIDE the git repo
# (852 tracked files, no LFS), so the commit pins the generator and the data together. A different commit
# silently yields a different trace from identical arguments, which is why it is recorded and checked.
SERVEGEN_PIN = "d70c8b2d5bc45f4b60a146fbc43dc5523f871c8b"
SERVEGEN_URL = "https://github.com/alibaba/ServeGen.git"


def servegen_provenance() -> dict:
    """Identify the exact generator+data used, for the trace's meta.

    Uses git tree objects rather than hashing the 875 MiB data directory: `HEAD:data` IS the content
    hash of the tracked data, and costs nothing to read.
    """
    import subprocess

    def git(*args: str) -> str:
        try:
            r = subprocess.run(["git", "-C", SERVEGEN_ROOT, *args],
                               capture_output=True, text=True, timeout=30)
            return r.stdout.strip() if r.returncode == 0 else ""
        except (OSError, subprocess.SubprocessError):
            return ""

    from importlib.metadata import version
    import importlib.util
    module = importlib.util.find_spec("servegen")
    if module is None or not Path(module.origin).resolve().is_relative_to(Path(SERVEGEN_ROOT).resolve()):
        raise RuntimeError("install the pinned checkout with pip install -e SERVEGEN_ROOT")
    for package, required in (("numpy", "2.5.2"), ("scipy", "1.18.1")):
        if version(package) != required:
            raise RuntimeError(f"world generation requires {package}=={required}; see _dev/generator-requirements.txt")
    commit = git("rev-parse", "HEAD")
    if not commit:
        # Not a checkout (e.g. a bare copy of data/). Record the absence rather than implying a version.
        print(f"  WARNING: {SERVEGEN_ROOT} is not a git checkout -- trace provenance will be incomplete",
              file=sys.stderr)
        return {"servegen_commit": None, "servegen_data_tree": None,
                "servegen_remote": None, "servegen_dirty": None}
    if commit != SERVEGEN_PIN:
        print(f"  WARNING: servegen at {commit[:12]}, pinned at {SERVEGEN_PIN[:12]} -- "
              f"regenerated traces will NOT match the frozen ones", file=sys.stderr)
    return {
        "servegen_commit": commit,
        "servegen_data_tree": git("rev-parse", "HEAD:data"),
        "servegen_remote": SERVEGEN_URL,  # transport/clone URL is not a data input
        "servegen_dirty": bool(git("status", "--porcelain")),
    }


def servegen_requests(sg_class: str, duration: float, rate: float, seed: str) -> list[dict]:
    """ClientPool construction takes tens of seconds, so cache aggressively: a trace is generated once
    and replayed many times, and nothing may build a pool inside a grading run."""
    os.makedirs(CACHE, exist_ok=True)
    key = hashlib.sha256(json.dumps(["window-rates-v3", SERVEGEN_PIN, sg_class, duration, rate, seed, seeds.MASTER_SEED]).encode()).hexdigest() + ".json"
    path = os.path.join(CACHE, key)
    if os.path.exists(path):
        return json.load(open(path))
    from servegen import Category, ClientPool
    from servegen.construct import generate_workload
    cat_name, pool_name = sg_class.split("/")
    cwd = os.getcwd()
    try:
        os.chdir(SERVEGEN_ROOT)
        pool = source_pool(sg_class)
        # SEED IT. `generate_workload(..., seed=None)` builds np.random.RandomState(None), which draws
        # from OS entropy, so the trace would not be a function of its seed -- and an on-disk cache would
        # hide that until the cache was cleared or the repository cloned fresh.
        sg_seed = seeds.legacy_int("servegen", sg_class, seed, f"{rate:.6f}", f"{duration:.0f}")
        rate_fn = {w.timestamp: rate for w in pool.span(0, duration).get()}
        reqs = generate_workload(pool, rate_fn, duration=duration, seed=sg_seed)
    finally:
        os.chdir(cwd)
    out = [{"t": float(r.timestamp), "in": int(r.data["input_tokens"]),
            "out": int(r.data["output_tokens"])} for r in reqs]
    # Atomic: write beside the target and rename. Cells generate in parallel and can share a cache key,
    # so a plain write could let one process read another's half-written file.
    tmp = f"{path}.{os.getpid()}.tmp"
    with open(tmp, "w") as f:
        json.dump(out, f)
    os.replace(tmp, path)
    return out


BATCH_TRACES = ["trace_d", "trace_e"]


# THINK TIME IS COMPRESSED, and the shape of the compression matters more than the amount.
#
# ServeGen's real inter-turn gaps are heavy-tailed: median 202 s, mean 1720 s, maximum 68,640 s. Replayed
# raw, a 600 s trace takes NINETEEN HOURS to drain, because the last turn of the slowest session is a day
# behind the first. So they have to be compressed. The question is how.
#
# A hard clamp is the obvious answer and the wrong one: 65.6% of gaps exceed a 120 s cap, so two thirds
# of every think time in the trace lands on EXACTLY the cap. That is not a tail, it is a spike, and a
# plane can tune its eviction threshold straight onto it.
#
# This map is strictly increasing over the whole range, so a longer gap stays longer and nothing can pile
# up at the cap -- measured, 0.0% of gaps land within half a second of it, against 66% for the clamp.
# Short gaps stay nearly linear; only the tail is squeezed. At (cap 300, scale 30) the median lands at
# 79 s against the real 202 s, p90 at 185 s, and the trace drains in 1602 s.
#
# What it does NOT change is the prize: prefix reuse (48.5%) is the same under any compression, because
# `prefix_of_prev_tokens` is a property of the conversation's token structure, not of when turns arrive.
THINK_CAP_S = float(os.environ.get("CLUSTER_THINK_CAP_S", "300"))
THINK_SCALE_S = float(os.environ.get("CLUSTER_THINK_SCALE_S", "30"))
THINK_MAX_RAW_S = 68640.0          # the corpus maximum, fixed so the map does not shift with the sample


def compress_think(seconds: float) -> float:
    import math
    if seconds <= 0:
        return 0.0
    return round(THINK_CAP_S * math.log1p(seconds / THINK_SCALE_S)
                 / math.log1p(THINK_MAX_RAW_S / THINK_SCALE_S), 3)


def profile_id(conversation: list[dict]) -> str:
    shape = [(t["input_token_count"], t["output_token_count"],
              round(t["timestamp"] - conversation[0]["timestamp"], 6)) for t in conversation]
    return hashlib.sha256(json.dumps(shape, separators=(",", ":")).encode()).hexdigest()


def in_split(identity: str, split: str) -> bool:
    if split not in ("public", "private"):
        raise ValueError("source split must be public or private")
    return (int(identity[:16], 16) % 5 == 0) == (split == "public")


def conversations(source_split: str = "private", max_turns: int = 8) -> list[list[dict]]:
    """ServeGen's multi-turn sessions: real turn counts, real per-turn lengths, real think times."""
    with open(f"{SERVEGEN_ROOT}/data/conversations/conversations_hashed.json") as f:
        return [c for c in json.load(f) if in_split(profile_id(c[:max_turns]), source_split)]


@lru_cache(maxsize=2)
def offline(name: str) -> list[dict]:
    """A batch trace. NOTE the field names: offline uses input_length/output_length where the online
    path uses input_tokens/output_tokens -- same quantity, different vocabulary, easy to get wrong."""
    with open(f"{SERVEGEN_ROOT}/data/offline/{name}.jsonl") as f:
        return sorted([json.loads(l) for l in f if l.strip()], key=lambda r: r["request_id"])


def conversation_shape(c: list[dict]) -> list[dict]:
    result = []
    for k, turn in enumerate(c):
        raw_input, raw_output = int(turn["input_token_count"]), max(1, int(turn["output_token_count"]))
        previous_raw = int(c[k - 1]["input_token_count"]) + int(c[k - 1]["output_token_count"]) if k else 0
        shared = result[-1]["prompt_tokens"] + prompts.CTRL_TOKENS + result[-1]["out_tokens"] if k and raw_input >= previous_raw else 0
        new_user = raw_input - previous_raw if shared else raw_input
        result.append({"prompt_tokens": shared + new_user, "prefix_of_prev_tokens": shared,
                       "new_user_tokens": new_user, "out_tokens": raw_output})
    return result


def build_sessions(convs, models, weights, rng, duration, n_sessions, max_turns,
                   tier_rng) -> list[dict]:
    """CLOSED-LOOP conversational sessions -- the class that makes prefix caching worth anything.

    Session START is open-loop: a user opens a chat when they open it, independently of our latency.
    Everything after the first turn is closed-loop -- turn N+1 is sent `think_after_s` after turn N
    COMPLETES, so a slow fleet genuinely delays the conversation instead of piling requests up.

    Turn N+1's prompt extends turn N's wire prompt and its answer, token for token (see _kit/prompts.py),
    so the engine derives its own KV blocks by hashing content, exactly as a real one does. Nothing in the
    request says which requests are related; a plane that wants the cache hit has to keep the session on
    the same engine because it noticed, not because the trace told it.
    """
    # REPEAT THE CORPUS, don't cap at it. ServeGen ships 1616 conversations, and the load axis can ask for
    # more (`sessions * load`). Capping at the corpus size would pin the conversational class at a
    # constant count while every other class scales, so the class mix -- and with it prefix reuse -- would
    # ride the load axis as a second variable. Whole permutations rather than sampling with replacement,
    # so coverage stays uniform: at load 2 every conversation is used twice, not 63% of them ~1.6 times
    # each.
    # A repeat is a DIFFERENT user, not the same request twice: `stream` below is redrawn per session and
    # prompt bytes derive from it (prompts.tokens), so a reused conversation shares its shape -- turn
    # count, lengths, think times -- and none of its tokens.
    eligible = []
    for c in convs:
        shapes = conversation_shape(c[:max_turns])
        lengths = [t["prompt_tokens"] + prompts.CTRL_TOKENS for t in shapes]
        need = max(n + t["out_tokens"] for n, t in zip(lengths, shapes))
        if any(int(m.get("context_len") or CONTEXT_LIMIT) >= need
               and service_contract.available(m, _TIERS, lengths) for m in models):
            eligible.append(c)
    convs = eligible
    if n_sessions and not convs:
        raise ValueError("no source conversations fit the advertised products")
    chosen = []
    while len(chosen) < n_sessions:
        chosen += rng.sample(range(len(convs)), min(n_sessions - len(chosen), len(convs)))
    out = []
    for i, ci in enumerate(sorted(chosen)):
        c = convs[ci][:max_turns]
        shapes = conversation_shape(c)
        need = max(t["prompt_tokens"] + prompts.CTRL_TOKENS + t["out_tokens"] for t in shapes)
        lengths = [t["prompt_tokens"] + prompts.CTRL_TOKENS for t in shapes]
        fits = [(m, w) for m, w in zip(models, weights)
                if int(m.get("context_len") or CONTEXT_LIMIT) >= need
                and service_contract.available(m, _TIERS, lengths)]
        if not fits:
            raise ValueError("session eligibility changed during generation")
        model = rng.choices([m for m, _ in fits], weights=[w for _, w in fits], k=1)[0]
        turns, stream = [], rng.getrandbits(62)
        service_tier = choose_tier(model, lengths, tier_rng)
        for k, (t, shape) in enumerate(zip(c, shapes)):
            shared = shape["prefix_of_prev_tokens"]
            # A break in accumulation starts a NEW stream: the client dropped history, so this prompt is
            # not an extension of the last one and must not share bytes with it.
            if k > 0 and shared == 0:
                stream = rng.getrandbits(62)
            new_user = shape["new_user_tokens"]
            n_out = shape["out_tokens"]
            ctx = int(model.get("context_len") or CONTEXT_LIMIT)
            ttft_ms, tbt_ms = slo_for_request(model, service_tier, shape["prompt_tokens"] + prompts.CTRL_TOKENS)
            turns.append({
                "prompt_tokens": shape["prompt_tokens"],
                "new_user_tokens": max(0, int(new_user)),
                "out_tokens": n_out,
                # The cap the PLANE sees: a round number at or above the truth. Baked into the trace,
                # not drawn at send time, because the trace is frozen test data -- two runs comparing
                # two agents have to be driven by identical bytes.
                "wire_max_tokens": prompts.bucket_for(
                    n_out, rng, ctx - shape["prompt_tokens"] - prompts.CTRL_TOKENS),
                "stream_id": stream,
                # Per TURN: each caller in a conversation waits its own length. `units()` spreads the
                # turn dict, so anything not set here is silently absent downstream and the deadline
                # falls back to the published constant -- which is the leak this exists to close.
                "deadline_slack_s": round(DEADLINE_SLACK_BASE_S * rng.uniform(*DEADLINE_JITTER), 2),
                "prefix_of_prev_tokens": int(shared),
                "service_tier": service_tier, "slo_ttft_ms": ttft_ms, "slo_tbt_ms": tbt_ms,
                # think time AFTER this turn completes; absent on the last turn
                "think_after_s": (compress_think(c[k + 1]["timestamp"] - t["timestamp"])
                                  if k + 1 < len(c) else None),
            })
        out.append({"kind": "session", "id": f"s{i:05d}", "class": "conversational",
                    "model": model["id"], "tier": model["tier"],
                    "context_len": ctx, "source_profile": profile_id(c),
                    "start_s": round(rng.random() * duration, 3), "turns": turns})
    return sorted(out, key=lambda x: x["start_s"])


@lru_cache(maxsize=128)
def bulk_source_runs(name, split, context, bins):
    """Eligible contiguous runs, confined to one disjoint source block.

    Eligibility is applied before a job is assigned. Lengths are never truncated
    and no admitted request is removed from a generated trace.
    """
    rows = offline(name)
    runs = []
    for block in range(0, len(rows), 1024):
        if not in_split(hashlib.sha256(f"{name}:{block}".encode()).hexdigest(), split):
            continue
        start = None
        for i in range(block, min(block + 1024, len(rows))):
            r = rows[i]
            size = int(r["input_length"]) + prompts.CTRL_TOKENS
            offered = next((ok for maximum, ok in bins if size <= maximum), False)
            eligible = size + max(1, int(r["output_length"])) <= context and offered
            if eligible and start is None:
                start = i
            if start is not None and (not eligible or i == min(block + 1024, len(rows)) - 1):
                end = i if not eligible else i + 1
                runs.append((start, end - start))
                start = None
    return tuple(runs)


def build_batch(models, weights, rng, duration, n_jobs, job_size, tier_rng, source_split="private") -> list[dict]:
    """OPEN-LOOP bulk sub-jobs sliced contiguously out of the offline traces.

    Contiguous by `request_id`, which preserves submission order in the original file, so a slice keeps
    whatever locality the real job had. Submission times are ours to choose -- batch submission records
    none, by its nature. Each request gets its OWN stream: batch jobs share no conversational history,
    so assuming no prefix reuse is the conservative direction.
    """
    out = []
    if not 0 < job_size <= 1024:
        raise ValueError("batch job size must fit the 1024-row source block")
    for j in range(n_jobs):
        model = rng.choices(models, weights=weights, k=1)[0]
        context = int(model.get("context_len") or CONTEXT_LIMIT)
        service_tier = 'bulk' if 'bulk' in _TIERS else 'standard'
        bins = tuple((r["max_input_tokens"], service_tier in r["tiers"]) for r in model.get("service_contracts", [])) or ((context, True),)
        choices = [(name, bulk_source_runs(name, source_split, context, bins)) for name in BATCH_TRACES]
        choices = [(name, runs) for name, runs in choices if runs]
        if not choices:
            raise ValueError(f"no eligible {source_split} bulk source for {model['id']}")
        name, runs = rng.choice(choices)
        size = min(job_size, max(length for _, length in runs))
        fits = [(start, length) for start, length in runs if length >= size]
        start, length = rng.choices(fits, weights=[length-size+1 for _,length in fits], k=1)[0]
        st = start + rng.randrange(length-size+1)
        rows = offline(name)
        sl = rows[st:st+size]
        # Batch is a workload class; it uses the same model SLO as all other requests.
        ttft_ms, tbt_ms = slo_for_request(model, service_tier, int(sl[0]["input_length"]) + prompts.CTRL_TOKENS)
        out.append({
            "kind": "batch", "id": f"b{j:03d}", "class": "batch", "model": model["id"],
            "tier": model["tier"], "source_trace": name,
            "source_range": [st, st + len(sl)], "context_len": int(model.get("context_len") or CONTEXT_LIMIT),
            "submit_s": round(rng.random() * duration * 0.6, 3),
            "service_tier": service_tier, "slo_ttft_ms": ttft_ms, "slo_tbt_ms": tbt_ms,
            "requests": [{"id": f"b{j:03d}-{k:05d}", "prompt_tokens": int(r["input_length"]),
                          "new_user_tokens": int(r["input_length"]),
                          "out_tokens": max(1, int(r["output_length"])),
                          "wire_max_tokens": prompts.bucket_for(
                              max(1, int(r["output_length"])), rng,
                              max(1, int(model.get("context_len") or CONTEXT_LIMIT)
                                  - int(r["input_length"]) - prompts.CTRL_TOKENS)),
                          "deadline_slack_s": round(DEADLINE_SLACK_BASE_S * rng.uniform(*DEADLINE_JITTER), 2),
                          "stream_id": rng.getrandbits(62),
                          "slo_ttft_ms": slo_for_request(model, service_tier, int(r["input_length"]) + prompts.CTRL_TOKENS)[0],
                          "slo_tbt_ms": slo_for_request(model, service_tier, int(r["input_length"]) + prompts.CTRL_TOKENS)[1]}
                         for k, r in enumerate(sl)],
        })
    return sorted(out, key=lambda x: x["submit_s"])


def zipf_weights(n: int, s: float = 1.1) -> list[float]:
    raw = [1.0 / (i + 1) ** s for i in range(n)]
    total = sum(raw)
    return [x / total for x in raw]


def generate(catalogue: dict, seed: str, duration: float, rate_per_model_unit: float,
             drift_at: float = 0.5, zipf_s: float = 1.8, *,
             n_sessions: int = 0, max_turns: int = 8,
             n_batch_jobs: int = 0, batch_job_size: int = 200, source_split: str = "private") -> list[dict]:
    """THREE CLASSES, because one of them is what makes the fleet's memory decisions matter.

    request   open-loop isolated requests from ServeGen's online categories. No session structure, so no
              cross-turn prefix: on their own they make vLLM's prefix cache worth nothing, leaving the
              cold load as the ONLY argument against sleeping a model.
    session   CLOSED-LOOP multi-turn conversations. Turn N+1's prompt extends turn N's, so keeping a
              session on one engine is worth real prefill.
    batch     open-loop bulk sub-jobs. No shared history, latency-tolerant: the work a plane should be
              packing into whatever capacity the interactive traffic is not using.
    """
    by_class: dict[str, list[dict]] = {}
    for m in catalogue["models"]:
        if "servegen_class" not in m:
            raise SystemExit(
                f"tracegen: model {m.get('id')!r} has no `servegen_class`. This is the AGENT'S redacted "
                f"catalogue (catalogue.redact_for_agent strips the trace-derived fields); generation "
                f"needs the full one from tests/. Pass --catalogue tests/catalogue-graded.json.")
        by_class.setdefault(m["servegen_class"], []).append(m)

    global _TIERS
    if "service_tiers" not in catalogue:
        # Deliberately fatal rather than falling back to catalogue.py's constants. A fallback would
        # silently resolve the trace's targets from unscaled numbers while the plane read scaled ones
        # from the file it was handed -- the exact mismatch this field exists to prevent.
        raise SystemExit("[tracegen] this catalogue has no resolved SLO targets (no 'service_tiers'). "
                         "Rebuild it: catalogue.build(n_gpus, slo_scale=...).")
    _TIERS = catalogue["service_tiers"]
    out: list[dict] = []
    rng_stream = seeds.rng("stream", seed)
    # Its own stream, seeded from the trace seed, so the jitter is reproducible for a given trace and
    # independent of every other draw -- changing the load must not change which callers are patient.
    rng_dl = seeds.rng("deadline", seed)
    for sg_class, models in by_class.items():
        # Rate scales with the class's share of POPULARITY, not of memory. Scaling by footprint would make
        # traffic proportional to model size, which is backwards -- in a real serverless fleet the volume
        # is in the small models and the big ones are rare -- and would concentrate demand on the models
        # the fleet can hold the fewest of. The normaliser keeps total load proportional to fleet size, so
        # oversubscription still scales.
        pop = sum(m["popularity_weight"] for m in models)
        all_pop = sum(m["popularity_weight"] for m in catalogue["models"])
        all_units = sum(m["min_util"] * m["tp"] for m in catalogue["models"])
        rate = rate_per_model_unit * all_units * (pop / all_pop)
        reqs = servegen_requests(sg_class, duration, rate, seed)

        # INDEPENDENT RNG STREAMS, one per purpose, each seeded from (seed, class) only.
        #
        # A single shared generator would interleave the popularity ORDERING with the per-request draws,
        # so the number of requests in one class would shift the RNG state for the next. Request count
        # depends on the arrival rate, so cells at different loads would get different popularity
        # structures, and a difference between cells would not be attributable to load.
        #
        # Split by purpose and the ordering becomes a pure function of (seed, class), identical at every
        # rate; only the volume changes.
        rng_order = seeds.rng("order", seed, sg_class)
        rng_pick = seeds.rng("pick", seed, sg_class)
        rng_tier = seeds.rng("tier", seed, sg_class)

        # Popularity: skewed, and PERTURBED at `drift_at` so a fitted placement rots. The ranking is
        # the catalogue's own, jittered -- not a reshuffle, which would make `expected_share`, the only
        # thing a plane can read before traffic starts, uncorrelated with what arrives; see
        # _kit/popularity.py. Jitter keeps the hint worth having and still worth checking.
        #
        # Ranked over the WHOLE catalogue, then restricted to this class: a model's weight relative to
        # another's must not depend on which class happens to admit it, or the published share would stop
        # matching for every class that filters the catalogue.
        allm = catalogue["models"]
        w_glob_a = popularity.jittered_weights(allm, rng_order, popularity.RANK_JITTER)
        w_glob_b = popularity.jittered_weights(allm, rng_order,
                                               popularity.RANK_JITTER + popularity.DRIFT_JITTER)
        order_a = order_b = models
        w_a = [w_glob_a[m["id"]] for m in models]
        w_b = [w_glob_b[m["id"]] for m in models]
        cls = CLASS_OF[sg_class]

        for r in reqs:
            first = r["t"] < duration * drift_at
            order, w = (order_a, w_a) if first else (order_b, w_b)
            model = rng_pick.choices(order, weights=w, k=1)[0]
            # Clamp BEFORE deriving the SLO: the target is written against the prompt that will actually
            # be sent, not the one ServeGen drew.
            # Per MODEL, not one global figure: a request is clamped to the context of the model it was
            # drawn for, so nothing is ever asked that its engine would refuse.
            # Which tier of service this request bought. Drawn from the published shares, so the mix is
            # stable in distribution and unknown in detail -- a plane has to read the tier off the
            # request rather than assume one.
            service_tier = rng_tier.choices(TIER_NAMES, weights=TIER_SHARES, k=1)[0]
            limit = int(model.get("context_len") or CONTEXT_LIMIT)
            in_tok = max(1, min(int(r["in"]), limit - prompts.CTRL_TOKENS - 1))
            out_tok = max(1, min(int(r["out"]), limit - in_tok - prompts.CTRL_TOKENS))
            service_tier = choose_tier(model, [in_tok + prompts.CTRL_TOKENS], rng_tier)
            ttft_ms, tbt_ms = slo_for_request(model, service_tier, in_tok + prompts.CTRL_TOKENS)
            out.append({
                "request_id": "", "arrival_s": round(r["t"], 4), "model": model["id"],
                "input_tokens": in_tok, "max_tokens": out_tok,
                "context_len": limit,
                "slo_ttft_ms": ttft_ms, "slo_tbt_ms": tbt_ms, "service_tier": service_tier,
                "class": cls, "weight": 1.0, "tier": model["tier"],
                # HOW LONG THIS CALLER WAITS, jittered per request and carried in the trace.
                #
                # A published constant (the SLO budget plus a fixed slack) would be computable from
                # fields a plane already holds. Knowing precisely when a request stops paying would let
                # it abandon work at the last profitable moment and plan around the scoring rule rather
                # than around the service. Real callers do not share one timeout, and the plane is not
                # told theirs: it observes a disconnect.
                "deadline_slack_s": round(DEADLINE_SLACK_BASE_S * rng_dl.uniform(*DEADLINE_JITTER), 2),
            })

    out.sort(key=lambda r: r["arrival_s"])
    for i, r in enumerate(out):
        r["request_id"] = f"r-{i:06d}"
        # Its own stream: an isolated request shares history with nothing, so its prompt must share no
        # bytes with any other. `new_user_tokens` is the whole prompt for the same reason.
        r["kind"] = "request"
        r["stream_id"] = rng_stream.getrandbits(62)
        r["new_user_tokens"] = r["input_tokens"]
        r["out_tokens"] = r.pop("max_tokens")
        r["wire_max_tokens"] = prompts.bucket_for(
            r["out_tokens"], rng_stream, r["context_len"] - r["input_tokens"] - prompts.CTRL_TOKENS)

    items: list[dict] = list(out)
    # Model choice for sessions and batch uses the SAME popularity structure as the request class, so
    # all three classes concentrate on the same models. Drawing them uniformly would spread the session
    # traffic across the whole catalogue and quietly remove the co-location pressure.
    all_models = [m for ms in by_class.values() for m in ms]
    all_w = [m["popularity_weight"] for m in all_models]
    if n_sessions:
        items += build_sessions(conversations(source_split, max_turns), all_models, all_w,
                                seeds.rng("sess", seed), duration, n_sessions, max_turns,
                                seeds.rng("sesstier", seed))
    if n_batch_jobs:
        items += build_batch(all_models, all_w, seeds.rng("batch", seed), duration,
                             n_batch_jobs, batch_job_size, seeds.rng("batchtier", seed), source_split)
    return items


# The flattener lives in traceunits.py so the grader and the agent's replay driver can have it without
# shipping the generator. Re-exported here so code that imports it from tracegen uses the same single
# definition.
from traceunits import units, units_with_arrival, validate_wire_context  # noqa: E402,F401


def summarise(trace: list[dict], catalogue: dict) -> str:
    import collections, statistics
    reqs = units(trace)
    by_model = collections.Counter(r["model"] for r in reqs)
    by_class = collections.Counter(r["class"] for r in reqs)
    tot_in = sum(r["input_tokens"] for r in reqs)
    tot_out = sum(r["out_tokens"] for r in reqs)
    dur = max([r.get("arrival_s", 0.0) for r in trace]
              + [i.get("start_s", 0.0) for i in trace] + [i.get("submit_s", 0.0) for i in trace])
    cold = [m["id"] for m in catalogue["models"] if by_model[m["id"]] <= 3]
    sess = [i for i in trace if i.get("kind") == "session"]
    # What the session class exists to create: prompt tokens that a previous turn already put in the
    # engine's cache. Reported here because if it is near zero the class is not doing its job.
    reusable = sum(t["prefix_of_prev_tokens"] for i in sess for t in i["turns"])
    lines = [f"{len(reqs)} requests over {dur:.0f}s ({len(reqs)/max(dur,1):.1f} req/s) across "
             f"{len(by_model)}/{len(catalogue['models'])} models",
             f"  items: {collections.Counter(i.get('kind', 'request') for i in trace)}",
             f"  by class: {dict(by_class)}",
             f"  tokens: {tot_in/1e6:.2f}M in, {tot_out/1e6:.2f}M out "
             f"({tot_out/max(len(reqs),1):.0f} out/req mean)",
             f"  sessions: {len(sess)}, "
             f"{statistics.mean([len(i['turns']) for i in sess]):.1f} turns mean, "
             f"reusable prefix {reusable/max(tot_in,1):.1%} of all prompt tokens" if sess else
             "  sessions: none",
             f"  popularity: top model {by_model.most_common(1)[0][1]} reqs, "
             f"median {statistics.median(by_model.values()):.0f}, "
             f"cold tail (<=3 reqs): {len(cold)} models",
             f"  top 5: {by_model.most_common(5)}"]
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--catalogue", required=True)
    ap.add_argument("--seed", required=True)
    ap.add_argument("--duration", type=float, default=600.0)
    # THE CLASS MIX IS A DESIGN PARAMETER, not something the corpus chose. ServeGen supplies real
    # conversations with real timestamps, batch traces with no timestamps at all, and online traffic as a
    # GENERATOR whose arrival rate is our input -- it models client populations, not production volumes.
    # So the shapes are sourced and the proportions are ours, and they are set here deliberately:
    #
    #   ~10% session   53% of PROMPT TOKENS -- the caching prize. A conversation's prompt accumulates
    #                  every earlier turn, so the class weighs about 4x its unit count; at 18% of units
    #                  it would reach 70% of prompt tokens and decide the measurement on its own.
    #   ~75% request   baseline load and coverage of the catalogue's cold tail; 42% of prompt tokens
    #   ~15% batch     34% of OUTPUT tokens -- schedulable slack worth packing around the rest
    #
    # The rule that produced these: size each class by the dimension it exists to exercise, and let no
    # class exceed ~60% of any dimension. The unit shares fall OUT of that; they are not the target.
    # Dropping any class still costs real score, which is what stops it being decoration.
    ap.add_argument("--rate", type=float, default=0.35,
                    help="req/s per unit of demanded util at load 1.0")
    ap.add_argument("--out", required=True)
    # ONE KNOB FOR OFFERED LOAD, applied to every class. Scaling the isolated-request stream alone would
    # change the CLASS MIX with load, so a difference between two cells would not be attributable to
    # load. The figures below are the shares at load 1.0; the load multiplies all three.
    ap.add_argument("--sessions", type=int, default=1616,
                    help="closed-loop multi-turn conversations at load 1.0 (0 disables the class)")
    ap.add_argument("--max-turns", type=int, default=8)
    ap.add_argument("--batch-jobs", type=int, default=30)
    ap.add_argument("--batch-job-size", type=int, default=200)
    ap.add_argument("--source-split", choices=("public", "private"), default="private")
    ap.add_argument("--load", type=float, default=1.0,
                    help="scale ALL THREE classes together -- request arrival rate, session count and "
                         "batch job count. This is the sweep axis; --rate sets the per-unit rate at "
                         "load 1.0 and should not be swept on its own.")
    a = ap.parse_args()
    provenance = servegen_provenance()
    if provenance.get("servegen_commit") != SERVEGEN_PIN or provenance.get("servegen_dirty"):
        raise SystemExit("ServeGen must be a clean checkout at the pinned commit before generation")
    cat = json.load(open(a.catalogue))
    trace = generate(cat, a.seed, a.duration, a.rate * a.load,
                     n_sessions=max(0, round(a.sessions * a.load)), max_turns=a.max_turns,
                     n_batch_jobs=max(0, round(a.batch_jobs * a.load)),
                     batch_job_size=a.batch_job_size, source_split=a.source_split)
    validate_wire_context(trace)
    with open(a.out, "w") as f:
        for r in trace:
            f.write(json.dumps(r) + "\n")
    # Provenance beside the trace. The trace is TEST DATA: authored once, frozen, and replayed
    # identically by every run. Two runs that compare two agents must be driven by the same bytes, so the
    # thing that makes them the same bytes is recorded next to them and checked at grade time.
    blob = open(a.out, "rb").read()
    meta = {
        "trace_sha256": hashlib.sha256(blob).hexdigest(),
        # UNITS, not items: a session is one item and many requests, so counting items would report a
        # fleet doing a fraction of the work it does.
        "n_requests": len(units(trace)), "n_items": len(trace),
        # THE TWO SWEEP AXES. `rate` is the per-unit rate at load 1.0 and is the same in every cell -- it
        # is not the axis; `load` is. The SLA is a whole catalogue rather than a multiplier, so it is
        # identified by which catalogue was used.
        "seed": a.seed, "duration_s": a.duration, "rate": a.rate, "load": a.load,
        # The MASTER seed, not just this stage's label. Every stream descends from it (see seeds.py), so
        # regenerating the world with CLUSTER_MASTER_SEED set to anything else produces a different trace
        # that is internally consistent and silently incomparable. Record it so the mismatch is visible
        # in the artifact.
        "master_seed": seeds.MASTER_SEED,
        "sla_catalogue": os.path.basename(a.catalogue),
        "catalogue_sha256": hashlib.sha256(open(a.catalogue, "rb").read()).hexdigest(),
        "slo_source": "fixed constants in catalogue.py (hardware-independent)",
        "slo_targets": "resolved in the catalogue; see catalogue['service_tiers']",
        "context_limit": CONTEXT_LIMIT,
        "source_split": a.source_split,
        "source_split_rule": "SHA256 profile or 1024-row offline block; fold 0/5 public, others private",
        "prompt_protocol": "retained-wire-history-v2",
        # Basenames, not the paths as typed: this meta SHIPS, and host layout must not travel with it.
        "argv": [os.path.basename(x) if x.startswith("/") else x for x in sys.argv[1:]],
        **provenance,
    }
    json.dump(meta, open(a.out.replace(".jsonl", "") + ".meta.json", "w"), indent=1)
    print(summarise(trace, cat))
    print(f"  sha256 {meta['trace_sha256'][:32]}  slo: fixed constants (no profiling)")


if __name__ == "__main__":
    main()
