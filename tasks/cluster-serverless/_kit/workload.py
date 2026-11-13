"""Construct the three-cell 16-GPU suite from fixed inputs and ServeGen shapes.

Private construction code: never copy into the submitted program's environment.
The phase transformation and model assignments are synthetic, explicitly so.
"""
from __future__ import annotations

import hashlib
import itertools
import json
import math
from pathlib import Path

import popularity
import prompts
import seeds
import service_contract
import tracegen
import traceunits

LOADS = (.75, 1., 1.25)
CELL_NAMES = ("load-0.75", "load-1.0", "load-1.25")


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def profile(path=None):
    p = Path(path) if path else Path(__file__).with_name("workload-profile.json")
    result = json.loads(p.read_text())
    if result.get('schema_version') != 4 or result.get('status') not in (
            'development-provisional', 'target-qualified'):
        raise ValueError('invalid workload profile')
    for key in ('duration_s', 'phase_s', 'transition_s', 'base_logical_rps', 'setup_s'):
        if not math.isfinite(result[key]) or result[key] <= 0:
            raise ValueError(f'invalid {key}')
    if not result['transition_s'] < result['phase_s'] <= result['duration_s']:
        raise ValueError('invalid demand transition schedule')
    if result['duration_s'] % result['phase_s'] != 0:
        raise ValueError('duration must contain a whole number of phases')
    fractions = result['logical_class_shares']
    if (set(fractions) != {'isolated', 'session', 'batch'}
            or any(not math.isfinite(v) or v <= 0 for v in fractions.values())
            or not math.isclose(math.fsum(fractions.values()), 1.)):
        raise ValueError('invalid class shares')
    if result['status'] == 'target-qualified' and not result.get('evidence_sha256'):
        raise ValueError('target-qualified rates require raw public measurement evidence fingerprints')
    return result


def pack(core, cohort, n_gpus, gpus_per_node=4):
    """A modeled HBM placement, not a throughput/host-memory certificate."""
    free = [.9] * n_gpus
    result = []
    order = lambda m: (-m["min_tp"], -m["min_util"], m["id"])
    for m in sorted(core, key=order) + sorted(cohort, key=order):
        tp, util = int(m["min_tp"]), float(m["min_util"])
        if tp >= gpus_per_node:
            choices = [tuple(range(i, i + tp)) for i in range(0, n_gpus - tp + 1, gpus_per_node)]
        else:
            choices = [g for node in range(n_gpus // gpus_per_node)
                       for g in itertools.combinations(range(node * gpus_per_node, (node + 1) * gpus_per_node), tp)]
        choices = [g for g in choices if all(free[x] + 1e-9 >= util for x in g)]
        if not choices:
            return None
        choice = min(choices, key=lambda g: (sum(free[x] - util for x in g), g))
        for x in choice:
            free[x] -= util
        by_node = {}
        for x in choice:
            by_node.setdefault(f"node{x // gpus_per_node}", []).append(x % gpus_per_node)
        result.append({"model": m["id"], "tp": tp, "util": util,
                       "placement": [{"node": n, "gpus": g} for n, g in by_node.items()]})
    return result


def normalize(values):
    total = math.fsum(values.values())
    if total <= 0:
        raise ValueError("empty demand distribution")
    return {k: v / total for k, v in values.items()}


def segments(catalogue, seed, settings):
    """All models remain offered while synthetic popularity changes smoothly.

    ServeGen provides source arrival order and shapes. Model identities and the
    changing demand mix are benchmark synthesis, not a claim about source users.
    """
    models = catalogue['models']
    duration, phase, transition = (settings[k] for k in ('duration_s', 'phase_s', 'transition_s'))
    probabilities = []
    for i in range(int(duration / phase)):
        weights = popularity.jittered_weights(models, seeds.rng('v4-popularity', seed, i),
                                              popularity.RANK_JITTER)
        # Multiplicative shifts retain the real provider-count popularity prior
        # and a positive tail, without protecting a preselected set of models.
        rng = seeds.rng('v4-demand-shift', seed, i)
        probabilities.append(normalize({mid: w * rng.choice((.25, 1., 4.))
                                        for mid, w in sorted(weights.items())}))
    out = [(0., phase, probabilities[0], probabilities[0])]
    for i in range(1, len(probabilities)):
        out += [(i*phase, i*phase+transition, probabilities[i-1], probabilities[i]),
                (i*phase+transition, (i+1)*phase, probabilities[i], probabilities[i])]
    return out


def masses(parts):
    out = {}
    for start, end, p, q in parts:
        for mid in p.keys() | q.keys():
            out[mid] = out.get(mid, 0.) + (end - start) * (p.get(mid, 0.) + q.get(mid, 0.)) / 2
    return out


def event_time(parts, mid, fraction):
    """Inverse cumulative intensity; keeps source burst ordering within a model."""
    fraction = min(max(float(fraction), 0.), math.nextafter(1., 0.))
    left = masses(parts)[mid] * fraction
    for start, end, p, q in parts:
        a, b = p.get(mid, 0.), q.get(mid, 0.)
        length = end - start
        area = length * (a + b) / 2
        if area <= 0:
            continue
        if left > area:
            left -= area
            continue
        slope = (b - a) / length
        if abs(slope) < 1e-15:
            delta = left / a
        elif left == 0:
            delta = 0.
        else:
            delta = 2 * left / (a + math.sqrt(max(0., a * a + 2 * slope * left)))
        return min(end, max(start, start + delta))
    raise ValueError("arrival probability lies outside the phase schedule")


def allocate(total, weights, *, coverage=False):
    total = int(total)
    weights = normalize(weights)
    base = 1 if coverage and total else 0
    if total < base * len(weights):
        raise ValueError("offered count is too small for declared model coverage")
    remaining = total - base * len(weights)
    exact = {m: w * remaining for m, w in weights.items()}
    counts = {m: base + math.floor(v) for m, v in exact.items()}
    for m in sorted(exact, key=lambda m: (-(exact[m] - math.floor(exact[m])), m))[:total - sum(counts.values())]:
        counts[m] += 1
    return counts


def generate_case(catalogue, settings, load_multiplier, seed, source_split):
    if load_multiplier not in LOADS:
        raise ValueError("unknown load cell")
    tracegen._TIERS = catalogue["service_tiers"]
    models = {m["id"]: m for m in catalogue["models"]}
    parts = sessions_parts = segments(catalogue, seed, settings)
    duration = parts[-1][1]
    reference = settings['base_logical_rps']
    rate = reference * load_multiplier
    shares = settings["logical_class_shares"]
    mixture, session_mix = masses(parts), masses(sessions_parts)
    online = allocate(round(rate * duration * shares["isolated"]), mixture, coverage=True)
    convs = tracegen.conversations(source_split, 8) if shares["session"] else []
    mean_turns = math.fsum(min(8, len(c)) for c in convs) / len(convs) if convs else 1.
    sessions = allocate(round(rate * duration * shares["session"] / mean_turns), session_mix, coverage=True)
    batches = allocate(round(rate * duration * shares["batch"]), mixture, coverage=True)
    # Draw one adequately sized empirical arrival reservoir per source class.
    # Seeded thinning to fixed model counts avoids low-rate client-rounding
    # artifacts from constructing one ServeGen client pool per rare model.
    reservoirs, offsets = {}, {}
    for cls in sorted({models[mid]["servegen_class"] for mid, n in online.items() if n}):
        count = sum(n for mid, n in online.items() if models[mid]["servegen_class"] == cls)
        rows = tracegen.servegen_requests(cls, duration, max(16., 2 * count / duration), f"v4:{seed}:{cls}")
        if len(rows) < count:
            raise ValueError("ServeGen arrival reservoir has too few requests")
        seeds.rng("v4-reservoir", seed, cls).shuffle(rows)
        reservoirs[cls], offsets[cls] = rows, 0
    items, clipped, conditioned = [], 0, 0
    for mid in sorted(mixture):
        model = models[mid]
        rng = seeds.rng("v4-model-stream", seed, mid)
        cls = model["servegen_class"]
        rows = reservoirs.get(cls, [])[offsets.get(cls, 0):offsets.get(cls, 0) + online[mid]]
        offsets[cls] = offsets.get(cls, 0) + online[mid]
        for row in rows:
            limit = int(model["context_len"])
            for attempt in range(10000):
                n_in = max(1, min(int(row["in"]), limit - prompts.CTRL_TOKENS - 1))
                n_out = max(1, min(int(row["out"]), limit - n_in - prompts.CTRL_TOKENS))
                if service_contract.available(model, tracegen._TIERS, [n_in + prompts.CTRL_TOKENS]):
                    break
                conditioned += 1
                row = rng.choice(reservoirs[cls])
            else:
                raise ValueError(f"no source shapes support an advertised product for {mid}")
            clipped += int(n_in != row["in"] or n_out != row["out"])
            tier = tracegen.choose_tier(model, [n_in + prompts.CTRL_TOKENS], rng)
            ttft, tpot = tracegen.slo_for_request(model, tier, n_in + prompts.CTRL_TOKENS)
            items.append({"kind": "request", "request_id": "", "model": mid, "tier": model["tier"],
                "class": tracegen.CLASS_OF[cls], "weight": 1., "context_len": limit,
                "arrival_s": min(duration - .0001, round(event_time(parts, mid, row["t"] / duration), 4)),
                "input_tokens": n_in, "new_user_tokens": n_in, "out_tokens": n_out,
                "wire_max_tokens": prompts.bucket_for(n_out, rng, limit - n_in - prompts.CTRL_TOKENS),
                "stream_id": rng.getrandbits(62), "service_tier": tier,
                "slo_ttft_ms": ttft, "slo_tbt_ms": tpot,
                "deadline_slack_s": round(tracegen.DEADLINE_SLACK_BASE_S * rng.uniform(*tracegen.DEADLINE_JITTER), 2)})
        remaining = batches.get(mid, 0)
        while remaining:
            size = min(200, remaining)
            job = tracegen.build_batch([model], [1.], rng, duration, 1, size, rng, source_split)[0]
            # build_batch draws submission times uniformly over 60% of the window;
            # this suite maps them through the full demand-intensity curve instead.
            job["submit_s"] = min(duration - .001, round(event_time(parts, mid, rng.random()), 3))
            items.append(job)
            remaining -= len(job["requests"])
    for mid, count in sorted(sessions.items()):
        if count:
            rng = seeds.rng("v4-sessions", seed, mid)
            drawn = tracegen.build_sessions(convs, [models[mid]], [1.], rng, duration, count, 8, rng)
            if len(drawn) != count:
                raise ValueError("session generation changed the declared offered load")
            for session in drawn:
                session["start_s"] = min(duration - .001, round(event_time(sessions_parts, mid, session["start_s"] / duration), 3))
            items += drawn
    items.sort(key=lambda r: r.get("arrival_s", r.get("start_s", r.get("submit_s", 0.))))
    for i, item in enumerate(items):
        if item["kind"] == "request":
            item["request_id"] = f"r-{i:07d}"
        elif item["kind"] == "session":
            item["id"] = f"s-{i:07d}"
        else:
            item["id"] = f"b-{i:07d}"
            for j, r in enumerate(item["requests"]):
                r["id"] = f"b-{i:07d}-{j:05d}"
    validate_trace(items, catalogue)
    assigned = traceunits.units(items)
    if set(r["model"] for r in assigned) != set(mixture) | {m for m, n in sessions.items() if n}:
        raise ValueError("trace omitted an advertised active model")
    metadata = {"schema_version": 4, "load_multiplier": load_multiplier, "source_split": source_split, "seed": seed,
                "master_seed": seeds.MASTER_SEED, "duration_s": duration,
                "nominal_logical_rps": rate, "reference_logical_rps": reference,
                "n_requests": len(assigned), "clipped_isolated_shapes": clipped, "rejected_unoffered_isolated_shapes": conditioned,
                "active_models": sorted({r["model"] for r in assigned}),
                "profile_sha256": digest(settings), "catalogue_sha256": digest(catalogue),
                "calibration_status": settings["status"], "prompt_protocol": "retained-wire-history-v3",
                "source_split_rule": "SHA256 profile or 1024-row offline block; fold 0/5 public, others private"}
    return items, metadata


def validate_trace(items, catalogue):
    traceunits.validate_wire_context(items)
    models = {m["id"]: m for m in catalogue["models"]}
    ids = set()
    for r in traceunits.units(items):
        if r["request_id"] in ids:
            raise ValueError("duplicate logical request ID")
        ids.add(r["request_id"])
        targets = service_contract.resolve(models[r["model"]], catalogue["service_tiers"],
                                            r["service_tier"], r["input_tokens"] + prompts.CTRL_TOKENS)
        if targets != (r["slo_ttft_ms"], r["slo_tbt_ms"]):
            raise ValueError("trace and advertised contract disagree")


def build_suite(catalogue, settings, seed, source_split):
    if catalogue['n_gpus'] != 16 or catalogue.get('slo_scope') != 'model':
        raise ValueError('this suite requires the 16-GPU catalogue with per-model SLOs')
    cases = [{'name': name, 'load_multiplier': factor, 'seed': f'{seed}:{name}'}
             for name, factor in zip(CELL_NAMES, LOADS)]
    return {'schema_version': 4, 'suite_seed': seed, 'source_split': source_split,
            'model_ids': sorted(m['id'] for m in catalogue['models']),
            'profile_sha256': digest(settings), 'catalogue_sha256': digest(catalogue),
            'qualification': {'workload': settings['status'], 'catalogue': catalogue['calibration']['status']},
            'setup_s': settings['setup_s'], 'session_retries': 2,
            'cell_weights': dict.fromkeys(CELL_NAMES, 1/3), 'cases': cases}
