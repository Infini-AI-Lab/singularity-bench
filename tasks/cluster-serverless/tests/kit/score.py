"""Three-cell serverless utility: strict attainment times model completion.

Only the trusted client writes records. The scorer checks original-clock timing
again and retains the trial-wide completed-response integrity gate.
"""
from __future__ import annotations
import argparse
import collections
import json
import math
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import traceunits

SCORE_VERSION = "attainment-completion-raw-v5"
CELLS = ("load-0.75", "load-1.0", "load-1.25")


def finite(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value) and value >= 0


def model_reward(model_stats: dict) -> float:
    """A is request-weighted; C gives each offered model equal weight."""
    total = strict = 0.
    completion = []
    for mid, s in model_stats.items():
        w, q, c, n = s['weight'], s['strict_weight'], s['completed'], s['requests']
        if (not all(finite(v) for v in (w, q, c, n)) or w <= 0 or n <= 0
                or q > w + 1e-9 or c > n or c != int(c) or n != int(n)):
            raise ValueError(f"invalid sufficient statistics for {mid}")
        total += w
        strict += q
        completion.append(c / n)
    return (strict / total) * (math.fsum(completion) / len(completion)) if total else 0.


def summarize(records: list[dict], trace: list[dict], stats: dict | None = None) -> dict:
    by_id = {r['request_id']: r for r in records if 'request_id' in r}
    units = traceunits.units(trace)
    if len({t['request_id'] for t in units}) != len(units):
        raise ValueError('duplicate logical request IDs')
    per_model, per_class = {}, {}
    responses = valid_receipts = complete = 0
    for t in units:
        w = t['weight']
        if not finite(w) or w <= 0:
            raise ValueError('request weights must be finite and positive')
        pm = per_model.setdefault(t['model'], dict(weight=0., strict_weight=0.,
            complete_weight=0., requests=0, completed=0))
        pc = per_class.setdefault(t['class'], dict(weight=0., strict_weight=0.))
        pm['weight'] += w
        pm['requests'] += 1
        pc['weight'] += w
        rec = by_id.get(t['request_id'])
        if rec is None:
            continue
        responded = rec.get('status') == 200 and rec.get('stream_done')
        responses += bool(responded)
        valid_receipts += bool(responded and rec.get('receipt_ok'))
        e2e = rec.get('e2e_ms')
        completed = (responded and rec.get('receipt_ok') and rec.get('complete')
                     and finite(e2e) and e2e <= traceunits.deadline_ms(t))
        if not completed:
            continue
        complete += 1
        pm['complete_weight'] += w
        pm['completed'] += 1
        ttft, tpot = rec.get('ttft_ms'), rec.get('tpot_ms')
        output = int(t.get('out_tokens', t.get('max_tokens', 0)))
        strict = (finite(ttft) and ttft <= e2e and ttft <= t['slo_ttft_ms']
                  and (output == 1 or (finite(tpot) and tpot <= t['slo_tbt_ms'])))
        if strict:
            pm['strict_weight'] += w
            pc['strict_weight'] += w
    total_weight = math.fsum(v['weight'] for v in per_model.values())
    strict_weight = math.fsum(v['strict_weight'] for v in per_model.values())
    attainment = strict_weight / total_weight if total_weight else 0.
    completion_model = {m: v['completed'] / v['requests'] for m, v in sorted(per_model.items())}
    fair = math.fsum(completion_model.values()) / len(per_model) if per_model else 0.
    attain_model = {m: v['strict_weight'] / v['weight'] for m, v in sorted(per_model.items())}
    starved = [m for m, v in completion_model.items() if v == 0.]
    tampered = [r.get('request_id', '?') for r in records if r.get('tampered')]
    utility = model_reward(per_model)
    return {
        'score_version': SCORE_VERSION, 'n_requests': len(units), 'responses': responses,
        'total_weight': total_weight, 'served_value': strict_weight,
        'model_stats': per_model, 'utility': utility, 'M': utility,
        'strict_attainment': attainment, 'attainment': attainment,
        'mean_model_completion': fair, 'fairness_factor': fair,
        'aborted': [r['aborted'] for r in records if 'aborted' in r],
        'gates': {'G1_receipts': valid_receipts / responses if responses else 0.,
                  'completion': complete / len(units) if units else 0.,
                  'tampered': bool(tampered), 'tampered_n': len(tampered),
                  'tampered_examples': tampered[:5]},
        'fairness': {'worst_model': min(completion_model.values(), default=0.),
                     'starved_models': len(starved), 'starved_examples': starved[:8],
                     'models_scored': len(per_model), 'min_samples': 0,
                     'below_25pct': sum(v < .25 for v in completion_model.values()),
                     'below_50pct': sum(v < .5 for v in completion_model.values())},
        'completion_by_model': completion_model, 'attainment_by_model': attain_model,
        'attainment_by_class': {c: v['strict_weight']/v['weight'] for c, v in per_class.items()},
        'worst_models': sorted(attain_model.items(), key=lambda kv: kv[1])[:5],
        'latency': _latency(list(by_id.values())), 'failures': _failures(list(by_id.values())),
        'fleet': _fleet(stats),
    }


def _latency(records: list[dict]) -> dict:
    import statistics
    # .get, not []: the records file can carry a non-request line (see summarize).
    ok = [r for r in records if r.get("ttft_ms") is not None]
    if not ok:
        return {}
    ttft = sorted(r["ttft_ms"] for r in ok)
    tbt = sorted(r["tpot_ms"] for r in ok if r.get("tpot_ms") is not None)
    q = lambda a, p: a[min(len(a) - 1, int(p * len(a)))]
    out = {"ttft_p50": round(statistics.median(ttft), 1), "ttft_p95": round(q(ttft, .95), 1)}
    if tbt:
        out |= {"tbt_p50": round(statistics.median(tbt), 2), "tbt_p95": round(q(tbt, .95), 2)}
    return out


def _failures(records: list[dict]) -> dict:
    c = collections.Counter()
    for r in records:
        if "request_id" not in r:      # a non-request line, e.g. the abort marker; see summarize
            continue
        if r["served"]:
            continue
        if r["note"]:
            c[r["note"] if not r["note"].startswith("truncated") else "truncated"] += 1
        elif r["status"] != 200:
            c[f"status {r['status']}"] += 1
        elif not r.get("on_time"):
            c["late (SLO missed)"] += 1
        else:
            c["other"] += 1
    return dict(c.most_common(8))


def _fleet(stats: dict | None) -> dict:
    if not stats:
        return {}
    served = aborted = 0
    for w in stats.get("worker_stats") or []:
        for m in (w.get("models") or {}).values():
            served += m.get("served", 0); aborted += m.get("aborted", 0)
    return {"worker_served": served, "worker_aborted": aborted}


def gates_pass(summary: dict) -> bool:
    """ONE GATE: nothing was tampered with. Completion and G1 are reported; neither gates."""
    g = summary["gates"]
    # RECEIPTS ONLY. Fairness is scored, not gated; fraud is a validity question, not a quality one --
    # a forged or replayed response is not degraded service, it is a false claim that service happened.
    return bool(not g["tampered"])


def reward(summary: dict) -> dict:
    """Absolute utility in [0, 1], subject to the response-integrity gate."""
    passed = gates_pass(summary)
    return {
        "reward": round(summary["M"] if passed else 0.0, 6),
        # The request-weighted attainment, kept as a DIAGNOSTIC under the name `served_fraction`. It is
        # not the reward: duplicating the reward here would make the two indistinguishable in every
        # downstream report.
        "served_fraction": summary["attainment"],
        "fairness_factor": summary["fairness_factor"],
        "gates_pass": passed,
        "G1_receipts": summary["gates"]["G1_receipts"],
        "completion": summary["gates"]["completion"],
        "tampered": summary["gates"]["tampered"],
        **{f"ttft_{k}": v for k, v in summary["latency"].items() if k.startswith("ttft")},
        **{f"tbt_{k.split('_')[1]}": v for k, v in summary["latency"].items() if k.startswith("tbt")},
    }


def write_reward(d: dict, outdir: str = "/logs/verifier") -> None:
    """reward.json holds numeric fields only: consumers parse every field as a float."""
    os.makedirs(outdir, exist_ok=True)
    numeric = {k: float(v) for k, v in d.items() if isinstance(v, (int, float, bool))}
    with open(os.path.join(outdir, "reward.json"), "w") as f:
        json.dump(numeric, f, indent=1)
    with open(os.path.join(outdir, "reward.txt"), "w") as f:
        f.write(f"{numeric.get('reward', 0.0)}\n")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--records", required=True)
    ap.add_argument("--trace", required=True)
    ap.add_argument("--stats")
    ap.add_argument("--out")
    a = ap.parse_args()
    records = [json.loads(l) for l in open(a.records)]
    trace = [json.loads(l) for l in open(a.trace)]
    stats = json.load(open(a.stats)) if a.stats and os.path.exists(a.stats) else None
    s = summarize(records, trace, stats)
    print(json.dumps({k: v for k, v in s.items() if k != "attainment_by_model"}, indent=1))
    if a.out:
        json.dump(s, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
