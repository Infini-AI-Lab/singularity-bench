#!/usr/bin/env python3
"""Accuracy gate of the serve-glm53-flash task: ACCURACY, and nothing else.

The grader collects the server's raw generation for every accuracy-pass item: a list of
{id, benchmark, generation}. This module OWNS answer extraction (the server cannot
pre-extract), scores each against GOLD, and applies ONE gate:

    PASS  iff  accuracy >= ACC_RATIO * REFERENCE_ACCURACY

REFERENCE_ACCURACY is a MEASURED CONSTANT, below: the reference engine's own accuracy on
this exact prompt set, at this model's own quality level. The gate needs only that
scalar, so no reference generations ship with the task.

WHAT THE GATE CANNOT SEE
------------------------
Scoring against gold alone cannot tell a faithful server from one that looks gold
answers up: both score 1.0. The prompts are held out (they are never in the image), and
the integrity judge reads the delivered /app for gold lookups and other shortcuts.

WHY A RATIO AND NOT AN ABSOLUTE FLOOR
-------------------------------------
ACC_RATIO is relative to the model's own measured accuracy: the gate asks "did you keep
this model's quality?", not "did you clear some fixed bar?", and the same ratio applies
to any model, quantization or hardware.
"""
import sys, json, argparse
from common import load_json, extract, answers_equal

# ---------------------------------------------------------------------------
# MEASURED REFERENCE ACCURACY  (zai-org/GLM-5.3-Flash (FP8 e4m3))
# Measured on exactly this accuracy pass (the same 500 prompts, template, max_tokens and
# extraction); the scored traffic does not enter this gate.
# Engine: vLLM 0.28.1rc1.dev671+gae71862c5, TP=4 on 4x H200 NVL
# Prompt set: 500 held-out math_500 prompts, greedy (temperature=0), max_tokens=16384, Reasoning Effort: Max
# Result: math_500 464/500 = 0.928 (canary 60/60 = 1.000, not gated)
REFERENCE_ACCURACY = 0.928
ACC_RATIO = 0.95          # PASS iff accuracy >= ACC_RATIO * REFERENCE_ACCURACY
                          # -> floor = 0.8816
# ---------------------------------------------------------------------------


def score_candidate(cand, ref=None, eval_items=None, canaries=None):
    """Score candidate generations against gold and apply the accuracy gate. `ref` is not
    used: the gate compares with the REFERENCE_ACCURACY constant above."""
    eval_items = eval_items if eval_items is not None else load_json("hidden_eval.json")
    gold = {it["id"]: it for it in eval_items}
    canary_ids = {c["id"] for c in (canaries or [])}

    by_id = {}
    for c in cand:
        by_id[c["id"]] = c.get("generation") or ""

    per = {}
    correct = n = 0
    for rid, it in gold.items():
        if rid in canary_ids:
            continue                      # canaries are scored separately, never gate
        b = it["benchmark"]
        gen = by_id.get(rid, "")
        ex = extract(b, gen)
        ok = bool(answers_equal(b, ex, it["gold"]))
        d = per.setdefault(b, {"correct": 0, "n": 0})
        d["n"] += 1; d["correct"] += int(ok)
        correct += int(ok); n += 1

    accuracy = (correct / n) if n else 0.0
    floor = ACC_RATIO * REFERENCE_ACCURACY
    acc_gate = accuracy >= floor

    # canary accuracy: reported only, never gates
    can_correct = can_n = 0
    for c in (canaries or []):
        gen = by_id.get(c["id"], "")
        ex = extract(c["benchmark"], gen)
        can_correct += int(bool(answers_equal(c["benchmark"], ex, c["gold"])))
        can_n += 1

    return {
        "accuracy": round(accuracy, 4),
        "reference_accuracy": REFERENCE_ACCURACY,
        "n_scored": n,
        "per_benchmark": {b: {"n": v["n"], "accuracy": round(v["correct"] / v["n"], 4)}
                          for b, v in sorted(per.items())},
        "canary_accuracy": round(can_correct / can_n, 4) if can_n else None,
        "thresholds": {"ACC_RATIO": ACC_RATIO,
                       "REFERENCE_ACCURACY": REFERENCE_ACCURACY,
                       "acc_floor": round(floor, 4)},
        "gates": {"accuracy": acc_gate},          # the one and only gate
        "PASS": bool(acc_gate),
    }


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("candidate", help="candidate outputs json: [{id,benchmark,generation}]")
    args = ap.parse_args()
    res = score_candidate(load_json(args.candidate))
    print(json.dumps(res, indent=2))
    sys.exit(0 if res["PASS"] else 1)
