"""What a trace ROW is, versus what a REQUEST is -- the one definition, in the one place.

A trace line is not a request. A `session` line is one conversation and many turns; a `batch` line is one
job and many sub-requests; only an isolated `request` line is itself. Every consumer that scores, bounds
or drives a trace needs it flattened, and anything that rolls its own risks taking a different subset of
the fields -- working on the isolated requests and failing on the first conversation it meets.

Flattening is defined here, once, so that everything reading a trace agrees on what one request is.
"""
from __future__ import annotations

import os
import prompts


def validate_wire_context(items: list[dict]) -> None:
    for unit in units(items):
        count = int(unit.get("out_tokens", unit.get("max_tokens", 0)))
        cap = int(unit.get("wire_max_tokens") or count)
        size = int(unit["input_tokens"])
        if count <= 0 or cap < count or size < 0 or size + prompts.CTRL_TOKENS + cap > int(unit.get("context_len", 32768)):
            raise ValueError(f"invalid wire context or output cap: {unit['request_id']}")
    for item in items:
        if item.get("kind") != "session":
            continue
        previous = None
        for turn in item["turns"]:
            if "new_user_tokens" not in turn:
                continue  # Small test fixtures may omit it; generated traces always carry both fields.
            shared = int(turn.get("prefix_of_prev_tokens", 0))
            expected = (previous["prompt_tokens"] + prompts.CTRL_TOKENS + previous["out_tokens"]) if previous else 0
            if shared and shared != expected:
                raise ValueError("conversation does not retain exact prior wire history")
            if shared + turn["new_user_tokens"] != turn["prompt_tokens"]:
                raise ValueError("conversation input length disagrees with history")
            previous = turn


def units(items: list[dict]) -> list[dict]:
    """Every individually-served request, flattened out of whatever item carries it.

    A session is one item and many requests; a batch job likewise. Anything that counts requests --
    the summary, the feasibility bound, the scorer's denominator -- has to flatten first or it counts a
    12-turn conversation as one request and reports a fleet doing a twelfth of the work it does.

    CARRIES EVERYTHING, deliberately: a fixed subset would drop fields such as `service_tier` and the
    resolved targets from session and batch units, and every consumer that looks for them would fail on
    the first conversation it met. The three classes also do not name their fields alike --
    `prompt_tokens` against `input_tokens` -- which is exactly why there is one flattener and not one per
    caller.
    """
    out = []
    for it in items:
        kind = it.get("kind", "request")
        if kind == "session":
            for k, t in enumerate(it["turns"]):
                out.append({**t, "input_tokens": t["prompt_tokens"], "model": it["model"],
                            "context_len": it.get("context_len", 32768),
                            "class": "conversational", "tier": it["tier"],
                            # The id LOADGEN will issue this turn under. It is the join key between the
                            # trace and the records; only the isolated-request class carries one in the
                            # trace itself. Any change here must match loadgen._session / _batch exactly
                            # or the scorer silently sees every conversation as unserved.
                            "request_id": f"{it['id']}-t{k}",
                            "weight": 1.0, "turn": k, "session": it["id"]})
        elif kind == "batch":
            for r in it["requests"]:
                out.append({**r, "input_tokens": r["prompt_tokens"], "model": it["model"],
                            "context_len": it.get("context_len", 32768),
                            "request_id": r["id"],
                            "class": "batch", "tier": it["tier"], "weight": 1.0, "job": it["id"],
                            "service_tier": it["service_tier"],
                            "slo_ttft_ms": r.get("slo_ttft_ms", it["slo_ttft_ms"]),
                            "slo_tbt_ms": r.get("slo_tbt_ms", it["slo_tbt_ms"])})
        else:
            out.append(dict(it))
    return out


def units_with_arrival(items: list[dict]) -> list[dict]:
    """`units()`, plus the EARLIEST time each request could be sent.

    Built ON TOP of `units()` rather than beside it: a second flattener would carry its own hand-picked
    subset of fields and drift from the first. There is one definition of what a unit is; this only adds a
    column.

    WHY THE EARLIEST. Two of the three classes declare their arrival. A conversation does not: it is
    closed-loop, so turn N+1 goes out `think_after_s` after turn N *comes back*, and when that happens
    depends on how fast the fleet is. Assume every turn is served instantly and turn k lands at the
    session's start plus the think times before it. Real arrivals can only be later, which spreads the
    work out -- so anything computed from this sees the most concentrated schedule the trace can
    produce, which is the conservative direction for a feasibility bound.
    """
    out = units(items)
    i = 0
    for it in items:
        kind = it.get("kind", "request")
        if kind == "session":
            t = float(it.get("start_s", 0.0))
            for turn in it["turns"]:
                out[i]["arrival_s"] = round(t, 3)
                t += turn.get("think_after_s") or 0.0
                i += 1
        elif kind == "batch":
            for _ in it["requests"]:
                out[i]["arrival_s"] = float(it.get("submit_s", 0.0))
                i += 1
        else:
            i += 1                      # an isolated request already carries its own arrival_s
    return sorted(out, key=lambda r: r.get("arrival_s", 0.0))


# ---------------------------------------------------------------- the deadline, in one place
#
# Both the driver and the scorer need it: the driver to know when to abandon a request, the scorer to
# know how late a late answer was. Two copies of this arithmetic would be two rules, and a request could
# be abandoned under one and credited under the other.
#
# A request whose own e2e budget plus this many seconds elapses is abandoned and counted as not served.
# 
# ADDITIVE, not multiplicative. This is a LIVENESS floor, not a latency target -- latency is scored in
# the served fraction and is not gated at all -- and what a request has to survive is ONE COLD LOAD,
# which costs the same wall-clock whatever the request asked for. That cost is additive, so the slack
# has to be additive too. A multiplier would get it backwards at both ends: it would hand hours to
# requests with large token budgets and less than one load's worth to short chat turns, so the requests
# needing slack would have the least of it.
#
# The VALUE is a published constant of the task and the per-request figure is in the trace. What it
# costs and what completion ceiling it implies are not recorded here: those are properties of a fleet
# under load, and the way to know them is to run against one.
HARD_DEADLINE_SLACK_S = float(os.environ.get("CLUSTER_DEADLINE_SLACK_S", "300.0"))


def e2e_budget_ms(unit: dict) -> float:
    """TTFT plus the token budget. A 1005-token reasoning answer cannot be judged on a chat deadline.

    From the budget the PLANE WAS SHOWN (`wire_max_tokens`), never the true length. A rule you are held
    to has to be one you can evaluate, and the true length is deliberately concealed. The cap is an
    upper bound on the truth -- median 2.25x, p90 5.3x -- so a budget from the true length would leave a
    plane that reads the cap believing it had over twice the slack it is judged on, with the error
    running in the unsafe direction. Using the cap is also strictly more generous, which is right for a
    liveness floor: this is not the latency measure. Latency is scored, not gated."""
    cap = int(unit.get("wire_max_tokens") or unit.get("max_tokens") or unit.get("out_tokens") or 0)
    return float(unit["slo_ttft_ms"]) + cap * float(unit["slo_tbt_ms"])


def deadline_ms(unit: dict) -> float:
    """When the client gives up on this request and stops listening.

    PER REQUEST, from the trace. A single published constant would make every deadline computable from
    fields the plane already holds, so it could abandon work at the last profitable instant and optimise
    against the scoring rule instead of the service. Real callers do not share one timeout. The plane is
    not told this number; it observes a disconnect, which is all a real server gets.

    The constant is only a fallback for trace items without the field.
    """
    slack = unit.get("deadline_slack_s")
    return e2e_budget_ms(unit) + (float(slack) if slack is not None else HARD_DEADLINE_SLACK_S) * 1000.0


def replay_bound_s(items: list[dict], setup_s: float = 900., retries: int = 2) -> float:
    """Conservative fleet-time bound; retries share each logical turn's patience."""
    if setup_s < 0 or retries < 0:
        raise ValueError("invalid replay bounds")
    ends = [0.]
    for item in items:
        flat = units([item])
        if item.get("kind") == "session":
            ends.append(float(item["start_s"]) + sum(
                deadline_ms(u) / 1000 + float(u.get("think_after_s") or 0.) for u in flat))
        elif item.get("kind") == "batch":
            ends.append(float(item["submit_s"]) + max((deadline_ms(u) / 1000 for u in flat), default=0.))
        else:
            ends.append(float(item["arrival_s"]) + deadline_ms(item) / 1000)
    return setup_s + max(ends)
