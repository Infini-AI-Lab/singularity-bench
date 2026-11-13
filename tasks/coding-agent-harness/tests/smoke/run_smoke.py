#!/usr/bin/env python3
r"""Drive a harness through a handful of ordinary requests and report what it did.

    python3 run_smoke.py <agent.py> [scenarios.json] [--json]

This answers one question: **is this a working harness?** It never produces a score. The score for this
task comes from the evaluation grid -- real tasks, three drivers -- and nothing here is folded into it.

Why it exists. Grading a submission means building multi-gigabyte images and running a grid of GPU
cells. A submission that will not import, defines no `run()`, or raises on the first turn burns all of
that and returns 0.000 in every cell -- a result indistinguishable from a harness that ran fine and found
the work too hard. Five seconds of CPU here tells the two apart.

Three scenarios decide validity: a harness has to import, loop, offer tools, do the work it was asked
to do, and not raise. Three more are **reported and never gated** -- acting on a call written in prose,
recovering from a transport fault, stopping when the budget runs out. Those are quality properties, and
this is not the place they are scored: a harness that only reads structured tool calls is still a
working harness, it just predicts trouble with the one evaluation driver whose dialect is its own.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

MARKER = "\x00SMOKE\x00"
PER_SCENARIO_TIMEOUT = int(os.environ.get("SMOKE_TIMEOUT_SEC", "300"))
# A hang catcher, not a budget: a harness with a runaway loop would otherwise sit here until the
# subprocess timeout, seven times over. It raises inside complete(), which the harness sees as an
# ordinary model failure. Nothing about the graded run limits calls.
HARD_CALL_CAP = 400
# What calls_left()/tokens_left() report. Neither is limited, here or at grading time.
UNBOUNDED = 10 ** 9


class _Budget:
    """Time, and only time -- the same surface a graded run hands the harness."""

    def __init__(self, seconds: float = 240.0) -> None:
        self.deadline = time.time() + seconds

    def seconds_left(self) -> float:
        return max(0.0, self.deadline - time.time())

    def exhausted(self) -> bool:
        return self.seconds_left() <= 0

    # Compatibility only -- see harness_api.Budget.
    def calls_left(self) -> int:
        return UNBOUNDED

    def tokens_left(self) -> int:
        return UNBOUNDED


def _child(payload_path: str) -> int:
    """Run one scenario in its own process, so a crash is a data point rather than the end of the run."""
    from scripted import ScriptedClient

    sc = json.load(open(payload_path))
    ws = sc["ws"]
    out: dict = {"id": sc["id"], "raised": None, "n_calls": 0, "tools": [],
                 "imported": False, "has_run": False}
    sys.path.insert(0, os.path.dirname(os.path.abspath(sc["module"])))
    import importlib.util
    spec = importlib.util.spec_from_file_location("submitted_harness", sc["module"])
    mod = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(mod)
    except BaseException as e:                                       # noqa: BLE001
        out["fatal"] = f"import failed: {type(e).__name__}: {e}"[:300]
        print(MARKER + json.dumps(out))
        return 0
    out["imported"] = True
    fn = getattr(mod, "run", None)
    if not callable(fn):
        out["fatal"] = "agent.py defines no callable run()"
        print(MARKER + json.dumps(out))
        return 0
    out["has_run"] = True

    client = ScriptedClient(sc["script"], sc.get("faults"))
    budget = _Budget(float(sc.get("seconds", 240)))
    real = client.complete
    seen = [0]

    def counted(messages, tools=None, _r=real, **kw):
        seen[0] += 1
        if seen[0] > HARD_CALL_CAP:
            raise RuntimeError(f"smoke call cap reached ({HARD_CALL_CAP})")
        return _r(messages, tools, **kw)

    client.complete = counted                                        # type: ignore[method-assign]
    try:
        fn(sc["task"], ws, client, budget)
    except BaseException as e:                                       # noqa: BLE001
        out["raised"] = f"{type(e).__name__}: {e}"[:300]
    out["n_calls"] = len(client.calls)
    out["tools"] = client.declared_tools()
    out["acted_on_prose"] = bool(client.acted_on_prose)
    print(MARKER + json.dumps(out))
    return 0


def _check(sc: dict, ws: str, rec: dict) -> dict:
    """Did the requested work actually happen on disk?"""
    e = sc.get("expect") or {}
    checks: dict = {}
    for rel, want in (e.get("file_is") or {}).items():
        try:
            checks[f"file_is:{rel}"] = open(os.path.join(ws, rel)).read() == want
        except OSError:
            checks[f"file_is:{rel}"] = False
    for rel in e.get("file_exists") or []:
        checks[f"exists:{rel}"] = os.path.exists(os.path.join(ws, rel))
    if e.get("file_contains"):
        rel, needle = e["file_contains"]
        try:
            checks[f"contains:{rel}"] = needle in open(os.path.join(ws, rel), errors="replace").read()
        except OSError:
            checks[f"contains:{rel}"] = False
    if e.get("max_model_calls") is not None:
        checks["within_budget"] = int(rec.get("n_calls", 10 ** 6)) <= int(e["max_model_calls"])
    return checks


def run_one(sc: dict, module: str, root: str) -> dict:
    work = tempfile.mkdtemp(dir=root)
    ws = os.path.join(work, "ws")
    os.makedirs(ws, exist_ok=True)
    for rel, body in (sc.get("files") or {}).items():
        p = os.path.join(ws, rel)
        os.makedirs(os.path.dirname(p) or ws, exist_ok=True)
        with open(p, "w") as f:
            f.write(body)
    payload = dict(sc, ws=ws, module=os.path.abspath(module))
    pf = os.path.join(work, "sc.json")
    json.dump(payload, open(pf, "w"))
    try:
        p = subprocess.run([sys.executable, os.path.abspath(__file__), "--child", pf],
                           capture_output=True, text=True, timeout=PER_SCENARIO_TIMEOUT)
        lines = [l for l in (p.stdout or "").splitlines() if l.startswith(MARKER)]
        rec = json.loads(lines[-1][len(MARKER):]) if lines else {
            "id": sc["id"], "imported": False,
            "fatal": ("no result; stderr: " + (p.stderr or "")[-300:]).strip()}
    except subprocess.TimeoutExpired:
        rec = {"id": sc["id"], "imported": True,
               "fatal": f"run() did not return within {PER_SCENARIO_TIMEOUT}s"}
    rec["checks"] = _check(sc, ws, rec)
    rec["reports"] = sc.get("reports")
    return rec


def summarise(rows: list[dict]) -> dict:
    """Five signals decide validity; the rest are recorded so a zero is diagnosable."""
    gated = [r for r in rows if not r.get("reports")]
    imported = all(r.get("imported") for r in rows) and bool(rows)
    no_fatal = not any(r.get("fatal") for r in rows)
    # Raising is judged on the GATED scenarios only. The reported-only three inject a transport fault
    # and starve the budget on purpose; a harness that completes ordinary work but lets an injected
    # ConnectionError escape has a real defect -- the spec says `run` must not raise -- but it is a
    # harness, and the grid is what prices the defect. Counting those rows here would refuse to grade
    # a submission over the very property the grid measures.
    did_not_raise = not any(r.get("raised") for r in gated)
    offered_tools = any(r.get("tools") for r in rows)
    looped = any(int(r.get("n_calls") or 0) >= 2 for r in rows)
    did_work = any(all(r["checks"].values()) for r in gated if r.get("checks"))

    # Requirement 11, and the single cheapest thing to check about a harness. Breaking the loop on
    # `if not tool_calls` is the natural mistake and a costly one: with a real model, a reply carrying no
    # tool call is usually a plan or a question, and a run that stops on it has done nothing.
    prose = next((r for r in rows if r.get("id") == "prose_turn_is_not_done"), None)
    continues_past_prose = bool(prose and prose.get("checks")
                                and all(prose["checks"].values()) and not prose.get("raised"))

    def _rep(name: str) -> float:
        r = next((x for x in rows if x.get("reports") == name), None)
        if not r or r.get("fatal"):
            return 0.0
        return 1.0 if (r.get("checks") and all(r["checks"].values()) and not r.get("raised")) else 0.0

    # GRADEABLE is the only signal that should ever stop a grid, and it is structural: each grid cell
    # imports the submission and calls run(). If that import fails or there is no callable run(), it
    # cannot call anything, so every cell is a guaranteed zero and the GPUs would be spent proving
    # it. Everything else below is a REPORTED diagnostic: a harness that does not loop, offer tools
    # or edit a file in the *scripted* smoke may still do so on a real repository -- the smoke is a
    # proxy, and a proxy must not decide the score.
    gradeable = bool(rows) and all(r.get("imported") and r.get("has_run") for r in rows)
    valid = all([imported, no_fatal, did_not_raise, offered_tools, looped, did_work,
                 continues_past_prose])
    return {
        "gradeable": 1.0 if gradeable else 0.0,
        "valid": 1.0 if valid else 0.0,
        "imported": float(imported), "no_fatal_scenario": float(no_fatal),
        "did_not_raise": float(did_not_raise), "offered_tools": float(offered_tools),
        "looped": float(looped), "did_the_work": float(did_work),
        "continues_past_prose": float(continues_past_prose),
        "gated_scenarios_passed": float(sum(1 for r in gated
                                            if r.get("checks") and all(r["checks"].values()))),
        "gated_scenarios_total": float(len(gated)),
        # Reported, never gated. See the module docstring.
        "handles_prose_calls": _rep("handles_prose_calls"),
        "recovers_from_fault": _rep("recovers_from_fault"),
        "stops_when_exhausted": _rep("stops_when_exhausted"),
        "distinct_tools_offered": float(len({t for r in rows for t in (r.get("tools") or [])})),
        "model_calls_total": float(sum(int(r.get("n_calls") or 0) for r in rows)),
    }


def main(argv: list[str]) -> int:
    if len(argv) > 1 and argv[1] == "--child":
        return _child(argv[2])
    module = argv[1] if len(argv) > 1 else "/app/harness/agent.py"
    cases = argv[2] if len(argv) > 2 and not argv[2].startswith("-") \
        else os.path.join(HERE, "scenarios.json")
    as_json = "--json" in argv
    if not os.path.exists(module):
        print(json.dumps({"valid": 0.0, "error": f"no submission at {module}"}))
        return 0
    root = os.environ.get("SMOKE_SCRATCH", tempfile.mkdtemp(prefix="smoke_"))
    os.makedirs(root, exist_ok=True)
    rows = [run_one(sc, module, root) for sc in json.load(open(cases))]
    summary = summarise(rows)
    if as_json:
        print(json.dumps({"summary": summary, "detail": rows}, indent=1))
        return 0
    print(f"harness: {module}\n")
    for r in rows:
        tag = "reported" if r.get("reports") else "validity"
        ok = "ok " if (r.get("checks") and all(r["checks"].values())
                       and not r.get("fatal") and not r.get("raised")) else "FAIL"
        note = r.get("fatal") or r.get("raised") or ""
        failed = [k for k, v in (r.get("checks") or {}).items() if not v]
        print(f"  [{ok}] {r['id']:24} ({tag})  calls={r.get('n_calls', 0):>3}"
              + (f"  {note}" if note else "")
              + (f"  unmet: {', '.join(failed)}" if failed and not note else ""))
    print(f"\n  gradeable = {summary['gradeable']:.0f}"
          + ("   (imports, and run() is callable)" if summary["gradeable"]
             else "   <-- a grid cell cannot call anything; every cell would be a certain zero"))
    print(f"  valid     = {summary['valid']:.0f}"
          + ("" if summary["valid"] else "   (a diagnostic, not a gate -- grading proceeds)"))
    if not summary["continues_past_prose"]:
        print("  it ended the run on a turn that asked for no tool. See requirement 11: with a real\n"
              "  model that turn is usually a plan or a question, not a completion signal.")
    print("  reported only: prose-dialect calls %.0f | fault recovery %.0f | stops when out of time"
          " %.0f" % (summary["handles_prose_calls"], summary["recovers_from_fault"],
                     summary["stops_when_exhausted"]))
    print("\nThese are ordinary requests. They are not the graded set, and passing them is not a score.")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
