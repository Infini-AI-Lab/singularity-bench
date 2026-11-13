#!/usr/bin/env python3
r"""Drive a harness with a REAL model, so you can watch it meet one that does not cooperate.

    python3 live.py <agent.py> [--model chat|chat-b] [--calls 30]

The scripted smoke next door is deterministic and free; it tells you the harness works. This tells you
something the scripted client cannot: what your harness does when the model paraphrases the task,
answers in prose, calls a tool you did not define, or produces nothing useful for three turns. That is
the situation it will be graded in.

Two honest caveats. This spends from the same metered allowance as the harness's own endpoint calls,
because it is the same gateway. And the models reachable here are two of the three that will drive you
at grading time -- the third is not served here, deliberately.

Nothing here is scored. It prints what happened and stops.
"""
from __future__ import annotations

import json
import os
import shutil
import sys
import tempfile
import time
import urllib.error
import urllib.request

BASE = os.environ.get("HARNESS_LLM_BASE_URL", "http://gateway:8000/v1").rstrip("/")

# A small repo with a real defect: `total()` sums the wrong column, and the test says so. Deliberately
# tiny -- the point is to watch the loop, not to be a benchmark.
REPO = {
    "billing.py": (
        "ROWS = [\n"
        "    {'item': 'widget', 'qty': 3, 'price': 4.0},\n"
        "    {'item': 'gadget', 'qty': 1, 'price': 9.5},\n"
        "]\n\n\n"
        "def total(rows):\n"
        "    # BUG: ignores qty\n"
        "    return sum(r['price'] for r in rows)\n"
    ),
    "test_billing.py": (
        "from billing import ROWS, total\n\n\n"
        "def test_total():\n"
        "    assert total(ROWS) == 21.5\n"
    ),
    "README.md": "# billing\n\nRun `python3 -m pytest -q` to check.\n",
}
TASK = ("The test in test_billing.py fails. Find out why, fix billing.py so the test passes, "
        "and do not change the test.")


class LiveModel:
    """The same surface as the scripted client, backed by the served endpoint."""

    def __init__(self, model: str, max_calls: int | None, verbose: bool = True) -> None:
        self.model, self.max_calls, self.verbose = model, max_calls, verbose
        self.calls = 0
        self.tokens = 0
        self.denied = 0

    def complete(self, messages, tools=None, **kw):
        if self.max_calls is not None and self.calls >= self.max_calls:
            raise RuntimeError("budget exhausted")
        if 'model' in kw:
            raise ValueError('The driver model is assigned by the evaluator')
        body = {**kw, "model": self.model, "messages": messages}
        if not {'max_tokens', 'max_completion_tokens'} & body.keys():
            body['max_tokens'] = 4096
        if tools:
            body['tools'] = tools
            body.setdefault('tool_choice', 'auto')
        self.calls += 1
        req = urllib.request.Request(BASE + "/chat/completions", data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=300) as r:
                out = json.loads(r.read())
        except urllib.error.HTTPError as e:
            detail = e.read()[:200].decode("utf-8", "replace")
            if e.code == 429:
                self.denied += 1
            # One attempt. A retry the harness did not ask for would hide the fault handling this is
            # meant to show you.
            raise RuntimeError(f"model endpoint returned {e.code}: {detail}") from None
        except Exception as e:                                       # noqa: BLE001
            raise RuntimeError(f"{type(e).__name__}: {e}") from None
        self.tokens += int((out.get("usage") or {}).get("total_tokens") or 0)
        if self.verbose:
            msg = ((out.get("choices") or [{}])[0].get("message") or {})
            names = [c.get("function", {}).get("name") for c in (msg.get("tool_calls") or [])]
            head = (msg.get("content") or "").strip().replace("\n", " ")[:90]
            print(f"  [{self.calls:>3}] sent {len(messages):>3} msgs"
                  f"  {sum(len(str(m.get('content') or '')) for m in messages):>7} chars"
                  f"  -> {('calls ' + ','.join(n for n in names if n)) if names else 'text'}"
                  + (f"  {head!r}" if head else ""))
        return out


class LiveBudget:
    def __init__(self, m: LiveModel, seconds: float = 1800.0) -> None:
        self.m = m
        self.deadline = time.time() + seconds

    def seconds_left(self) -> float:
        return max(0.0, self.deadline - time.time())

    def exhausted(self) -> bool:
        return self.seconds_left() <= 0 or (self.m.max_calls is not None and self.m.calls >= self.m.max_calls)

    # Compatibility only -- see harness_api.Budget. `--calls` here is a cost guard on a local
    # rehearsal, not something the graded run has.
    def calls_left(self) -> int:
        return 10 ** 9

    def tokens_left(self) -> int:
        return 10 ** 9


def main(argv: list[str]) -> int:
    module = argv[1] if len(argv) > 1 else "/app/harness/agent.py"
    model = "chat"
    max_calls = 30
    for i, a in enumerate(argv):
        if a == "--model" and i + 1 < len(argv):
            model = argv[i + 1]
        if a == "--calls" and i + 1 < len(argv):
            max_calls = int(argv[i + 1])
    if not os.path.exists(module):
        print(f"no harness at {module}")
        return 1

    work = tempfile.mkdtemp(prefix="live_")
    ws = os.path.join(work, "ws")
    os.makedirs(ws)
    for rel, body in REPO.items():
        with open(os.path.join(ws, rel), "w") as f:
            f.write(body)
    before = {rel: body for rel, body in REPO.items()}

    sys.path.insert(0, os.path.dirname(os.path.abspath(module)))
    import importlib.util
    spec = importlib.util.spec_from_file_location("submitted_harness", module)
    mod = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(mod)
    except BaseException as e:                                       # noqa: BLE001
        print(f"import failed: {type(e).__name__}: {e}")
        return 1
    fn = getattr(mod, "run", None)
    if not callable(fn):
        print("agent.py defines no callable run()")
        return 1

    print(f"model={model}  endpoint={BASE}  budget={max_calls} calls  workspace={ws}\n")
    m = LiveModel(model, max_calls)
    t0 = time.time()
    raised = None
    try:
        fn(TASK, ws, m, LiveBudget(m))
    except BaseException as e:                                       # noqa: BLE001
        raised = f"{type(e).__name__}: {e}"

    print(f"\n  {m.calls} model calls, {m.tokens} tokens, {time.time() - t0:.0f}s"
          + (f", 429s: {m.denied}" if m.denied else "")
          + (f"\n  run() raised: {raised}   <-- the spec says it must not" if raised else ""))
    changed = []
    for root, _d, files in os.walk(ws):
        for fnm in files:
            p = os.path.join(root, fnm)
            rel = os.path.relpath(p, ws)
            try:
                now = open(p, errors="replace").read()
            except OSError:
                continue
            if before.get(rel) != now:
                changed.append(rel)
    print(f"  files changed: {', '.join(sorted(changed)) if changed else '(none — nothing was done)'}")
    if "billing.py" in changed:
        print("\n  --- billing.py now ---")
        print("  " + open(os.path.join(ws, "billing.py"), errors="replace").read().replace("\n", "\n  "))
    if "test_billing.py" in changed:
        print("  NOTE: the test was modified. The task said not to.")
    print(f"\n  workspace kept at {ws} — inspect it, then remove it.")
    shutil.rmtree(work, ignore_errors=True) if os.environ.get("LIVE_CLEAN") else None
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
