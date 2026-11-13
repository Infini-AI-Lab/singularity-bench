r"""A ModelClient whose replies are fixed in advance, and which records everything it was asked.

Used two ways: by `selfcheck.py`, so an author can watch their harness work, and by the verifier, to
decide whether a submission is a working harness at all before a GPU grid is spent on it. It never
produces a score.

The harness names its own tools, so a scenario cannot script a call by tool name. It scripts an
*intent* -- "run this shell command", "write this file" -- and maps that onto whatever schema the
harness actually declared. A harness that describes its tools sensibly is driveable; one that does not
is a harness a real model could not drive either.

**The reply is a function of the conversation, never of a call counter.** Indexing a script by call
ordinal quietly breaks every harness that does something reasonable and unanticipated: an opening
planning probe, a classifier pass, an LLM-based compaction call, a hedged retry. Each shifts the whole
script by one, and the harness is then judged on a conversation that never happened. What this client
proposes next is the first scripted step the conversation does not already show as requested, so extra
probes, retries, reordering and a summarised history are harmless by construction rather than by
detection.

Deliberately small: the projection, the intent-to-schema mapping, the prose dialects and one
transport fault. That is everything a validity check needs and nothing more -- it is not a grader.
"""
from __future__ import annotations

import json
import re
from typing import Any

SHELL = re.compile(r"(?i)\b(bash|shell|sh|exec|run|command|terminal|cmd)\b")
WRITE = re.compile(r"(?i)\b(write|edit|create|save|patch|apply|modify)\b")
READ = re.compile(r"(?i)\b(read|cat|open|view|get|show)\b")
KIND = {"shell": SHELL, "write": WRITE, "read": READ}


def _fn(t: dict) -> dict:
    return t.get("function", t) if isinstance(t, dict) else {}


def pick_tool(tools: list[dict], kind: str) -> tuple[str, dict] | None:
    """The declared tool that best matches an intent, by name first then description."""
    pat = KIND[kind]
    best = None
    for t in tools or []:
        f = _fn(t)
        name, desc = str(f.get("name", "")), str(f.get("description", ""))
        score = (2 if pat.search(name) else 0) + (1 if pat.search(desc) else 0)
        if score and (best is None or score > best[0]):
            best = (score, name, f.get("parameters") or {})
    return (best[1], best[2]) if best else None


def _arg_names(params: dict) -> list[str]:
    return list((params.get("properties") or {}).keys())


def build_call(tools: list[dict], step: dict) -> dict | None:
    """Turn a scripted intent into a tool call in the harness's own vocabulary."""
    got = pick_tool(tools, step["intent"])
    if not got:
        return None
    name, params = got
    names = _arg_names(params)
    payload: dict[str, Any] = {}
    if step["intent"] == "shell":
        key = next((n for n in names if re.search(r"(?i)cmd|command|script|input", n)),
                   names[0] if names else "command")
        payload[key] = step["command"]
    elif step["intent"] == "write":
        pk = next((n for n in names if re.search(r"(?i)path|file", n)), "path")
        ck = next((n for n in names if re.search(r"(?i)content|text|body|data|new", n)), "content")
        payload[pk], payload[ck] = step["path"], step["content"]
    else:
        pk = next((n for n in names if re.search(r"(?i)path|file", n)), "path")
        payload[pk] = step["path"]
    return {"id": f"call_{step.get('n', 0)}", "type": "function",
            "function": {"name": name, "arguments": json.dumps(payload)}}


def render_as_text(tools: list[dict], step: dict) -> str | None:
    """The same call, written the way a model that does not emit structured calls writes it.

    A harness that only reads `message.tool_calls` sees nothing here and does nothing. That is not a
    validity failure -- it is reported separately -- but it predicts trouble with a driver whose
    tool-call dialect is its own, which is exactly what one of the three evaluation drivers is.
    """
    got = pick_tool(tools, step["intent"])
    if not got:
        return None
    name, _params = got
    call = build_call(tools, step)
    args = json.loads(call["function"]["arguments"])
    d = step.get("dialect", "xml")
    if d == "xml":
        inner = "".join(f"<{k}>{v}</{k}>" for k, v in args.items())
        return f"I will do that now.\n<tool_call><name>{name}</name>{inner}</tool_call>"
    if d == "bare_json":
        return json.dumps({"tool": name, "parameters": args})
    if d == "func_syntax":
        inner = ", ".join(f"{k}={json.dumps(v)}" for k, v in args.items())
        return f"Calling {name}({inner}) to make progress."
    if d == "react":
        return "Thought: I should act.\nAction: " + name + "\nAction Input: " + json.dumps(args)
    return None


def step_signature(step: dict) -> tuple:
    """What distinguishes one scripted step from another, once the harness has asked for it.

    Matched on argument VALUES, never tool names -- the harness names its own tools. A read and a write
    of the same file share a path, so the write also carries its content; otherwise the read appearing
    in the conversation marks the write as already requested and the file never gets written.
    """
    if step.get("intent") == "shell":
        return ("shell", str(step.get("command", "")), "")
    if step.get("intent") == "write":
        return ("write", str(step.get("path", "")), str(step.get("content", ""))[:40])
    return ("read", str(step.get("path", "")), "")


def _realises(sig: tuple, text: str) -> bool:
    kind, a, b = sig
    if not a:
        return False

    def present(v):
        return bool(v) and (v in text or json.dumps(v)[1:-1] in text)

    if kind == "write":
        return present(a) and present(b)
    return present(a)


class ScriptedClient:
    """Replies come from the scenario. Every request is recorded verbatim."""

    def __init__(self, script: list[dict], faults: dict | None = None) -> None:
        self.script, self.faults = list(script), faults or {}
        self.proposed: set[int] = set()
        self.calls: list[dict] = []
        self.fired: set[str] = set()
        self.permanent = False
        self.calls_after_first_fault: int | None = None
        self.saw_prose_step = False
        self.acted_on_prose = False

    def requested(self, messages: list[dict]) -> set[int]:
        """Which scripted steps this conversation already shows the harness asking for."""
        blob = []
        for m in messages or []:
            for c in (m.get("tool_calls") or []):
                blob.append(str((c.get("function") or {}).get("arguments", "")))
            if m.get("role") == "assistant" and m.get("content"):
                blob.append(str(m["content"]))          # a call written in prose counts too
        text = "\n".join(blob)
        return {i for i, st in enumerate(self.script)
                if st.get("intent") in KIND and _realises(step_signature(st), text)}

    def _next_index(self, messages) -> int:
        asked = self.requested(messages) | self.proposed
        for i, _st in enumerate(self.script):
            if i not in asked:
                return i
        return len(self.script)

    def complete(self, messages: list[dict], tools: list[dict] | None = None, **kw: Any) -> dict:
        self.calls.append({"messages": json.loads(json.dumps(messages, default=str)),
                           "tools": json.loads(json.dumps(tools or [], default=str))})
        if self.calls_after_first_fault is not None:
            self.calls_after_first_fault += 1

        # A call that offers no tools is not asking for the next action -- it is a probe, a
        # classification, a summarisation. Answering it must not consume a scripted step, or a harness
        # that thinks before it acts runs out of script before it starts.
        if not tools:
            return {"choices": [{"message": {"role": "assistant", "content": "Understood."},
                                 "finish_reason": "stop"}]}

        # Did the harness act on the prose-dialect step it was handed earlier? Recorded, never gated.
        if self.saw_prose_step and not self.acted_on_prose:
            for i, st in enumerate(self.script):
                if st.get("dialect") and i in self.proposed and i in self.requested(messages):
                    self.acted_on_prose = True

        n = self._next_index(messages)
        progress = len(self.requested(messages))

        # Faults are keyed to PROGRESS -- "after the step carrying X has landed" -- not to a call
        # ordinal, so a harness that probes first or retries meets the same fault at the same point.
        key = str(progress)
        if self.permanent:
            raise RuntimeError("simulated permanent upstream failure (do not retry)")
        fault = self.faults.get(key)
        if fault in ("error", "permanent") and key not in self.fired:
            self.fired.add(key)
            if self.calls_after_first_fault is None:
                self.calls_after_first_fault = 0
            if fault == "permanent":
                self.permanent = True
                raise RuntimeError("simulated permanent upstream failure (do not retry)")
            raise ConnectionError("simulated upstream failure")

        if n < len(self.script):
            self.proposed.add(n)
        step = self.script[n] if n < len(self.script) else {"intent": "stop", "content": "done"}
        msg: dict[str, Any] = {"role": "assistant", "content": step.get("content")}
        if step.get("intent") in KIND and step.get("dialect"):
            self.saw_prose_step = True
            text = render_as_text(tools or [], dict(step, n=n))
            msg["content"] = (msg.get("content") or "") + ("\n" + text if text else "")
        elif step.get("intent") in KIND:
            call = build_call(tools or [], dict(step, n=n))
            if call:
                msg["tool_calls"] = [call]
            else:
                msg["content"] = ((msg.get("content") or "")
                                  + "\n[no tool matching intent %s was offered]" % step["intent"])
        return {"choices": [{"message": msg, "finish_reason": "stop"}]}

    def sent_text(self) -> str:
        out = []
        for c in self.calls:
            for m in c["messages"]:
                out.append(str(m.get("content", "")))
                for tc in (m.get("tool_calls") or []):
                    out.append(str((tc.get("function") or {}).get("arguments", "")))
        return "\n".join(out)

    def declared_tools(self) -> list[str]:
        return sorted({str(_fn(t).get("name", "")) for c in self.calls for t in c["tools"] if _fn(t)})
