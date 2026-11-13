"""A competent loop with one defect: it treats a reply carrying no tool call as completion.

Not a strawman -- it is the natural mistake, and the reasoning behind it is superficially sound: with
a scripted client the only no-call step IS the end of the script. With a real model a no-call turn is
usually a plan, a question, or a turn spent thinking aloud, and ending there throws the run away.
"""
import json, os, subprocess

TOOLS = [
    {"type": "function", "function": {"name": "read_file", "description": "read a file",
     "parameters": {"type": "object", "properties": {"path": {"type": "string"}}}}},
    {"type": "function", "function": {"name": "write_file", "description": "write a file",
     "parameters": {"type": "object", "properties": {"path": {"type": "string"},
                                                     "content": {"type": "string"}}}}},
    {"type": "function", "function": {"name": "run_shell_command", "description": "run a command",
     "parameters": {"type": "object", "properties": {"command": {"type": "string"}}}}},
]


def run(task, workspace, model, budget):
    msgs = [{"role": "system", "content": "You are a coding agent."},
            {"role": "user", "content": task}]
    for _ in range(30):
        if budget.exhausted():
            return
        try:
            r = model.complete(msgs, tools=TOOLS)
        except BaseException:
            return
        m = (r.get("choices") or [{}])[0].get("message") or {}
        calls = m.get("tool_calls") or []
        msgs.append({"role": "assistant", "content": m.get("content"), "tool_calls": calls})
        if not calls:
            return                      # <-- the defect
        for c in calls:
            fn = c.get("function") or {}
            try:
                args = json.loads(fn.get("arguments") or "{}")
            except ValueError:
                args = {}
            name = fn.get("name")
            try:
                if name == "read_file":
                    out = open(os.path.join(workspace, args["path"]), errors="replace").read()
                elif name == "write_file":
                    p = os.path.join(workspace, args["path"])
                    os.makedirs(os.path.dirname(p) or workspace, exist_ok=True)
                    open(p, "w").write(args.get("content", ""))
                    out = "written"
                elif name == "run_shell_command":
                    out = subprocess.run(args.get("command", ""), shell=True, cwd=workspace,
                                         capture_output=True, text=True, timeout=30).stdout
                else:
                    out = "unknown tool"
            except BaseException as e:
                out = f"error: {type(e).__name__}"
            msgs.append({"role": "tool", "tool_call_id": c.get("id", "x"),
                         "content": "<tool_output>\n" + str(out)[:4000] + "\n</tool_output>"})
