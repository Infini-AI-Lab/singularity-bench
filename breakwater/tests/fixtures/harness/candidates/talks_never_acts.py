import json
def run(task, workspace, model, budget):
    tools = [{"type": "function", "function": {"name": "run_shell_command",
              "description": "run a shell command", "parameters": {"type": "object",
              "properties": {"command": {"type": "string"}}}}},
             {"type": "function", "function": {"name": "write_file",
              "description": "write a file", "parameters": {"type": "object",
              "properties": {"path": {"type": "string"}, "content": {"type": "string"}}}}}]
    msgs = [{"role": "user", "content": task}]
    for _ in range(4):
        if budget.exhausted():
            return
        r = model.complete(msgs, tools=tools)
        m = r["choices"][0]["message"]
        msgs.append({"role": "assistant", "content": m.get("content")})
        for c in (m.get("tool_calls") or []):
            msgs.append({"role": "tool", "tool_call_id": c["id"], "content": "(not executed)"})
