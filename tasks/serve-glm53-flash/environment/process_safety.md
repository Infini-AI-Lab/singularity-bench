# Stopping processes safely (READ THIS before you run `kill`/`pkill`)

You run inside the **same container** as the *harness process that is executing you*
(the agent) and the *shell helper that runs each of your Bash commands*. Broad
process-killing commands can match **those** processes as easily as they match your
server — and if you kill them, **your session ends instantly, mid-task**, with no
chance to finalize a correct server. Three ways to kill yourself by accident:

## Footgun 0 — `pkill python3` (NO `-f`) kills the shell running your commands
Every Bash command you run is executed by a helper process **literally named `python3`**.
So a plain NAME match -- no `-f` anywhere -- is fatal:

```
pkill python3        # ❌ kills the helper running THIS command
pkill -9 python3     # ❌ same, unsurvivable
killall python3      # ❌ same
kill -9 $(pgrep python3)   # ❌ same
```

This is a SEPARATE hazard from Footgun 1 below. Reasoning "I am not using `-f`, so the
command-line matching problem does not apply" is exactly the trap: avoiding `-f` and then
running `pkill -9 python3` returns `Exit code 137` (SIGKILL) and ends the session.

If you want the model server dead, target it SPECIFICALLY -- by recorded PID, by process
group, or by a full path that cannot match anything else (see the safe patterns below).

## Footgun 1 — `pkill -f <name>` matches your own agent process
`pkill -f` (and `pgrep -f`) match against the **entire command line**, not just the
program name. Your agent process is launched roughly as one of:

```
claude ... --print -- "<the full task instructions>"      # Claude Code
codex exec ... -- "<the full task instructions>"          # Codex
```

That means **every word in this task text is on your agent's command line** —
including `server.py`, `simulator.py`, `serve.sh`, `python`, etc. So:

```
pkill -f simulator.py    # ❌ ALSO matches the agent (task text mentions simulator.py) -> kills you
pkill -f server.py       # ❌ same
pkill -f python          # ❌ matches the agent AND the shell running this command -> kills you
```

## Footgun 2 — `ps | grep <pattern> | kill` matches the harness shell
Each Bash command you issue is executed by a small helper process (it shows up in
`ps` as **`python3 -`**). A "kill everything that looks like X" pipeline will sweep
it up too:

```
for p in $(ps aux | grep "[p]ython3 -" | awk '{print $2}'); do kill -9 $p; done   # ❌ kills the shell running THIS command -> kills you
ps aux | grep python | grep -v grep | awk '{print $2}' | xargs kill -9            # ❌ same
pkill -9 python3                                                                   # ❌ same
```

Even the `[p]ython3` bracket trick (which avoids matching the *grep* itself) does
**not** save you here — the victim is the real `python3 -` exec helper, not the grep.

## The correct way to stop the server

**Prefer the provided scripts — they kill only the server's own process group by PID:**

```
bash /app/stop_server.sh        # stop the server (no restart)
bash /app/start_server.sh       # restart-safe: stops the old server, starts a fresh one
```

`/tmp/server.pid` holds the server's process **group** id, written by `start_server.sh`. Do
not overwrite it with a PID of your own: `stop_server.sh` would then report "stopped" while
the server keeps the GPU and port 8000.

**If you must kill manually, be surgical — target the group the script recorded:**

```
kill -TERM -- -"$(cat /tmp/server.pid)"            # ✅ exactly the server's process group

# Or find the server by its exact, fully-qualified command (not a bare word):
pgrep -af 'python3 /app/server.py'                 # ✅ inspect first
pkill -f 'python3 /app/server.py'                  # ✅ full path is specific to the server
```

**Rules of thumb**
- Never `pkill`/`kill` on a bare word that also appears in this task text
  (`server`, `simulator`, `python`, `serve`, `uvicorn`, …) — it can match your agent.
- Never sweep `python3 -` / `python` broadly — that is the shell running your commands.
- Prefer `/app/stop_server.sh` and `/app/start_server.sh`. If killing by hand, kill the
  **recorded process group** or a **full path** (`python3 /app/server.py`), and `pgrep -f` it
  first to confirm you are about to kill *only* the server.
