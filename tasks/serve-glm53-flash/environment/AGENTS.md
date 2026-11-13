# Working in this environment

## ⚠️ CRITICAL — stop/restart your server without killing yourself

You are launched with the task text on your own command line — `claude --print -- '<the task
instruction>'` under Claude Code, `codex exec … -- '<the task instruction>'` under Codex — so the
task text, **including `/app/server.py`**, is on **your own** process's argv. Any command that finds processes by
matching that text will match and can **`kill` the agent itself**, ending your run
and grading you on an unfinished server.

**Never** do these. There are TWO separate ways to kill yourself, and avoiding one
does NOT protect you from the other:

**(a) matching the COMMAND LINE** — `pkill -f` searches full argv, and your own agent
process is `claude … -- '<the task text>'` or `codex exec … -- '<the task text>'`, so any
word from this task matches it:
- `pkill -f server.py` / `pkill -f uvicorn` / `pkill -f python` / `pkill -f simulator.py`
- `pgrep -f server.py | xargs kill` / `for p in $(pgrep -f server.py); do kill $p; done`
- a `/proc` scan that greps cmdlines for `*server.py*` and kills matches
- `fuser -k`, or `kill`-ing PIDs from an `nvidia-smi` name match

**(b) matching the PROCESS NAME** — this needs no `-f` at all. Every Bash command you
run is executed by a helper process literally named `python3`, so a bare name match
kills the shell running your own command (and can take the session with it):
- `pkill python3` / `pkill -9 python3` / `killall python3` / `kill -9 $(pgrep python3)`

Heeding the `-f` warnings, avoiding `-f`, and then running `pkill -9 python3` gets your own
shell SIGKILLed (exit 137) and ends the session. Dropping `-f` does not make it safe -- it
makes it worse, because the match is now on the very name your shell helper uses.

**Always** manage your server with the two provided scripts. They launch `/app/serve.sh` —
the exact command the grader runs — in its own process group, record that group in
`/tmp/server.pid`, and stop it by that group and nothing else:
```bash
bash /app/start_server.sh    # stops any server it started before, then starts a fresh one (logs: /tmp/server.log)
bash /app/stop_server.sh     # stops it
```
- Do **not** start the server by hand with `python3 /app/server.py &` and write its PID to
  `/tmp/server.pid` yourself: that file holds a process GROUP id, a bare PID in it makes
  `stop_server.sh` report "stopped" while the server keeps the GPU and port 8000, and a server
  that does not come up through `serve.sh` is not the server the grader will launch.
- **Free the GPU** with `stop_server.sh` and `sleep 5` — make the server release CUDA on SIGTERM.
  Do **not** sweep `/proc` or `pkill` by name to reclaim VRAM.
- If you must enumerate processes, match a **unique token that is NOT anywhere in this prompt**
  (give your server a unique name) and **exclude any cmdline containing `claude`, `codex`,
  `--print` or `exec`**;
  `pgrep -af` and look before you kill.

One such command ends your run on the spot. Full details + recipes: **`/app/process_safety.md`**.
