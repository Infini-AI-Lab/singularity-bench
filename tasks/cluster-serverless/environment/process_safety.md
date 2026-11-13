# Start and stop the control plane safely

Your coding agent, command helpers and local control plane share a container.
Broad process-killing commands can terminate your coding session immediately.

`pkill -f` searches entire command lines. An agent or parent shell may carry the
task instructions as a command-line argument, including names such as
`control_plane.py` and even full paths. Matching a full path does not make this
safe. `pgrep -f ... | xargs kill` has the same problem.

Dropping `-f` does not solve it: `pkill python3` and `killall python3` can kill
Python command helpers. Avoid name-based cleanup, process-list killing pipelines,
and automatic port-owner killing.

Use the provided development helpers:

```bash
bash /app/start_control_plane.sh  # stops its previous launch, then runs /app/run.sh
curl -fsS http://127.0.0.1:8000/health
tail -n 100 /tmp/cluster-control-plane/control-plane.log
bash /app/stop_control_plane.sh
```

Startup detaches the control plane into its own session and returns promptly.
Only that launch and its children receive a fresh identity in their environment.
Stopping sends TERM, allows five seconds to exit, then sends KILL to remaining
processes carrying that exact identity. Signals use stable process handles, so
reused PIDs do not select unrelated processes. Concurrent starts/stops serialize.
Do not edit `/tmp/cluster-control-plane/state.json` or reuse its identity elsewhere.

These helpers only manage processes they launched. If you manually launch another
experiment, retain its process handle or exact PID and terminate only that owned
process. Do not discover kill targets by searching names or prompt text.

GPU engines live on the remote fleet. Stopping the local control plane does not
unload them. Inspect `GET /workers` and use the controller's `POST /unload` with
the specific worker ID when an engine should be removed.

Keep `/app/run.sh` as the foreground serving entrypoint. The start/stop helpers
are development conveniences; do not invoke them from `run.sh`.
