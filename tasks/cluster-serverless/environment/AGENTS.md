# Process management

Before stopping or restarting the local control plane, read
`/app/process_safety.md`.

Your agent and command helpers share the container with your server. Their
command lines may contain the full task instructions, including server paths.
`pkill -f`, `pgrep -f ... | xargs kill`, and process-name killing pipelines can
therefore kill your own coding session. A full-path pattern is not sufficient.
`pkill python3` and `killall python3` can also kill Python command helpers.

Use these development commands:

```bash
bash /app/start_control_plane.sh  # restart safely; launches /app/run.sh
bash /app/stop_control_plane.sh   # stop only its recorded launch and children
```

Logs are at `/tmp/cluster-control-plane/control-plane.log`. Keep `run.sh` in the
foreground; do not call the development helpers from it. Do not edit the helpers'
state file. Remote GPU engines are separate: remove them through the controller's
`POST /unload` for a specific worker ID, not by killing local processes.
