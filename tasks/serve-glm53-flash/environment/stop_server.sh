#!/bin/bash
# Stop the server started by /app/start_server.sh -- by its PROCESS GROUP (recorded in
# /tmp/server.pid). This kills ONLY the server (and any workers it spawned), never the
# agent harness, so it is always safe.
#
# Do NOT stop the server with a broad `pkill -f` (e.g. `pkill -f server.py`): `pkill -f`
# matches full command lines, and the agent harness's own command line contains the task
# text (which mentions server.py) -- a broad pkill will kill the agent itself and end
# your session. Use this script instead.
PIDFILE=/tmp/server.pid
if [ -f "$PIDFILE" ]; then
  pgid=$(cat "$PIDFILE" 2>/dev/null)
  if [ -n "$pgid" ]; then
    kill -TERM -"$pgid" 2>/dev/null
    for _ in $(seq 1 40); do kill -0 -"$pgid" 2>/dev/null || break; sleep 0.5; done
    kill -KILL -"$pgid" 2>/dev/null
    echo "server stopped (pgid $pgid)"
  else
    echo "empty pidfile; nothing to stop"
  fi
  rm -f "$PIDFILE"
else
  echo "no server pidfile ($PIDFILE); nothing to stop"
fi
