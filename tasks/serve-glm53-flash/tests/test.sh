#!/usr/bin/env bash
# Harbor verifier entry point. This task runs only under Breakwater; inside its verifier stage this hands over
# to the Breakwater verifier.
set -euo pipefail
[[ "${BREAKWATER_STAGE:-}" == verifier ]] || { echo 'Launch this task with breakwater/run.py: the agent, grading and the verifier run in three separate containers with sealed external evidence.' >&2; exit 2; }
[[ -n "${BREAKWATER_AUTHORITY_SHA256:-}" ]] || { echo 'Missing host authority digest' >&2; exit 2; }
exec python3 -I /opt/breakwater/verifier.py --authority "$BREAKWATER_AUTHORITY_SHA256"
