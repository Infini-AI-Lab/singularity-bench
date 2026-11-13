#!/usr/bin/env bash
# Harbor verifier entry point. This task is graded only by Breakwater's verifier stage.
set -euo pipefail
[[ "${BREAKWATER_STAGE:-}" == verifier ]] || { echo 'midtrain-for-rl runs only under Breakwater: launch it with breakwater/run.py.' >&2; exit 2; }
[[ -n "${BREAKWATER_AUTHORITY_SHA256:-}" ]] || { echo 'Missing Breakwater authority digest: launch the task with breakwater/run.py.' >&2; exit 2; }
exec python3 -I /opt/breakwater/verifier.py --authority "$BREAKWATER_AUTHORITY_SHA256"
