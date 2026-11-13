#!/usr/bin/env bash
set -euo pipefail
[[ "${BREAKWATER_STAGE:-}" == verifier ]] || { echo 'Launch this task with breakwater/run.py: three separate containers and sealed external evidence are mandatory.' >&2; exit 2; }
[[ -n "${BREAKWATER_AUTHORITY_SHA256:-}" ]] || { echo 'Missing host authority digest' >&2; exit 2; }
exec python3 -I /opt/breakwater/verifier.py --authority "$BREAKWATER_AUTHORITY_SHA256"
