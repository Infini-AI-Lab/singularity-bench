#!/usr/bin/env bash
set -euo pipefail
# Harbor verifier entry point. Grading runs only under Breakwater (breakwater/run.py): the three graded cells
# run in the separate intermediate container, and this read-only verifier stage recomputes the reward from the
# sealed raw records and runs the integrity judge.
[[ "${BREAKWATER_STAGE:-}" == verifier ]] || { echo 'Launch this task with breakwater/run.py; it does not grade outside Breakwater.' >&2; exit 2; }
[[ -n "${BREAKWATER_AUTHORITY_SHA256:-}" ]] || { echo 'Missing host authority digest' >&2; exit 2; }
exec python3 -I /opt/breakwater/verifier.py --authority "$BREAKWATER_AUTHORITY_SHA256"
