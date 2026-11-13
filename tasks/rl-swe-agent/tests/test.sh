#!/usr/bin/env bash
# Harbor verifier entry point. It runs only in the Breakwater verifier stage, which seals the submission, runs the
# integrity judge and writes reports/host-evaluation.json; the held-out score is then computed on the host by
# evaluation/evaluate_checkpoint.py and evaluation/score.py (evaluation/README.md).
set -euo pipefail
[[ "${BREAKWATER_STAGE:-}" == verifier ]] || { echo 'Launch this task with breakwater/run.py: it runs the agent, evaluation and verifier stages in separate containers.' >&2; exit 2; }
[[ -n "${BREAKWATER_AUTHORITY_SHA256:-}" ]] || { echo 'Missing host authority digest: launch the task with breakwater/run.py.' >&2; exit 2; }
exec python3 -I /opt/breakwater/verifier.py --authority "$BREAKWATER_AUTHORITY_SHA256"
