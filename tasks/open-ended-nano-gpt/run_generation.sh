#!/usr/bin/env bash
# Supplied generation evaluator. Implement generation_model.py, not this runner.
set -euo pipefail
TASK_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
exec python "$TASK_DIR/generation_eval.py" "$@"
