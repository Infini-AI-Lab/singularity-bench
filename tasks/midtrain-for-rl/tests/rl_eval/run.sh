#!/usr/bin/env bash
# Usage: bash run.sh CHECKPOINT_DIR OUTPUT_DIR [operator options]
set -euo pipefail
RL_ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
exec python3 "${RL_ROOT}/launch.py" "$@"
