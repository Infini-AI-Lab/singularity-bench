#!/usr/bin/env bash
# Download the pinned RL sources and prepare the fixed 43K mixture once, before grading.
# Usage: bash download_data.sh --base BASE_TOKENIZER --output DATA_DIR [--cache-dir CACHE]
set -euo pipefail
RL_ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
export HF_HUB_OFFLINE=0 HF_DATASETS_OFFLINE=0 TRANSFORMERS_OFFLINE=0
exec python3 "${RL_ROOT}/prepare_data.py" "$@"
