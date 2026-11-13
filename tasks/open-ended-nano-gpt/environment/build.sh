#!/usr/bin/env bash
# Build the task runtime image (default tag open-ended-nano-gpt:local; override with TASK_IMAGE)
# from environment/Dockerfile. MEGATRON_SOURCE is an export of Megatron-LM at the pinned commit
# (see the task README); the Dockerfile copies it to /opt/megatron-lm.
set -euo pipefail
environment=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
: "${MEGATRON_SOURCE:?set MEGATRON_SOURCE to the pinned Megatron-LM source tree}"
exec docker build --platform linux/arm64 \
  --file "${environment}/Dockerfile" \
  --build-context "megatron=${MEGATRON_SOURCE}" \
  --tag "${TASK_IMAGE:-open-ended-nano-gpt:local}" \
  "${environment}"
