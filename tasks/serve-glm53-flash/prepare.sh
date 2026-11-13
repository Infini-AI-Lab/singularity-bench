#!/usr/bin/env bash
# Prepare the task runtime: fetch and verify the pinned checkpoint, then build the native ARM64
# image (serve-glm53-flash:local, or $SERVING_IMAGE). The Breakwater author image
# (breakwater/author/Dockerfile) is built on top of it; see README.md. --check only validates.
set -euo pipefail
task_root=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
[[ $(uname -m) == aarch64 ]] || { echo 'Prepare this task on its native ARM64 GB300 host.' >&2; exit 1; }
case "${1:-}" in ''|--check) ;; *) echo "Usage: $0 [--check]" >&2; exit 2 ;; esac
python3 "${task_root}/prepare_model.py" "$@"
image=${SERVING_IMAGE:-serve-glm53-flash:local}
if [[ "${1:-}" != --check ]]; then
  docker build --platform linux/arm64 --tag "$image" "${task_root}/environment"
fi
[[ $(docker image inspect --format '{{.Architecture}}' "$image") == arm64 ]]
docker run --rm --network none --entrypoint python3 "$image" -c \
  'import torch, transformers, triton; assert torch.version.cuda; print(torch.__version__, transformers.__version__, triton.__version__)'
