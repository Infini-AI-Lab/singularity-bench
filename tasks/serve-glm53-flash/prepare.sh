#!/usr/bin/env bash
# Prepare the task runtime: fetch and verify the pinned checkpoint, then build the native ARM64
# image (serve-glm53-flash:local, or $SERVING_IMAGE). The Breakwater author image
# (breakwater/author/Dockerfile) is built on top of it; see README.md. --check only validates.
# On a host without Docker, PREPARE_RUNTIME=apptainer TASK_SIF=serve-glm53-flash.sif checks the SIF instead.
set -euo pipefail
task_root=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
[[ $(uname -m) == aarch64 ]] || { echo 'Prepare this task on its native ARM64 GB300 host.' >&2; exit 1; }
case "${1:-}" in ''|--check) ;; *) echo "Usage: $0 [--check]" >&2; exit 2 ;; esac
python3 "${task_root}/prepare_model.py" "$@"
if [[ "${PREPARE_RUNTIME:-docker}" == apptainer ]]; then
  # A host without Docker: the runtime is the SIF from environment/build-apptainer.sh (TASK_SIF); check its stack.
  [[ -f "${TASK_SIF:-}" ]] || { echo 'Set TASK_SIF to the SIF built by environment/build-apptainer.sh' >&2; exit 1; }
  apptainer exec --containall --no-home --cleanenv "${TASK_SIF}" python3 -c \
    'import platform, torch, transformers, triton; assert platform.machine() == "aarch64" and torch.version.cuda; print(torch.__version__, transformers.__version__, triton.__version__)'
  exit 0
fi
image=${SERVING_IMAGE:-serve-glm53-flash:local}
if [[ "${1:-}" != --check ]]; then
  docker build --platform linux/arm64 --tag "$image" "${task_root}/environment"
fi
[[ $(docker image inspect --format '{{.Architecture}}' "$image") == arm64 ]]
docker run --rm --network none --entrypoint python3 "$image" -c \
  'import torch, transformers, triton; assert torch.version.cuda; print(torch.__version__, transformers.__version__, triton.__version__)'
