#!/usr/bin/env bash
# Build the task runtime as an Apptainer image (default serve-glm53-flash.sif in the current directory;
# override with TASK_SIF) from environment/apptainer.def, without Docker or root. The build context is
# environment/, as for prepare.sh's docker build; APT_MIRROR, when set, is passed as the definition's mirror build
# argument. The checkpoint is not part of the image (see the task README). Put APPTAINER_TMPDIR on local disk: the
# build unpacks the base image there, and some shared filesystems reject its extended attributes.
set -euo pipefail
environment=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
output=$(realpath -m "${TASK_SIF:-serve-glm53-flash.sif}")
context=$(mktemp -d "${APPTAINER_TMPDIR:-${TMPDIR:-/tmp}}/serve-glm53-flash-build.XXXXXX")
trap 'rm -rf "${context}"' EXIT
# tar, not cp -a: it copies modes and times but not ACLs, which a local build directory may not support.
tar -C "${environment}" -cf - apptainer.def prefetch_fp8_kernel.py server.py serve.sh start_server.sh \
  stop_server.sh example_requests.py simulator.py agentx process_safety.md instruction.md CLAUDE.md AGENTS.md \
  restrict-network.sh | tar -C "${context}" --no-same-owner -xf -
cd "${context}"
apptainer build --fakeroot --build-arg "APT_MIRROR=${APT_MIRROR:-}" "${output}" apptainer.def
