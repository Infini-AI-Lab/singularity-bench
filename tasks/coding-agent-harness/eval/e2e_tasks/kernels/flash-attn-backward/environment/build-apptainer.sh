#!/usr/bin/env bash
# Build the task runtime as an Apptainer image (default flash-attn-backward.sif in the current directory; override with
# TASK_SIF) from environment/apptainer.def, without Docker or root. The build context is environment/, as for the
# coding-agent-harness README's docker build; the grid's cell image is built from the result with the harness grid's
# images/apptainer.def. CUDA_TAG, TORCH_CUDA and QUACK_SPEC, when set, are passed as the definition's build
# arguments of the same names (the Dockerfile's ARGs). Put APPTAINER_TMPDIR on local disk with room for the image tree,
# which also holds the build's temporary downloads: the build unpacks the base image there, and some shared filesystems
# reject its extended attributes.
set -euo pipefail
environment=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
output=$(realpath -m "${TASK_SIF:-flash-attn-backward.sif}")
context=$(mktemp -d "${APPTAINER_TMPDIR:-${TMPDIR:-/tmp}}/flash-attn-backward-build.XXXXXX")
trap 'rm -rf "${context}"' EXIT
# tar, not cp -a: it copies modes and times but not ACLs, which a local build directory may not support.
tar -C "${environment}" -cf - apptainer.def reference.py measure.py attn_backward.py \
  | tar -C "${context}" --no-same-owner -xf -
build_args=()
for name in CUDA_TAG TORCH_CUDA QUACK_SPEC; do
  if [[ -n "${!name:-}" ]]; then build_args+=(--build-arg "${name}=${!name}"); fi
done
cd "${context}"
apptainer build --fakeroot "${build_args[@]}" "${output}" apptainer.def
