#!/usr/bin/env bash
# Build the task runtime as an Apptainer image (default pr-vllm-35022.sif in the current directory; override with
# TASK_SIF) from environment/apptainer.def, without Docker or root. The build context is environment/, as for the
# coding-agent-harness README's docker build; it holds only the Dockerfile, so only the definition is staged, and the
# build downloads everything else. The grid's cell image is built from the result with the harness grid's
# images/apptainer.def.
# VLLM_REF and VLLM_COMMIT, when set, are passed as the definition's build arguments of the same names (the
# Dockerfile's ARGs). Build on a GPU node: uv picks the PyTorch wheels from the build host's NVIDIA driver. Put
# APPTAINER_TMPDIR on local disk with room for the image tree, which also holds the build's temporary downloads: the
# build unpacks the base image there, and some shared filesystems reject its extended attributes.
set -euo pipefail
environment=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
output=$(realpath -m "${TASK_SIF:-pr-vllm-35022.sif}")
context=$(mktemp -d "${APPTAINER_TMPDIR:-${TMPDIR:-/tmp}}/pr-vllm-35022-build.XXXXXX")
trap 'rm -rf "${context}"' EXIT
# tar, not cp -a: it copies modes and times but not ACLs, which a local build directory may not support.
tar -C "${environment}" -cf - apptainer.def | tar -C "${context}" --no-same-owner -xf -
build_args=()
for name in VLLM_REF VLLM_COMMIT; do
  if [[ -n "${!name:-}" ]]; then build_args+=(--build-arg "${name}=${!name}"); fi
done
cd "${context}"
apptainer build --fakeroot "${build_args[@]}" "${output}" apptainer.def
