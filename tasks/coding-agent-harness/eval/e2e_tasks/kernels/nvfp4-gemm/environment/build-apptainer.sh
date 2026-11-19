#!/usr/bin/env bash
# Build the task runtime as an Apptainer image (default nvfp4-gemm.sif in the current directory; override with
# TASK_SIF) from environment/apptainer.def, without Docker or root. The build context is environment/, as for the
# coding-agent-harness README's docker build; the grid's cell image is built from the result with the harness grid's
# images/apptainer.def. Put APPTAINER_TMPDIR on local disk with room for the image tree, which also holds the build's
# temporary downloads: the build unpacks the base image there, and some shared filesystems reject its extended
# attributes.
set -euo pipefail
environment=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
output=$(realpath -m "${TASK_SIF:-nvfp4-gemm.sif}")
context=$(mktemp -d "${APPTAINER_TMPDIR:-${TMPDIR:-/tmp}}/nvfp4-gemm-build.XXXXXX")
trap 'rm -rf "${context}"' EXIT
# tar, not cp -a: it copies modes and times but not ACLs, which a local build directory may not support.
tar -C "${environment}" -cf - apptainer.def oracle-requirements.txt reference.py measure.py nvfp4_gemm.py \
  | tar -C "${context}" --no-same-owner -xf -
cd "${context}"
apptainer build --fakeroot "${output}" apptainer.def
