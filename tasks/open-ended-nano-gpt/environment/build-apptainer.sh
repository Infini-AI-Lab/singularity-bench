#!/usr/bin/env bash
# Build the task runtime as an Apptainer image (default open-ended-nano-gpt.sif in the current directory;
# override with TASK_SIF) from environment/apptainer.def, without Docker or root. MEGATRON_SOURCE is an
# export of Megatron-LM at the pinned commit (see the task README); it is staged as "megatron" next to the
# definition file, which copies it to /opt/megatron-lm. Put APPTAINER_TMPDIR on local disk: the build
# unpacks the base image there, and some shared filesystems reject its extended attributes.
set -euo pipefail
environment=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
: "${MEGATRON_SOURCE:?set MEGATRON_SOURCE to the pinned Megatron-LM source tree}"
output=$(realpath -m "${TASK_SIF:-open-ended-nano-gpt.sif}")
context=$(mktemp -d "${APPTAINER_TMPDIR:-${TMPDIR:-/tmp}}/open-ended-nano-gpt-build.XXXXXX")
trap 'rm -rf "${context}"' EXIT
cp "${environment}/apptainer.def" "${environment}/requirements.txt" "${environment}/triton_ptxas.py" \
  "${environment}/patch_megatron.py" "${context}/"
# tar, not cp -a: it copies modes and times but not ACLs, which a local build directory may not support.
mkdir "${context}/megatron"
tar -C "${MEGATRON_SOURCE}" -cf - . | tar -C "${context}/megatron" --no-same-owner -xf -
cd "${context}"
apptainer build --fakeroot "${output}" apptainer.def
