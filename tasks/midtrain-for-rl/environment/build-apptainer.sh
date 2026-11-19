#!/usr/bin/env bash
# Build the task runtime as an Apptainer image (default midtrain-for-rl.sif in the current directory; override
# with TASK_SIF) from environment/apptainer.def, without Docker or root. The build context is tests/rl_eval, as for
# prepare.sh's docker build of the agent target, without the bytecode its .dockerignore excludes. Put
# APPTAINER_TMPDIR on local disk with room for two unpacked copies of the base image (the vendor check is a stage of
# its own): the build unpacks them there, and some shared filesystems reject their extended attributes.
set -euo pipefail
environment=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
rl_eval=$(cd "${environment}/../tests/rl_eval" && pwd)
output=$(realpath -m "${TASK_SIF:-midtrain-for-rl.sif}")
context=$(mktemp -d "${APPTAINER_TMPDIR:-${TMPDIR:-/tmp}}/midtrain-for-rl-build.XXXXXX")
trap 'rm -rf "${context}"' EXIT
cp "${environment}/apptainer.def" "${context}/"
# tar, not cp -a: it copies modes and times but not ACLs, which a local build directory may not support.
tar -C "${rl_eval}" --exclude=__pycache__ --exclude='*.pyc' -cf - vendor verify_vendor.py vendor_manifest.json \
  checkpoint_check.py requirements-midtrain.txt check_runtime.py | tar -C "${context}" --no-same-owner -xf -
cd "${context}"
apptainer build --fakeroot "${output}" apptainer.def
