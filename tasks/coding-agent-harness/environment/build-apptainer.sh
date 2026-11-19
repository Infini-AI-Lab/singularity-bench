#!/usr/bin/env bash
# Build the task runtime as an Apptainer image (default coding-agent-harness.sif in the current directory;
# override with TASK_SIF) from environment/apptainer.def, without Docker or root. The build context is environment/,
# as for the README's docker build, without the bytecode its .dockerignore excludes. Put APPTAINER_TMPDIR on local
# disk: the build unpacks both base images there, and some shared filesystems reject their extended attributes.
set -euo pipefail
environment=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
output=$(realpath -m "${TASK_SIF:-coding-agent-harness.sif}")
context=$(mktemp -d "${APPTAINER_TMPDIR:-${TMPDIR:-/tmp}}/coding-agent-harness-build.XXXXXX")
trap 'rm -rf "${context}"' EXIT
# tar, not cp -a: it copies modes and times but not ACLs, which a local build directory may not support.
tar -C "${environment}" --exclude=__pycache__ --exclude='*.pyc' -cf - apptainer.def harness spec smoke examples \
  harness_api.py selfcheck.py | tar -C "${context}" --no-same-owner -xf -
cd "${context}"
apptainer build --fakeroot "${output}" apptainer.def
