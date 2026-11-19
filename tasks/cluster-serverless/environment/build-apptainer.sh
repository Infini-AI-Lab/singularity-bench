#!/usr/bin/env bash
# Build the task runtime as an Apptainer image (default cluster-serverless.sif in the current directory; override
# with TASK_SIF) from environment/apptainer.def, without Docker or root. The build context is environment/, as for
# the README's docker build. The fleet image (environment/fleetsvc) is separate. Put APPTAINER_TMPDIR on local disk:
# the build unpacks the base image there, and some shared filesystems reject its extended attributes.
set -euo pipefail
environment=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
output=$(realpath -m "${TASK_SIF:-cluster-serverless.sif}")
context=$(mktemp -d "${APPTAINER_TMPDIR:-${TMPDIR:-/tmp}}/cluster-serverless-build.XXXXXX")
trap 'rm -rf "${context}"' EXIT
cd "${environment}"
# tar, not cp -a: it copies modes and times but not ACLs, which a local build directory may not support.
tar -cf - apptainer.def reference_cp.py run.sh selfcheck.py control_plane_process.py start_control_plane.sh \
  stop_control_plane.sh process_safety.md AGENTS.md CLAUDE.md catalogue-dev.json fleet-dev.public.json \
  trace-public-*.jsonl suite-public.json _kitcopy | tar -C "${context}" --no-same-owner -xf -
cd "${context}"
apptainer build --fakeroot "${output}" apptainer.def
