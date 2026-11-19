#!/usr/bin/env bash
# Build the task runtime as an Apptainer image (default rl-swe-agent.sif in the current directory; override with
# TASK_SIF) from environment/apptainer.def, without Docker or root. The build context is the task directory, as for
# build.sh's docker build: it is staged as "context" next to the definition file, without the paths its .dockerignore
# lists, which Docker matches from the root of the context. Put APPTAINER_TMPDIR on local disk: the build unpacks the
# base image there, and some shared filesystems reject its extended attributes.
set -euo pipefail
environment=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
task=$(dirname "${environment}")
output=$(realpath -m "${TASK_SIF:-rl-swe-agent.sif}")
build=$(mktemp -d "${APPTAINER_TMPDIR:-${TMPDIR:-/tmp}}/rl-swe-agent-build.XXXXXX")
trap 'rm -rf "${build}"' EXIT
excludes=()
while read -r pattern || [[ -n "${pattern}" ]]; do
  [[ -z "${pattern}" || "${pattern}" == '#'* ]] && continue
  if [[ "${pattern}" == *'**'* || "${pattern}" == '!'* ]]; then
    echo "Unsupported .dockerignore pattern: ${pattern}" >&2
    exit 1
  fi
  pattern=${pattern#/}
  excludes+=("--exclude=./${pattern%/}")
done < "${task}/.dockerignore"
cp "${environment}/apptainer.def" "${build}/"
mkdir "${build}/context"
# tar, not cp -a: it copies modes and times but not ACLs, which a local build directory may not support.
tar -C "${task}" --anchored --no-wildcards-match-slash "${excludes[@]}" -cf - . \
  | tar -C "${build}/context" --no-same-owner -xf -
cd "${build}"
apptainer build --fakeroot "${output}" apptainer.def
