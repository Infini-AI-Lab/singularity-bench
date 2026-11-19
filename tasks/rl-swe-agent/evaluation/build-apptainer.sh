#!/usr/bin/env bash
# Build the held-out evaluation runtime as an Apptainer image (default rl-swe-agent-eval.sif in the current directory;
# override with EVAL_SIF) from evaluation/apptainer.def, without Docker or root, on the GB300 (arm64) host. The build
# context is evaluation/, as for its docker build; only the definition and the SGLang patch are staged. Put
# APPTAINER_TMPDIR on local disk: the build unpacks the base image there, and some shared filesystems reject its
# extended attributes.
set -euo pipefail
evaluation=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
output=$(realpath -m "${EVAL_SIF:-rl-swe-agent-eval.sif}")
build=$(mktemp -d "${APPTAINER_TMPDIR:-${TMPDIR:-/tmp}}/rl-swe-agent-eval-build.XXXXXX")
trap 'rm -rf "${build}"' EXIT
cp "${evaluation}/apptainer.def" "${evaluation}/patches/sglang-context-clip.patch" "${build}/"
cd "${build}"
apptainer build --fakeroot "${output}" apptainer.def
