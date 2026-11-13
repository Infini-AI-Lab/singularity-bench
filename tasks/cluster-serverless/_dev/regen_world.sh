#!/usr/bin/env bash
# Rebuild the generated world (catalogues, fleets, suites, traces) from the frozen inputs and the pinned
# ServeGen checkout, verify it against _dev/world-manifest.json, then publish it into the task tree.
# CLUSTER_WORLD_MANIFEST=write records a new manifest instead of verifying (an intentional revision).
set -euo pipefail
T="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$T/_dev/pyenv.sh"
PY="${CLUSTER_GENERATOR_PYTHON:-$(_pick_python "$T" "$(dirname "$T")")}"
ARGS=()
if [ "${CLUSTER_WORLD_MANIFEST:-verify}" = write ]; then ARGS+=(--write-manifest); fi
exec "$PY" "$T/_dev/regen_world.py" "${ARGS[@]}" "$@"
