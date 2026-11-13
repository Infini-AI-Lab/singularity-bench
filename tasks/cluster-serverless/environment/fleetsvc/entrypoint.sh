#!/usr/bin/env bash
# One switch between the emulated and the real fleet. Both expose the identical HTTP surface; graded
# runs use the real fleet (CLUSTER_FLEET_MODE=real), the emulator is a development utility.
set -euo pipefail
MODE="${CLUSTER_FLEET_MODE:-sim}"
# `python3` explicitly: the vLLM image has both, and a bare `python` resolves differently across bases.
case "$MODE" in
  # One MANAGER for this machine, plus -- on the first node only -- the fleet controller beside it.
  # CLUSTER_ONLY_NODE picks which node's slice this container is.
  sim)  MGR=/fleet/kit/simmanager.py ;;
  real) MGR=/fleet/kit/manager.py ;;
  *)    echo "CLUSTER_FLEET_MODE must be sim or real, got '$MODE'" >&2; exit 2 ;;
esac

ARGS=(--fleet "$CLUSTER_NODES" --catalogue "$CLUSTER_CATALOGUE")
if [ "${CLUSTER_RUN_CONTROLLER:-0}" = "1" ]; then
  python3 /fleet/kit/controller.py "${ARGS[@]}" --port "${CLUSTER_CONTROLLER_PORT:-9000}" &
fi
exec python3 "$MGR" "${ARGS[@]}"
