#!/usr/bin/env bash
# How the platform starts your control plane. It must listen on port 8000 and stay in the foreground.
# Change the command if you rewrite the plane in something other than Python.
set -euo pipefail
export CLUSTER_NODES="${CLUSTER_NODES:-/app/fleet.json}"
export CLUSTER_CATALOGUE="${CLUSTER_CATALOGUE:-/app/catalogue.json}"
export PORT="${PORT:-8000}"
exec python3 /app/control_plane.py
