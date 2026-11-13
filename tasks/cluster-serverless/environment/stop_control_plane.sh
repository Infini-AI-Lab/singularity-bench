#!/usr/bin/env bash
# Stop only processes belonging to the recorded local development launch.
set -euo pipefail
exec python3 "$(dirname "${BASH_SOURCE[0]}")/control_plane_process.py" stop
