#!/usr/bin/env bash
# Restart the local development control plane without matching process names.
set -euo pipefail
exec python3 "$(dirname "${BASH_SOURCE[0]}")/control_plane_process.py" start
