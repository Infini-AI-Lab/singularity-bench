#!/usr/bin/env bash
# Builds the task runtime image rl-swe-agent:local with the task directory as the build context.
set -euo pipefail
cd "$(dirname "$0")/.."
exec docker build -f environment/Dockerfile -t rl-swe-agent:local .
