#!/usr/bin/env bash
# A downstream task of the coding-agent-harness grid (tasks/coding-agent-harness/grid/), which grades it inside a
# Breakwater cell with a sealed submitted harness. It has no standalone verifier entry point.
echo 'Graded only by the coding-agent-harness grid: tasks/coding-agent-harness/grid/grid.py.' >&2
exit 2
