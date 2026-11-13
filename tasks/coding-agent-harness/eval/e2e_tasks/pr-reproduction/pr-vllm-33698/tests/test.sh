#!/usr/bin/env bash
# A downstream task of the coding-agent-harness grid: breakwater/grid.py grades it inside a Breakwater cell
# (breakwater/cell.py) with a sealed submitted harness. It has no standalone verifier entry point.
echo 'This downstream task is graded only by the coding-agent-harness grid: run it with breakwater/grid.py.' >&2
exit 2
