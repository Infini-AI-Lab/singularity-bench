#!/bin/bash
# Starter server: a NAIVE but correct OpenAI-compatible API server for
# /models/GLM-5.3-Flash, built with plain torch + transformers (port 8000,
# /v1/completions). This is your baseline -- edit it (or replace server.py
# entirely, add files, write custom CUDA kernels) to earn as much as you can on the
# agentic traffic in /app/simulator.py WITHOUT breaking correctness. NO serving
# frameworks are installed and the container has no internet, so you cannot
# pip-install them: build the fast path yourself, with torch + transformers + triton
# (and CUDA C++ if you like).
#
# NOTE: the ~328 GB of FP8 weights do NOT fit on one GPU. The starter shards with
# device_map="auto" (pipeline placement, one GPU busy at a time). Anything you launch
# from here must shard too.
#
# The verifier runs exactly `bash /app/serve.sh`, waits for /v1/models or /health, then
# sends /v1/completions requests: pre-templated strings and token-id prompts, streamed
# and not (see /app/example_requests.py for every form).
set -e
exec python3 /app/server.py
