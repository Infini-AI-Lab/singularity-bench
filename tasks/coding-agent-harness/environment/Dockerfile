# Task runtime for coding-agent-harness (amd64 and arm64; GB300 hosts are arm64):
#   docker build -t coding-agent-harness:local tasks/coding-agent-harness/environment
# The protected author image (breakwater/author/Dockerfile) is built on top of it. The harness must run on
# CPython 3.11, so the interpreter comes from the official Python 3.11 image; the CUDA 13 toolkit provides a
# compiler that recognizes the evaluation GPUs (sm_103 on GB300) for GPU development and the examples.
FROM python:3.11.14-slim-bookworm AS python
FROM nvidia/cuda:13.0.2-devel-ubuntu24.04
ENV DEBIAN_FRONTEND=noninteractive PYTHONDONTWRITEBYTECODE=1
RUN apt-get update && apt-get install -y --no-install-recommends \
      iptables iproute2 ca-certificates build-essential git procps ripgrep tmux jq curl util-linux \
      libbz2-1.0 libffi8 libgdbm6t64 libgdbm-compat4t64 liblzma5 libncursesw6 \
      libreadline8t64 libsqlite3-0 libssl3t64 libuuid1 zlib1g \
    && rm -rf /var/lib/apt/lists/*
COPY --from=python /usr/local/ /usr/local/
RUN ldconfig && python3 -c 'import ssl, sqlite3, ctypes, bz2, lzma, readline; import sys; assert sys.version_info[:2] == (3, 11)'
ENV CUDA_HOME=/usr/local/cuda PYTHONUNBUFFERED=1
RUN python3 -m pip install --no-cache-dir torch==2.11.0 --index-url https://download.pytorch.org/whl/cu130 \
    && python3 -m pip install --no-cache-dir numpy==2.2.6 ninja==1.11.1.4 pytest==8.3.5 \
    && python3 -c 'import torch, triton; assert torch.version.cuda == "13.0"; print(torch.__version__, triton.__version__)'
WORKDIR /app
COPY harness/ /app/harness/
COPY spec/ /app/spec/
COPY smoke/ /app/smoke/
COPY examples/ /app/examples/
COPY harness_api.py selfcheck.py /app/
# Sticky, group-writable parent: the author can create scratch files but cannot replace the root-owned
# spec/smoke/API paths by renaming their directory entries. GID 1200 is the shared workspace group that the
# author image's installer creates.
RUN chown root:1200 /app && chmod 3775 /app \
    && chown -R root:1200 /app/harness && chmod 2775 /app/harness \
    && chmod g+w /app/harness/agent.py
