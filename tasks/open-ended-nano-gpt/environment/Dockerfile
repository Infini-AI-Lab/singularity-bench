# The open-ended-nano-gpt task runtime (open-ended-nano-gpt:local). Build it with
# environment/build.sh, which supplies the pinned Megatron-LM source as the "megatron"
# build context. Breakwater's author image (breakwater/author/Dockerfile) and verifier image
# (breakwater/Dockerfile.verifier) are built on top of it.
# GB300: native Grace ARM64 userspace and Blackwell-capable CUDA.
FROM --platform=linux/arm64 nvidia/cuda:12.9.1-cudnn-devel-ubuntu24.04

# Fail early if the base image is not arm64.
RUN test "$(dpkg --print-architecture)" = arm64

# CUDA extensions build Blackwell code (10.0) plus PTX; both run on GB300 (10.3).
ENV TORCH_CUDA_ARCH_LIST="10.0+PTX"

ARG DEBIAN_FRONTEND=noninteractive

SHELL ["/bin/bash", "-o", "pipefail", "-c"]

RUN apt-get update \
    && apt-get install --yes --no-install-recommends \
        python3 \
        python3-dev \
        python3-venv \
        build-essential \
        ca-certificates \
        curl \
        git \
        iptables \
        numactl \
        nodejs \
        npm \
        pciutils \
        procps \
        ripgrep \
    && rm -rf /var/lib/apt/lists/*

RUN python3 -m venv /opt/venv
ENV PATH="/opt/venv/bin:${PATH}"
ENV VIRTUAL_ENV="/opt/venv"
RUN python -m pip install --no-cache-dir --upgrade pip setuptools wheel \
    && python -m pip install --no-cache-dir torch==2.9.1 \
        --index-url https://download.pytorch.org/whl/cu129

COPY requirements.txt /tmp/requirements.txt

RUN python -m pip install --no-cache-dir \
        --requirement /tmp/requirements.txt \
    && rm /tmp/requirements.txt \
    && python -c 'import torch; assert torch.__version__ == "2.9.1+cu129", torch.__version__; assert torch.version.cuda == "12.9", torch.version.cuda'

# Triton bundles a CUDA 12.8 ptxas, which cannot target GB300 (sm_103): link it to this image's CUDA 12.9 ptxas.
COPY triton_ptxas.py /tmp/triton_ptxas.py
RUN python /tmp/triton_ptxas.py && rm /tmp/triton_ptxas.py

# Optional Megatron library for agent-designed training and evaluation code.
COPY --from=megatron . /opt/megatron-lm
COPY patch_megatron.py /tmp/patch_megatron.py
RUN python /tmp/patch_megatron.py && rm /tmp/patch_megatron.py

ENV HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1 \
    TOKENIZERS_PARALLELISM=false \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/opt/megatron-lm

WORKDIR /app

# Breakwater seeds the task tree into /app. Tests, the private validation split and
# LAMBADA reach only the evaluation and verifier stages; never bake them into this image.
RUN mkdir -p /app/out

# The coding-agent CLI is installed by the Breakwater author image, not here.
