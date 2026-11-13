# serve-glm53-flash runtime: build an OpenAI-compatible /v1/completions server for
# /models/GLM-5.3-Flash (glm5_next: hybrid linear + sparse attention, 288-expert top-8 MoE,
# FP8 e4m3; about 17B active of about 320B total parameters) ACROSS FOUR GPUS, FROM SCRATCH
# with torch + HuggingFace transformers (+ triton for custom kernels), and make it EARN as
# much as it can serving replayed agentic coding sessions (simulator.py: the AgentX corpus on
# a fixed number of session lanes, priced per token). The serving/attention FRAMEWORKS are NOT
# installed (no vLLM, SGLang, flashinfer, flash-attn) and commands have no network at run time,
# so the agent CANNOT pip-install or download them -- the fast path must be written by hand
# (continuous batching, a paged KV cache, prefix caching, custom triton/CUDA kernels). triton
# ships with torch; nvcc (CUDA 13.0 devel) + cmake + ninja are present so the agent CAN also
# compile a CUDA extension.
#
# HARDWARE: FOUR GB300 GPUs (NVIDIA Grace Blackwell Ultra, compute capability 10.3 / sm_103,
# Grace ARM64 CPU).
#
# WHY FOUR GPUS ARE MANDATORY, NOT OPTIONAL: GLM-5.3-Flash is ~328 GB of FP8 e4m3 weights.
# That does not fit on one GB300 (~288 GB), so any working server -- the naive starter
# server included -- must shard the model. The agent therefore has to implement multi-GPU
# parallelism itself, which is a real part of this task and not a detail. Tensor-parallel
# size must divide num_attention_heads (64).
#
# Two platform details matter:
#   1. ARCHITECTURE. Grace is ARM64, so the whole userspace must be aarch64 --
#      --platform=linux/arm64 plus the guard below. Build with buildx on an
#      aarch64 host (or with qemu-user-static registered for cross-build).
#   2. sm_103. NVIDIA lists GB300 (and B300) at compute capability 10.3, versus 10.0
#      for B200/GB200 [developer.nvidia.com/cuda-gpus]. CUDA 12.9 is the first toolkit
#      that can emit it -- "CUDA Toolkit 12.9 adds compiler target support for SM
#      architecture 10.3 (sm_103, sm_103f, sm_103a)" [CUDA 12.9 release notes]. This image
#      is CUDA 13.0, and TORCH_CUDA_ARCH_LIST below lists BOTH 10.0 and 10.3, which makes an
#      agent-compiled extension correct on either capability.
#
# On stock torch: the install below is FROZEN at torch==2.14.0+cu130 so that scores stay
# comparable across runs (a torch bump moves prefill and decode speed, and with them what a
# server earns on the fixed traffic window). That version brings torch.version.cuda 13.0 and
# triton 3.8.0, and:
#   torch.cuda.get_arch_list() -> ['sm_80','sm_90','sm_100','sm_110','sm_120']
# i.e. the wheel ships an sm_100 cubin and no sm_103 one. NVIDIA's Blackwell Compatibility
# Guide states: "A cubin generated for a certain compute capability is supported to run on
# any GPU with the same major revision and same or higher minor revision of compute
# capability" -- 10.0 -> 10.3 satisfies that, so stock torch ops run unchanged. Check a host:
#   python3 -c "import torch; print(torch.cuda.get_device_capability())"   # expect (10, 3)
#
# The accuracy gate's anchor (tests/score.py REFERENCE_ACCURACY = 0.928) and its provenance
# are documented in tests/score.py. If you change the model, the prompt set or the reasoning
# effort, that number must be re-measured the same way.
#
# Plain -devel, NOT -cudnn-devel: torch brings its own cuDNN as a pip dependency
# (nvidia-cudnn-cu13), so a cuDNN-carrying base is redundant weight in the image.
FROM --platform=linux/arm64 nvidia/cuda:13.0.1-devel-ubuntu24.04

# Fail early if an old local image or an incorrect base is selected.
RUN test "$(dpkg --print-architecture)" = arm64

ENV DEBIAN_FRONTEND=noninteractive

# OPTIONAL APT MIRROR OVERRIDE. The build needs archive.ubuntu.com at image-build time.
# On some hosts/networks the ubuntu.com domains are unreachable or blocked, which fails
# the build with "Could not connect to archive.ubuntu.com:80 ... connection timed out"
# even though Docker Hub, PyPI and NVIDIA's repos all work. Rather than hard-depending on
# one mirror, allow it to be swapped at build time:
#     docker build --build-arg APT_MIRROR=https://mirrors.kernel.org/ubuntu ...
# Left empty, the image uses the distro default and nothing changes.
ARG APT_MIRROR=""
RUN if [ -n "$APT_MIRROR" ]; then \
        echo "using apt mirror: $APT_MIRROR" && \
        sed -i "s|http://archive.ubuntu.com/ubuntu|$APT_MIRROR|g; s|http://security.ubuntu.com/ubuntu|$APT_MIRROR|g; s|http://ports.ubuntu.com/ubuntu-ports|$APT_MIRROR|g" \
            /etc/apt/sources.list.d/ubuntu.sources 2>/dev/null || true; \
        sed -i "s|http://archive.ubuntu.com/ubuntu|$APT_MIRROR|g; s|http://security.ubuntu.com/ubuntu|$APT_MIRROR|g; s|http://ports.ubuntu.com/ubuntu-ports|$APT_MIRROR|g" \
            /etc/apt/sources.list 2>/dev/null || true; \
    fi
# Ensure agent-compiled CUDA extensions target Blackwell even when no GPU is visible
# at build/JIT time (torch reads TORCH_CUDA_ARCH_LIST for nvcc -gencode). Both 10.0
# (B200) and 10.3 (Blackwell Ultra) are listed so a JIT-built extension has a native
# cubin either way; verified with the resolved torch (2.14.0+cu130) that this list emits
# a compute_103/sm_103 gencode pair. "10.3a" is also accepted if you want arch-specific
# features.
ENV TORCH_CUDA_ARCH_LIST="10.0 10.3"

# ncu ships in the devel image (/usr/local/cuda/bin/ncu); add nsys + build/firewall deps.
# nsight-compute / nsight-systems come from Ubuntu noble multiverse, which is published
# for arm64 on ports.ubuntu.com too.
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential python3 python3-pip python3-dev git curl ca-certificates \
        cmake ninja-build iptables iproute2 nsight-compute nsight-systems && \
    rm -rf /var/lib/apt/lists/*

# DL stack: torch (ships triton for custom kernels) + transformers (+ fastapi/uvicorn
# to serve, numpy/safetensors). Deliberately NO vLLM / SGLang / flashinfer / flash-attn.
# torch comes from the CUDA 13.0 wheel index so it matches the nvcc in this image AND
# ships Blackwell (sm_100) kernels -- the default PyPI wheel may not include them, which
# would make every CUDA op on this GPU fall back / fail. See the sm_103 note in the
# header for what this wheel does and does not contain.
# REPRODUCIBILITY: the torch version is FROZEN, deliberately. torch version affects prefill
# and decode speed, i.e. what a server earns on the fixed traffic window, so a floating
# version would make runs months apart non-comparable. The local-version suffix (+cu130) is
# spelled out so the pin can only ever resolve to the CUDA 13.0 build. To move to a newer
# torch, bump this deliberately, re-record the version with the results, and treat scores
# from before and after the bump as separate series. triton is pinned transitively (3.8.0).
RUN pip3 install --break-system-packages --no-cache-dir \
        --index-url https://download.pytorch.org/whl/cu130 \
        "torch==2.14.0+cu130" && \
    pip3 install --break-system-packages --no-cache-dir \
        "transformers[kernels]==5.17.0" "kernels==0.16.0" \
        accelerate fastapi "uvicorn[standard]" numpy safetensors \
        huggingface_hub hf_transfer "aiohttp==3.13.2" && \
    python3 -m pip check
# aiohttp is the simulator's HTTP client (/app/simulator.py streams many long-lived
# requests at once); it is a client library, not a serving framework.

# THE FP8 KERNEL transformers needs is NOT in the wheel: it is fetched from the HF Hub at
# first use, and the container has no internet at run time. Bake it in, and prove at BUILD
# time that it loads offline (see prefetch_fp8_kernel.py for what was measured and why).
# Pin transformers and kernels together above: transformers 5.17.0 accepts kernels
# >=0.16.0,<0.17.0. Floating transformers can silently move that required range.
# The kernels extra also makes pip enforce compatibility during resolution.
COPY prefetch_fp8_kernel.py /opt/prefetch_fp8_kernel.py
RUN python3 /opt/prefetch_fp8_kernel.py
# Everything the image needs from the Hub is now in its cache; say so, so that no library
# sits in a connect timeout against the firewall before falling back to it.
ENV HF_HUB_OFFLINE=1

# THE MODEL IS NOT BAKED INTO THE IMAGE -- IT IS BIND-MOUNTED AT RUN TIME.
# GLM-5.3-Flash is ~328 GB. Baking it makes the image enormous: slow to build, slow to
# push/pull, and duplicated on disk in every image that serves the same weights.
# Instead Breakwater mounts the host directory holding the snapshot READ-ONLY at
# /models/GLM-5.3-Flash (see ../bindings.example.json), so the image stays small and the
# weights exist once per machine.
#
# /models is READ-ONLY to the agent and commands have no network at run time, so the
# agent can neither modify the weights nor fetch anything else.
#
# REQUIRED ON THE HOST: the snapshot must already exist at the mounted path (MODEL_HOST_DIR
# names its parent directory). Fetch and verify the pinned revision once with the task's
# prepare_model.py (see the task README).

# Agent CLIs are installed only in the Breakwater author image built on top of this one
# (breakwater/author/Dockerfile).
RUN apt-get update && apt-get install -y --no-install-recommends procps ripgrep nodejs npm && rm -rf /var/lib/apt/lists/*

# tmux: required by terminal-multiplexer agent harnesses (e.g. terminus); a few hundred
# KB, no effect on the serving benchmark. Kept as a late layer so the layers above stay
# cached across rebuilds.
RUN apt-get update && apt-get install -y --no-install-recommends tmux && rm -rf /var/lib/apt/lists/*

# Task workspace: the NAIVE starter server (the agent's starting point) plus
# the dev-loop helpers (start_server.sh detaches the server; simulator.py scores it).
WORKDIR /app
COPY server.py        /app/server.py
COPY serve.sh         /app/serve.sh
COPY start_server.sh  /app/start_server.sh
COPY stop_server.sh   /app/stop_server.sh
COPY example_requests.py /app/example_requests.py
# simulator.py IS the scored benchmark and is deliberately agent-visible: agents are
# expected to read it and specialize for its traffic. The grader runs this same file, on
# a seed that lives only in tests/. agentx/ is its data (recorded sessions + code text).
COPY simulator.py     /app/simulator.py
COPY agentx           /app/agentx
# process_safety.md: how to stop the server without killing your own agent process.
# instruction.md: a readable copy of the task prompt inside /app. NOTE: this is a COPY
# of ../instruction.md (build context is environment/, so the canonical task-root file
# is out of reach). Re-sync `cp ../instruction.md environment/instruction.md` whenever
# the task prompt changes, or /app/instruction.md goes stale (prepare_model.py refuses
# copies that differ).
COPY process_safety.md /app/process_safety.md
COPY instruction.md   /app/instruction.md
# CLAUDE.md: auto-loaded by Claude Code from the working dir (/app) at startup. Front-loads
# the #1 footgun -- self-kill via `pkill -f server.py` matching the agent's own
# `claude --print -- '<prompt>'` argv -- so CC internalizes safe PID-based server management.
COPY CLAUDE.md        /app/CLAUDE.md
# AGENTS.md: the same notes, read by Codex from the working dir (/app).
COPY AGENTS.md        /app/AGENTS.md
RUN chmod +x /app/serve.sh /app/start_server.sh /app/stop_server.sh

# restrict-network.sh: the standalone outbound firewall (LLM API endpoints only; github,
# pypi and HF blocked), listed in instruction.md. Under Breakwater the protected bootstrap
# installs the worker firewall instead, and this script is not run.
COPY restrict-network.sh /app/restrict-network.sh
RUN chmod +x /app/restrict-network.sh

# Strip apt sources: nothing installs packages at run time (breakwater/author/Dockerfile
# restores them to build the author image).
RUN rm -rf /etc/apt/sources.list /etc/apt/sources.list.d /var/lib/apt/lists/*
