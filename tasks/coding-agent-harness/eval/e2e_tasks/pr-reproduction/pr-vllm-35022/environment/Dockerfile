# Reproduce-a-PR task — vLLM PR #35022 ([Core] Support structured outputs for beam search).
#
# Stages vLLM at the commit BEFORE the PR, EDITABLE + PRECOMPILED; the feature is pure-Python (the beam-search
# loop + a params field) so the agent's edits under /app/vllm/vllm/ are live with NO CUDA rebuild. Single GPU.
# Offline at run time (no network). xgrammar + jsonschema come in with vLLM's deps.
#
# This is a FULL-GENERATION task (it runs a real model end-to-end, not a synthetic kernel), so:
#   * TinyLlama-1.1B-Chat is baked into the HF cache at build time (build has network; run time does not) and
#     HF_HUB_OFFLINE is set afterwards, so the graded test — and the agent's own iteration — run fully offline.
#   * The CUDA compiler matching torch's CUDA is installed: vLLM's attention on Blackwell (FlashInfer) compiles its
#     kernel launchers at first use. VLLM_USE_FLASHINFER_SAMPLER=0 keeps the Triton sampler.
#
# ─────────────────────────────────────────────────────────────────────────────────────────────────────
# BASE (torch-2.11). The image builds standalone; no pre-built local base image is required:
#   docker build -t pr-vllm-35022:local tasks/coding-agent-harness/eval/e2e_tasks/pr-reproduction/pr-vllm-35022/environment
#
# The CUDA *runtime* image: torch ships its own CUDA runtime libs, and the compiler FlashInfer needs is installed
# after torch, at torch's own CUDA version (a 12.8 devel base could not target the GB300's sm_103).
FROM nvidia/cuda:12.8.1-runtime-ubuntu24.04
ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
        python3 python3-dev git curl ca-certificates build-essential \
        patch iptables iproute2 procps && rm -rf /var/lib/apt/lists/*

RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH=/opt/venv/bin:/root/.local/bin:$PATH
# Don't bake uv's ~8 GB download cache into the image.
ENV UV_NO_CACHE=1

# Install torch + every vLLM dep ONCE at a fixed reference commit. The task layer below then re-runs the
# editable install at ITS OWN parent commit with --no-deps, so only vLLM's package + that commit's .so differ.
ARG VLLM_REF=635c38338afe132f9555ebf4a7c8ac7dfb05b2a0
RUN git clone --filter=blob:none https://github.com/vllm-project/vllm.git /opt/vllm-ref && \
    cd /opt/vllm-ref && git checkout ${VLLM_REF}
RUN uv venv /opt/venv --python 3.12
ENV VLLM_USE_PRECOMPILED=1
RUN cd /opt/vllm-ref && \
    VLLM_USE_PRECOMPILED=1 VLLM_PRECOMPILED_WHEEL_COMMIT=${VLLM_REF} \
        uv pip install --python /opt/venv/bin/python -e . --torch-backend=auto && \
    uv pip install --python /opt/venv/bin/python pytest gguf && \
    /opt/venv/bin/python -c "import vllm; print('base vLLM', vllm.__version__)"

# vLLM's attention on Blackwell (the sm_100 family, which includes the GB300's sm_103) is FlashInfer's, whose kernel
# launchers compile with nvcc at first use; vLLM's own images ship the compiler for it. Install the CUDA toolkit
# matching torch's CUDA (13.0 on GB300) and make it /usr/local/cuda. CUDA 12.8's compiler cannot target sm_103.
RUN TORCH_CUDA=$(/opt/venv/bin/python -c "import torch; print(torch.version.cuda or '')") && \
    if [ -n "${TORCH_CUDA}" ]; then \
        apt-get update && \
        apt-get install -y --no-install-recommends "cuda-toolkit-$(echo "${TORCH_CUDA}" | tr . -)" && \
        rm -rf /var/lib/apt/lists/* && \
        ln -sfn "/usr/local/cuda-${TORCH_CUDA}" /usr/local/cuda && \
        echo "torch CUDA ${TORCH_CUDA} -> $(/usr/local/cuda/bin/nvcc --version | tail -1)"; \
    fi

# ──────────────────────────────────── end base · task layer below ───────────────────────────────────
ARG VLLM_COMMIT=fbc3a1907aeb6beff59461e535045f17ac14306e

RUN git clone --filter=blob:none https://github.com/vllm-project/vllm.git /app/vllm && \
    cd /app/vllm && git checkout ${VLLM_COMMIT}
RUN cd /app/vllm && \
    VLLM_USE_PRECOMPILED=1 VLLM_PRECOMPILED_WHEEL_COMMIT=${VLLM_COMMIT} \
        uv pip install --python /opt/venv/bin/python -e . --no-deps && \
    /opt/venv/bin/python -c "import vllm; print('task vLLM', vllm.__version__)"

# Bake the test model into the HF cache (build-time network only), safetensors-only to stay small.
RUN /opt/venv/bin/python -c "\
from huggingface_hub import snapshot_download; \
p = snapshot_download('TinyLlama/TinyLlama-1.1B-Chat-v1.0', \
        ignore_patterns=['*.bin', '*.pth', '*.pt', '*.gguf', '*.onnx']); \
print('baked model at', p)"

# gguf: the parent commit's transformers_utils.config eagerly `import gguf`, so make sure it is installed and
# importable here. xgrammar + jsonschema are already in the base.
RUN uv pip install --python /opt/venv/bin/python gguf && \
    /opt/venv/bin/python -c "import gguf, jsonschema, xgrammar; print('deps ok: gguf jsonschema xgrammar')"

# From here on the container is offline — resolve the baked model from cache, never the Hub.
ENV HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
# vLLM's default flashinfer sampler would compile one more kernel at first use. Force the Triton/torch sampler
# (Triton bundles ptxas). Orthogonal to the beam-search grammar feature — purely a sampler backend choice — so it
# does not hint at the solution. Applies to both the agent's iteration and the grader.
ENV VLLM_USE_FLASHINFER_SAMPLER=0

# Drop .git so the agent cannot check out the fix commit or read history.
RUN rm -rf /app/vllm/.git


# ANTI-LEAK: DROP THE REFERENCE CHECKOUT. The base layers cloned vLLM into /opt/vllm-ref at a fixed
# REFERENCE commit so the dependency install happens once, before the task commit is checked out. That
# tree is a full working checkout WITH its .git, it is not this task's commit, and the reference can be a
# LATER commit than the one being reproduced -- which puts the post-fix source of the very file the agent
# is asked to fix into the image as a plain readable file, next to a commit graph that names the PR.
# Dropping /app/vllm/.git above closes the obvious door and leaves this one open. Nothing needs the tree
# at run time: its dependencies are installed into /opt/venv as ordinary wheels, and this task layer's
# own editable install of /app/vllm replaced its editable vllm. Deleted, then proven -- the check fails
# the BUILD, not a grade.
RUN rm -rf /opt/vllm-ref && \
    /opt/venv/bin/python -c "import vllm, os; assert not os.path.exists('/opt/vllm-ref'); \
print('reference checkout dropped; vLLM still importable from', os.path.dirname(vllm.__file__))"

# strip apt sources
RUN rm -rf /var/lib/apt/lists/* /etc/apt/sources.list /etc/apt/sources.list.d 2>/dev/null || true

WORKDIR /app/vllm
