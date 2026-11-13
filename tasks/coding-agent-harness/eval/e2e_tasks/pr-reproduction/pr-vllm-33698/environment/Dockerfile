# Reproduce-a-PR task — vLLM PR #33698 ([Core][BugFix] Fix PP KV-cache sharding memory validation).
#
# Stages vLLM at the commit BEFORE the PR, EDITABLE + PRECOMPILED; the fix is pure-Python (kv_cache_utils.py) so
# the agent's edits under /app/vllm/vllm/ are live with NO CUDA rebuild. Offline at run time (no network).
#
# The fix + grader run NO CUDA kernels (KV-cache config arithmetic, no model weights) — but a GPU is still
# REQUIRED: vLLM's VllmConfig does device inference at construction and raises "Failed to infer device type"
# with no visible GPU. The graded test builds `ModelConfig(max_model_len=512)`, whose default model is
# Qwen/Qwen3-0.6B, so its *config* (config.json + tokenizer, NOT the ~1.2 GB weights — the model is never
# loaded) is baked into the HF cache at build time and HF_HUB_OFFLINE is set so it resolves offline.
#
# ─────────────────────────────────────────────────────────────────────────────────────────────────────
# BASE (torch-2.10). The image builds standalone; no pre-built local base image is required:
#   docker build -t pr-vllm-33698:local tasks/coding-agent-harness/eval/e2e_tasks/pr-reproduction/pr-vllm-33698/environment
#
# The torch version is load-bearing: VLLM_USE_PRECOMPILED fetches the task commit's .so (built against THAT
# commit's torch) and installs it --no-deps onto this base's torch, so it only loads if the torch matches.
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
ARG VLLM_REF=a572baff5e5fde4aa3fb92961de04eb043dc5cf4
RUN git clone --filter=blob:none https://github.com/vllm-project/vllm.git /opt/vllm-ref && \
    cd /opt/vllm-ref && git checkout ${VLLM_REF}
RUN uv venv /opt/venv --python 3.12
ENV VLLM_USE_PRECOMPILED=1
RUN cd /opt/vllm-ref && \
    VLLM_USE_PRECOMPILED=1 VLLM_PRECOMPILED_WHEEL_COMMIT=${VLLM_REF} \
        uv pip install --python /opt/venv/bin/python -e . --torch-backend=auto && \
    uv pip install --python /opt/venv/bin/python pytest gguf && \
    /opt/venv/bin/python -c "import vllm; print('base vLLM', vllm.__version__)"

# ──────────────────────────────────── end base · task layer below ───────────────────────────────────
ARG VLLM_COMMIT=afdce12c89555ce7b7bd4f3215b5d844de0a32ed

RUN git clone --filter=blob:none https://github.com/vllm-project/vllm.git /app/vllm && \
    cd /app/vllm && git checkout ${VLLM_COMMIT}
RUN cd /app/vllm && \
    VLLM_USE_PRECOMPILED=1 VLLM_PRECOMPILED_WHEEL_COMMIT=${VLLM_COMMIT} \
        uv pip install --python /opt/venv/bin/python -e . --no-deps && \
    /opt/venv/bin/python -c "import vllm; from vllm.config import ModelConfig, VllmConfig; \
from vllm.v1.core.kv_cache_utils import get_kv_cache_configs; \
from vllm.v1.kv_cache_interface import FullAttentionSpec, KVCacheConfig, KVCacheGroupSpec, KVCacheTensor; \
print('task vLLM', vllm.__version__, '+ config/kv-cache imports OK')"

# Bake ONLY the default model's config + tokenizer (not the ~1.2GB weights — the test never loads the model).
RUN /opt/venv/bin/python -c "\
from huggingface_hub import snapshot_download; \
p = snapshot_download('Qwen/Qwen3-0.6B', \
        ignore_patterns=['*.safetensors', '*.bin', '*.pth', '*.pt', '*.gguf', '*.onnx']); \
print('baked config at', p)"

# From here on the container is offline — resolve the baked model config from cache, never the Hub.
ENV HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1

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
