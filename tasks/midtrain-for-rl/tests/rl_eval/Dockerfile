# syntax=docker/dockerfile:1.7
# Build context: this folder. Targets Grace/GB300 (ARM64).
ARG BASE_IMAGE=lmsysorg/sglang:v0.5.15.post1-cu130@sha256:00264faaade6f6fefb1d8f191147680ea49d9b03fd3871e5963bb9765412bb82
# Validate the source in a disposable stage; grading files never enter runtime.
FROM --platform=${TARGETPLATFORM} ${BASE_IMAGE} AS vendor-check
WORKDIR /src
COPY vendor vendor
COPY verify_vendor.py vendor_manifest.json checkpoint_check.py ./
RUN python3 verify_vendor.py

FROM --platform=${TARGETPLATFORM} ${BASE_IMAGE} AS runtime
SHELL ["/bin/bash", "-o", "pipefail", "-c"]
ARG TARGETARCH
ARG TORCH_ARCH=10.3
ARG NVTE_ARCH=103a
RUN test "$(dpkg --print-architecture)" = "${TARGETARCH}"
ENV DEBIAN_FRONTEND=noninteractive TORCH_CUDA_ARCH_LIST=${TORCH_ARCH} \
    OMP_NUM_THREADS=4 PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 \
    CUDA_DEVICE_MAX_CONNECTIONS=1 TOKENIZERS_PARALLELISM=false
RUN apt-get update && apt-get install -y --no-install-recommends \
    git build-essential ninja-build cmake python3-dev python3-venv \
    libnuma-dev libz3-dev curl ca-certificates iptables numactl pciutils procps ripgrep wget \
    && rm -rf /var/lib/apt/lists/*

# Do not let dependency resolution replace the image's CUDA/Torch kernel stack.
RUN python3 - <<'PY'
import importlib.metadata as m
names = ('torch', 'torchvision', 'torchaudio', 'triton', 'sglang',
         'sgl-kernel', 'sglang-kernel', 'flashinfer-python', 'flashinfer-cubin',
         'flashinfer-jit-cache', 'transformers', 'tilelang', 'nvidia-cutlass-dsl')
with open('/opt/runtime-constraints.txt', 'w') as out:
    for name in names:
        try: out.write(f'{name}=={m.version(name)}\n')
        except m.PackageNotFoundError: pass
PY
ENV PIP_CONSTRAINT=/opt/runtime-constraints.txt
ARG MAX_JOBS=8
ARG TE_COMMIT=c9877beb87ad7e711e1869dd0b5062167ede447a
ARG APEX_COMMIT=10417aceddd7d5d05d7cbf7b0fc2daad1105f8b4
ARG MEGATRON_COMMIT=1dcf0dafa884ad52ffb243625717a3471643e087
ARG TMS_COMMIT=8d30c59ca12a68d9deccbc9c6599076a1218cbc5
# CUDA-13 TE wheels can target a newer cuBLAS; compile against this base instead.
RUN NVTE_FRAMEWORK=pytorch NVTE_CUDA_ARCHS=${NVTE_ARCH} MAX_JOBS=${MAX_JOBS} \
    CMAKE_BUILD_PARALLEL_LEVEL=${MAX_JOBS} python3 -m pip install --no-build-isolation \
    "git+https://github.com/NVIDIA/TransformerEngine.git@${TE_COMMIT}"
RUN MAX_JOBS=${MAX_JOBS} NVCC_APPEND_FLAGS="--threads 2" \
    python3 -m pip install --no-build-isolation \
    --config-settings "--build-option=--cpp_ext --cuda_ext --parallel ${MAX_JOBS}" \
    "git+https://github.com/NVIDIA/apex.git@${APEX_COMMIT}"
RUN TMS_CUDA_MAJOR=13 MAX_JOBS=${MAX_JOBS} python3 -m pip install \
    "git+https://github.com/zhuzilin/torch_memory_saver.git@${TMS_COMMIT}"

WORKDIR /opt/midtrain-rl
COPY --from=vendor-check /src/vendor ./vendor
COPY requirements-midtrain.txt check_runtime.py ./
RUN git clone https://github.com/NVIDIA/Megatron-LM.git /opt/Megatron-LM \
    && cd /opt/Megatron-LM && git checkout ${MEGATRON_COMMIT} \
    && git apply --check /opt/midtrain-rl/vendor/slime/docker/patch/latest/megatron.patch \
    && git apply /opt/midtrain-rl/vendor/slime/docker/patch/latest/megatron.patch \
    && git apply --check /opt/midtrain-rl/vendor/slime/docker/patch/latest/megatron-sglang-aligned.patch \
    && git apply /opt/midtrain-rl/vendor/slime/docker/patch/latest/megatron-sglang-aligned.patch \
    && python3 -m pip install --no-deps -e .
# Use the upstream ARM64 router, not Slime's x86-only custom router wheel.
RUN python3 -m pip install --ignore-installed 'PyJWT>=2.8.0' \
    && python3 -m pip install -r vendor/slime/requirements.txt \
    'ray[default]==2.58.0' sglang-router==0.3.2 psutil \
    'numpy==1.26.4' 'scipy==1.17.1' -r requirements-midtrain.txt \
    && python3 -m pip install --no-deps -e vendor/slime
ENV PYTHONPATH=/opt/midtrain-rl:/opt/midtrain-rl/vendor/slime:/opt/Megatron-LM
# Keep imported source usable by the unprivileged agent and remove upstream agent hooks.
RUN rm -rf /opt/Megatron-LM/.agents /opt/Megatron-LM/.claude /opt/Megatron-LM/.cursorrules /opt/Megatron-LM/skills \
    && find /opt/Megatron-LM \( -name AGENTS.md -o -name CLAUDE.md \) -delete \
    && make -C /opt/Megatron-LM/megatron/core/datasets \
    && python3 -c 'from megatron.core.datasets import helpers_cpp' \
    && python3 -m pip check \
    && python3 check_runtime.py --image-only \
    && python3 vendor/slime/train.py --help > /opt/slime-help.txt \
    && python3 -m pip freeze > /opt/training-packages.txt
ENV HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1
ENTRYPOINT []
CMD ["bash"]

# The task runtime (environment/Dockerfile tags it midtrain-for-rl:local). Breakwater adds the author CLI
# in its protected overlay; its evaluation stage runs the RL stage in the same image, with this task's
# tests/ mounted read-only.
FROM runtime AS agent
ENV HF_DATASETS_OFFLINE=1
WORKDIR /app

# This directory at /opt/midtrain-rl: prepare.sh --private-data builds the RL mixture and the
# evaluation sets in this image, which can also run the RL stage by hand (README.md).
FROM runtime AS rl
COPY . .
RUN python3 verify_vendor.py

# A build without --target produces the rl image.
FROM rl AS default
