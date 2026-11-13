# Frontier-kernel task: write a FlashAttention BACKWARD that beats FlashAttention-2 (needs Hopper-specific
# optimizations — warp specialization + wgmma + TMA, i.e. FA-3-class work -> CUDA / CuTe DSL). Base =
# nvidia/cuda devel (ships nvcc), CUDA 13 by default. torch pinned to 2.11.0, from the cu130 index by default.
# The agent gets triton + the full CuTe DSL env (quack-kernels -> nvidia-cutlass-dsl + cuda-python + tvm_ffi) +
# nvcc; the FA-3 backward implementation (flash_attn.cute) is NOT installed and can't be fetched (no network).
#
# ARCHITECTURE-PORTABLE, and Blackwell-capable. Builds natively on x86_64 AND aarch64 (Grace) -- the base is
# multi-arch and every wheel below publishes manylinux_2_28_aarch64 -- and the stack carries device code for
# Hopper (sm_90) AND datacenter Blackwell incl. GB300 (sm_103). Two knobs, both ARGs:
#
#   * CUDA 13, not 12.8. GB300 is compute capability 10.3 and nvcc only learned sm_103 in CUDA 12.9. This
#     task hands the agent nvcc and the CuTe DSL; an nvcc that cannot name the target GPU is not a toolchain.
#   * torch from the cu130 index, not cu128. `cuobjdump --list-elf` over the two wheels: cu128 carries
#     sm_100/sm_100a but NO sm_103; cu130 adds sm_103a and keeps sm_90/sm_90a, so one wheel serves H100/H200
#     and GB300. quack takes its [cu13] extra to match, which pulls nvidia-cutlass-dsl's cu13 libs instead
#     of cu12 -- mixing those with a cu13 torch is how the CuTe JIT ends up loading two CUDA runtimes.
#
# Nothing here is pinned to a CUDA major by an ABI: this task's oracle is JIT-compiled Python, not a
# vendored .so, so the same CUDA 13 stack serves every architecture.
ARG CUDA_TAG=13.0.2-devel-ubuntu24.04
FROM nvidia/cuda:${CUDA_TAG}
ARG TORCH_CUDA=cu130
# FA-4 backward can hang on sm_103 with CuTe DSL 4.6.0/4.6.1. Quack 0.6.4 pins
# BOTH the DSL and its cu13 runtime to 4.6.2; upgrading only the meta-package can
# leave the broken shared library loaded. See Dao-AILab/flash-attention#2797.
ARG QUACK_SPEC=quack-kernels[cu13]==0.6.4
ENV DEBIAN_FRONTEND=noninteractive

# python + build tools (apt sources are removed at the end)
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3 python3-pip python3-dev git curl ca-certificates build-essential iptables iproute2

# torch (brings triton) + the CuTe DSL env (quack -> nvidia-cutlass-dsl + cuda-python + tvm_ffi) + numpy
RUN pip3 install --break-system-packages --no-cache-dir torch==2.11.0 --index-url https://download.pytorch.org/whl/${TORCH_CUDA} && \
    pip3 install --break-system-packages --no-cache-dir "${QUACK_SPEC}" numpy

# given to the agent: the torch REFERENCE (correctness spec) + the perf-baseline self-assessment tool + the stub
COPY reference.py /app/reference.py
COPY measure.py /app/measure.py
COPY attn_backward.py /app/attn_backward.py

# process tools (ps, kill, ...) for the harness and the commands it runs
RUN apt-get update && apt-get install -y procps

# Fail the BUILD, not the trial. The printed arch list is how you tell from a build log whether this image
# can drive the node's GPUs: sm_90 for H100/H200, sm_100 for datacenter Blackwell (sm_100 cubins run on
# GB300's sm_103 by CUDA's within-family minor-version compatibility, which is what torch itself relies on).
RUN python3 -c "import torch, quack, numpy; print('arch', __import__('platform').machine(), '| torch', torch.__version__, '| gpu archs', torch.cuda.get_arch_list())"
RUN python3 -c "from importlib.metadata import version; packages = ('quack-kernels', 'nvidia-cutlass-dsl', 'nvidia-cutlass-dsl-libs-base', 'nvidia-cutlass-dsl-libs-core', 'nvidia-cutlass-dsl-libs-cu12', 'nvidia-cutlass-dsl-libs-cu13'); versions = {p: version(p) for p in packages}; print('FA-4 runtime:', versions); assert all(v not in ('4.6.0', '4.6.1') for p, v in versions.items() if p != 'quack-kernels'), versions"

# strip apt sources LAST
RUN rm -rf /var/lib/apt/lists/* /etc/apt/sources.list /etc/apt/sources.list.d

WORKDIR /app
