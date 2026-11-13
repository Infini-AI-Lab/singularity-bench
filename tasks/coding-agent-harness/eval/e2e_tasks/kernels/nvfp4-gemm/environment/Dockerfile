# Frontier-kernel task: write an NVFP4 block-scaled GEMM that beats bf16 (needs Blackwell FP4 tensor cores
# -> Triton tl.dot_scaled / CuTe DSL / CUDA). Base = nvidia/cuda 12.8 devel on x86_64, CUDA 13 on aarch64
# (ships nvcc); torch pinned to 2.11.0+cu128 / +cu130. The agent gets Triton + CuTe DSL + nvcc.
#
# ARCHITECTURE-AWARE, but not because of a vendored binary: this task's oracle is FlashInfer's `mm_fp4`,
# which is pure python over JIT/cuDNN kernels, so there is nothing here to ABI-match. What the
# architecture decides is the CUDA toolkit:
#
#   x86_64   -> CUDA 12.8 + torch cu128.
#   aarch64  -> CUDA 13 toolkit + torch cu130, because GB300 is sm_103 and nvcc only learned it in 12.9 --
#               and this task hands the agent nvcc, so a toolchain that cannot name the GPU is not a
#               toolchain.
#
# The task is Blackwell-only by construction: FP4 tensor cores arrived with sm_100, and on anything older
# tests/oracle_bench.py raises OracleUnavailable so the cell is VOIDED rather than scored against a ceiling
# that does not exist.
#
# nvidia-cutlass-dsl is pinned to 4.8.0.dev0 -- the version flashinfer-python resolves for itself -- and
# that pin is load-bearing, not incidental. With this stack `import torch, cutlass, cutlass.cute,
# flashinfer, numpy` succeeds. Under 4.7.1, `mm_fp4[cute-dsl]` fails to compile with an internal
# pass-option error (`no such option enable-pyir`, a duplicated `cubin-chip`) and tests/oracle_bench.py
# rejects it, which costs the ceiling a whole backend. Under 4.8.0.dev0 it is correct at every probe shape
# and the FASTEST route at large graded shapes on Blackwell.
#
# The `import` line at the end of the install RUN is the guard: if a dev build breaks the import, the
# BUILD fails, loudly, instead of every graded cell voiding on an oracle that will not load.
FROM nvidia/cuda:12.8.1-devel-ubuntu24.04
ENV DEBIAN_FRONTEND=noninteractive

# python + build tools (apt sources are removed at the end)
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3 python3-pip python3-dev git curl ca-certificates build-essential iptables iproute2

COPY oracle-requirements.txt /opt/oracle-requirements.txt

# torch + the agent's CuTe DSL + the Blackwell oracle, all chosen from the build host's architecture.
# One RUN so the case-variables survive; the echo is what a build log is read for afterwards.
RUN set -eu; \
    case "$(uname -m)" in \
      aarch64) IDX=cu130; CUDA_PY=13.4.1; CUDA_BIND=13.4.2; DSL='nvidia-cutlass-dsl[cu13]==4.8.0.dev0'; \
               apt-get update && apt-get install -y --no-install-recommends cuda-toolkit-13-0 && \
               ln -sfn /usr/local/cuda-13.0 /usr/local/cuda ;; \
      *)       IDX=cu128; CUDA_PY=12.9.7; CUDA_BIND=12.9.8; DSL='nvidia-cutlass-dsl==4.8.0.dev0' ;; \
    esac; \
    echo "build host $(uname -m) -> torch ${IDX}, ${DSL}, nvcc $(nvcc --version | tail -1)"; \
    pip3 install --break-system-packages --no-cache-dir torch==2.11.0 --index-url https://download.pytorch.org/whl/${IDX}; \
    pip3 install --break-system-packages --no-cache-dir "${DSL}" "cuda-python==${CUDA_PY}" "cuda-bindings==${CUDA_BIND}" -r /opt/oracle-requirements.txt; \
    python3 -c "import torch, cutlass, cutlass.cute, flashinfer, numpy; \
print('arch', __import__('platform').machine(), '| torch', torch.__version__, \
      '| gpu archs', torch.cuda.get_arch_list(), '| flashinfer', flashinfer.__version__)"

# Bake FlashInfer's prebuilt cubins at build time, while there is a network: its trtllm-gen backends load
# them at run time and the grading container has no network. The cutlass/cuTile backends JIT from bundled
# sources and need only nvcc, which is why the oracle wrappers ask for those where there is a choice.
ENV FLASHINFER_CUBIN_DIR=/opt/flashinfer-cubins
RUN python3 -m flashinfer download-cubin

# SEAL THE ORACLE. This image's GRADER uses FlashInfer's mm_fp4 as its oracle -- FlashInfer IS the
# ceiling -- and every reward on this task is a ratio against it. Installed normally, that package sits on
# the agent's import path and in `pip list`: the denominator is handed to the numerator, `import flashinfer`
# is a 1.000, and the task stops measuring anything. It is moved somewhere only the grader looks.
#
# The cell overlay (breakwater/cell/Dockerfile.overlay) makes this directory and the cubins root-only, and
# submitted code runs as the worker. A harness runs only in the overlaid cell image, never in this image.
#
# The dependencies stay in site-packages. What must not leak is the kernel, not numpy.
RUN set -eu; \
    PKG="$(python3 -c 'import flashinfer, os; print(os.path.dirname(os.path.abspath(flashinfer.__file__)))')"; \
    SITE="$(dirname "$PKG")"; \
    mkdir -p /opt/oracle-libs; \
    for d in "$SITE"/flashinfer*; do [ -e "$d" ] || continue; mv "$d" /opt/oracle-libs/; done; \
    if python3 -c 'import flashinfer' 2>/dev/null; then \
        echo "FAIL: flashinfer is still importable without /opt/oracle-libs -- this image would hand"; \
        echo "      the agent the grader's own oracle. Find the remaining copy and move it too."; \
        exit 1; \
    fi; \
    PYTHONPATH=/opt/oracle-libs python3 -c "import flashinfer; \
print('oracle-only flashinfer', flashinfer.__version__, '-> /opt/oracle-libs')"

# given to the agent: the torch REFERENCE (correctness spec) + the perf-baseline self-assessment tool + the stub
COPY reference.py /app/reference.py
COPY measure.py /app/measure.py
COPY nvfp4_gemm.py /app/nvfp4_gemm.py

# process tools (ps, kill, ...) for the harness and the commands it runs
RUN apt-get update && apt-get install -y procps

# strip apt sources LAST
RUN rm -rf /var/lib/apt/lists/* /etc/apt/sources.list /etc/apt/sources.list.d

WORKDIR /app
