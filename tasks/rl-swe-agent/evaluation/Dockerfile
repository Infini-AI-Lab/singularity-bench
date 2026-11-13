# Evaluation runtime for the held-out evaluation: the pinned multi-architecture Miles image (CUDA 13, SGLang,
# PyTorch) at Miles 53f5d942, with sglang-context-clip.patch applied. Build natively on the GB300 (arm64) host:
#   docker build -t rl-swe-agent-eval:local tasks/rl-swe-agent/evaluation
FROM radixark/miles@sha256:b8e4671c03adda98cc6165d5bbee3b322ce6baf7a8601dfe18a40da8bca376b8
WORKDIR /root/miles
RUN (git cat-file -e 53f5d94224b7fc584857eb2aba68ffdd364dc139^{commit} || \
     git fetch origin 53f5d94224b7fc584857eb2aba68ffdd364dc139) && \
    git checkout --detach 53f5d94224b7fc584857eb2aba68ffdd364dc139 && \
    python3 -m pip install --no-deps -e . && \
    test "$(python3 docker/image_inputs.py)" = 866414f4e508ee2ce12ef294560689010642754952a6e9795fc8c800fd326776
# The policy server clips each request's output budget to the context left after exact tokenization
# (EVAL_CONTEXT_LIMIT=65536, set by evaluate_checkpoint.py); it never truncates the conversation.
COPY patches/sglang-context-clip.patch /tmp/sglang-context-clip.patch
RUN cd /sgl-workspace/sglang && \
    git apply --check /tmp/sglang-context-clip.patch && \
    git apply /tmp/sglang-context-clip.patch && \
    rm /tmp/sglang-context-clip.patch
ENV PYTHONUNBUFFERED=1 CUDA_DEVICE_MAX_CONNECTIONS=1 TOKENIZERS_PARALLELISM=false \
    TORCH_CUDA_ARCH_LIST="10.3" CMAKE_CUDA_ARCHITECTURES=103 MAX_JOBS=8
# Records the package set, architecture and source commits of the built runtime.
RUN python3 -m pip freeze > /opt/runtime-packages.txt && \
    python3 -c 'import json,platform,subprocess,torch; from pathlib import Path; roots={"miles":"/root/miles","sglang":"/sgl-workspace/sglang","megatron":"/root/Megatron-LM"}; commits={k:subprocess.check_output(["git","-C",v,"rev-parse","HEAD"],text=True).strip() for k,v in roots.items()}; Path("/opt/eval-runtime.json").write_text(json.dumps({"architecture":platform.machine(),"torch":torch.__version__,"cuda":torch.version.cuda,"torch_cuda_architectures":torch.cuda.get_arch_list(),"source_commits":commits},indent=2)+"\n")'
ENTRYPOINT []
CMD ["bash"]
