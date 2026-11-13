# The rl-swe-agent task runtime: the pinned multi-architecture Miles image (CUDA 13, SGLang, Megatron-LM,
# PyTorch) at Miles 53f5d942, with the task tree at /app and mini_slime installed in editable mode.
# Build with the task directory as context: bash tasks/rl-swe-agent/environment/build.sh
FROM radixark/miles@sha256:b8e4671c03adda98cc6165d5bbee3b322ce6baf7a8601dfe18a40da8bca376b8
WORKDIR /root/miles
RUN git fetch origin 53f5d94224b7fc584857eb2aba68ffdd364dc139 && \
    git checkout --detach 53f5d94224b7fc584857eb2aba68ffdd364dc139 && \
    python3 -m pip install --no-deps -e .
WORKDIR /app
COPY . .
RUN python3 -m pip install --no-deps -e . && python3 -m pip freeze > /opt/rl-swe-agent-runtime-packages.txt
ENV PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
ENTRYPOINT []
CMD ["bash"]
