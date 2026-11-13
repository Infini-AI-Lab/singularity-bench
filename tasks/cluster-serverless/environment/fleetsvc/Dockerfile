# The fleet. ONE image for both backends: the emulator and the real vLLM supervisor ship together and are
# selected at run time by CLUSTER_FLEET_MODE, so the two cannot drift apart.
#
# Pinned serving version. This immutable multi-platform index contains both amd64 and arm64 builds.
# Each node builds its native image (`bin/cluster.sh stage`); GB300 trays use the arm64 build, and an
# amd64 image must not be copied to a Grace tray.
FROM vllm/vllm-openai:v0.28.0@sha256:61fc8a896b0a4fbbbdc063bc4b0dbc25ce98e02b5050c24aeb7830ac02039b14

# The image ships uvicorn/fastapi but not aiohttp, which both fleet backends use for their own server.
RUN pip install --no-cache-dir aiohttp==3.14.3
# Scheduler instrumentation: ai-dynamo's instrumented vLLM scheduler publishes native forward-pass
# metrics which, with vLLM's own KV-event publisher, feed the workers' read-only telemetry bridge. It is
# the same for every engine and policy; scheduling itself stays vLLM's.
RUN pip install --no-cache-dir --no-deps ai-dynamo==1.5.0 ai-dynamo-runtime==1.5.0
RUN pip install --no-cache-dir msgspec==0.19.0 pyzmq==27.1.0
ENV DYN_LOG=warn

# Record the installed runtime during build.
RUN python3 -c "import platform, torch; \
print('fleet image:', platform.machine(), '| torch', torch.__version__, \
      '| cuda', torch.version.cuda, '| gpu archs', torch.cuda.get_arch_list())"

WORKDIR /fleet
COPY kit/ /fleet/kit/
COPY entrypoint.sh /fleet/entrypoint.sh
RUN chmod +x /fleet/entrypoint.sh
# The base image's entrypoint is `vllm serve`; the fleet supervisor launches those itself.
ENTRYPOINT ["/bin/bash", "/fleet/entrypoint.sh"]
