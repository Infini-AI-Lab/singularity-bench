# The protected author image: the task runtime plus the selected agent CLI, installed privately.
#   docker build -f breakwater/author/Dockerfile --build-context task_environment=tasks/<name>/environment \
#     --build-arg TASK_BASE=<name>:local --build-arg AUTHOR_CLI=claude-code -t <name>-author:claude-code breakwater
# The agent runs as author (UID 1100); every command it issues runs as worker (UID 1101).
ARG TASK_BASE
FROM ${TASK_BASE}
USER root
ENV DEBIAN_FRONTEND=noninteractive
COPY author/restore_apt.py /opt/breakwater-restore-apt.py
RUN python3 -I /opt/breakwater-restore-apt.py && apt-get update && apt-get install -y --no-install-recommends \
    build-essential iptables iproute2 procps ca-certificates curl git ripgrep \
    python3-venv util-linux && rm -rf /var/lib/apt/lists/*
COPY author/ /opt/breakwater-runtime/
COPY capture/stdio_capture.c capture/stdio_capture.h /opt/breakwater-runtime/
COPY --from=task_environment agent-policy.json /etc/breakwater-task.json
ARG AUTHOR_CLI=claude-code
RUN --mount=type=tmpfs,target=/tmp,size=12884901888 \
    python3 -I /opt/breakwater-runtime/purge.py --task-container --preserve-runtimes \
    && bash /opt/breakwater-runtime/install.sh "$AUTHOR_CLI"
ENTRYPOINT ["python3", "-I", "/opt/breakwater-runtime/bootstrap.py"]
