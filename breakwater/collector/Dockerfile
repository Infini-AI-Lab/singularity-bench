FROM python:3.12-slim-bookworm
RUN apt-get update && apt-get install -y --no-install-recommends strace \
    && rm -rf /var/lib/apt/lists/*
COPY observe.py syscall_trace.py contract.py snapshots.py process_scope.py /opt/
ENTRYPOINT ["python3", "-I", "/opt/observe.py"]
