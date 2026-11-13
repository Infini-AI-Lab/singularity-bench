FROM python:3.12-slim-bookworm
RUN apt-get update && apt-get install -y --no-install-recommends gcc libc6-dev && rm -rf /var/lib/apt/lists/*
COPY stdio_capture.c stdio_capture.h /src/
RUN gcc -Wall -Wextra -Werror -O2 -fPIC -shared /src/stdio_capture.c -o /opt/stdio_capture.so
RUN mkdir -p /opt/frozen/sglang && printf 'def genuine(): return 42\n' > /opt/frozen/sglang/__init__.py
COPY fixture.py /opt/fixture.py
ENTRYPOINT []
CMD ["sleep", "infinity"]
