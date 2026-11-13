#!/bin/bash
set -euo pipefail
profile=$1
groupadd --gid 1100 author
groupadd --gid 1101 worker
groupadd --gid 1200 workspace
useradd --uid 1100 --gid author --groups workspace --create-home author
useradd --uid 1101 --gid worker --groups workspace --create-home worker
chown -R root:author /home/author
chmod 750 /home/author
case "$profile" in
  claude-code) installer=claude ;;
  codex)
    apt-get update
    apt-get install -y --no-install-recommends cmake pkg-config libssl-dev protobuf-compiler
    rm -rf /var/lib/apt/lists/*
    installer=codex ;;
  *) echo 'Unsupported protected CLI' >&2; exit 1 ;;
esac
python3 -I "/opt/breakwater-runtime/install_${installer}.py"
printf '%s\n' "$profile" > /opt/author-cli/harness
chown -R root:author /opt/author-cli
chmod 750 /opt/author-cli
if [ "$profile" = codex ]; then chmod 4750 /opt/author-cli/codex-worker; fi
gcc -Wall -Wextra -Werror -O2 -fPIC -shared /opt/breakwater-runtime/stdio_capture.c -o /opt/breakwater-runtime/stdio_capture.so
gcc -Wall -Wextra -Werror -O2 /opt/breakwater-runtime/worker-exec.c /opt/breakwater-runtime/stdio_capture.c -o /opt/breakwater-runtime/worker-exec
# Pin the interpreter from this image at build time. Runtime PATH, workspace
# files and the author environment must never select a privileged entrypoint.
author_python=$(python3 -I - <<'PY'
import json
from pathlib import Path
import subprocess
import sys
interpreter = Path(sys.executable).resolve(strict=True)
for path in (interpreter, *interpreter.parents):
    info = path.stat()
    if info.st_uid != 0 or info.st_mode & 0o022:
        raise SystemExit(f'Author interpreter path must be root-owned and not group/world-writable: {path}')
# Match the launcher's cleared environment and author identity. Root ownership
# alone does not make an interpreter below a private directory executable.
try:
    subprocess.run([str(interpreter), '-I', '-c', 'pass'], check=True,
                   cwd='/', env={}, user=1100, group=1100, extra_groups=[1100, 1200])
except (OSError, subprocess.CalledProcessError) as error:
    raise SystemExit(f'Author cannot execute image interpreter {interpreter}: {error}')
print(json.dumps(str(interpreter)))
PY
)
gcc -Wall -Wextra -Werror -O2 "-DAUTHOR_PYTHON=$author_python" /opt/breakwater-runtime/author-exec.c /opt/breakwater-runtime/stdio_capture.c -o /opt/breakwater-runtime/author-exec
find /usr /bin /sbin -xdev -type f -perm /6000 -exec chmod a-s '{}' +
chown root:author /opt/breakwater-runtime/worker-exec
chmod 4750 /opt/breakwater-runtime/worker-exec
chown root:author /opt/breakwater-runtime/author-exec
chmod 4750 /opt/breakwater-runtime/author-exec
mkdir -p /etc/claude-code /opt/author-tools
cp /opt/breakwater-runtime/managed-settings.json /etc/claude-code/managed-settings.json
gcc -Wall -Wextra -Werror -O2 -DGIT_MODE /opt/breakwater-runtime/worker-exec.c /opt/breakwater-runtime/stdio_capture.c -o /opt/author-tools/git
chown root:author /opt/author-tools/git
chmod 4750 /opt/author-tools/git
chown root:author /opt/author-tools
chmod 750 /opt/author-tools
cp /opt/author-cli/harness /etc/breakwater-author-cli
python3 -I /opt/breakwater-runtime/install_task.py
