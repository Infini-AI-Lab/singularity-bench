"""Build-time task policy and offline cache ownership; never accepts model input."""
import json
import os
from pathlib import Path
import shutil
import sys

root = Path('/opt/breakwater-runtime')
config = json.loads(Path('/etc/breakwater-task.json').read_text())
if not config['name'] or not Path(config['workdir']).is_absolute():
    raise SystemExit('A task-owned agent-policy.json is required')
Path('/etc/breakwater-task.json').chmod(0o644)
(root / 'worker-python').symlink_to(sys.executable)
# Keep prefetched CUDA/HF kernels available offline, without opening /root.
for relative in ('.cache/huggingface', '.cache/torch', '.cache/torch_extensions', '.triton'):
    source, target = Path('/root') / relative, Path('/home/worker') / relative
    if source.exists():
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(source), target)
for directory, dirs, files in os.walk('/home/worker'):
    for path in [Path(directory), *(Path(directory, name) for name in files)]:
        os.chown(path, 1101, 1200, follow_symlinks=False)
