#!/usr/bin/env python3
"""prefetch_fp8_kernel.py -- IMAGE BUILD STEP. Not part of the task; never runs at grade time.

transformers does not ship the kernel it uses for this checkpoint's fine-grained FP8
weights. It fetches it from the HF Hub at first use, through the `kernels` package
(transformers.integrations.hub_kernels: "finegrained-fp8" -> a pinned version of
kernels-community/finegrained-fp8). The container has NO INTERNET at run time, so without
this step the provided /app/server.py loads the weights, reports ready, and then answers
every request with

    ImportError: finegrained-fp8 kernel unavailable: `kernels` is either not installed ...

With transformers 5.17.0 and this checkpoint's quant_method ("fp8", e4m3, dynamic), two things are
needed for the OFFLINE load to work, and `kernels` does neither on its own:
  1. the COMPLETE repo snapshot. `kernels` downloads only the build variant it needs, and
     an offline snapshot lookup then refuses the cache as incomplete
     (IncompleteSnapshotError: 24 files missing).
  2. a refs/v<N> file naming the commit. Offline, `kernels` resolves "version N" from the
     cache's refs/ directory, which a by-commit download never writes.

The last step PROVES the result: it re-loads the kernel in a fresh process with
HF_HUB_OFFLINE=1, and the build fails if that does not return a module. The Dockerfile
then sets HF_HUB_OFFLINE=1 for the image, so nothing waits on a blocked network.
"""
import os, pathlib, subprocess, sys

from huggingface_hub import HfApi
from transformers.integrations.hub_kernels import _HUB_KERNEL_MAPPING

NAME = "finegrained-fp8"
spec = _HUB_KERNEL_MAPPING[NAME]          # the version THIS transformers release asks for
repo, version = spec["repo_id"], spec["version"]
api = HfApi()
commit = next(b.target_commit for b in api.list_repo_refs(repo, repo_type="kernel").branches
              if b.name == f"v{version}")
snap = pathlib.Path(api.snapshot_download(repo, repo_type="kernel", revision=commit))
refs = snap.parents[1] / "refs"
refs.mkdir(exist_ok=True)
(refs / f"v{version}").write_text(commit)
print(f"[prefetch] {repo} v{version} -> {commit} ({sum(1 for p in snap.rglob('*') if p.is_file())} files)")

check = ("from transformers.integrations.hub_kernels import lazy_load_kernel; "
         f"k = lazy_load_kernel({NAME!r}); assert k is not None, 'kernel did not load offline'; print(k.__file__)")
out = subprocess.run([sys.executable, "-c", check], env={**os.environ, "HF_HUB_OFFLINE": "1"},
                     capture_output=True, text=True)
print(f"[prefetch] offline load: {(out.stdout.strip().splitlines() or ['<no output>'])[-1]}")
if out.returncode != 0:
    sys.exit(f"[prefetch] OFFLINE LOAD FAILED -- the starter server would not serve:\n{out.stderr[-2000:]}")
