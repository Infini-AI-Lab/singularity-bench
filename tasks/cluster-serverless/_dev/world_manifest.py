#!/usr/bin/env python
"""Hash every generated artifact so a world rebuilt on another machine can be PROVED identical.

The generated world, traces included, ships with the task, and it is reproducible: one master seed, a
pinned ServeGen checkout, pinned numerical packages and the frozen inputs listed below. But reproducible
is not a check. Someone regenerating the traces has no way to know the result matches the reference: the
per-trace `.meta.json` carries a `trace_sha256`, but it is computed from the trace just built, so it
agrees with itself no matter what went wrong. A DIFFERENT env var, a `CLUSTER_MASTER_SEED` left over in a
shell, a ServeGen checkout that moved, another CPU architecture -- each can produce a complete,
internally consistent, silently incomparable world.

So: `--write` records the hashes here, and `--verify` compares the files on disk (shipped or rebuilt)
against them and names every file that moved.
"""
from __future__ import annotations
import argparse, hashlib, json, os, sys

# Generated artifacts, relative to the task root. Globs, so every generated trace and its meta are covered.
PATTERNS = [
    "tests/catalogue-graded*.json",
    "tests/fleet-graded.json", "tests/fleet-graded.public.json",
    "tests/trace-graded-*.jsonl", "tests/trace-graded-*.meta.json",
    "environment/catalogue-dev*.json", "environment/fleet-dev.public.json",
    "environment/trace-public-*.jsonl", "environment/trace-public-*.meta.json",
    "tests/public/catalogue.json", "tests/suite-graded.json", "environment/suite-public.json",
]
# Inputs that DEFINE the world. A manifest that only covers outputs cannot tell "the generator changed"
# from "the inputs changed", and those need different responses.
INPUTS = ["_kit/models-openrouter-2026-09-08.json", "_kit/true-sizes.json",
          "_dev/snapshots/openrouter-2026-09-08.json.gz", "_kit/checkpoints.lock.json",
          "_kit/catalogue.py", "_kit/tracegen.py", "_kit/traceunits.py", "_kit/prompts.py",
          "_kit/seeds.py", "_kit/popularity.py", "_kit/physics.py", "_kit/checkpoints.py",
          "_kit/fleetgen.py", "_kit/stage_weights.py",
          "_dev/generator-requirements.txt", "_dev/regen_world.py",
          "_kit/workload.py", "_kit/workload-profile.json", "_kit/calibration-profile.json",
          "_kit/service_contract.py", "_kit/engine_args.py", "_kit/catalogue_profile.py", "_kit/market-anchors.json",
          "_dev/build_market_anchors.py", "_dev/market-requirements.txt",
          "_kit/model-architecture.json", "_dev/build_architecture.py", "_dev/resolve_architecture.py",
          "_kit/runtime_compat.py"]
# `.meta.json` files embed the sha of the trace beside them; hashing them too means a meta/trace
# mismatch is caught, not just a trace change.


def sha(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def collect(root: str) -> dict:
    import glob
    out = {}
    for name in INPUTS:
        if not os.path.isfile(os.path.join(root, name)):
            raise FileNotFoundError(name)
    for pat in PATTERNS + INPUTS:
        for p in sorted(glob.glob(os.path.join(root, pat))):
            if os.path.isfile(p):
                out[os.path.relpath(p, root)] = {"sha256": sha(p), "bytes": os.path.getsize(p)}
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
    ap.add_argument("--out", default="")
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--write", action="store_true")
    g.add_argument("--verify", action="store_true")
    a = ap.parse_args()
    root = os.path.abspath(a.root)
    path = a.out or os.path.join(root, "_dev", "world-manifest.json")

    sys.path.insert(0, os.path.join(root, "_kit"))
    import seeds                                                       # noqa: E402
    have = collect(root)

    if a.write:
        json.dump({"master_seed": seeds.MASTER_SEED, "n_files": len(have), "files": have},
                  open(path, "w"), indent=1)
        print(f"[manifest] {len(have)} files, master_seed {seeds.MASTER_SEED} -> {path}")
        return 0

    if not os.path.exists(path):
        print(f"[manifest] FATAL: no manifest at {path}; run --write on the reference machine first")
        return 2
    want = json.load(open(path))
    if want.get("master_seed") != seeds.MASTER_SEED:
        print(f"[manifest] FATAL: master seed is {seeds.MASTER_SEED}, manifest was built at "
              f"{want.get('master_seed')}. Unset CLUSTER_MASTER_SEED and rebuild.")
        return 2
    wf = want["files"]
    missing = [k for k in wf if k not in have]
    extra = [k for k in have if k not in wf]
    moved = [k for k in wf if k in have and have[k]["sha256"] != wf[k]["sha256"]]
    for k in missing:
        print(f"  MISSING   {k}")
    for k in moved:
        print(f"  DIFFERS   {k}  ({wf[k]['bytes']} -> {have[k]['bytes']} bytes)")
    for k in extra:
        print(f"  untracked {k}")
    ok = not missing and not moved and not extra
    print(f"[manifest] {len(wf) - len(missing) - len(moved)}/{len(wf)} match"
          f"{'' if ok else '  -- WORLD DIFFERS'}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
