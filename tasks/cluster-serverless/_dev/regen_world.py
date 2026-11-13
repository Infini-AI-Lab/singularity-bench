"""Rebuild the public/private world from pinned sources; publish only on success."""
from __future__ import annotations
import argparse
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

T = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(T / "_kit"))
import catalogue
import fleetgen
import tracegen
import workload
import world_manifest


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=1) + "\n")


def generate(job):
    root, cat, settings, suite, case, provenance = job
    tracegen.CACHE = str(Path(root) / "cache")
    items, meta = workload.generate_case(cat, settings, case["load_multiplier"],
                                         case["seed"], suite["source_split"])
    public = suite["source_split"] == "public"
    path = Path(root) / ("environment" if public else "tests") / (
        f"trace-{'public' if public else 'graded'}-{case['name']}.jsonl")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as out:
        for item in items:
            out.write(json.dumps(item) + "\n")
    sha = hashlib.sha256(path.read_bytes()).hexdigest()
    write(path.with_suffix(".meta.json"), meta | provenance | {"trace_sha256": sha, "n_items": len(items)})
    print(f"generated {path.name} ({suite['source_split']}): {meta['n_requests']} requests", flush=True)
    return suite["source_split"], case["name"], sha


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--write-manifest", action="store_true")
    ap.add_argument("--jobs", type=int, default=2)
    ap.add_argument("--output-root", type=Path, help="rebuild and verify in a separate directory")
    a = ap.parse_args()
    if not 1 <= a.jobs <= 4:
        raise SystemExit("use 1–4 generation jobs on a shared host")
    provenance = tracegen.servegen_provenance()
    if provenance.get("servegen_commit") != tracegen.SERVEGEN_PIN or provenance.get("servegen_dirty"):
        raise SystemExit("clean pinned ServeGen checkout required")
    initial_inputs = {name: world_manifest.sha(T/name) for name in world_manifest.INPUTS}
    settings = workload.profile()
    calibration = json.loads((T / "_kit/calibration-profile.json").read_text())
    (T / "runs").mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="serverless-regen-", dir=T / "runs") as tmp:
        root = Path(tmp)
        cat = catalogue.build(16, max_gpus_per_model=4, calibration_profile=calibration)
        write(root / "tests/catalogue-graded.json", cat)
        public_cat = catalogue.redact_for_agent(cat)
        write(root / "environment/catalogue-dev.json", public_cat)
        write(root / "tests/public/catalogue.json", public_cat)
        nodes = [f"node{i}" for i in range(4)]
        graded = fleetgen.build(16, "graded-1", 9100, "simfleet", .10, 0, nodes, 4)
        public = fleetgen.build(16, "dev-1", 9100, "127.0.0.1", .10, 0, nodes, 4, port_stride=10)
        for n in public["nodes"]:
            n["host"] = n["addr"] = "127.0.0.1"
        public["controller"]["host"] = "127.0.0.1"
        write(root / "tests/fleet-graded.json", graded)
        write(root / "tests/fleet-graded.public.json", fleetgen.public_view(graded))
        write(root / "environment/fleet-dev.public.json", fleetgen.public_view(public))
        suites = {split: workload.build_suite(cat, settings, seed, split)
                  for split, seed in (("private", "graded-v4"), ("public", "public-v4"))}
        for suite in suites.values():
            suite["public_catalogue_sha256"] = workload.digest(public_cat)
        jobs = [(str(root), cat, settings, suite, case, provenance)
                for suite in suites.values() for case in suite["cases"]]
        with ProcessPoolExecutor(a.jobs) as pool:
            for split, name, sha in pool.map(generate, jobs):
                next(c for c in suites[split]["cases"] if c["name"] == name)["trace_sha256"] = sha
        for split, suite in suites.items():
            if split == "public":
                suite = {k: suite[k] for k in ("schema_version", "model_ids", "cell_weights", "qualification", "cases", "public_catalogue_sha256", "setup_s", "session_retries")}
                suite["cases"] = [{k: c[k] for k in ("name", "load_multiplier", "trace_sha256")} for c in suite["cases"]]
            write(root / ("environment/suite-public.json" if split == "public" else "tests/suite-graded.json"), suite)
        for name in world_manifest.INPUTS:
            dest = root / name
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(T / name, dest)
        changed = [name for name, sha in initial_inputs.items()
                   if world_manifest.sha(root/name) != sha or world_manifest.sha(T/name) != sha]
        if changed or tracegen.servegen_provenance() != provenance:
            raise RuntimeError(f"generation inputs changed while rebuilding: {changed}; nothing published")
        have = world_manifest.collect(str(root))
        if not a.write_manifest:
            want = json.loads((T / "_dev/world-manifest.json").read_text())["files"]
            if want != have:
                changed = sorted(k for k in want.keys() | have.keys() if want.get(k) != have.get(k))
                raise RuntimeError(f"regenerated world differs: {changed}; intentional revisions require --write-manifest")
        destination = a.output_root or T
        # After all new cases succeeded, remove generated files with other naming patterns, so only this
        # world's catalogues and traces remain beside the manifest.
        if destination == T:
            for pattern in ("tests/catalogue-graded-s*.json", "environment/catalogue-dev-s*.json",
                            "tests/public/catalogue-s*.json", "tests/trace-graded-r*-s*.*",
                            "environment/trace-public-light.*", "environment/trace-public-heavy.*",
                            "environment/trace-public-tight.*", "environment/trace-public-loose.*",
                            "tests/trace-graded-control-*", "tests/trace-graded-shift-*", "tests/trace-graded-pressure-*",
                            "environment/trace-public-control-*", "environment/trace-public-shift-*", "environment/trace-public-pressure-*"):
                for old in T.glob(pattern):
                    old.unlink()
        paths = list(have) if a.output_root else [k for k in have if k.startswith(("tests/", "environment/"))]
        for name in paths:
            dest = destination / name
            dest.parent.mkdir(parents=True, exist_ok=True)
            staged = dest.with_suffix(dest.suffix + ".new")
            shutil.copyfile(root / name, staged)
            staged.replace(dest)
        write(destination / "_dev/world-manifest.json", {
            "master_seed": tracegen.seeds.MASTER_SEED, "n_files": len(have), "files": have})
    if not a.output_root:
        subprocess.run(["bash", str(T / "_dev/sync_kit.sh")], check=True)


if __name__ == "__main__":
    main()
