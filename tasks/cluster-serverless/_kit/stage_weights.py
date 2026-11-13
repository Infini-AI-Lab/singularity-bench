"""Stage the catalogue's checkpoints onto local disk. Idempotent; safe to re-run and to interrupt.

The fleet needs every checkpoint, visible at the same path on every node, so this runs once and takes
hours. It is kept separate from the catalogue definition on purpose: the catalogue is what the task IS,
staging is an operational chore, and the two move at different speeds.
"""
from __future__ import annotations
import argparse, json, os, sys, time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import checkpoints

# BOTH names: `bin/cluster.sh` uses CLUSTER_WEIGHTS_DIR, and CLUSTER_WEIGHTS also works. The launcher
# passes --dest; reading both keeps a direct invocation of this script (which is how you check what is
# staged) pointed at the same directory instead of ~/hf-weights.
ROOT = Path(os.environ.get("CLUSTER_WEIGHTS") or os.environ.get("CLUSTER_WEIGHTS_DIR")
            or Path.home() / "hf-weights")
# Local directory names that differ from the lower-cased repository name (the catalogue's `local_dir`).
ALIASES = {"llama-3.3-70b-instruct": "llama33-70b", "qwen3-32b": "qwen3-32b"}


def patterns_for(checkpoint: str) -> list[str]:
    """Download exactly the files in the immutable lock, including every shard."""
    return list(checkpoints.entry(checkpoint)["files"])


def local_name(checkpoint: str) -> str:
    name = checkpoint.split("/", 1)[1].lower()
    return ALIASES.get(name, name)


def already(path: Path, checkpoint: str | None = None, *, full: bool = False) -> bool:
    try:
        checkpoint = checkpoint or json.loads((path / checkpoints.RECEIPT).read_text())["checkpoint"]
        return checkpoints.verify(path, checkpoint, checkpoints.entry(checkpoint), full=full)
    except (OSError, ValueError, KeyError):
        return False


def fetch(checkpoint: str) -> tuple[str, str, float]:
    from huggingface_hub import snapshot_download
    dest = ROOT / local_name(checkpoint)
    locked = checkpoints.entry(checkpoint)
    if already(dest, checkpoint):
        return checkpoint, "already staged", 0.0
    t0 = time.monotonic()
    try:
        snapshot_download(checkpoint, revision=locked["revision"], local_dir=str(dest), allow_patterns=patterns_for(checkpoint),
                          max_workers=8)
        checkpoints.certify(dest, checkpoint, locked)
        return checkpoint, "ok", time.monotonic() - t0
    except Exception as e:                                            # noqa: BLE001
        return checkpoint, f"FAILED {type(e).__name__}: {str(e)[:110]}", time.monotonic() - t0


def access(checkpoint: str) -> tuple[str, str]:
    """Can THIS machine's credentials actually fetch this repo? Returns (checkpoint, verdict).

    Several catalogue checkpoints are gated (the meta-llama and google/gemma repositories, for example),
    and some are `gated=manual`: a human must request access and be approved, which is not something you
    want to discover after hours of downloading everything else. Fetching one small file with the ambient
    token answers it in one request.
    """
    # FETCH A FILE. `model_info()` is NOT an access check: it returns metadata for a gated repo you have
    # no grant on, so it cannot fail on the case this probe exists to catch. config.json is ~1 KB and
    # decisive.
    try:
        from huggingface_hub import hf_hub_download, HfApi
        from huggingface_hub.utils import GatedRepoError, RepositoryNotFoundError
    except Exception as e:                                            # noqa: BLE001
        return checkpoint, f"UNKNOWN ({type(e).__name__})"
    try:
        gated = bool(getattr(HfApi().model_info(checkpoint), "gated", None))
    except Exception:                                                 # noqa: BLE001
        gated = False
    try:
        hf_hub_download(checkpoint, "config.json", revision=checkpoints.entry(checkpoint)["revision"])
    except GatedRepoError:
        return checkpoint, "GATED - licence not accepted"
    except RepositoryNotFoundError:
        # A gated repo read without a token answers 404, not 403; do not report it as deleted.
        return checkpoint, "GATED or missing - no access"
    except Exception as e:                                            # noqa: BLE001
        return checkpoint, f"ERROR {type(e).__name__}"
    return checkpoint, ("ok (gated, granted)" if gated else "ok")


def main() -> None:
    global ROOT
    ap = argparse.ArgumentParser()
    ap.add_argument("--catalogue", default="", help="the SHIPPED catalogue to stage for. Prefer this "
                                                   "over --gpus: it stages exactly what will be graded, "
                                                   "rather than whatever build() produces today.")
    ap.add_argument("--dest", default="", help=f"where to stage (default $CLUSTER_WEIGHTS, else {ROOT})")
    ap.add_argument("--gpus", type=int, default=64, help="build a catalogue for this fleet size instead")
    ap.add_argument("--plan", action="store_true", help="say what is missing and its size; download nothing")
    ap.add_argument("--only", default="", help="fetch just these checkpoints (comma-separated local "
                                               "dir names, or 'probe' for the smallest one). The "
                                               "multi-node preflight needs one small checkpoint and "
                                               "should not wait for the other 3.4 TiB.")
    ap.add_argument("--parallel", type=int, default=3)
    ap.add_argument("--verify", action="store_true", help="fully hash existing staged files against the lock; download nothing")
    a = ap.parse_args()

    if a.dest:
        ROOT = Path(a.dest)
    import catalogue as cat_mod
    cat = json.load(open(a.catalogue)) if a.catalogue else cat_mod.build(a.gpus)
    wanted = sorted({m["checkpoint"] for m in cat["models"]})
    if a.only:
        if a.only == "probe":
            keep = {min(cat["models"], key=lambda m: m["weights_gib"])["local_dir"]}
        else:
            keep = {x.strip() for x in a.only.split(",") if x.strip()}
        wanted = [c for c in wanted if local_name(c) in keep]
        if not wanted:
            raise SystemExit(f"--only {a.only} matched no checkpoint in this catalogue")
    # Size per CHECKPOINT, not per model: many logical models share one set of weights on disk, and
    # summing over models overstates the download several times over.
    # DOWNLOAD size, which is not the catalogue's `weights_gib`. That field is GPU-resident bytes, and
    # vLLM downcasts unquantised fp32 at load -- `ReMM-SLERP-L2-13B` is 24.2 GiB in HBM and 48.5 on the
    # wire. true-sizes.json holds the measured on-disk bytes; fall back to the catalogue when it is absent.
    true = {}
    try:
        with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "true-sizes.json")) as f:
            true = json.load(f)["sizes"]
    except (OSError, KeyError, ValueError):
        pass
    size = {}
    for m in cat["models"]:
        rec = true.get(m["checkpoint"])
        size[m["checkpoint"]] = rec["weights_gib"] if rec else m["weights_gib"]
    # Only what we are actually fetching, so `--only` reports the size of that subset rather than the
    # whole catalogue's.
    size = {c: g for c, g in size.items() if c in set(wanted)}
    # --plan is READ-ONLY. Creating the destination to report on it would mean `--plan /shared/weights`
    # against a typo silently makes the typo, and the next run stages the whole catalogue into it.
    if not (a.plan or a.verify):
        ROOT.mkdir(parents=True, exist_ok=True)
    for c in wanted:
        checkpoints.entry(c)  # Missing pins are errors, never an implicit download from main.
    missing = [c for c in wanted if not already(ROOT / local_name(c), c, full=a.verify)]
    print(f"[stage] {len(wanted)} distinct checkpoints, {sum(size.values())/1024:.2f} TiB total "
          f"({len(cat['models'])} logical models) -> {ROOT}", flush=True)
    for c in wanted:
        print(f"         {'staged' if c not in missing else 'NEEDED'}  {size[c]:7.1f} GiB  {c}",
              flush=True)
    print(f"[stage] {len(missing)} missing, {sum(size[c] for c in missing)/1024:.2f} TiB to fetch",
          flush=True)
    if a.verify:
        raise SystemExit(1 if missing else 0)
    if a.plan:
        # Access BEFORE bytes. A missing checkpoint you cannot download is a different problem from a
        # missing checkpoint you simply have not fetched yet, and only one of them is fixed by waiting.
        print("[stage] checking repo access with the ambient HF credentials ...", flush=True)
        with ThreadPoolExecutor(max_workers=8) as pool:
            verdicts = dict(pool.map(access, wanted))
        blocked = {c: v for c, v in verdicts.items() if not v.startswith("ok")}
        for c, v in sorted(verdicts.items()):
            if not v.startswith("ok") or "gated" in v:
                print(f"         {v:28} {size.get(c, 0):7.1f} GiB  {c}", flush=True)
        if blocked:
            print(f"[stage] {len(blocked)} checkpoint(s), "
                  f"{sum(size.get(c, 0) for c in blocked)/1024:.2f} TiB, are NOT reachable. Accept each "
                  f"licence on huggingface.co and set HF_TOKEN, then re-run --plan.", flush=True)
        else:
            print("[stage] all checkpoints reachable", flush=True)
        # Exit non-zero when something is missing so a launcher can branch on it without parsing text.
        raise SystemExit(1 if (missing or blocked) else 0)

    results = []
    with ThreadPoolExecutor(max_workers=a.parallel) as pool:
        futures = {pool.submit(fetch, c): c for c in wanted}
        for fut in as_completed(futures):
            ckpt, status, secs = fut.result()
            results.append({"checkpoint": ckpt, "status": status, "seconds": round(secs, 1)})
            print(f"[stage] {status:52} {secs/60:6.1f} min  {ckpt}", flush=True)
            json.dump(results, open(ROOT / "stage_results.json", "w"), indent=1)

    bad = [r for r in results if r["status"].startswith("FAILED")]
    print(f"\n[stage] {len(results)-len(bad)}/{len(results)} staged; {len(bad)} failed", flush=True)
    for r in bad:
        print(f"         {r['checkpoint']}: {r['status']}", flush=True)
    if bad:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
