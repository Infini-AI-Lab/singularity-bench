#!/usr/bin/env bash
# _kit/ is the single source of truth. The agent image, the fleet image and the grader get COPIES, made
# here, so a fix cannot land in one place and not the other. Run after touching anything in _kit/.
set -euo pipefail
T="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

mkdir -p "$T/environment/_kitcopy" "$T/environment/fleetsvc/kit" "$T/tests/kit"

# WHAT THE AGENT GETS: the controller, receipt verification, the prompt/trace definitions and the driver.
# Not the emulator, not physics.py, not the scorer. This staging directory is the agent image's build
# context, so it holds exactly what environment/Dockerfile COPYs and nothing else.
cp "$T/_kit"/{controller.py,receipts.py,prompts.py,loadgen.py,response_io.py,looplag.py,fleetaddr.py,engine_args.py,traceunits.py,service_contract.py} \
   "$T/environment/_kitcopy/"

# The fleet image needs both backends and their shared support: a manager per node (real or emulated),
# the worker each real deploy launches, the controller above them, and nodeinfo for the measurements.
# seeds.py rides along because fleetgen.py imports it: every generation stream descends from
# CLUSTER_MASTER_SEED, including the fleet's node-speed perturbation.
cp "$T/_kit"/{simmanager.py,sim_fabric.py,kv_layout.py,manager.py,worker.py,inference_audit.py,telemetry.py,telemetry_scheduler.py,response_io.py,controller.py,nodeinfo.py,physics.py,receipts.py,prompts.py,protocol.py,manager_auth.py,looplag.py,fleetaddr.py,cluster_launch.py,fleetgen.py,engine_args.py,seeds.py,checkpoints.py,checkpoints.lock.json,runtime_compat.py,owned_processes.py} \
   "$T/environment/fleetsvc/kit/"

# The grader needs the driver and the scorer (breakwater/cluster_cells.py and breakwater/score_workload.py
# import them from tests/kit).
cp "$T/_kit"/{loadgen.py,response_io.py,score.py,reference.py,receipts.py,prompts.py,physics.py,fleetaddr.py,traceunits.py,looplag.py,service_contract.py} "$T/tests/kit/"

# Dockerfiles are authored build recipes. This step synchronizes kit data and records the build inputs
# in tests/build-fingerprint.json; it never regenerates build code.
python3 - "$T" <<'PY'
from pathlib import Path
import hashlib
import json
import sys
t = Path(sys.argv[1])
(t / "tests/public").mkdir(exist_ok=True)
(t / "tests/public/catalogue.json").write_bytes((t / "environment/catalogue-dev.json").read_bytes())
paths = sorted((t / "_kit").glob("*.py")) + [t / "_kit/checkpoints.lock.json", t / "_kit/market-anchors.json", t / "_kit/model-architecture.json", t / "_kit/workload-profile.json", t / "_kit/calibration-profile.json", t / "environment/Dockerfile"]
paths += [t / "task.toml", t / "environment/agent-policy.json",
    t / "environment/fleetsvc/Dockerfile", t / "environment/fleetsvc/entrypoint.sh"]
paths += [t / "environment" / n for n in ("control_plane_process.py", "start_control_plane.sh",
    "stop_control_plane.sh", "process_safety.md", "AGENTS.md", "CLAUDE.md")]
(t / "tests/build-fingerprint.json").write_text(json.dumps({str(p.relative_to(t)):
    hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}, indent=1) + "\n")
PY

# Every copy must be IMPORTABLE, not just present: check the import closure of each destination.
python3 "$T/_dev/check_imports.py" || exit 1

echo "synced kit -> environment/_kitcopy, environment/fleetsvc/kit, tests/kit"
