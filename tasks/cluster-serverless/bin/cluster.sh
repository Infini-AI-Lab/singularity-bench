#!/usr/bin/env bash
# Operate the cluster-serverless GPU fleet from a launch host, against already-allocated GPU nodes.
#
#   ./bin/cluster.sh weights [--plan] [--probe-only]            download the weights -- NO NODES NEEDED
#                                                              --probe-only: just the checkpoint the
#                                                              multinode preflight needs
#   ./bin/cluster.sh check   --nodes n1,n2,...                 are the nodes usable, before anything else
#   ./bin/cluster.sh stage   --nodes n1,n2,...                 build the fleet image on them
#   ./bin/cluster.sh multinode --nodes n1,n2                   optional cross-node probe (after `stage`)
#   ./bin/cluster.sh up      --nodes n1,n2,n3,n4              start a fleet for breakwater/run.py
#   ./bin/cluster.sh down    --fleet /path/to/fleet.json       stop only that fleet
#
# Agent runs use breakwater/run.py against a fleet started with `up` (see README.md). Export
# CLUSTER_RECEIPT_KEY and CLUSTER_GRADER_TOKEN before `up`: breakwater/run.py needs the same values.
#
# `--nodes` takes the hostnames inline, comma- or space-separated, and may be repeated. Nothing here
# allocates anything: the nodes must already be yours.
#
# `weights` comes FIRST and deliberately takes no nodes. It is ~5.4 TiB across 28 checkpoints and many
# hours of download, it goes to a shared path, and it is reused by every later run -- so it should not be
# holding 16 GPUs while it happens. Everything after it needs the nodes. `weights --plan` prints the exact
# figure for the shipped catalogue and checks every repo is reachable with your HF credentials before any
# bytes move: some repositories are gated and need an accepted licence plus HF_TOKEN.
#
# WHAT RUNS WHERE
#   launch host     the agent, grading and verifier containers (breakwater/run.py), CPU only
#   GPU nodes       one fleet container each, owning that node's GPUs, reached over HTTP
#
# The agent's container talks to the GPU nodes by address. It never has a GPU: the task is to run a
# fleet, not to sit on one.
set -uo pipefail

TASK="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUNS="${CLUSTER_RUNS:-$TASK/runs}"
# Must be the SAME PATH on the launch host and on every GPU node -- the fleet containers bind-mount
# it. On a cluster with a shared home this is automatic; if not, set CLUSTER_RUN_ROOT to a shared path.
RUN_ROOT="${CLUSTER_RUN_ROOT:-$RUNS}"
WEIGHTS="${CLUSTER_WEIGHTS_DIR:-$HOME/hf-weights}"
IMAGE="${CLUSTER_FLEET_IMAGE:-cluster-fleet:v1}"
GPUS_PER_NODE="${GPUS_PER_NODE:-4}"
PY="${CLUSTER_PYTHON:-python3}"
# The card size the shipped catalogue was built against. Read from the artifact, never re-typed:
# a second copy of this number is how the fleet and the catalogue drift apart.
CARD_GIB=$("$PY" -c "import json;print(int(json.load(open('$TASK/tests/catalogue-graded.json'))['gpu_gib']))" 2>/dev/null || echo 288)
# Distinct checkpoints the catalogue needs staged. Derived from the catalogue, not typed.
N_CKPTS=$("$PY" -c "import json;print(len({m['checkpoint'] for m in json.load(open('$TASK/tests/catalogue-graded.json'))['models']}))" 2>/dev/null || echo 28)
# How to run a command on a named node. ssh by default; override for pdsh, srun, a site wrapper.
if [ -z "${CLUSTER_REMOTE_LAUNCH:-}" ]; then
  CLUSTER_REMOTE_LAUNCH='ssh -o BatchMode=yes -o StrictHostKeyChecking=no {node} {cmd}'
fi
export CLUSTER_REMOTE_LAUNCH

NODES=""; ONE_BOX=0; ONE_BOX_NODES=4; ONE_BOX_STRIDE=10; TAG=""
SKIP_BUILD=0; FLEET_MODE=real

die() { echo "$*" >&2; exit 2; }

normalise_nodes() {   # whatever was given -> a clean comma-separated list in $NODES
  # Commas, spaces or both, and --nodes may be repeated. All the same thing, because the hostnames are
  # the input and how they were typed should not be a second thing to get right.
  NODES="$(printf '%s' "$NODES" | tr -s '[:space:],' '\n' | sed '/^$/d' | paste -sd, -)"
  [ -n "$NODES" ] || die "give the nodes inline: --nodes host1,host2,... (or --nodelist FILE)"
  local dup
  dup="$(printf '%s' "$NODES" | tr ',' '\n' | sort | uniq -d | paste -sd, -)"
  [ -z "$dup" ] && return 0
  # A repeated hostname is silently half a fleet: the launcher would start one container per NAME and
  # the second start would remove the first.
  die "these hostnames appear twice: $dup"
}

parse() {
  while [ $# -gt 0 ]; do
    case "$1" in
      --nodes)     NODES="$NODES,$2"; shift 2 ;;
      # A file of hostnames, for a site that keeps one; --nodes is the usual way in.
      --nodelist)  [ -f "$2" ] || die "no such nodelist file: $2"
                   NODES="$NODES,$(sed 's/#.*//' "$2" | tr -s '[:space:],' ' ')"; shift 2 ;;
      --tag)       TAG="$2"; shift 2 ;;
      --gpus-per-node) GPUS_PER_NODE="$2"; shift 2 ;;
      --weights)   WEIGHTS="$2"; shift 2 ;;
      --image)     IMAGE="$2"; shift 2 ;;
      --sim)       FLEET_MODE=sim; shift ;;
      # Emulate the whole graded fleet on this machine: no nodes, no GPUs, no weights, but the real
      # catalogue. A development utility; it measures nothing about GB300 speed.
      --one-box)   ONE_BOX=1; FLEET_MODE=sim; NODES="${NODES:-localhost}"; shift ;;   # emulator; TIME_SCALE=n divides every simulated cost
      --skip-build) SKIP_BUILD=1; shift ;;
      *) die "unknown option $1" ;;
    esac
  done
  normalise_nodes
  N_NODES=$(awk -F, '{print NF}' <<< "$NODES")
  N_GPUS=$(( N_NODES * GPUS_PER_NODE ))
}

run_on() {   # run_on <node> <command...>  -- synchronous, output returned
  local n="$1"; shift
  local tmpl="${CLUSTER_REMOTE_LAUNCH}"
  tmpl="${tmpl//\{node\}/$n}"
  tmpl="${tmpl//\{cmd\}/$(printf '%q' "$*")}"
  eval "$tmpl"
}

# Docker names architectures differently from `uname`. One place to say so.
docker_arch_of() { case "$1" in aarch64) echo arm64 ;; x86_64) echo amd64 ;; *) echo "$1" ;; esac; }

# Is $IMAGE on a machine built FOR that machine? The fleet image is built per node and never pushed, so
# the way this goes wrong is not a bad build -- it is an OLD one: an image left from before the node was
# reimaged, or carried over with `docker save`. Docker runs a foreign-architecture image under emulation
# if binfmt is installed (slowly, and vLLM finds no GPU) and dies with an exec-format error if it is not.
# Both read as "the fleet never came up", an hour later, on all four nodes at once.
# An ABSENT image passes: `multinode` reports that separately, and better.
fleet_image_arch_ok() {   # $1 = node, or "" for this machine
  local got want
  if [ -z "$1" ]; then
    got=$(docker image inspect --format '{{.Architecture}}' "$IMAGE" 2>/dev/null)
    want=$(docker_arch_of "$(uname -m)")
  else
    got=$(run_on "$1" "docker image inspect --format '{{.Architecture}}' $IMAGE" 2>/dev/null | tr -dc 'a-z0-9_')
    want=$(docker_arch_of "$(run_on "$1" 'uname -m' 2>/dev/null | tr -dc 'a-zA-Z0-9_')")
  fi
  [ -z "$got" ] || [ -z "$want" ] || [ "$got" = "$want" ]
}

# ------------------------------------------------------------------------------------------ check
cmd_check() {
  parse "$@"
  echo "== $N_NODES nodes x $GPUS_PER_NODE GPUs = $N_GPUS GPUs"
  local fail=0 ARCHES=""
  IFS=',' read -ra NL <<< "$NODES"
  printf "%-18s %-6s %-8s %-5s %-9s %-9s %-8s %s\n" node reach arch gpus "gpu GiB" "RAM GiB" docker ckpts
  for n in "${NL[@]}"; do
    local reach=no arch=- ngpu=- mem=- ram=- dock=- wt=- memok=no memory_rows
    if run_on "$n" true >/dev/null 2>&1; then
      reach=yes
      # The fleet image is BUILT ON each node, so its architecture is simply whatever the node is.
      # That makes a mixed node set invisible: every build succeeds, and half the fleet then runs an
      # aarch64 binary and half an x86_64 one against a catalogue that is a single fixed artifact.
      # Read it, print it, refuse below.
      arch=$(run_on "$n" 'uname -m' 2>/dev/null | tr -dc 'a-zA-Z0-9_')
      [ -n "$arch" ] || arch=unknown
      ngpu=$(run_on "$n" 'nvidia-smi --query-gpu=index --format=csv,noheader' 2>/dev/null | wc -l)
      if memory_rows=$(run_on "$n" 'nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits' 2>/dev/null); then
        if mem=$("$PY" "$TASK/bin/check_gpu_memory.py" "$GPUS_PER_NODE" "$CARD_GIB" <<< "$memory_rows"); then
          memok=yes
        else
          mem=invalid
        fi
      fi
      ram=$(run_on "$n" "awk '/MemTotal/{printf \"%d\", \$2/1048576}' /proc/meminfo" 2>/dev/null | tr -dc '0-9')
      dock=$(run_on "$n" 'docker info --format "{{.ServerVersion}}"' 2>/dev/null | head -1)
      [ -z "$dock" ] && dock=MISSING
      # COUNT the checkpoint directories, do not `du` them. A du of ~20 TiB walks every file, which on a
      # shared filesystem takes minutes per node and tells you nothing the count does not.
      wt=$(run_on "$n" "ls $(printf '%q' "$WEIGHTS") 2>/dev/null | wc -l" 2>/dev/null | tr -dc '0-9')
      [ -z "$wt" ] || [ "$wt" = 0 ] && wt=absent || wt="$wt/$N_CKPTS"
    fi
    printf "%-18s %-6s %-8s %-5s %-9s %-9s %-8s %s\n" "$n" "$reach" "$arch" "$ngpu" "$mem" "$ram" "$dock" "$wt"
    [ "$reach" = yes ] && ARCHES="$ARCHES $arch"
    # Every card must supply valid telemetry. This coarse inventory range does
    # not substitute for measuring usable memory.
    [ "$reach" = yes ] && [ "$ngpu" = "$GPUS_PER_NODE" ] && [ "$dock" != MISSING ] \
      && [ "$memok" = yes ] || fail=1
    [ "$memok" = yes ] || echo "    $n: GPU memory $mem GiB, catalogue is sized for $CARD_GIB GiB" >&2
  done
  echo
  # Three things silently change what the task IS rather than breaking it loudly: the GPU memory size,
  # a mixed architecture, and a fleet of a different SIZE from the one the catalogue was built against.
  # All three are refused, not just printed. This one exits non-zero on a size mismatch.
  "$PY" - "$TASK" "$N_GPUS" <<'PYCHK' || fail=1
import json, sys
cat = json.load(open(f"{sys.argv[1]}/tests/catalogue-graded.json"))
n = int(sys.argv[2])
print(f"  catalogue: {len(cat['models'])} models, {cat['oversubscription']}x oversubscribed, "
      f"sized for {cat['n_gpus']} GPUs at {cat['gpu_gib']:.0f} GiB")
ck = sorted({m["local_dir"] for m in cat["models"]})
gib = sum({m["local_dir"]: m["weights_gib"] for m in cat["models"]}.values())
print(f"  weights: {len(ck)} checkpoints, {gib/1024:.2f} TiB, needed on EVERY node")
if n != cat["n_gpus"]:
    print(f"  MISMATCH: this fleet has {n} GPUs, the catalogue is sized for {cat['n_gpus']}."
          f" The catalogue is a fixed artifact and does not resize; a {n}-GPU fleet running it is a"
          f" different task and its score is not comparable to anyone else's.", file=sys.stderr)
    print(f"    Either bring {cat['n_gpus']} GPUs (the shipped fleet is 4 nodes x 4), or set"
          f" --gpus-per-node so that nodes x gpus-per-node = {cat['n_gpus']}.", file=sys.stderr)
    sys.exit(1)
PYCHK
  # One fleet, one architecture. This is the same class of silent change as a wrong GPU size: it does
  # not break the task, it makes the score describe something other than the task.
  local n_arch
  n_arch=$(echo $ARCHES | tr ' ' '\n' | sort -u | grep -c . || true)
  if [ "${n_arch:-0}" -gt 1 ]; then
    echo "  MIXED ARCHITECTURES: $(echo $ARCHES | tr ' ' '\n' | sort -u | tr '\n' ' ')" >&2
    echo "    Split the node set; a fleet spanning two architectures is a different problem." >&2
    fail=1
  fi
  [ "$fail" = 0 ] && echo "  nodes look usable" || echo "  SOME NODES ARE NOT USABLE" >&2
  return $fail
}

# -------------------------------------------------------------------------------------- multinode
cmd_multinode() {
  parse "$@"
  # Wider engines span more than one four-GPU node. Test actual cross-node launch, serving and cleanup
  # before reserving the fleet for an agent run; modeled packing cannot establish those properties.
  # This deploys one real cross-node model on two of your nodes and serves one request through it.
  IFS=',' read -ra NL <<< "$NODES"
  [ "${#NL[@]}" -ge 2 ] || die "give at least two nodes: --nodes n1,n2"
  local a="${NL[0]}" b="${NL[1]}"
  # Its OWN prerequisites, deliberately minimal: the fleet image, and the one small checkpoint it
  # probes with. This is the check you want to run EARLY -- while `weights` is still downloading -- so
  # it must not inherit `stage`'s requirement that every checkpoint be present.
  local ck; ck=$("$PY" -c "
import json,sys
c=json.load(open('$TASK/tests/catalogue-graded.json'))
print(min(c['models'], key=lambda m: m['weights_gib'])['local_dir'])")
  for n in "$a" "$b"; do
    run_on "$n" "docker image inspect $IMAGE >/dev/null 2>&1" >/dev/null 2>&1 || \
      die "$IMAGE is not on $n. Run: ./bin/cluster.sh stage --nodes $a,$b"
    fleet_image_arch_ok "$n" || \
      die "$IMAGE on $n was built for a different architecture than $n. Remove it and re-stage: ./bin/cluster.sh stage --nodes $a,$b"
    run_on "$n" "test -d $(printf '%q' "$WEIGHTS/$ck")" >/dev/null 2>&1 || \
      die "$n cannot see $WEIGHTS/$ck -- the smallest checkpoint. Let ./bin/cluster.sh weights get that far first."
  done
  echo "== probing cross-node vLLM on $a + $b (needs only the $ck checkpoint)"
  mkdir -p "$RUN_ROOT" || die "cannot create run root: $RUN_ROOT"
  local rd
  rd=$(mktemp -d "$RUN_ROOT/multinode-check-XXXXXXXX") || die "cannot create a fresh preflight directory"
  mkdir -p "$rd/logs"; keys

  # A catalogue holding exactly one model that must span both machines.
  "$PY" - "$TASK" "$rd" "$GPUS_PER_NODE" <<'PYCAT'
import json, sys, pathlib
task, rd, gpn = sys.argv[1], sys.argv[2], int(sys.argv[3])
cat = json.loads((pathlib.Path(task) / "tests/catalogue-graded.json").read_text())
big = [m for m in cat["models"] if m["gpus"] > gpn]
if not big:
    print("no model in the catalogue needs more than one node; multi-node is optional here")
    raise SystemExit(3)
m = min(big, key=lambda x: x["weights_gib"])
cat["models"] = [m]
json.dump(cat, open(f"{rd}/catalogue.json", "w"), indent=1)
print(f"  probing with {m['id']} ({m['weights_gib']:.0f} GiB over {m['gpus']} GPUs, "
      f"{gpn} per node -> spans {-(-m['gpus'] // gpn)} nodes)")
PYCAT
  rc=$?; [ "$rc" = 3 ] && return 0; [ "$rc" = 0 ] || return 1

  "$PY" "$TASK/_kit/cluster_launch.py" up --nodes "$a,$b" --gpus-per-node "$GPUS_PER_NODE" \
      --catalogue "$rd/catalogue.json" --out "$rd/fleet.json" --public "$rd/fleet.public.json" \
      --image "$IMAGE" --run-dir "$rd" --weights "$WEIGHTS" --worker-script manager.py \
      --seed multinode-check --perturb 0 \
      --receipt-key "$CLUSTER_RECEIPT_KEY" --grader-token "$CLUSTER_GRADER_TOKEN" \
      --wait "${FLEET_WAIT:-1800}" || { echo "the two-node fleet did not come up" >&2; return 1; }

  local ok=0
  "$PY" "$TASK/bin/check_cluster.py" "$rd" "$TASK/_kit" && ok=1
  "$PY" "$TASK/_kit/cluster_launch.py" down --nodes "$a,$b" >/dev/null 2>&1 || true
  if [ "$ok" = 1 ]; then
    echo "  MULTI-NODE WORKS on these nodes."
  else
    echo "  MULTI-NODE DOES NOT WORK on these nodes." >&2
    echo "  Cross-node engines will fail on this fleet; fix the node interconnect before a graded run." >&2
    return 1
  fi
}

# ---------------------------------------------------------------------------------------- weights
cmd_weights() {
  # NO parse, NO nodes. This is the step that should happen before you hold any hardware: ~5.4 TiB over 28
  # checkpoints, many hours on a good link, written to a shared path that every later run reads. Doing it
  # from inside an allocation means 16 GPUs sit idle for the download, and doing it per run means doing it
  # again. The exact size comes from the catalogue, so it is not restated here.
  local plan=0 only=""
  while [ $# -gt 0 ]; do
    case "$1" in
      --dest) WEIGHTS="$2"; shift 2 ;;
      --plan) plan=1; shift ;;
      # Just the checkpoint the multi-node preflight probes with -- minutes, not hours, so the go/no-go
      # on cross-node vLLM can happen while the real download is still running.
      --probe-only) only="--only probe"; shift ;;
      *) die "unknown option $1 (weights takes --dest DIR and --plan)" ;;
    esac
  done
  echo "== weights -> $WEIGHTS"
  echo "   This path must be readable at the SAME location on every compute node."
  echo "   A shared filesystem counts as every node; otherwise repeat this on each."
  echo
  if [ "$plan" = 1 ]; then
    # $only here too, so `--probe-only --plan` plans just the probe checkpoint.
    "$PY" "$TASK/_kit/stage_weights.py" --catalogue "$TASK/tests/catalogue-graded.json" \
          --dest "$WEIGHTS" --plan $only
    return $?
  fi
  "$PY" "$TASK/_kit/stage_weights.py" --catalogue "$TASK/tests/catalogue-graded.json" \
        --dest "$WEIGHTS" $only || return 1
  echo
  echo "   Re-running this is cheap and safe: a checkpoint already there is skipped, and an interrupted"
  echo "   download resumes. Check what is missing at any time with: ./bin/cluster.sh weights --plan"
}


# ------------------------------------------------------------------------------------------ stage
cmd_stage() {
  parse "$@"
  IFS=',' read -ra NL <<< "$NODES"
  if [ "$SKIP_BUILD" = 0 ]; then
    echo "== building $IMAGE on $N_NODES nodes (in parallel; the vLLM base is ~9 GB)"
    local pids=() bad=0
    for n in "${NL[@]}"; do
      ( run_on "$n" "docker build -t $IMAGE $TASK/environment/fleetsvc" \
          > "/tmp/cluster-build-$n.log" 2>&1 ) &
      pids+=($!)
    done
    for i in "${!pids[@]}"; do
      if ! wait "${pids[$i]}"; then bad=1; echo "  BUILD FAILED on ${NL[$i]} -- /tmp/cluster-build-${NL[$i]}.log" >&2; fi
    done
    [ "$bad" = 0 ] || return 1
    echo "  built on every node"
  fi
  # From the NODES, not from here. A shared filesystem usually means the launch host's view is every
  # node's view -- usually. When it is not, the failure is a fleet that starts and then cannot load a
  # model, an hour in.
  echo "== can each node see the weights at $WEIGHTS?"
  local bad=0 zero=0
  for n in "${NL[@]}"; do
    local got
    got=$(run_on "$n" "ls $(printf '%q' "$WEIGHTS") 2>/dev/null | wc -l" 2>/dev/null | tr -dc '0-9')
    [ -z "$got" ] && got=0
    printf "  %-18s %s checkpoints\n" "$n" "$got"
    [ "$got" -ge "$N_CKPTS" ] || bad=1
    [ "$got" -gt 0 ]  || zero=1
  done
  # A node seeing ZERO checkpoints is a wrong path and is fatal. A node seeing SOME is a download still
  # in progress, which is normal -- `weights` takes hours and the image build should not wait for it,
  # nor should the multi-node preflight, which needs one small checkpoint and nothing else.
  if [ "$zero" != 0 ]; then
    echo
    echo "  Some node cannot see $WEIGHTS at all. Check it is the same path there as here." >&2
    return 1
  fi
  if [ "$bad" != 0 ]; then
    echo
    echo "  Not every node has every checkpoint yet. That is fine if ./bin/cluster.sh weights is still"
    echo "  running -- the image is built and you can run the multi-node preflight now. The graded run"
    echo "  needs them all."
  else
    echo "  every node sees all the weights"
  fi
}

# ------------------------------------------------------------------------------------------ fleet
fleet_up() {   # $1 = run directory
  local rd="$1"
  [ ! -e "$rd/fleet.json" ] || { echo "run directory already has a fleet manifest; use a fresh run" >&2; return 1; }
  mkdir -p "$rd/logs"
  cp "$TASK/tests/catalogue-graded.json" "$rd/catalogue.json"

  if [ "$ONE_BOX" = 1 ]; then
    # The whole graded fleet, emulated on this machine: 16 GPUs over four named nodes, addressed at
    # 127.0.0.1 with a per-node port stride so they do not all try to bind 9100. Node identity is kept,
    # so per-node host-RAM budgets and cross-node sharding still behave as they do on a cluster.
    # It cannot tell you anything about speed -- the numbers come from a cost model.
    local onebox_addr
    onebox_addr="${CLUSTER_ONEBOX_HOST:-$(docker network inspect bridge --format '{{(index .IPAM.Config 0).Gateway}}')}"
    [ -n "$onebox_addr" ] || { echo "set CLUSTER_ONEBOX_HOST to a host IPv4 address reachable from containers" >&2; return 1; }
    local nodes; nodes=$(python3 -c "print(','.join(f'node{i}' for i in range($ONE_BOX_NODES)))")
    "$PY" "$TASK/_kit/fleetgen.py" --workers $((ONE_BOX_NODES * GPUS_PER_NODE)) \
        --seed "cluster-${TAG:-1}" --perturb 0.10 --nodes "$nodes" \
        --gpus-per-node "$GPUS_PER_NODE" --one-host "$onebox_addr" --port-stride "$ONE_BOX_STRIDE" \
        --out "$rd/fleet.json" --public "$rd/fleet.public.json" > /dev/null
    local run_id fleet_image_id
    fleet_image_id=$(docker image inspect --format '{{.Id}}' "$IMAGE") || return 1
    run_id=$(python3 -c 'import secrets; print(secrets.token_hex(16))')
    "$PY" - "$rd/fleet.json" "$run_id" "$fleet_image_id" "$TASK/_kit" <<'OWNED'
import json, sys
p, identity, image, kit = sys.argv[1:]
sys.path.insert(0, kit)
from cluster_launch import healthy
f = json.load(open(p))
f.update(run_id=identity, launch={"mode": "container", "cleanup_nodes": ["localhost"], "images": {"localhost":image}})
json.dump(f, open(p, "w"), indent=1)
if any(healthy(n["host"], n["port"]) for n in f["nodes"]) or healthy(f["controller"]["host"], f["controller"]["port"]):
    raise SystemExit("fleet endpoints already occupied; no existing processes stopped")
OWNED
    [ "$?" = 0 ] || return 1
    # One container holds the fleet. simmanager supervises a separate process and
    # event loop for each logical node, matching the distributed simulator path.
    docker run --rm --name "cluster-fleet-$run_id" --label "cluster.run=$run_id" -d --network host \
      --cpus 8 --memory 32g --memory-swap 32g --pids-limit 4096 --ulimit nofile=65536:65536 -e CLUSTER_RUN_ID="$run_id" \
      -e CLUSTER_FLEET_MODE=sim -e CLUSTER_RUN_CONTROLLER=1 -e CLUSTER_NODES=/fleet/run/fleet.json \
      -e CLUSTER_CATALOGUE=/fleet/run/catalogue.json -e CLUSTER_LOG_DIR=/fleet/run/logs \
      -e CLUSTER_RECEIPT_KEY="$CLUSTER_RECEIPT_KEY" -e CLUSTER_GRADER_TOKEN="$CLUSTER_GRADER_TOKEN" \
      -e CLUSTER_TIME_SCALE="${TIME_SCALE:-1}" \
      -v "$rd":/fleet/run "$fleet_image_id" >/dev/null || {
        echo "could not start the emulated fleet; is $IMAGE built here? (./bin/cluster.sh stage --nodes localhost)" >&2
        return 1; }
    if ! "$PY" - "$TASK/_kit" "$rd/fleet.json" <<'READY'
import json, sys
sys.path.insert(0, sys.argv[1])
from cluster_launch import wait_fleet
missing = wait_fleet(json.load(open(sys.argv[2])), 120)
if missing:
    raise SystemExit(f"owned simulator endpoints did not become ready: {missing}")
READY
    then
      echo "the emulated fleet never became ready; docker logs cluster-fleet-$run_id" >&2
      return 1
    fi
    echo "[cluster] emulated fleet up: $((ONE_BOX_NODES * GPUS_PER_NODE)) workers on this machine"
    return 0
  fi

  "$PY" "$TASK/_kit/cluster_launch.py" up \
      --nodes "$NODES" --gpus-per-node "$GPUS_PER_NODE" \
      --catalogue "$rd/catalogue.json" \
      --out "$rd/fleet.json" --public "$rd/fleet.public.json" \
      --image "$IMAGE" --run-dir "$rd" --weights "$WEIGHTS" \
      --worker-script "$([ "$FLEET_MODE" = sim ] && echo simmanager.py || echo manager.py)" \
      --time-scale "$([ "$FLEET_MODE" = sim ] && echo "${TIME_SCALE:-1}" || echo 1)" \
      --seed "cluster-${TAG:-1}" --perturb 0.10 \
      --receipt-key "$CLUSTER_RECEIPT_KEY" --grader-token "$CLUSTER_GRADER_TOKEN" \
      --wait "${FLEET_WAIT:-1800}"
}

cmd_up() {
  # A fleet for breakwater/run.py. Both secrets must be exported first: run.py reads the same values
  # (the grader token resets and rekeys the fleet; the development receipt key goes to the agent).
  [ -n "${CLUSTER_RECEIPT_KEY:-}" ] && [ -n "${CLUSTER_GRADER_TOKEN:-}" ] || \
    die "export CLUSTER_RECEIPT_KEY and CLUSTER_GRADER_TOKEN first; breakwater/run.py needs the same values"
  parse "$@"; mkdir -p "$RUN_ROOT"; local rd; rd=$(mktemp -d "$RUN_ROOT/fleet-XXXXXXXX")
  fleet_up "$rd" && echo "fleet up: --fleet $rd/fleet.json --public-fleet $rd/fleet.public.json"
}
cmd_down() {
  [ "$#" = 2 ] && [ "$1" = --fleet ] || die "down requires --fleet /path/to/the/run/fleet.json"
  local manifest="$2"
  if python3 - "$manifest" <<'LOCAL'
import json, sys
f = json.load(open(sys.argv[1]))
sys.exit(0 if f.get("launch", {}).get("cleanup_nodes") == ["localhost"] else 1)
LOCAL
  then
    CLUSTER_REMOTE_LAUNCH='bash -c {cmd}' "$PY" "$TASK/_kit/cluster_launch.py" down --fleet "$manifest"
  else
    "$PY" "$TASK/_kit/cluster_launch.py" down --fleet "$manifest"
  fi
}

keys() {
  # Fresh per invocation unless exported. The grader token is the fleet's admin capability: it goes to
  # the fleet and to trusted grading code, never to the agent's container. The receipt key here is the
  # public development key; grading rekeys every node with a fresh secret before each cell.
  export CLUSTER_RECEIPT_KEY="${CLUSTER_RECEIPT_KEY:-$(head -c 32 /dev/urandom | base64 | tr -d '=+/')}"
  export CLUSTER_GRADER_TOKEN="${CLUSTER_GRADER_TOKEN:-$(head -c 24 /dev/urandom | base64 | tr -d '=+/')}"
}

case "${1:-}" in
  weights) shift; cmd_weights "$@" ;;
  multinode) shift; cmd_multinode "$@" ;;
  check) shift; cmd_check "$@" ;;
  stage) shift; cmd_stage "$@" ;;
  run)   die "Agent runs use breakwater/run.py against a fleet started with \`cluster.sh up\`; see README.md." ;;
  up)    shift; cmd_up    "$@" ;;
  down)  shift; cmd_down  "$@" ;;
  *) sed -n '2,14p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'; exit 2 ;;
esac
