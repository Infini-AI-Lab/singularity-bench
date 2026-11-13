# cluster-serverless

The cluster management task of Singularity-Bench. Within four hours, and beginning from a naive policy, the
agent builds a serverless inference orchestrator: an online control plane that serves 28 models through the
provided vLLM interface on 16 GB300 GPUs distributed across four nodes, under mixed online and batch traffic at
three load levels. The score combines request SLO attainment with completion across models.

Paths in the text are relative to `tasks/cluster-serverless/` unless they start with `breakwater/` or `tasks/`;
commands run from the repository root.

## The task

The agent implements the control plane for **four GB300 nodes (compute trays) with four GPUs each**: model
placement, replica counts, tensor/pipeline/expert parallelism, GPU colocation, model residency (deploy, sleep,
wake, unload), admission, queueing and request routing. The catalogue holds **28 models (14 mixture-of-experts,
14 dense) with about 5.4 TiB of weights**, more than the fleet's serving memory, so keeping every model ready is
infeasible: retaining models avoids cold starts but consumes capacity, while eviction and colocation free
capacity at the cost of loading delays and contention. The trusted fleet executes the plane's deploy and
lifecycle calls and authenticates every vLLM response.

| | |
| --- | --- |
| Budget | 4 hours of development and experiments (`agent_timeout_sec` 14,400), then grading on the same fleet |
| Fleet | 4 nodes x 4 GB300 GPUs (288 GiB each); a 968 GiB host-memory budget per node; vLLM 0.28.0 |
| Agent container | CPU only (8 CPUs, 32 GiB, 512 processes); the GPUs are the remote fleet, reached over HTTP |
| Starting point | A naive working control plane (`environment/reference_cp.py`, installed as `/app/control_plane.py`) |
| Objective | Mean over three load cells of strict SLO attainment x mean per-model completion (higher is better) |
| Human baseline | **0.4599** (NVIDIA Dynamo with the KAI Scheduler; shown as 0.460 in the paper) |

[instruction.md](instruction.md) is the full statement the agent receives.

## What the agent receives

The task image (`environment/Dockerfile`, Python 3.12) seeds `/app`:

| Path | Contents |
| --- | --- |
| `/app/catalogue.json` | The 28 models: pinned checkpoints, memory estimates (`weights_gib`, `active_gib`), `min_tp`, context limits, one TTFT/mean-TPOT target pair per model |
| `/app/fleet.json` | The public fleet view: controller address and four nodes with their GPU and host-memory budgets |
| `/app/control_plane.py`, `/app/run.sh` | The naive starter plane and its entry point |
| `/app/start_control_plane.sh`, `/app/stop_control_plane.sh`, `/app/process_safety.md`, `/app/AGENTS.md`, `/app/CLAUDE.md` | Safe local restart helpers and process-management notes |
| `/app/trace-public-load-{0.75,1.0,1.25}.jsonl`, `/app/suite-public.json` | Three public workload realizations from source partitions disjoint from the graded ones |
| `/app/kit/loadgen.py` (+ its modules) | The public replay driver, the same driver that grades |
| `/app/controller.py`, `/app/engine_args.py`, `/app/service_contract.py`, `/app/receipts.py`, `/app/prompts.py`, `/app/fleetaddr.py` | The fleet controller, the accepted engine arguments and the public protocol code |
| `/app/selfcheck.py` | A basic protocol check |

Installed packages: aiohttp, uvloop, httpx, uvicorn, fastapi, orjson, numpy, pulp and sortedcontainers; runtime
packages are fixed, and there is no internet access. The coding agent (Claude Code or Codex) runs as the
Breakwater author and reaches its model only through the author gateway; its commands run as the worker and
reach only local services and the fleet controller and engine ports listed in `/app/fleet.json` (node managers
are rejected). It receives the public development receipt key (`CLUSTER_RECEIPT_KEY`), which is replaced
before grading. Catalogue models are test workloads, not authoring assistants: custom latency and correctness
probes are allowed, but asking a catalogue model to design, write, debug or review the submission is not. The
fleet journals every inference request and response for the integrity review.

The submission is `/app` plus the worker's home directory (`/home/worker`), together at most 8 GiB and 100,000
entries; symbolic links must stay inside them. `bash /app/run.sh` must start a server on port 8000 with
`GET /health` (healthy within the 900-second setup window) and streaming `POST /v1/completions`, and must
forward prompts, token caps, models, streams and the signed `cluster_receipt` unchanged.

## The fleet and its interface

The fleet runs outside every task container: one container per node (`environment/fleetsvc`, built from the
pinned `vllm/vllm-openai:v0.28.0` multi-platform image, arm64 on GB300), each with a node manager, plus one
controller beside the first node's manager.

| Component | Operations |
| --- | --- |
| Controller | `POST /deploy`, `POST /unload`, `GET /nodes`, `GET /workers`, `GET /logs`, `GET /health` |
| Worker (one per engine, on the engine's head node) | `POST /v1/completions`, `POST /admin/sleep`, `POST /admin/wake`, `GET /stats`, `GET /health`, `GET /telemetry?kind=kv\|fpm` |
| Node manager (private) | Engine launch, readiness, placement measurement, reset and key rotation; every call except health requires the grader token |

- **Deployment.** `POST /deploy` takes a catalogue model ID, engine arguments and a list of `{node, gpus}`
  placements. The manager resolves the model through the trusted catalogue, checks the checkpoint against
  `_kit/checkpoints.lock.json` and sets the context length. Arguments are checked against the allowlist in
  `_kit/engine_args.py` (TP/PP sizes, expert parallelism, memory utilization, batching and sequence limits,
  block size, prefix caching, chunked prefill, eager mode, sleep mode, `fcfs`/`priority` scheduling); unknown,
  abbreviated or duplicated flags return 400. Model identity, numerics (BF16 activations, the checkpoint's
  native weight format), sampling and endpoints are fixed; the platform uses vLLM's built-in model
  implementation, sends and verifies native token IDs, and loads tokenizer files only for the checkpoints whose
  vLLM 0.28 input processors need them (`_kit/runtime_compat.py`). Defaults are the catalogue's `min_tp`, 4,096
  batched tokens and 256 sequences. Data-parallel replicas are separate deployments. An engine may span nodes
  when TP x PP is divisible by the node count; the head node serves and its peers run headless. Engines may
  share GPUs; memory failures surface as crashed engines (logs stay available through `/logs` after unload).
- **Requests and receipts.** Requests carry `model`, a list of integer `prompt` IDs, a positive `max_tokens`
  (an upper bound, not the answer length), `stream: true` and `routing` metadata with a fresh per-attempt
  nonce. The driver appends an authenticated, encrypted length marker (six tokens) to every prompt. The worker
  signs an HMAC over the served model, the SHA-256 of the received prompt IDs, the wire token cap, the SHA-256
  of the emitted output IDs, the finish reason and the nonce; the driver reconstructs these fields itself. A
  completion needs a valid receipt, `[DONE]`, finish reason `length` and the intended number of output IDs.
  Streams are bounded at `65536 + 16*(I+O)` bytes per line and `65536 + 16*I + 1024*O` bytes in total for
  `I` prompt and `O` wire output tokens. Only token-ID frames count for TTFT and TPOT; mean TPOT is
  `(last_token_time - first_token_time) / (output_tokens - 1)`.
- **Telemetry.** Every engine runs the same instrumented vLLM scheduler; the worker exposes its native KV
  events and forward-pass metrics through a bounded, read-only HTTP bridge with explicit sequence gaps and
  cursor expiry.
- **Host memory.** Each node's fleet container runs under a cgroup equal to the published budget (968 GiB, swap
  disabled), so sleeping weights and every other fleet process share an enforced limit.

## Workload

Each graded cell offers the whole catalogue for 900 seconds of arrivals in three 300-second popularity phases
with 60-second transitions; every model keeps a positive share. About 60% of logical requests are isolated
requests, 20% conversation turns and 20% batch requests. ServeGen (pinned at commit
`d70c8b2d5bc45f4b60a146fbc43dc5523f871c8b`) supplies the arrival and token-length distributions, conversation
profiles and batch jobs; model assignment, popularity shifts and token contents are synthetic. The base rate
is 2 logical requests per second (`_kit/workload-profile.json`), and the three cells run at 0.75, 1.0 and 1.25
times it, each with its own seed; the graded traces plan 1,327, 1,768 and 2,219 logical requests. Conversations
are closed loop: a turn is sent after the previous answer and a predetermined think time, carrying the full
previous wire prompt and the actual answer; a turn may retry a fast failure twice within one patience budget
anchored at its first arrival. Callers abandon at a private, randomized deadline. After the arrival window,
outstanding work drains until completion or abandonment.

Public and private traces come from disjoint sources: conversation profiles and 1024-row batch blocks are
assigned by `int(sha256[:16], 16) % 5 == 0` (public) versus the other folds (private), and the public traces use
their own seeds.

**SLO targets.** Each model has one TTFT and one mean-TPOT target for all request classes and input lengths
(TTFT 2.0 to 3.5 s, TPOT 8 to 26 ms per token). The base TTFT is 2 s and the base TPOT comes from the model's
active-parameter band: 8, 12 or 18 ms below 10B, 30B or 60B active parameters, 25 ms above. For the 14 models
with a matching Artificial Analysis entry in the pinned OpenRouter snapshot (`_kit/market-anchors.json`), the
TTFT target is raised to the market median TTFT rounded up to 250 ms, and the TPOT target to
`ceil(1000 / (market output tokens/s x native-per-standardized token ratio))`, whichever is larger.

## Scoring

For each cell, with all planned logical requests (every batch element and conversation turn, issued or not) in
the denominators:

```text
A = requests that completed and met both their TTFT and mean-TPOT targets / all planned requests
C = mean over the 28 models of (completed requests / planned requests)
U = A x C
reward = mean of U over the cells load-0.75, load-1.0 and load-1.25
```

A completion is an authentic, complete response received before the caller's abandonment deadline; SLO
attainment is measured from the request's original arrival, including queueing, loading and retries. A late
completion counts toward C only. The reward is in [0, 1] with no normalization. It is 0 when:

- any completed response fails the receipt check (a stream ending in `[DONE]` without a valid receipt for that
  exact model, prompt, cap, output and nonce), which zeroes the whole trial;
- the submission is invalid (over 8 GiB or 100,000 entries, a link that escapes `/app` and the home directory,
  or no `/app/run.sh`); it is still reviewed;
- Breakwater's environment-integrity or process-cleanup checks fail, or the integrity judge does not pass.

**Human baseline: 0.4599** (NVIDIA Dynamo with the KAI Scheduler, scored with the same reward; the paper
displays 0.460). In the Singularity Index this task contributes the agent's reward divided by 0.4599.

## Preparing assets

Run from the repository root. The launch host needs Docker, Python 3.12 or newer, `git`, and a way to run
commands on the GPU nodes (`ssh` by default; set `CLUSTER_REMOTE_LAUNCH`, for example
`'srun --nodelist={node} --nodes=1 {cmd}'`). The repository, `tasks/cluster-serverless/runs/` and the weights
must be visible at the same absolute paths on the launch host and on every node.

**1. The workload ships with the task.** The generated world -- catalogues, fleets, suite definitions and the
six traces with their `.meta.json` files (`tests/trace-graded-load-*`, `environment/trace-public-load-*`, about
3.8 MB in total) -- is part of the task tree, and `_dev/world-manifest.json` records the SHA-256 of every file
and every generation input. The graded suite (`tests/suite-graded.json`) also binds the private traces by
SHA-256, and grading refuses any other trace. Check the shipped files (needs numpy):

```bash
python3 tasks/cluster-serverless/_dev/world_manifest.py --verify      # [manifest] 49/49 match
```

Regenerating the world is an optional reproducibility check. The released world was generated on an arm64
(aarch64) host with ServeGen at commit `d70c8b2d5bc45f4b60a146fbc43dc5523f871c8b`, numpy 2.5.2 and scipy 1.18.1;
regeneration needs the same (a clean pinned ServeGen checkout, whose data directory is about 0.9 GB, and
`_dev/generator-requirements.txt`, on an arm64 host) and must reproduce the manifest exactly:

```bash
git clone https://github.com/alibaba/ServeGen.git ~/servegen-root     # or export SERVEGEN_ROOT=/path/to/checkout
git -C ~/servegen-root checkout d70c8b2d5bc45f4b60a146fbc43dc5523f871c8b
python3 -m pip install -e ~/servegen-root -r tasks/cluster-serverless/_dev/generator-requirements.txt
bash tasks/cluster-serverless/_dev/regen_world.sh --output-root /tmp/cluster-world-check
```

`regen_world.sh` rebuilds every generated file in a scratch directory under `runs/` and compares each file and
each generation input with `_dev/world-manifest.json`; any difference is an error. With `--output-root` it
writes the verified copy there; without it, it republishes the (identical) files into the task tree and
refreshes the kit copies (`_dev/sync_kit.sh`).

**2. Stage the weights** (about 5.4 TiB over 28 checkpoints; some repositories are gated and need an accepted
licence and `HF_TOKEN`; needs `huggingface_hub` on the launch host). This takes no GPU nodes, is resumable, and
verifies every file against `_kit/checkpoints.lock.json`:

```bash
export CLUSTER_WEIGHTS_DIR=/shared/hf-weights
tasks/cluster-serverless/bin/cluster.sh weights --plan      # size and access check, no download
tasks/cluster-serverless/bin/cluster.sh weights
```

**3. Check and stage the nodes.** `check` refuses a node set whose GPU count or GPU memory does not match the
catalogue (16 GPUs of about 288 GiB) or that mixes CPU architectures; `stage` builds the fleet image
(`cluster-fleet:v1`, from `environment/fleetsvc`) natively on every node; `multinode` optionally deploys one
model across two nodes (`bin/check_cluster.py`) to confirm cross-node vLLM works before a run:

```bash
export NODES=node1,node2,node3,node4
tasks/cluster-serverless/bin/cluster.sh check --nodes "$NODES"
tasks/cluster-serverless/bin/cluster.sh stage --nodes "$NODES"
tasks/cluster-serverless/bin/cluster.sh multinode --nodes node1,node2
```

## Building the images

The task image copies the public traces, and the verifier image copies `tests/`, including the private graded
traces.

```bash
docker build -t cluster-serverless:local tasks/cluster-serverless/environment
docker build -f breakwater/author/Dockerfile --build-context task_environment=tasks/cluster-serverless/environment \
  --build-arg TASK_BASE=cluster-serverless:local --build-arg AUTHOR_CLI=claude-code \
  -t cluster-serverless-author:claude-code breakwater
docker build -f breakwater/Dockerfile.verifier --build-context task=tasks/cluster-serverless \
  --build-arg TASK_BASE=cluster-serverless:local -t cluster-serverless-verifier:local breakwater
docker build -t breakwater-observer:local breakwater/collector
docker build -f breakwater/Dockerfile.gateway -t breakwater-author-gateway:local breakwater
```

Use `--build-arg AUTHOR_CLI=codex` (tag `cluster-serverless-author:codex`) for a Codex author.

## Running under Breakwater

Start a fleet first. `bin/cluster.sh up` and `breakwater/run.py` must share two secrets: the development
receipt key (given to the agent) and the fleet's grader token (trusted code only).

```bash
export CLUSTER_RECEIPT_KEY=$(openssl rand -hex 32)
export CLUSTER_GRADER_TOKEN=$(openssl rand -hex 24)
export FIREWORKS_API_KEY=...            # the integrity judge
tasks/cluster-serverless/bin/cluster.sh up --nodes "$NODES"
# prints: fleet up: --fleet RUN/fleet.json --public-fleet RUN/fleet.public.json

cp tasks/cluster-serverless/bindings.example.json bindings.json     # an empty list: no asset bindings
python3 breakwater/run.py --task tasks/cluster-serverless \
  --agent-image cluster-serverless-author:claude-code --verifier-image cluster-serverless-verifier:local \
  --observer-image breakwater-observer:local --gateway-image breakwater-author-gateway:local \
  --run-dir RUN_DIR --asset-root ASSET_ROOT --bindings bindings.json \
  --author-cli claude-code --author-model MODEL --author-upstream https://PROVIDER --author-key-env KEY_VAR \
  --fleet RUN/fleet.json --public-fleet RUN/fleet.public.json

tasks/cluster-serverless/bin/cluster.sh down --fleet RUN/fleet.json
```

- The task binds no host assets and uses no local GPU devices; any dedicated empty directory serves as
  `--asset-root`, and `--gpus` is not needed. `RUN_DIR` must be new and outside the task tree.
- `run.py` checks before authoring that the fleet has four nodes of four GPUs matching the catalogue and that
  the public manifest names the same nodes without secrets. The graded suite is development-provisional, which
  `breakwater.json` permits with `cluster.allow_unqualified`; measurements are marked `development_only`.
- Fleet hosts are written as IPv4 addresses so the agent's container can reach them. The fleet run
  directory's `inference-audit/` (recorded in the private manifest) must be readable on the launch host.
- Budgets: up to 4 hours of authoring, up to 6 hours for the three graded cells (`evaluation_timeout_sec`) and
  up to 3 hours for the integrity review (`judge_timeout_sec`), plus startup and cleanup. At the authoring
  deadline the submission is graded as it stands (`agent_deadline: evaluate`).
- Reports land in `RUN_DIR/reports/` (`mechanical-check.json`, `judge.json`, `reward.json`). See
  [breakwater/README.md](../../breakwater/README.md) for the generic stages, safeguards and evidence.

### What the intermediate stage and the verifier do

The intermediate stage (`breakwater/cluster_cells.py`, a fresh container) grades the sealed submission.
Root keeps the private traces (the task's `tests/`), the private fleet manifest, the grader token and every
per-cell receipt key in directories the worker cannot read. Before the cells it checks the fleet's shape and
hardware identity through the controller (four trays of four B300-named GPUs, 16 distinct UUIDs), the trace,
suite and catalogue fingerprints, and the submission limits; it then restricts worker egress to loopback, the
controller and the engine ports (node managers rejected, IPv6 closed). For each cell (`load-0.75`, `load-1.0`,
`load-1.25`; one repeat) it:

1. resets the fleet (every engine unloaded) and rekeys every node with a fresh secret;
2. removes worker-owned temporary files and installs fresh worker-owned copies of `/app` and `/home/worker`,
   plus the graded public catalogue and fleet view;
3. starts `bash /app/run.sh` as the unprivileged worker (UID 1101) with `RLIMIT_NPROC` 512 on 8 pinned CPUs;
4. runs the trusted load driver (`tests/kit/loadgen.py`) as root on the other CPUs: a fixed 900-second setup
   window, 900 seconds of private-trace arrivals, then the drain;
5. stops every worker process and writes the driver's records, statistics and the plane log (stdout capped at
   8 MiB) to the root-only evaluation directory.

After the cells, the host resets the fleet again, copies only this run's fleet inference journals into the
sealed evidence and checks their structure. The read-only verifier (no GPU) recomputes every cell's utility
from the root-written driver records and the private traces with `tests/kit/score.py` and
`tests/kit/reference.py`, requires them to equal the recorded values, and computes the reward. The integrity
judge (Codex with DeepSeek V4.1 Flash) then applies `breakwater/checklist.md` and
`tests/submission_judge.md` to the author transcript, the observer's activity journal, the submission, the
plane logs and the fleet inference journals; agent-stage requests to catalogue models are shown in full.

### Inference journals

Every real worker writes a private journal to the fleet run directory's `inference-audit/`, mounted only in
the trusted fleet containers. Each worker incarnation creates a new JSONL journal whose header identifies the
run, node, worker, model, pinned checkpoint and tokenizer revision. Each accepted request is recorded with its
full prompt token IDs, routing metadata and output cap before vLLM is contacted, and each response frame,
including output token IDs, before it is forwarded; records carry sequence numbers and worker-generated
`audit_id` values. Journals survive unload, fleet reset and container removal, and a failed write stops
inference on that worker. For manual review, decode prompts and output `token_ids` with the recorded tokenizer
revision.

### Development fleet emulator

`bin/cluster.sh up --one-box` starts an emulated 16-GPU fleet on one machine (`_kit/simmanager.py`): same API,
no GPUs, timings from a cost model rather than GB300 hardware, and a memory model that refuses with 507 where
the real fleet would crash. It is a development utility for exercising a control plane locally; it produces no
comparable scores and cannot be used for graded runs.

## Files

| Path | Contents |
| --- | --- |
| `instruction.md` | The statement the agent receives |
| `task.toml`, `breakwater.json`, `bindings.example.json` | Harbor task declaration, Breakwater task contract and (empty) asset bindings |
| `environment/` | Task image: `Dockerfile`, starter plane, helpers, public catalogue, fleet view, suite and traces, agent policy; `_kitcopy/` holds the kit modules the image copies; `fleetsvc/` is the fleet image |
| `tests/` | Grading inputs: driver and scorer (`kit/`), graded and public catalogues, graded fleet manifests, graded suite, private traces, build fingerprint, judge rubric, `test.sh` |
| `_kit/` | Source of truth for the fleet (controller, managers, workers, inference journals, telemetry, emulator), the driver, the scorer, the catalogue and workload generators and their frozen inputs |
| `bin/` | `cluster.sh` (weights, node checks, fleet up/down), `check_gpu_memory.py`, `check_cluster.py` |
| `_dev/` | World verification and optional regeneration: `world_manifest.py`, `world-manifest.json`, `regen_world.{sh,py}`, the OpenRouter snapshot, requirements, the frozen-input builders the manifest records, `sync_kit.sh` and `check_imports.py` |
