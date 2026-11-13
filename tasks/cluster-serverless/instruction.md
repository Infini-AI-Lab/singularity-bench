## Environment integrity

Runtime executables, shared libraries and installed packages are fixed. Do not
replace or shadow supplied packages (including `aiohttp`, `uvloop` and `numpy`) or
the interpreter, or install import/startup hooks outside `/app` and your home directory.
The pre-attempt environment inventory is available read-only at
`/opt/environment-baseline/environment.json`. The host checks it again after
authoring, and checks the fresh grading container before evaluation. A failed
integrity or process-cleanup check stops the run with reward **0**, reason
**violation of rules**. The container transition must leave no live authoring
processes. Keep your implementation and its dependencies under `/app` or your
home directory (`$HOME`), as described below.

# Serverless multi-model inference: implement the control plane

Build an online control plane for **four GB300 trays, each with four B300 GPUs**.
Serve the models in `/app/catalogue.json` while demand changes. You control model
placement, replica counts, engine configuration, admission, queuing, and routing.
You may deploy, sleep, wake, or unload engines throughout a run. The catalogue is
larger than the fleet's serving memory; keeping every model ready is infeasible.

Your goal is to meet the published model latency targets while completing work
across the entire catalogue. Future requests and individual caller deadlines are
not revealed. Grading uses real GPU serving.

## Files and entrypoint

Your commands have access to the task workspace, local control-plane services,
and the controller and engine endpoints supplied in `/app/fleet.json`. Remote
GPU experiments use those APIs. The coding-agent installation and its model
connection are private benchmark services. Do not invoke or copy an installed
coding agent; those installations are inaccessible to commands and removed before
grading. Implement and test the control plane directly. These controls do not
require you to add restrictions to your control plane's permitted fleet access.

Catalogue models are available as **test workloads, not authoring assistants**.
You may send custom prompts to measure a particular model's latency or throughput
and test your implementation's routing, caching, streaming, and protocol
correctness. Do not ask catalogue models to design, write, debug, optimize, or
review your submission, interpret experimental results for you, or otherwise
help solve this benchmark. This includes delegation through a temporary script,
agent loop, or another model. Treat generated responses as test data; do not use
them as implementation code or advice for your solution.
You may inspect test responses and measurements and use your own reasoning to
improve the implementation.

The trusted workers retain inference requests and responses, including direct
probes and partial responses from cancelled requests. These records and your
authoring trajectory are reviewed for compliance before results are accepted.
Custom test traffic remains permitted; the review considers its purpose and how
you use the responses.

| Path | Purpose |
| --- | --- |
| `/app/catalogue.json` | Model IDs, pinned checkpoint properties, memory estimates, context limits, and TTFT/TPOT targets |
| `/app/fleet.json` | Controller address and four nodes with GPU and host-memory budgets |
| `/app/control_plane.py` | A naive working starter that you may replace |
| `/app/run.sh` | Starts your implementation |
| `/app/start_control_plane.sh`, `/app/stop_control_plane.sh` | Safely restart/stop the local server during development |
| `/app/process_safety.md` | Process-management guidance |
| `/app/trace-public-*.jsonl` | Three public workload realizations |
| `/app/suite-public.json` | Public load multipliers and arrival-window metadata |
| `/app/kit/loadgen.py` | Public replay driver |
| `/app/service_contract.py` | Public target lookup |
| `/app/controller.py`, `/app/engine_args.py` | Interface implementation and accepted engine arguments |
| `/app/selfcheck.py` | Basic protocol check |

`bash /app/run.sh` must start a server on port **8000**, with `GET /health` and
streaming `POST /v1/completions`. Health must return 200 within the **900-second
setup window**, measured from process launch. Every evaluation cell starts with
an empty fleet and a fresh process/filesystem. Initialization consumes setup time.
Loading can continue once arrivals begin.

You have four hours for development and experiments, followed by grading.
Keep your implementation and dependencies under `/app` or your home directory
(`$HOME`, `/home/worker`): their combined submission limit is **8 GiB and 100,000
entries**. Every cell installs fresh copies of both at the same paths. Regular
files, directories and symlinks are supported; special files are not, and every
symlink must resolve inside `/app` or `$HOME`. Each cell gives your server 8 CPUs,
512 processes (threads included) and 65,536 file descriptors per process, in a
40 GiB grading container shared with the load driver. You run unprivileged;
stdout retention is 8 MiB per cell. Internet access is restricted. No upstream
router or expert reference policy is installed.

## Controller and worker APIs

The controller address is in `fleet.json`. It creates and destroys engines:

```text
POST /deploy
  {"model":"...", "engine_args":[...],
   "placement":[{"node":"node0","gpus":[0,1]}]}
  -> {"worker":"w...", "url":"http://...:91xx", "model":"...",
      "requested":[...], "placement":[...]}
POST /unload {"worker":"w..."}       # idempotent
GET /nodes                           # measured GPU usage, host RAM, runtime
GET /workers                         # deployments and actual live placement
GET /logs?worker=w...                 # bounded per-node logs, retained after unload
GET /health
```

An engine may span nodes; its worker URL is the single serving/lifecycle endpoint:

```text
POST /v1/completions
POST /admin/sleep {}
POST /admin/wake {}
GET /stats
GET /health
GET /telemetry?kind=kv&after=0
GET /telemetry?kind=fpm&after=0
```

Managers are private execution services. Use the controller and worker APIs.
There is no fixed routing policy: routing is part of your implementation.

`placement` lists visible devices in order. It controls `CUDA_VISIBLE_DEVICES`;
TP/PP arguments determine how many are used. Omitting explicit GPU lists can make
multiple engines contend on the first cards. `/workers` reports actual placement.
Multiple engines may share a physical GPU. Track memory yourself: deployments and
wakes can fail if their requested reservations do not fit.

Accepted vLLM options are listed in `engine_args.py`:

- Tensor and pipeline parallel sizes (`--tensor-parallel-size`, `--pipeline-parallel-size`).
- `--enable-expert-parallel` for supported model/runtime configurations.
- GPU utilization, CPU offload/swap, sequence and batching limits, block size,
  and `fcfs` or `priority` scheduling.
- Prefix caching, chunked prefill, eager execution, and sleep-mode switches.

Unknown, abbreviated, duplicated, or invalid arguments return 400. The platform
fixes model identity, context length, numerics, sampling, and endpoints. Activation
dtype is BF16; the checkpoint's native weight format remains in effect. Deploy
separate workers for data-parallel replicas; embedded data-parallel size must be 1.
FSDP is not an inference engine option. Defaults are the catalogue's `min_tp`,
4,096 batched tokens and 256 sequences. Supported parallelism depends on the model;
an accepted argument is not a promise that a configuration will start or fit.

For N nodes, TP × PP must be divisible by N, with enough visible GPUs per node.
The first node hosts the API; peers are headless. Lifecycle operations apply to
the entire engine. Cross-tray communication uses the actual cluster fabric.

`weights_gib` estimates residency/load cost; `active_gib` estimates the weights
read per token. These differ for MoE models. Neither is a complete KV-cache or
runtime-overhead budget. `/nodes` provides measured memory and the enforced
node-wide host-RAM budget. Sleeping weights share that budget with other fleet
processes. Exceeding it can trigger an OOM kill.

Engines start awake. `--enable-sleep-mode` makes sleep/wake available; without it
those calls return 409. Sleep interrupts active requests and moves weights to
host RAM. Unload removes the engine. Loading and waking serialize on occupied
GPUs, so issue independent operations concurrently. A crashed engine's worker
remains available for diagnostics but returns 503 for inference; redeploy to
recover. Also handle unavailable/asleep workers, memory exhaustion, launch
failures, and upstream transport failures. Measure actual transition costs.

## Requests, streaming and telemetry

Requests contain only `model`, integer-list `prompt`, positive `max_tokens`,
`stream: true`, and `routing`. Preserve all of them when forwarding. Routing
metadata includes a logical request ID, a fresh per-attempt nonce, and
`service_tier: "standard"`. Each model has **one fixed TTFT/mean-TPOT target pair**
for all classes and input lengths. Read `slo_ttft_ms` and `slo_tpot_ms`; the public
`service_contracts` table repeats that pair across supported input bins.

`max_tokens` is an upper bound, not the actual answer length. Workers produce
native token IDs. Forward the complete, uncompressed SSE stream, including its
`cluster_receipt` and `[DONE]`. The receipt authenticates the actual model, prompt,
wire cap, output IDs, finish reason and attempt nonce. It cannot be reused for
another attempt. Do not modify prompts, change caps, substitute models, fabricate
answers, shorten streams, or replay receipts.

For I input tokens and wire cap O, each line is bounded by
`65536 + 16*(I+O)` bytes, and the whole stream by `65536 + 16*I + 1024*O` bytes.
The worker's initial prompt-ID frame can be large. Only received output-token IDs
start TTFT or contribute to TPOT; text and metadata do not.

All policies have access to the same read-only telemetry bridge. `kind=kv` carries
actual vLLM KV events; `kind=fpm` carries native forward-pass measurements.
Each event contains base64-encoded native multipart frames and a cursor. Advance
using `next_cursor`; `latest_cursor` identifies outstanding data. Pages include
resolved engine/cache capabilities, source-gap counts and stream age. The history
is bounded: expired cursors return 409. Do not assume an old cache index remains
valid after event loss, unload or restart. Telemetry is optional for your policy.

## Workload and timing

Each cell mixes roughly 60% isolated requests, 20% conversation turns and 20%
batch requests by logical request count. Arrival/token distributions, session
lengths/timing and batch structure come from empirical ServeGen sources. Model
assignment and token contents are synthesized. Popularity changes smoothly across
three 300-second phases, with 60-second transitions; every model remains offered.

Isolated arrivals and batch submissions are independent of your responses.
Conversations are closed-loop: after a completed turn and its predetermined think
time, the next turn arrives. Its prompt retains the previous wire prompt and the
**actual generated answer**, then adds new user tokens. Source history resets are
preserved. Prefix reuse therefore depends on where previous turns were served.

A conversation turn may retry a fast failure up to twice. All attempts share one
patience budget anchored at its original arrival; retries never renew that budget.
Their prompts match and their nonces differ. A fast successful retry can still
meet the original SLO. An unanswered turn ends the conversation, but every planned
later turn remains in the score denominator as incomplete.

Callers disconnect when their private patience expires. The latency allowance is
based on the model targets and wire output cap plus randomized patience. You see
the public rule and public examples, but not a live request's exact deadline.
Arrivals stop after 900 seconds; accepted batches/conversations continue until
their own completion or abandonment. There is no arrival-window cutoff.

## Reward

Every planned logical request has weight one, including each batch element and
all conversation turns. A completion requires an authentic complete response
received before abandonment. Strict attainment additionally requires both model
SLOs, measured from the request's original arrival, including queuing/loading and
retries. Mean TPOT is `(last_token_time - first_token_time)/(output_tokens - 1)`.
A one-token response has no TPOT condition. Buffering counts toward TTFT.

For each cell:

```text
A = requests meeting both SLOs / all planned requests
C = mean over all offered models of (completed requests / planned requests)
U = A * C
cell_reward = U
final_reward = mean(cell_reward at loads 0.75, 1.0, 1.25)
```

Each model has equal weight in C regardless of popularity. Completing work outside
an SLO can improve C, but does not improve A. There is no linear late-credit curve
or minimum-completion threshold. Discarding an unpopular model reduces C even if
it barely changes the global request count. Failed and unissued requests remain
in both denominators. Complete repeat utilities are averaged per cell before
the arithmetic mean across load cells. Reward is already in [0, 1]; no baseline
normalization or clipping is applied.

The separately measured baseline is a Dynamo-based system with native cache-aware routing and native
SLA replica recommendations, plus an adapter for overcommitted model residency
and physical placement. It receives the same public inputs, targets, telemetry,
setup window, resource limits and worker APIs. Its score does not affect yours.
Its code is not a starter.

Any completed-response integrity violation zeros the **entire trial**, even if a
retry later succeeds. Ordinary timeouts, refusal and genuine incomplete streams
are service failures. Missing infrastructure invalidates a measurement.

## Development loop

Use the helpers below when restarting your local server. Broad commands such as
`pkill -f control_plane.py` or `pkill python3` can kill your coding session or its
command helpers. Even a full-path pattern can match task text on a parent command
line. Read `/app/process_safety.md`; use the controller API for remote GPU engines.

```bash
bash /app/start_control_plane.sh
python3 /app/selfcheck.py
python3 /app/kit/loadgen.py --trace /app/trace-public-load-1.0.jsonl \
  --fleet /app/fleet.json --target http://127.0.0.1:8000 \
  --key "$CLUSTER_RECEIPT_KEY" --warmup 10 \
  --out /tmp/records.jsonl --stats-out /tmp/stats.json
bash /app/stop_control_plane.sh
```

The public development key is replaced before grading. Private signing keys,
traces and results are inaccessible to your implementation. Public driver records
include `served`, `complete`, `receipt_ok`, `ttft_ms`, `tpot_ms`, `e2e_ms`, `status`
and `note`. Compare measured `tpot_ms` to trace `slo_tbt_ms` (the same quantity as
catalogue `slo_tpot_ms`). Compute U using all planned requests. Inspect per-model completion as well as
aggregate attainment. Public traces teach workload structure; the private
realizations and future requests are different.
