# Breakwater

Breakwater is the evaluation and safety framework of Singularity-Bench. Its tasks are open-ended: an agent spends
hours developing on real GPUs, and its submission is a working system (a model, a serving engine, a harness, a control
plane) whose evaluation may have to run the code the agent wrote. Breakwater runs such tasks, written in
[Harbor](https://github.com/harbor-framework/harbor)'s task format, with five safeguards:

1. **Environment fingerprinting** ([environment_guard.py](environment_guard.py)): the runtime's executables, libraries,
   package trees and import paths are inventoried before the agent phase and checked after it, and again in the fresh
   evaluation container. Any difference fails the submission.
2. **API routing** ([author_gateway.py](author_gateway.py)): the agent's single model connection goes through a gateway
   that forwards only the assigned model and protocol. Every other model or API access is blocked.
3. **Process wall** ([process_guard.py](process_guard.py), [collector/](collector/)): when the agent phase ends, every
   live process is stopped and the drain is proven before anything else starts, which rules out sidecar processes.
4. **External journal** ([collector/](collector/), the gateways): model traffic, tool calls, command output, process
   and file events are recorded outside the agent's reach. The agent can neither read nor write this record.
5. **Restricted intermediate stage** ([intermediate.py](intermediate.py), [workers.py](workers.py)): submitted code
   runs as an unprivileged worker in a fresh container between the agent container and the verifier. The verifier
   ([verifier.py](verifier.py)) is a separate read-only container that recomputes the score from sealed raw outputs and
   runs an LLM integrity judge, without executing submitted code.

One copy of the framework serves all six tasks. Each task declares its contract in `tasks/<name>/breakwater.json` and
the prepared assets each stage may see in `tasks/<name>/bindings.example.json`.

## How a run proceeds

1. **Agent container.** The protected author image ([author/](author/), [PROFILES.md](PROFILES.md)) with its own
   writable workspace and local native log. The agent CLI runs as `author` (UID 1100) and every command it issues runs
   as `worker` (UID 1101). The container has no external-evidence, authority or report mount and no provider
   credential; model requests go through the separate author gateway.
2. **Intermediate container.** A fresh image instance with private PID and IPC namespaces, started after the agent
   container has stopped. Only the sealed submission crosses, read-only. The trusted coordinator handles private inputs
   as root; submitted code (pre-training generation adapters, serving engines, harnesses, control planes) runs as the
   worker, with no access to labels, tests, traces, keys or raw result storage under `/private`. No external logs,
   agent-local logs or judge key enter this container.
3. **Verifier container.** A third, distinct container after evaluation stops: UID 1102, read-only root filesystem,
   no capabilities, no GPU, `no-new-privileges`. External evidence, the local-log snapshot, the submission, raw results
   and the host authority are read-only mounts; `/reports` and temporary judge state are the only writable paths. The
   verifier reconciles the logs and checks the seals, recomputes the score from the sealed raw outputs, then runs the
   integrity judge (Codex with DeepSeek V4.1 Flash) on the mandatory [checklist](checklist.md).

The gateways and observers are trusted logging sidecars, not extra workload phases. The observer has host PID
visibility and `SYS_PTRACE` only to attach before worker execution; neither its storage nor its namespace is exposed
to either execution container. The host seals evidence only after those containers stop. The agent CLI's stdout and
stderr have independent host-owned copies; the agent-visible copies are untrusted and are compared byte for byte and by
tool call, command, output and exit status. Observer and gateway sequence/hash chains, completion records and capture
gaps are checked too. A failure, missing capture, truncated log, extra or deleted event, wrong run identity, write
access or contradictory checklist cannot yield a passing grade. A provider error on an author model call is recorded
evidence, not missing capture; the verifier reports it in `mechanical-check.json`.

### Environment integrity and process drain

Before the agent starts, the trusted controller inventories runtime binaries, Python package trees, metadata,
permissions and import paths from Docker's filesystem export. The full SHA-256 inventory is mounted read-only in the
agent container at `/opt/environment-baseline/environment.json`. After stopping the agent, the controller hashes its
filesystem again without executing its interpreter or importing its packages. It also hashes the fresh intermediate
container after GPU initialization, while it runs only a trusted `sleep` and before any submitted code; both must match
the initial inventory. Only the fresh-container comparison lets NVIDIA's two random loader-config filenames differ,
with identical content hashes, ownership, permissions and roles; the alias mapping is recorded. Task-local dependency
impersonations, alternate site-packages or virtual environments and startup hooks are checked separately against the
initial workspace, under the task's declared shadow policy. Symbolic links count like files at their path.

After each execution container stops, the independent observer checks its pinned PID namespace and container cgroup,
including descendant cgroups and nested PID namespaces. Two consecutive empty process scans and Docker's
stopped/PID-zero state are required before the next stage starts. Only a sealed file snapshot crosses the boundary;
writable root filesystems, processes and sockets are never reused. A missing or failed process-drain or
environment-integrity proof stops the run immediately, skips the judge, and writes `reward.json` with the task's
rule-violation reward (`rule_violation_reward`: 0, or -1 for open-ended-nano-gpt, whose score is a perplexity) plus
`rule-violation.json` with reason `violation of rules`.

### Authoring deadline and submitted links

Every task grades what exists when the authoring budget ends (`agent_deadline: evaluate`). A CLI still running at the
deadline is stopped exactly like a normal exit (process wall, drain, environment check, snapshot), and a CLI that exits
with an error also proceeds to evaluation. The agent stage records `exit_code` and `deadline_reached`; the verifier
accepts that record only under a sealed `evaluate` policy. Only after a recorded deadline may the native log end with
an unfinished final line and at most one tool call without a result, the one in flight; the byte-for-byte comparison
with the external copy still applies.

Symbolic links in the workspace (and, for cluster-serverless, in the worker's home directory) are not sealed as files.
Each one is recorded with its link text in `external/workspace-links.json` and in the seal manifest, counted by the
import-shadow check like a file at its path, and shown to the judge. Only serve-glm53-flash and cluster-serverless
recreate links at evaluation, never following them, and only when they resolve inside the installed submission
(serve-glm53-flash: or the read-only model directory); any other link makes the submission invalid. Hard links and
special files always fail the snapshot.

### Integrity judge

[checklist.md](checklist.md) and [judge.schema.json](judge.schema.json) apply to every task, together with the task's
`tests/submission_judge.md`. Command execution, subagents, images and web search are disabled in the judge. Under the
`indexed` review profile ([review_index.py](review_index.py)), used by all six tasks, the judge sees a security-event
index of the raw syscall traces (every exec, write-capable open, mutation, signal, prctl, mount, io_uring and
shared-mapping syscall, failed write and probe outside system roots, each with the raw file's SHA-256), model journals
as per-lane conversation deltas plus complete responses, the workspace as a diff against the pre-agent inventory,
byte-identical copies once, and tensor payloads as metadata; the `exhaustive` profile reviews every retained
non-tensor byte in bounded packets instead. Safetensors are inspected structurally and represented by their complete
headers plus file hashes. Other data payloads (files with a NUL byte in their first 8 KB, common tensor, array and
checkpoint suffixes, and paths listed in `review_metadata_paths`) are represented by SHA-256, size and a short head;
pickles, including those inside checkpoint archives, stay reviewable in full. Raw evaluation outputs that the verifier
recomputes mechanically are summarized (hash, size, rows, bounded head). At most eight judge calls run at once, each
with its own retries (at most three attempts) and Codex home, under the task's judge budget. A valid verdict, including
FAIL or INCONCLUSIVE, is final; the final reward is the recomputed score only on PASS.

## Task contracts (`breakwater.json`)

| Task | `kind` | Workspace | GPUs | Authoring / evaluation / judge (s) | Failure / violation reward | Evaluation stage |
|---|---|---|---|---|---|---|
| open-ended-nano-gpt | `nano` | task tree at `/app` | 4 | 43,200 / 21,600 / 10,800 | -1 / -1 | perplexity and LAMBADA generation through the submitted adapter |
| midtrain-for-rl | `midtrain` | task tree at `/app` | 4 | 43,200 / 86,400 / 10,800 | 0 / 0 | the fixed GRPO stage, then post-RL evaluation; only the code sandbox is reachable |
| rl-swe-agent | `rl` | task tree at `/app` | 4 | 43,200 / 16,200 / 10,800 | 0 / 0 | none: scored on the host by `tasks/rl-swe-agent/evaluation` |
| serve-glm53-flash | `serving` | image `/app`, graded from a fresh writable copy | 4 | 43,200 / 21,600 / 10,800 | 0 / 0 | the traffic replay and accuracy check against `bash /app/serve.sh` |
| coding-agent-harness | `harness` | image `/app` | 1 | 21,600 / 3,600 / 10,800 | 0 / 0 | the validity battery; graded grid cells run afterwards (`grid.py`) |
| cluster-serverless | `cluster` | image `/app` plus the worker home | 0 (remote 4 x 4 fleet) | 14,400 / 21,600 / 10,800 | 0 / 0 | three graded load cells against the remote fleet |

The keys, with their defaults in `run.py` `DEFAULTS` and strict validation in `load_config`:

| Key | Meaning |
|---|---|
| `kind`, `task`, `workdir` | the evaluation adapter, task name and agent workspace path |
| `gpus`, `gpu_name_patterns`, `gpu_type` | the declared CDI devices (0 to 4; default 4 GB300) |
| `agent_timeout_sec`, `evaluation_timeout_sec`, `judge_timeout_sec` | the three stage budgets |
| `agent_deadline` | `fail` (default) or `evaluate`: what happens when authoring reaches its budget |
| `failure_reward`, `rule_violation_reward` | the reward of a failed run and of a rule violation (default 0) |
| `workspace_source`, `workspace_excludes` | seed the workspace from the task tree (`task`) or the image workdir (`image`); a plain exclude is a name anywhere in the task tree, a leading `/` anchors a path at the task root (the task's `tests/`, `breakwater.json` and `bindings.example.json` are never seeded) |
| `submission_mount` | where the sealed workspace is mounted for evaluation when it is not `workdir` |
| `snapshot_excludes`, `snapshot_worker_home` | workspace paths left out of the sealed submission; also seal the worker's home directory (mounted at `/submission-home`) |
| `protected_workspace_paths` | task files the agent may read but must not change |
| `dataset_manifest`, `dataset_roots` | an agent-written training-dataset manifest that the observer verifies (declared `sha256` values) and archives when authoring ends, and the workspace roots its files must lie under (midtrain-for-rl) |
| `shadow_policy` | the packages, hooks and virtual-environment markers the import-shadow check covers; empty lists and `environments: false` turn parts off |
| `resources`, `agent_resources` | `cpus`, `cpuset`, `memory_mb`, `pids`, `nofile`, `memlock`, `stack`, `shm_size` for both execution stages; agent-only overrides |
| `intermediate_network`, `intermediate_cap_add`, `intermediate_env`, `intermediate_add_hosts` | the evaluation stage's network, network capabilities, named host variables (never credentials) and the host-gateway alias |
| `review_profile`, `review_metadata_paths` | `exhaustive` or `indexed` review; paths the judge sees as metadata |
| `development_gateway`, `cluster` | coding-agent-harness's budgeted development models; cluster-serverless's grading options |
| `sandbox_worker` | rl-swe-agent's host sandbox worker port and the agent's session-server port range: the agent container joins a dedicated bridge network, the range is published on the host, and the authoring firewall admits only the worker endpoint and replies on those ports (draft, not yet validated on GB300 hosts) |

Every key that changes what is graded or accepted (`task`, `kind`, `gpus`, `workdir`, `review_profile`,
`agent_deadline`, `rule_violation_reward`, `snapshot_excludes`, `snapshot_worker_home`, `review_metadata_paths`) is
sealed into the host authority, and the verifier refuses a contract that differs from the copy in its image.

## Running a task

**Requirements.** A trusted Linux host with NVIDIA GB300 GPUs (cluster-serverless also needs a four-node fleet),
Docker with BuildKit, the NVIDIA Container Toolkit with a generated CDI specification, and Python 3.11 or newer on the
host. Credentials: the author model's provider key (any variable, named with `--author-key-env`) and
`FIREWORKS_API_KEY` for the integrity judge; coding-agent-harness also needs a key for its development models
(`--dev-key-env`, default `FIREWORKS_API_KEY`). Keep run directories and the prepared asset root outside the
repository. Run every command from the repository root.

1. **Prepare the task's assets and its runtime image** as described in the task README (`tasks/<name>/README.md`).
2. **Build the protected author image** on top of the task runtime, choosing the agent CLI (`claude-code` or `codex`):
   ```bash
   docker build -f breakwater/author/Dockerfile --build-context task_environment=tasks/<name>/environment \
     --build-arg TASK_BASE=<name>:local --build-arg AUTHOR_CLI=claude-code -t <name>-author:claude-code breakwater
   ```
3. **Build the task's verifier image and the shared logging images** (the observer and gateways once per host):
   ```bash
   docker build -f breakwater/Dockerfile.verifier --build-context task=tasks/<name> \
     --build-arg TASK_BASE=<name>:local -t <name>-verifier:local breakwater
   docker build -t breakwater-observer:local breakwater/collector
   docker build -f breakwater/Dockerfile.gateway -t breakwater-author-gateway:local breakwater
   docker build -f breakwater/Dockerfile.dev-gateway -t breakwater-dev-gateway:local breakwater  # coding-agent-harness
   ```
4. **Bind the assets.** Copy `tasks/<name>/bindings.example.json` and point each entry's `source` at the prepared
   asset, relative to the asset root. Targets and per-stage visibility are fixed by the task contract; no private path
   can be bound into the agent container.
5. **Launch:**
   ```bash
   python3 breakwater/run.py --task tasks/<name> \
     --agent-image <name>-author:claude-code --verifier-image <name>-verifier:local \
     --observer-image breakwater-observer:local --gateway-image breakwater-author-gateway:local \
     --author-cli claude-code --author-model MODEL \
     --author-upstream https://api.anthropic.com --author-key-env ANTHROPIC_API_KEY \
     --asset-root /data/prepared --bindings /data/bindings/<name>.json --run-dir /data/runs/RUN_ID
   ```
   `--author-upstream` is the provider's HTTPS endpoint for the selected protocol (for example
   `https://api.fireworks.ai/inference` for an Anthropic-compatible Fireworks model with Claude Code, or
   `https://api.openai.com` with Codex). `--gpus device=0,1,2,3` selects the task's declared CDI devices (by default
   the first `gpus` devices; none for cluster-serverless). `--intermediate-image` defaults to the author image.
   coding-agent-harness adds `--dev-gateway-image breakwater-dev-gateway:local`; cluster-serverless adds `--fleet` and
   `--public-fleet` (below).

The controller resolves image tags to immutable IDs, checks the actual Docker mounts, namespaces and credentials
before execution, runs live negative access probes before either untrusted stage, and requires a write to a sealed
canary inside the verifier to fail. Only the gateways receive provider keys; only the verifier receives
`FIREWORKS_API_KEY`, for the judge. Reports land in `RUN_DIR/reports/`: `mechanical-check.json`, the checklist
verdicts (`judge.json`), packet coverage, `llmjudgecomments.md` and the final `reward.json`. External evidence is
kept separately under `RUN_DIR/external/`. A raw intermediate reward is never the final reward.

## Task notes

- **open-ended-nano-gpt.** The coordinator prepares answer-free generation operands in a root-owned `/eval-inputs`
  and runs the submitted generation adapter as the worker through the fixed runners in [drivers/](drivers/);
  perplexity is computed by trusted code ([nano_stages.py](nano_stages.py)). Lower is better, so a failed run or a
  rule violation scores -1, never 0. The judge also receives the perplexity metrics, the generation result and the
  gate.
- **midtrain-for-rl.** The evaluation stage runs the task's grading command (`tests/grade.py`): the fixed five-hour
  GRPO stage from the submitted checkpoint, then post-RL evaluation on AMC, HumanEval+ and GPQA-Diamond. Its only
  network peer is the operator's code sandbox (`MIDTRAIN_CODE_SANDBOX_URL` and `MIDTRAIN_CODE_SANDBOX_TOKEN` on the
  host, reached through `host.docker.internal`); IPv6 is closed. The verifier recomputes AMC and GPQA from the sealed
  responses and takes the sandbox-graded HumanEval+ rewards as recorded.
- **rl-swe-agent.** During authoring, the agent's rollout code reaches the host's Harbor sandbox worker at
  `http://sandbox-worker:11000`, and the task sandboxes the worker starts reach the agent's session servers at
  `host.docker.internal:30000-30015`, published from the agent container (`sandbox_worker`). The Breakwater run covers
  authoring, sealing and the integrity review. The held-out suites need one
  Harbor sandbox per task, which the evaluation container cannot start, so the verifier writes
  `reports/host-evaluation.json` instead of a reward. The trusted host evaluator
  (`tasks/rl-swe-agent/evaluation/evaluate_checkpoint.py`) then serves the sealed checkpoint
  (`RUN_DIR/submission/out/final_checkpoint`) on the held-out suites, and `tasks/rl-swe-agent/evaluation/score.py
  TRIAL_DIR EVAL_OUT --integrity RUN_DIR/reports/judge.json` writes the final reward, which is 0 unless the review
  passed.
- **serve-glm53-flash.** The sealed `/app` is mounted at `/submission`; each evaluation installs a fresh, worker-owned,
  writable copy at `/app` and starts `bash /app/serve.sh` through the worker wrapper. The verifier recomputes income
  per GPU-hour from the complete traffic capture (four GPUs, 900 s window, 300 s drain, 16 lanes) and the accuracy
  generations, and requires it to equal the coordinator's reward.
- **coding-agent-harness.** Authoring has a budgeted development gateway ([dev_gateway.py](dev_gateway.py), its own
  provider key, never in the agent container). The evaluation stage runs the validity battery
  ([harness_battery.py](harness_battery.py)) on one GPU with no network. The graded grid then runs each (downstream
  task, driver model) cell as its own three-stage run ([grid.py](grid.py), [cell.py](cell.py),
  [cell-pipeline.json](cell-pipeline.json)); the cell images are built with [cell/Dockerfile.overlay](cell/Dockerfile.overlay)
  and [cell/Dockerfile.verifier](cell/Dockerfile.verifier), listed in a file like
  [images.example.json](images.example.json). See the task README for the commands.
- **cluster-serverless.** Start the fleet with `tasks/cluster-serverless/bin/cluster.sh up --nodes n1,n2,n3,n4` (four
  GPUs per node) after exporting `CLUSTER_RECEIPT_KEY` and `CLUSTER_GRADER_TOKEN`, and pass its manifests to `run.py`
  with `--fleet` and `--public-fleet`. The launcher requires four nodes of four GPUs before authoring and again before
  grading. Authoring has 8 CPUs, 32 GiB and 512 processes, and the worker's home directory is sealed with the
  workspace. Before each graded cell, every worker-owned file in `/tmp`, `/var/tmp`, `/dev/shm`, `/run/lock` and
  `/dev/mqueue` is removed, fresh copies of `/app` and the home directory are installed at their authoring paths, and
  the control plane starts with `RLIMIT_NPROC` 512 on its own half of the container's `cpuset`. A submission over
  8 GiB or 100,000 entries, or with a link that escapes `/app` and the home directory, is invalid: every cell scores 0
  and it is still judged. The graded suite is a development suite, accepted because the contract declares
  `cluster.allow_unqualified`.

## Layout

| Path | Responsibility |
|---|---|
| [run.py](run.py) | the launcher: contract validation, image and mount checks, the three stages, sealing |
| [contract.py](contract.py), [bundle_storage.py](bundle_storage.py) | seals, inventories, boundary inspection, strict JSON |
| [environment_guard.py](environment_guard.py), [process_guard.py](process_guard.py) | environment fingerprinting and import-shadow policy; process drain |
| [collector/](collector/), [observed.py](observed.py), [capture/](capture/) | the external observer, the observed-stage entry point, the stdio supervisor |
| [author_gateway.py](author_gateway.py), [dev_gateway.py](dev_gateway.py) | the author gateway; the development and driver-model gateway |
| [author/](author/), [PROFILES.md](PROFILES.md) | the protected author image and its agent CLI profiles |
| [intermediate.py](intermediate.py), [workers.py](workers.py) | the evaluation-stage coordinator; worker installation and teardown |
| [score_workload.py](score_workload.py), [tensor_inventory.py](tensor_inventory.py) | deterministic score recomputation; safetensors inventory |
| [verifier.py](verifier.py), [reconcile.py](reconcile.py), [review_index.py](review_index.py), [checklist.md](checklist.md), [judge.schema.json](judge.schema.json) | the verifier, log reconciliation and the integrity judge |
| [nano_stages.py](nano_stages.py), [drivers/](drivers/) | open-ended-nano-gpt evaluation |
| [harness_battery.py](harness_battery.py), [harness_child.py](harness_child.py) | coding-agent-harness validity battery |
| [grid.py](grid.py), [cell.py](cell.py), `cell_*.py`, [cell/](cell/), [downstream/](downstream/), [worker/](worker/) | coding-agent-harness grid cells |
| [cluster_cells.py](cluster_cells.py), [cluster_fleet.py](cluster_fleet.py) | cluster-serverless graded cells and fleet checks |
| [tests/](tests/) | the framework's test suite |

## Tests

```bash
cd breakwater && python3 -m unittest discover -s tests -p 'test_*.py' -v
```

The CPU suite covers tampering, missing, extra or reordered events, broken hash chains, wrong identities, mount aliases
and ancestor mounts, credential and socket exposure, reused containers, overlapping stages, package and import-hook
tampering, recorded links, contradictory checklist verdicts, malformed judge replies and bounded retries, unknown
native event shapes, both gateway roles against a fake upstream, the author gateway against a fake provider, the
harness validity battery against known-good and hostile harnesses, cluster scoring gates and fleet-journal structure,
kernel and PR cell recomputation, overlay smuggling, the review index with its metadata and concurrency limits, the
authoring-deadline record, Docker resource flags, the rule-violation reward, cluster submission limits and fleet
shape, serving recomputation with the task's simulator, and every task's contract. The live Docker tests
(`test_docker_*.py`) skip without Docker and test the read-only verifier boundary, the recorder with a planted
detached process, image workspace seeding, the validity battery inside a protected intermediate container and the
graded-cell coordinator against a stub fleet. A skip is not evidence of runtime isolation.

The optional GPU test runs BF16 matrix multiplication on four GB300 GPUs in separate agent and intermediate containers
with independent capture, and checks process drain, unchanged runtime hashes, fresh temporary state and the rejection
of a no-op Torch replacement:

```bash
docker build --build-arg TASK_BASE=<name>:local -f breakwater/capture/Dockerfile.gpu-fixture \
  -t breakwater-gpu-fixture:local breakwater/capture
cd breakwater && BREAKWATER_GPU_TEST_IMAGE=breakwater-gpu-fixture:local \
  python3 -m unittest discover -s tests -p test_gpu_runtime.py -v
```

## Known limits

- Candidate kernel timing in harness grid cells includes one pipe round trip per timed batch: a small bias against the
  candidate that cannot be subtracted without trusting the worker.
- In-process cheats, such as content-keyed memoization or faked synchronization inside a kernel, rely on the judge
  reading the sealed overlay.
- A harness that binds `run` statically but hijacks its runner's channel in-process to forge gradeability is caught
  only by the judge; it gains nothing in the grid, which re-imports it.
- Cluster prompts are token IDs: the verifier has no tokenizer, so the judge reads response text and the author
  transcript.
- Fleet-journal phase windows use host wall clocks.
- The rl-swe-agent sandbox network path (`sandbox_worker`) is a draft. It publishes the session ports on the host's
  loopback and default-bridge addresses and relies on the host forwarding sandbox traffic to them and allowing the
  dedicated network to reach the worker; it has not been validated on GB300 hosts.
- At the authoring deadline at most one tool call may be in flight; a log with several unfinished calls ends
  INCONCLUSIVE. Agent processes run for up to ten seconds of `docker stop` after the deadline, as after a normal exit,
  and the observer records them.
- Cluster cells share one evaluation container. Worker files are removed between cells, but SysV IPC objects, `/proc`
  and `/sys` are not isolated per cell, and `RLIMIT_NPROC` counts every UID-1101 process on the host. The `cpuset` must
  name CPUs the fleet does not use.
- Data payloads and paths under `review_metadata_paths` reach the judge as metadata, together with the code that reads
  them. A very large pickle can exhaust the judge budget, which ends INCONCLUSIVE.
- An agent file named `manifest.json` at the workspace root is replaced by the seal manifest.
- These are Docker, UID and mount boundaries under a trusted host and kernel, not a defense against kernel escapes or a
  malicious Docker administrator. External logging is not exhaustive memory replay; the collector contract records its
  visibility limits. The agent is never given evidence paths or access, but its knowledge that monitoring exists cannot
  be mechanically prevented.
