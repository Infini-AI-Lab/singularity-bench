# Breakwater

Breakwater is an evaluation and safety framework for open-ended agent tasks. In such a task an agent spends hours
developing on real GPUs, and its submission is a working system (a model, a serving engine, a harness, a control
plane) whose evaluation may have to run the code the agent wrote. Breakwater runs such tasks, written in
[Harbor](https://github.com/harbor-framework/harbor)'s task format, with five safeguards:

1. **Environment fingerprinting**: the runtime the agent works in cannot change unnoticed. Apptainer runs every stage
   from one read-only SIF whose digest is checked before and after each stage; Docker inventories executables,
   libraries, package trees and import paths before the agent phase and checks them after it and in the fresh
   evaluation container ([environment_guard.py](environment_guard.py)). Any difference fails the submission.
2. **API routing** ([author_gateway.py](author_gateway.py)): the agent's single model connection goes through a gateway
   that forwards only the assigned model and protocol. Every other model or API access is blocked.
3. **Process wall**: when the agent phase ends, every live process is stopped and the drain is proven before anything
   else starts, which rules out sidecar processes (Apptainer: the kernel tears down each sandbox's PID namespace;
   Docker: [process_guard.py](process_guard.py) and the observer prove the container empty).
4. **External journal**: model traffic, tool calls, command output, process and file events are recorded outside the
   agent's reach (Apptainer: brokers trace every command from outside its sandbox; Docker: the
   [collector/](collector/)), together with the gateways' journals. The agent can neither read nor write this record.
5. **Restricted intermediate stage** ([intermediate.py](intermediate.py), [workers.py](workers.py)): submitted code
   runs unprivileged, apart from the trusted coordinator and its private inputs, in a fresh sandbox after authoring.
   The verifier ([verifier.py](verifier.py)) runs in a separate read-only sandbox, recomputes the score from sealed raw
   outputs and runs an LLM integrity judge, without executing submitted code.

One copy of the framework serves every task. A task declares its contract, including the prepared assets each stage
may see, in the `[breakwater]` tables of its `task.toml`, and its evaluation and scoring in `tests/breakwater_task.py`
([SPEC.md](SPEC.md)).

## Documentation

| Document | Covers |
|---|---|
| this README | what Breakwater checks, how a run proceeds on Docker, and running a task |
| [apptainer/README.md](apptainer/README.md) | the Apptainer backend (the default): requirements, options, Slurm, and how it isolates a run |
| [WRITING_TASKS.md](WRITING_TASKS.md) | writing a task of your own, from the example task to testing it |
| [SPEC.md](SPEC.md) | the task specification: every `task.toml` setting and the interface of task code |
| [PROFILES.md](PROFILES.md) | the agent CLI profiles (Claude Code and Codex) |
| [checklist.md](checklist.md) | the integrity judge's checklist, which applies to every task |

## Backends

- **Apptainer** ([apptainer/](apptainer/README.md)), the default: for the many GPU clusters with Slurm and no Docker.
  No root, daemon or setuid helper: every stage runs in sandboxes of the task's SIF (unprivileged
  Linux namespaces), each memory-limited sandbox as its own Slurm step, and every network path is a relay from the
  host. [apptainer/README.md](apptainer/README.md) describes how a run is isolated and what is weaker and stronger
  than on Docker.
- **Docker** ([run.py](run.py)), on a host with a Docker daemon and the NVIDIA Container Toolkit. The agent and its
  commands run as separate users in containers with root-managed namespaces, cgroups and per-container GPU devices.
  The rest of this README describes it.

Both run the same tasks, seal the same evidence and use the same verifier and judge.

## How a run proceeds on Docker

The Apptainer backend's counterpart is [apptainer/README.md](apptainer/README.md#how-a-run-is-isolated).

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
rule-violation reward (`rule_violation_reward`: 0 by default; -1 suits a task whose score can be negative) plus
`rule-violation.json` with reason `violation of rules`.

### Authoring deadline and submitted links

A task with `agent_deadline = "evaluate"` grades what exists when the authoring budget ends. A CLI still running at
the deadline is stopped exactly like a normal exit (process wall, drain, environment check, snapshot), and a CLI that
exits with an error also proceeds to evaluation unless the task declares `agent_exit = "zero"`. The agent stage records
`exit_code` and `deadline_reached`; the verifier accepts that record only under the sealed policy. Only after a recorded deadline may the native log end with
an unfinished final line and at most one tool call without a result, the one in flight; the byte-for-byte comparison
with the external copy still applies.

Symbolic links in the workspace (and, with `workspace.snapshot_home`, in the worker's home directory) are not sealed
as files. Each one is recorded with its link text in `external/workspace-links.json` and in the seal manifest, counted
by the import-shadow check like a file at its path, and shown to the judge. Task code that recreates links at
evaluation must never follow them and should accept only links that resolve inside the installed submission (or a
read-only input it names). Hard links and special files always fail the snapshot.

### Integrity judge

[checklist.md](checklist.md) and [judge.schema.json](judge.schema.json) apply to every task, together with the task's
`tests/submission_judge.md`. Command execution, subagents, images and web search are disabled in the judge. Under the
`indexed` review profile ([review_index.py](review_index.py)), the judge sees a security-event
index of the raw syscall traces (every exec, write-capable open, mutation, signal, prctl, mount, io_uring and
shared-mapping syscall, failed write and probe outside system roots, each with the raw file's SHA-256; repeats are
counted, exec, signal, prctl, mount, chroot and ptrace lines only when identical apart from temp-file names in exec
arguments, other events when they differ only in descriptors, sizes or addresses; a directory tree with more than 50
distinct paths of one call is summarized with examples, and each execve environment is printed once), model journals
as per-lane conversation deltas plus complete responses (each streamed response reassembled into the message it
delivered), the workspace as a diff against the pre-agent inventory, byte-identical copies once, the agent CLI's
stream with each run of content-free progress events (Claude Code's per-token thinking ticks) as one line, each
Apptainer command's output headed by the command, working directory and exit status its broker journaled, the CLI's
own session copy of the conversation by digest and entry counts, model-journal blobs in the evidence manifest by count,
and the task's own evidence (`external/task/`) and evaluation outputs through the task's review views where it
declares them (each under a header with the raw file's digest); the `exhaustive` profile reviews every retained
non-tensor byte in bounded packets instead. Consecutive
packets under 8,000 characters share one packet, each headed by its reference; `coverage.json` lists every packet
and what each shared one holds. Safetensors are inspected structurally and represented by their complete
headers plus file hashes. Other data payloads (files with a NUL byte in their first 8 KB, common tensor, array and
checkpoint suffixes, and paths listed in `review_metadata_paths`) are represented by SHA-256, size and a short head;
pickles, including those inside checkpoint archives, stay reviewable in full. Raw evaluation outputs that the verifier
recomputes mechanically are summarized (hash, size, rows, bounded head). At most eight judge calls run at once
(`--judge-concurrency`, up to 64, within the judge provider's rate limit: a long run's review must fit the task's
judge budget), each with its own retries (at most five attempts) and Codex home. A reply is valid only if
every checklist row gives specific evidence, the verdict is the worst row status, cheating is claimed only with a FAIL
row, each FAIL or INCONCLUSIVE row of a packet review quotes the packet verbatim, a synthesis is no milder than the
reviews it combines and grounds each FAIL or INCONCLUSIVE row in quotes from them (a contradiction only between
passages that two reviews quoted and that were found in what those reviews read), and every digest, list or object a
FAIL or INCONCLUSIVE row cites appears in its input (lists and objects compared without whitespace). A quotation may be
elided with an ellipsis. A valid reply is never re-run to change it, but a new finding needs agreement: when a packet
review is not PASS, or a synthesis is harsher than every review it combines, two more independent reviews of the same
input run at once, and the median of the three verdicts stands (`judge-confirmations.json`). A reply that the judge's
provider cuts off at its output cap is retried, and after two such cut-offs a call's remaining attempts use low
reasoning effort. The final reward is the recomputed score only on PASS.

## Task contracts

Each task declares its contract in the `[breakwater]` tables of its `task.toml` and its trusted code in
`tests/breakwater_task.py`; [SPEC.md](SPEC.md) is the full specification. The launchers resolve the contract with
[spec.py](spec.py), hand it to the evaluation coordinator and the verifier as `contract.json`, and seal its digest
into the authority. A task that needs more than the workspace and the agent's model declares it there too: model access
for the agent's software, endpoints on the launcher's host or other machines, inbound ports, operator inputs and
secrets, and host hooks for setup and evidence collection outside the execution stages.
[WRITING_TASKS.md](WRITING_TASKS.md) shows when a task needs each.

Every setting that changes what is graded or accepted (`spec.SEALED_PIPELINE`) is sealed into the host authority,
and the verifier refuses a contract that differs from it.

## Running a task

`breakwater check tasks/<name>` validates a task and lists the backends that can run it; `breakwater run ...` hands
its arguments to the Apptainer launcher ([apptainer/run.py](apptainer/run.py)), and `breakwater run --backend docker
...` to the Docker launcher ([run.py](run.py)).

### On Apptainer

Install the pinned agent CLI and the judge runner once, build the task's SIF as its README describes, and run:

```bash
python3 -m breakwater install-cli --dest CLI_DIR [--agent codex]
python3 -m breakwater install-judge --dest JUDGE_DIR
python3 -m breakwater run --task tasks/<name> --image <name>.sif --cli-dir CLI_DIR --judge-dir JUDGE_DIR \
  --run-dir RUN_DIR --asset-root ASSET_ROOT --author-model MODEL --author-upstream https://PROVIDER \
  --author-key-env KEY_VAR
```

[apptainer/README.md](apptainer/README.md) has the requirements, the options and how to run inside Slurm.

### On Docker

**Requirements.** A trusted Linux host with the GPUs the task declares, Docker with BuildKit, the NVIDIA Container
Toolkit with a generated CDI specification, and Python 3.11 or newer on the host. Credentials: the author model's
provider key (any variable, named with `--author-key-env`) and `FIREWORKS_API_KEY` for the integrity judge; a task with
model access also needs its development models' key (its `model_access` `key_env`, or `--dev-key-env`), and a task
with operator secrets needs those (its `[breakwater.operator]` variables, or `--secret-env NAME=VAR`). Keep run
directories and the prepared asset root outside the repository. Run every command from the repository root.

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
   docker build -f breakwater/Dockerfile.dev-gateway -t breakwater-dev-gateway:local breakwater  # model access
   ```
4. **Prepare the assets** under one asset root, at the sources the task declares (`[[breakwater.assets]]` in its
   `task.toml`). Assets laid out differently can be moved with `--bindings FILE` (a list of `{source, target,
   stages}` with the task's own targets and stages). No private path can be bound into the agent container.
5. **Launch:**
   ```bash
   python3 -m breakwater run --backend docker --task tasks/<name> \
     --agent-image <name>-author:claude-code --verifier-image <name>-verifier:local \
     --observer-image breakwater-observer:local --gateway-image breakwater-author-gateway:local \
     --author-cli claude-code --author-model MODEL \
     --author-upstream https://api.anthropic.com --author-key-env ANTHROPIC_API_KEY --author-auth anthropic \
     --asset-root /data/prepared --run-dir /data/runs/RUN_ID
   ```
   `--author-auth anthropic` sends the key as Anthropic's `x-api-key` header (the default is a bearer token, which
   Codex requires). `--author-upstream` is the provider's HTTPS endpoint for the selected protocol (for example
   `https://api.fireworks.ai/inference` for an Anthropic-compatible Fireworks model with Claude Code, or
   `https://api.openai.com` with Codex). `--gpus device=0,1,2,3` selects the task's declared CDI devices (by default
   the first `gpus` devices; none for a CPU task). `--intermediate-image` defaults to the author image. A task with
   model access adds `--dev-gateway-image breakwater-dev-gateway:local`; one with operator inputs adds
   `--input NAME=PATH` for each; `--endpoint NAME=HOST:PORT` says where a declared endpoint really is.

   The launcher refuses a verifier image built from another Breakwater revision or from other task tests (a rubric or
   scorer changed since), and an author image whose bootstrap does not apply the launcher's authoring plan: rebuild
   them after upgrading Breakwater or editing the task's `tests/`.

The controller resolves image tags to immutable IDs, checks the actual Docker mounts, namespaces and credentials
before execution, runs live negative access probes before either untrusted stage, and requires a write to a sealed
canary inside the verifier to fail. Only the gateways receive provider keys; only the verifier receives
`FIREWORKS_API_KEY`, for the judge. Reports land in `RUN_DIR/reports/`: `mechanical-check.json`, the checklist
verdicts (`judge.json`), packet coverage, `llmjudgecomments.md` and the final `reward.json`. External evidence is
kept separately under `RUN_DIR/external/`. A raw intermediate reward is never the final reward.

## Layout

| Path | Responsibility |
|---|---|
| [run.py](run.py) | the launcher: contract validation, image and mount checks, the three stages, sealing |
| [contract.py](contract.py), [bundle_storage.py](bundle_storage.py) | seals, inventories, boundary inspection, strict JSON |
| [environment_guard.py](environment_guard.py), [process_guard.py](process_guard.py) | environment fingerprinting and import-shadow policy; process drain |
| [collector/](collector/), [observed.py](observed.py), [capture/](capture/) | the external observer, the observed-stage entry point, the stdio supervisor |
| [author_gateway.py](author_gateway.py), [dev_gateway.py](dev_gateway.py) | the author gateway; the model-access gateway (development and driver roles) |
| [plan.py](plan.py), [host_hooks.py](host_hooks.py) | endpoints per role; host hooks and operator inputs, values and secrets |
| [author/](author/), [PROFILES.md](PROFILES.md) | the protected author image and its agent CLI profiles |
| [cli.py](cli.py) | the `breakwater` command (`check`, `run`, `reverify`, the Apptainer installers) |
| [spec.py](spec.py), [SPEC.md](SPEC.md), [task_api.py](task_api.py), [WRITING_TASKS.md](WRITING_TASKS.md) | the task specification; the interface of task code; the guide to writing tasks |
| [intermediate.py](intermediate.py), [workers.py](workers.py) | the evaluation-stage coordinator; worker installation and teardown |
| [tensor_inventory.py](tensor_inventory.py) | safetensors inventory |
| [verifier.py](verifier.py), [reconcile.py](reconcile.py), [review_index.py](review_index.py), [checklist.md](checklist.md), [judge.schema.json](judge.schema.json) | the verifier, log reconciliation and the integrity judge |
| [apptainer/](apptainer/) | the Apptainer backend (sandboxes, brokers, relays, stage records and their checks) |
| `tasks/<name>/tests/breakwater_task.py` and its modules | each task's evaluation, scoring and review views; its host hooks in `tests/breakwater_host.py`; task-specific tools live in the task too |
| [tests/](tests/) | the framework's test suite |

## Tests

```bash
cd breakwater && python3 -m unittest discover -s tests -p 'test_*.py' -v
```

The CPU suite covers tampering, missing, extra or reordered events, broken hash chains, wrong identities, mount aliases
and ancestor mounts, credential and socket exposure, reused containers, overlapping stages, package and import-hook
tampering, recorded links, contradictory checklist verdicts, malformed judge replies and bounded retries, unknown
native event shapes, both gateway roles against a fake upstream, the author gateway against a fake provider, the
review index with its metadata and concurrency limits, the authoring-deadline record, Docker resource flags and the
rule-violation reward. The live Docker test (`test_docker_boundary.py`) skips without Docker and tests the read-only
verifier boundary, the recorder with a planted detached process and image workspace seeding. A skip is not evidence of
runtime isolation.

A task's own tests live with it, in its `tests/` directory (the example task's are in `tests/unit/`).

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

- At the authoring deadline at most one tool call may be in flight; a log with several unfinished calls ends
  INCONCLUSIVE. Agent processes run for up to ten seconds of `docker stop` after the deadline, as after a normal exit,
  and the observer records them.
- Data payloads and paths under `review_metadata_paths` reach the judge as metadata, together with the code that reads
  them. A very large pickle can exhaust the judge budget, which ends INCONCLUSIVE.
- An agent file named `manifest.json` at the workspace root is replaced by the seal manifest.
- Cheats confined to the submitted code's own process, such as content-keyed memoization or faked synchronization
  inside a kernel, leave no trace outside it: only the judge's reading of the sealed submission can catch them.
- These are Docker, UID and mount boundaries under a trusted host and kernel, not a defense against kernel escapes or a
  malicious Docker administrator. External logging is not exhaustive memory replay; the collector contract records its
  visibility limits. The agent is never given evidence paths or access, but its knowledge that monitoring exists cannot
  be mechanically prevented.
