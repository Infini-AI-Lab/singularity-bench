# Breakwater task specification

A Breakwater task is a [Harbor](https://github.com/harbor-framework/harbor) task directory whose `task.toml` also
carries `[breakwater]` tables, plus one Python module of trusted task code. Breakwater runs it in three stages, on
Docker or on Apptainer, with Claude Code or Codex as the agent:

1. **Authoring.** The agent works in its workspace for the task's budget. Every command it runs is traced from
   outside its sandbox, its model calls pass through a recording gateway, and when the budget ends every process is
   stopped and the workspace is sealed as the submission.
2. **Evaluation.** A fresh sandbox runs the task's `evaluate(run)` as the trusted coordinator, with the task's private
   inputs. Submitted code runs apart from those inputs. The raw outputs are sealed.
3. **Verification.** A read-only verifier checks the evidence mechanically, recomputes the score with the task's
   `score(scoring)` from the sealed outputs, and has the integrity judge review the whole run against Breakwater's
   checklist and the task's rubric. The reward is the recomputed score only if the verdict is PASS.

[`examples/word-count`](../examples/word-count) is a complete, CPU-only task to start from, and
[WRITING_TASKS.md](WRITING_TASKS.md) walks through writing a task; this document is the reference.

## Layout

```
my-task/
  task.toml                 Harbor sections plus the [breakwater] tables below
  instruction.md            what the agent is told
  environment/Dockerfile    the runtime the agent works in and the evaluation runs in (Apptainer: a SIF built from it,
                            or from an apptainer.def beside it)
  environment/agent-policy.json  the agent's workspace policy, which the Docker author image requires: exactly name,
                            workdir, write_roots and local_network (whether its commands may use the container's own
                            addresses)
  tests/                    private: never in the agent's workspace, never visible to submitted code
    breakwater_task.py      the task's trusted code: evaluate(run) and score(scoring)
    breakwater_host.py      optional ([breakwater.host]): the task's host hooks
    submission_judge.md     the integrity judge's rubric for this task
  runners/                  optional (evaluation.runners): trusted code that executes the submission
  solution/                 optional: Harbor's reference solution
```

With `workspace.source = "task"`, the agent's workspace is a copy of this directory without `tests/`, the runners,
`task.toml` and the declared excludes.

## task.toml

### Harbor sections

| Key | Default | Meaning |
|---|---|---|
| `[agent] timeout_sec` | required | the authoring budget, in seconds |
| `[environment] docker_image` | | the runtime image (Docker backend) |
| `[environment] gpus` | 4 | GPUs for authoring and evaluation (0 to 4); the verifier never gets one |
| `[environment] gpu_types` | `["GB300", "B300"]` | accepted GPU name patterns; the first is the reported type |
| `[environment] workdir` | `/app` | the agent's workspace |

Breakwater ignores Harbor's other fields. In particular `[verifier] timeout_sec` is Harbor's, not the judge budget.

### `[breakwater]`

| Key | Default | Meaning |
|---|---|---|
| `name` | Harbor's `[task] name` (its last part), else the directory name | the task's identity in run records; set it so that a copy of the task under another directory name keeps its identity |
| `failure_reward` | 0.0 | the reward of a run that fails, or whose verdict is not PASS |
| `rule_violation_reward` | 0.0 | the reward of a run that breaks a mechanically checked rule |
| `agent_deadline` | `"fail"` | `"evaluate"`: grade what exists when the budget ends; `"fail"`: a run that reaches the budget fails |
| `agent_exit` | `"any"` with `evaluate`, else `"zero"` | `"zero"`: the agent must exit with status 0 unless the deadline stopped it; `"any"`: any status (only with `evaluate`) |
| `review_profile` | `"exhaustive"` | `"indexed"`: the judge reviews syscall traces, model journals and the workspace as indexes and diffs; `"exhaustive"`: every retained byte |
| `review_metadata_paths` | `[]` | workspace paths of bulk data the judge reviews as metadata |

### `[breakwater.judge]`

| Key | Default | Meaning |
|---|---|---|
| `timeout_sec` | required | the integrity judge's budget, in seconds |

### `[breakwater.workspace]`

| Key | Default | Meaning |
|---|---|---|
| `source` | `"task"` | `"task"`: a copy of the task directory; `"image"`: the image's own workdir |
| `excludes` | `[]` | not copied into a `task` workspace: a plain name anywhere, or a path anchored at the task root with a leading `/` |
| `protected_paths` | `[]` | workspace paths the agent may not change (checked against the pre-attempt inventory) |
| `snapshot_excludes` | `[]` | workspace paths left out of the sealed submission |
| `snapshot_home` | false | also seal the worker's home directory |
| `submission_mount` | none | where evaluation sees the sealed submission, when the task code installs a writable copy at the workdir |
| `dataset_manifest`, `dataset_roots` | none | an agent-written manifest of its training data, checked and archived after authoring |
| `shadow_policy` | none | the import-shadow check: packages or hooks the workspace may define |

### `[breakwater.resources]`, `[breakwater.agent_resources]`

Container limits for every stage, and overrides for authoring only: `cpus`, `memory_mb`, `nofile`, `shm_size`,
`memlock`, `stack`, `pids`, `cpuset`.

### `[breakwater.evaluation]`

| Key | Default | Meaning |
|---|---|---|
| `timeout_sec` | required | the evaluation stage's budget, in seconds |
| `runners` | none | a task directory of trusted code that executes the submission, mounted read-only at `/opt/breakwater-drivers` in the evaluation stage (for the coordinator and for submitted code) and kept out of the agent's workspace |
| `worker_env` | a minimal environment | the environment submitted code runs with (no credentials); `BREAKWATER_WORKDIR` defaults to the workdir |
| `ports` | none | ports the submitted code serves (a server task), which the coordinator reaches on its own `127.0.0.1` (Apptainer relays each from the submission sandbox; under Docker they share a network) |
| `shared` | none | directories the coordinator prepares for the submitted code, writable by both (under Docker they share the container; Apptainer binds each into both sandboxes) |

### `[breakwater.scoring]`

| Key | Default | Meaning |
|---|---|---|
| `recomputed` | `[]` | `raw/...` or `inputs/...` paths (glob patterns allowed) that `score` recomputes row by row; the judge sees them summarized |

### `[[breakwater.assets]]`

Inputs the operator prepares and Breakwater binds read-only into the stages that may see them.

| Key | Meaning |
|---|---|
| `name` | a unique identifier (`run.asset(name)`, `scoring.asset(name)`) |
| `source` | the default location, relative to the operator's `--asset-root` |
| `target` | the absolute path in the sandbox; its first component is `private`, `models`, `assets`, `app` or `workspace` |
| `stages` | which of `agent`, `evaluation` and `verifier` see it; a `/private/...` target is never shown to the agent |

An operator whose files are laid out differently passes `--bindings FILE`: a list of `{source, target, stages}`
entries with the task's own targets and stages.

### Evaluation network

`[breakwater.evaluation] env` lists operator variables forwarded to the coordinator (the launcher refuses to start if
one is unset). With `network = "bridge"` (and `cap_add`, `add_hosts` for its firewall and host name), the Docker
coordinator joins a bridge network and the task's code limits its egress. On Apptainer the coordinator has no network
of its own: a forwarded variable holding a loopback URL (`http://127.0.0.1:PORT`) names a service on the launcher's
host, which the coordinator reaches at the same URL through a relay, and any other URL is refused (a code-execution
service the task's evaluation calls, say).

### `[breakwater.model_access]`

A budgeted development gateway the agent's software may call: an OpenAI-compatible endpoint, outside every sandbox,
that holds its own provider key, serves only the models the task declares and journals every request (the verifier
reconciles that journal). The agent's commands reach it at `http://HOST:PORT/v1`; tell them so with
`[breakwater.authoring.env]`.

| Key | Default | Meaning |
|---|---|---|
| `upstream` | required | the provider's OpenAI-compatible base URL |
| `chat_models` | required | a table of alias = model: the aliases the agent's software may request |
| `embed_model` | none | an embedding model, served as `embed`; without one there is no embedding route |
| `max_chat_tokens`, `max_embed_tokens` | none | token allowances for the whole run, reserved per request |
| `host`, `port` | `gateway`, 8000 | where the agent's commands reach it |
| `key_env` | `FIREWORKS_API_KEY` | the launcher's variable holding its provider key (`--dev-key-env` overrides); never in a sandbox |

Docker runs it as a sidecar on the agent container's internal network (`--dev-gateway-image`); Apptainer runs it on the
host, relayed into every command sandbox.

### `[breakwater.authoring]`

| Key | Default | Meaning |
|---|---|---|
| `env` | none | a table of variables for the agent's commands (never a credential, a loader or interpreter variable, or one the framework sets, such as `HOME` or `PATH`) |
| `inbound_ports` | none | a range `"LOW-HIGH"` (at most 64 ports, from 1024): servers the agent's commands run on these ports are reachable from the launcher's host |
| `inbound_host` | `host.docker.internal` | with `inbound_ports`: the name the agent's commands use for the launcher's host, where those servers are reached |
| `extra_networks` | none | Docker only: networks the agent container joins (only `bridge`), for endpoints on other hosts |

Docker publishes the inbound ports on the host's loopback and on the default bridge's gateway and lets only replies
out of them; Apptainer relays them from the host's loopback into each command sandbox, replacements included, where the
inbound host is the sandbox's own loopback.

### `[[breakwater.endpoints]]`

A host and port that sandboxes may reach, beyond the framework's own services: a service on the launcher's host (the
default) or on another machine.

| Key | Default | Meaning |
|---|---|---|
| `name` | required | the host name the sandboxes use (not `localhost`, `author-gateway`, the model-access host or the inbound host) |
| `port` | required | the port they use |
| `service` | `127.0.0.1:PORT` | where it really is; the operator's `--endpoint NAME=HOST:PORT` overrides it |
| `reach` | `["agent"]` | which of `agent` (the agent's commands), `submission` (submitted code under evaluation) and `coordinator` reach it |
| `health` | none | a path that must answer HTTP 200 before authoring starts |

Apptainer relays each declared port into the sandboxes of the roles that reach it, under the endpoint's name, with a
port map where the real service listens on another port. Docker hands the agent's endpoints to the author image's
firewall in the authoring plan (below): a service on the launcher's host is reached through a dedicated network whose
gateway is the host, one on another machine needs `extra_networks`, and the service port must equal `port`. On Docker,
endpoints the evaluation reaches need `[breakwater.evaluation] network = "bridge"`, and the task's coordinator limits
the submitted code's egress.

### `[breakwater.options]`

A free-form table of the task's own settings, which Breakwater seals with the run and never interprets. Task code reads
it as `contract['options']` (`run.contract`, `scoring.contract`, `host.contract`).

### `[breakwater.operator]`

What the operator supplies at launch, beyond the assets.

| Key | Meaning |
|---|---|
| `inputs` | names of files the operator passes as `--input NAME=PATH` (each one is required; never inside the task or the run directory) |
| `values` | a table of name = `"VARIABLE"`: non-secret values from the launcher's environment (`--value-env NAME=VAR` renames one) |
| `secrets` | a table of name = `"VARIABLE"`: credentials from the launcher's environment (`--secret-env NAME=VAR` renames one) |

Inputs, values and secrets reach only the task's host hooks, and whatever those hooks pass on in their plan. Secrets
leave the launcher's environment at start; their variable names join the credentials no sandbox environment may hold,
and no secret value may appear in a sandbox's environment, the authoring plan or a workspace file.

### `[breakwater.host]`: host hooks

| Key | Default | Meaning |
|---|---|---|
| `module` | none | a Python file under `tests/`, usually `tests/breakwater_host.py` |
| `timeout_sec` | 1800 | the time limit of each phase |

The module is trusted task code that runs on the launcher's host, outside the execution stages:

```python
from host_hooks import Endpoint, FromInput, FromSecret, HostPlan, Text

def preflight(host):                  # required: before the run directory exists; no infrastructure side effects
    return HostPlan(
        endpoints=[Endpoint('store', 'store.internal', [9000], reach=['agent', 'coordinator'])],
        agent_env={'STORE_URL': 'http://store:9000'},
        workspace_files={'config.json': FromInput('config')},          # or Text('...')
        private_files={'token': FromSecret('token')},                   # the coordinator's /private/operator/token
        metadata={'store_version': '3'})                                # sealed: scoring.metadata

def before_authoring(host): ...       # optional: after seeding, before the first sandbox
def after_evaluation(host, windows):  # optional: after both execution stages are proven empty, before sealing
    host.evidence.json('store-log.json', {...})                         # external/task/store-log.json
def on_failure(host): ...             # optional: best effort when a run fails; its errors are ignored
```

`host` has `task` (the task directory), `contract`, `backend` (`docker` or `apptainer`), `smoke`, `input(name)`,
`value(name)`, `secret(name)`, `state` (a private directory that persists across the phases and is never sealed) and,
in `before_authoring` and `after_evaluation`, `evidence`: `evidence.json(name, value)` and `evidence.path(name)` write
under the run's `external/task/`, which is sealed and reviewed. `windows` gives each execution stage's wall-clock span,
`{'agent': [start_ns, end_ns], 'intermediate': [start_ns, end_ns]}`.

Each phase runs in its own process (`python3 -I -B`), in the hook state directory, with only `PATH`, `LANG` and the
task's operator values and secrets in its environment; its result comes back as a file, never on stdout. The launcher
validates the plan (endpoint names and roles as for `[[breakwater.endpoints]]`, where an address may name a host: a
loopback address on Apptainer, another address on Docker; the environment as for `[breakwater.authoring] env`; file
paths and sources, secrets only into private files) and applies it: endpoints join the task's declared ones, merged per
role; workspace files are written before the pre-attempt inventory; private files are written read-only and bound only
into the coordinator, at `/private/operator`. It records the plan, with digests for file contents, as
`external/task-plan.json`, and seals the phases' times, which the verifier checks lie outside the execution stages. A
phase that fails or runs out of time stops the run (after evaluation: without a reward). Host hooks are trusted like
the task's evaluation: a contained, recorded place for host work, not a security boundary.

## Task code: `tests/breakwater_task.py`

```python
def evaluate(run):        # evaluation stage, the trusted coordinator
    ...

def score(scoring):       # verifier, sealed outputs only
    ...

def mechanical(scoring):  # optional: task facts for the mechanical report
    ...

def review_views(scoring):  # optional: how the judge sees the task's own evidence and outputs
    ...
```

The module runs in the task image's Python, with the task's `tests/` directory and Breakwater importable: `contract`
(`require`, `read_json`, `atomic_json`), `task_api` (`Void`, `jsonl`) and, for an evaluation that starts servers or
needs a writable copy of the submission, `workers` (`kill_workers(grace)` stops every submitted process, detached
ones included; `install_tree(pairs, allowed)` installs fresh worker-owned copies of sealed trees and raises
`InvalidSubmission` for a link that escapes them). It is loaded by path, so module names in `tests/` do not clash with
it.

### `evaluate(run)`

It prepares inputs, has the submission do its work and records everything the score needs. It never imports or
executes submitted code itself.

| Attribute | Meaning |
|---|---|
| `run.raw` | `/evaluation/raw`: the outputs the scorer reads, sealed when the stage ends |
| `run.inputs` | `/evaluation/inputs`: trusted inputs the coordinator prepares (create it first), sealed too |
| `run.submission` | the sealed submission (the workdir, or `workspace.submission_mount`) |
| `run.tests`, `run.contract` | the task's tests directory and its resolved contract |
| `run.asset(name)` | where a declared asset is visible to this stage |
| `run.smoke` | a smoke run (below): the task may use reduced evaluation settings |
| `run.publish(name, data)` | an answer-free file submitted code may read, at `/eval-inputs/NAME` |
| `run.run_submission(argv, **options)` | `subprocess.run` of submitted code, apart from the private inputs, with `worker_env` |
| `run.start_submission(argv, **options)` | the same as `subprocess.Popen` |
| `run.input_roots()` | the non-private inputs a submitted symbolic link may point into |

Submitted code is anything in the submission. Run it through a runner in `runners/` that imports or executes it, and
pass only answer-free data with `run.publish`.

### `score(scoring)`

It recomputes the reward from the sealed outputs. No submitted code runs in the verifier.

| Attribute | Meaning |
|---|---|
| `scoring.raw`, `scoring.inputs` | the sealed evaluation outputs and inputs |
| `scoring.submission` | the sealed submission, read-only |
| `scoring.evidence` | the external evidence (for `mechanical`) |
| `scoring.records` | a directory for the scorer's own records |
| `scoring.smoke` | a smoke run, as sealed in the authority: score with the same reduced settings |
| `scoring.metadata` | the task metadata sealed in the authority (its host hooks' plan, or a custom launcher's record), read-only |
| `scoring.tests`, `scoring.contract`, `scoring.asset(name)` | as in `run` |

It returns `{'reward': float, 'details': {...}, 'validated': [...]}`:

- **`details`** are shown to the judge with the reward.
- **`validated`** lists the checks the scorer applied to every record. The judge is told that a reading which
  contradicts one is a misreading.

A task whose reward is computed after the review (for example, on the host with resources the verifier lacks)
returns `{'reward': None, 'deferred': {'scored_by': ..., 'rule': ...}}`. The verifier then records the verdict for
that scorer instead of a reward.

An inconsistency raises (for example with `contract.require`), and the run ends without a reward. A scorer that could
not measure at all (an unavailable oracle, say) raises `task_api.Void`: the attempt then has no score, never a zero
(`reports/void.json`; the reward stays the failure reward).

### `mechanical(scoring)`

Task facts the judge must not be asked to establish (the task's own evidence, checked), added to the mechanical report.
It runs before `score`, so its facts are recorded even when scoring fails.

### `review_views(scoring)`

Under the `indexed` review profile, a task may present its own evidence (`external/task/...`) and its evaluation
outputs (`evaluation/...`) to the judge in a readable form. It returns `[(pattern, view)]`, glob patterns over those
references only; `view(path, scoring)` returns text, or None to leave the file to the generic presentation. The judge
sees the text under a header with the raw file's SHA-256 and size; a view that fails falls back to the file as it is,
and the failure is listed in `coverage.json`.

## `tests/submission_judge.md`

The task's rubric: what counts as an honest solution, what a violation looks like, and what the judge must check in
the submission's source. Breakwater's own checklist (isolation, capture, process wall, tampering) applies to every
task.

## Running a task

The `breakwater` command (`pip install -e .` at the repository root, or `python3 -m breakwater` from it):

```bash
breakwater check examples/word-count            # validate the task; say which backends can run it

# Apptainer, the default (no root or daemon): the runtime SIF, the pinned agent CLI and judge runner, then the run
apptainer build word-count.sif docker://python:3.12-slim
breakwater install-cli --dest AUTHOR_CLI_DIR [--agent codex]   # Claude Code by default
breakwater install-judge --dest JUDGE_DIR
breakwater run --task examples/word-count --image word-count.sif \
  --cli-dir AUTHOR_CLI_DIR --judge-dir JUDGE_DIR --run-dir RUN_DIR --asset-root examples/assets \
  [--author-cli codex] --author-model MODEL --author-upstream https://PROVIDER --author-key-env KEY_VAR

# Docker, on a host with a Docker daemon (README.md has the images to build)
breakwater run --backend docker --task examples/word-count --agent-image ... --run-dir RUN_DIR \
  --asset-root examples/assets ...
```

Reports land in `RUN_DIR/reports/`: the mechanical checks, the integrity verdict and `reward.json`.

`--smoke` (either launcher) makes a smoke run, for testing a setup cheaply: every stage runs, but the task's evaluation
uses its reduced smoke settings where it has them (`run.smoke`, `scoring.smoke`; a training task might train for a
few steps instead of its full budget). The run is sealed as a smoke run, the summary line and
`reward.json` say `"smoke": true`, and the judge is told; a smoke reward is never a benchmark score. The setup for
each backend (the agent CLI, the judge runner, the images) is in [README.md](README.md) and
[apptainer/README.md](apptainer/README.md).

### What a run seals

Besides the stages' records and the evidence digests, a launcher seals into the authority the contract's digest and
its pipeline settings (`spec.SEALED_PIPELINE`), the framework revision (`contract.FRAMEWORK_REVISION`) and a digest of
the task's `tests/` (`contract.tests_digest`), what it recorded (`recording`: whether the agent reached its model
through the author gateway, the service gateways' journals, and whether the agent's native output is a JSONL log or
byte streams), the task metadata and the host hooks' times. The verifier refuses a run sealed by another framework
revision or with other task tests: re-verify a run with the Breakwater it was launched with, and rebuild a Docker
verifier image after editing the task's tests. The Docker launcher checks the verifier image's revision and tests
before authoring, and the author image's bootstrap returns the digest of the authoring plan it applied (the agent's
services, endpoints, refused ports, host names, inbound reply ports and environment), which the launcher checks too.

## Custom launchers

A launcher of its own (one that runs a submitted agent as the agent, say) can reuse the backends' stages and the
verifier: Apptainer's `agent_stage.AuthoringStage` (with `gateway=None`, `native='stdout'` and its own `services`, such
as a driver-role gateway), `evaluation_stage.EvaluationStage` and `run.verify`; Docker's launcher helpers in `run.py`.
It writes the run's contract (with `submission_root` when the submission is not at the workdir, and `agent_deadline`,
`agent_exit`) and passes `run.verify` a verification task directory: an `instruction.md`, and `tests/` with the
`breakwater_task.py` the verifier runs and the rubric `tests/submission_judge.md`. It seals `recording` and
`task_metadata` through `run.verify`'s `extra`.
