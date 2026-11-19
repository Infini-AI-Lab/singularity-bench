# Breakwater on Apptainer

GPU clusters usually run Slurm with Apptainer and no Docker. This backend runs any task written to the
specification ([../SPEC.md](../SPEC.md)) with Breakwater's five safeguards, without
Docker, root, setuid helpers or subordinate UID ranges. Every process runs as the calling user, and Linux namespaces,
not separate UIDs, decide what each part of a run can see. The run directory, the evidence, the seals, the verifier
and the integrity judge are the Docker backend's; [checks.py](checks.py) replaces the Docker inspections with the
sandboxes' own records.

## Requirements

- Linux with unprivileged user namespaces (one level is enough; sandboxes never nest them). On Ubuntu 24.04 and newer,
  AppArmor may restrict them to binaries with a profile (`kernel.apparmor_restrict_unprivileged_userns`).
- `kernel.yama.ptrace_scope` 0 or 1: the launcher traces its own descendants.
- Apptainer 1.3 or newer with `--userns`, `--containall` and `--net --network none`; `nsenter` and `strace` on the
  host. GPUs come through `apptainer --nv`.
- Apptainer need not be installed by the site: a user-level, relocated, non-setuid install of the release `.deb` works
  (unpack it with `dpkg-deb -x`, add the symlinks `<root>/usr/etc -> ../etc` and `<root>/usr/var -> ../var`, and put
  its library directory on `LD_LIBRARY_PATH`, which its bundled build helpers need).
- Local disk for each sandbox's private `/tmp` (`--scratch DIR` or `$BREAKWATER_SCRATCH`; default the system temporary
  directory), removed when the sandbox stops. Where the system temporary directory is a tmpfs, as on some GPU nodes,
  it would hold sandbox files in memory; the launcher warns, and `--scratch` should name local disk.
- A run directory with room for the workspace, its checkpoints and the evidence.
- Optional, under Slurm: per-stage memory limits, with the launcher started on a node of the job (by `srun` or
  `sbatch`, not from an `salloc` shell on a login node). A sandbox with a memory limit is its own Slurm step on that
  node (`srun --overlap --nodes=1 --nodelist=$SLURMD_NODENAME --mem=... --cpus-per-task=... --oom-kill-step=0`),
  whose cgroup holds the container and every process in it. With `--oom-kill-step=0` (where srun has it), running out
  of memory kills only the process the kernel picks, as in a Docker container, even on sites whose Slurm sets
  OOMKillStep. If the sandbox is lost anyway (an older Slurm kills the whole step): during authoring, the launcher
  records it, starts a fresh command sandbox on the same workspace and keeps the stage going (the agent CLI is in its
  own sandbox and survives; a command sent meanwhile waits for the replacement); during evaluation, it fails that
  evaluation.

`python3 -I breakwater/apptainer/probe.py [--image TASK.sif]` checks a node for all of this without changing anything;
run it on a login node and inside an allocation on a GPU node.

**Images.** Each task's runtime has an Apptainer definition translated step for step from its Dockerfile
(`tasks/<name>/environment/apptainer.def`, built with `environment/build-apptainer.sh`; a task with more images lists
them in its README). Builds use `apptainer build --fakeroot`. Put `APPTAINER_TMPDIR` on local disk: the build unpacks
the base image there, and some shared filesystems reject its extended attributes. Without subordinate UID ranges or a
host `fakeroot` command, a build's user namespace maps only the calling user, as root, and dpkg cannot give package
files other owners (`error setting ownership ... Invalid argument`); the definitions that install such packages detect
this, run their package installs under a `fakeroot` they install first and remove it afterwards, so every file is
then root's. `apptainer build --fakeroot --ignore-fakeroot-command` reproduces that mode on a host that has `fakeroot`.
A build's `%post` would write temporary files to the build host's `/tmp`, which may be small: the definitions keep
theirs in the image tree (under `APPTAINER_TMPDIR`) and remove them at the end.

## Running a task

Install the pinned agent CLI and the pinned judge runner once, build the task's SIF, prepare its assets as its README
describes, then run from the repository root:

```bash
python3 -m breakwater install-cli --dest CLI_DIR [--agent codex]   # the pinned Claude Code (default) or Codex
python3 -m breakwater install-judge --dest JUDGE_DIR               # the pinned Codex judge runner (verifier only)
export FIREWORKS_API_KEY=...   # the integrity judge; only the verifier sandbox receives it
export AUTHOR_API_KEY=...      # the agent's model provider; only the author gateway receives it
python3 -m breakwater run --task TASK_DIR --image TASK.sif \
  --cli-dir CLI_DIR --judge-dir JUDGE_DIR --run-dir RUN_DIR --asset-root ASSET_ROOT \
  [--author-cli codex] --author-model MODEL --author-upstream https://PROVIDER --author-key-env AUTHOR_API_KEY \
  [--scratch LOCAL_DISK_DIR] [--judge-concurrency N]
```

`--judge-concurrency` (default 8, up to 64) is how many judge calls run at once: raise it, within the judge provider's
rate limit, so that a long run's review fits the task's judge budget.

With Claude Code, `--author-upstream` is an Anthropic Messages-API endpoint (`--author-auth anthropic` for an
`x-api-key` provider); with Codex, an OpenAI Responses-API endpoint with bearer authentication (for example
`https://api.fireworks.ai/inference` for a Fireworks model, or `https://api.openai.com`). Assets are found at the
sources the task declares (`[[breakwater.assets]]`) under `--asset-root`; `--bindings FILE` moves them. A task that
declares more takes more options: `--dev-key-env VAR` (model access, when its key is not in the task's `key_env`),
`--endpoint NAME=HOST:PORT` (where a declared endpoint really is), and `--input NAME=PATH`, `--value-env NAME=VAR` and
`--secret-env NAME=VAR` (its operator inputs, values and secrets). Each task's README says how to build its image (a
task usually has an `environment/build-apptainer.sh`) and what else it needs.

The launcher removes every key variable from its own environment before starting anything, so no spawner, Slurm step
or sandbox inherits one; the gateways and the verifier are given theirs explicitly. It prints a JSON summary (stage
exit codes, seal digests, the verifier's exit code) and returns the verifier's exit code: 0 for a PASS verdict, 1 for
FAIL, 2 for a rule violation or an incomplete run. `RUN_DIR/reports/reward.json` holds the reward; a run that stops
early prints a JSON line with its status and error instead, and leaves `reports/pipeline-error.json`. `--smoke` runs
every stage with the task's reduced evaluation settings, for testing a setup cheaply; the run is sealed as a smoke run
and its reward is not a benchmark score ([../SPEC.md](../SPEC.md)).

If verification itself fails (the judge's provider, a verifier error), `python3 -m breakwater reverify --run-dir
RUN_DIR` with the run's task, image, assets and judge runs only the verifier again, against `authority-reverify-N` and
`reports-reverify-N`, leaving the earlier reports in place. Evidence is left read-only: remove a run with `chmod -R
u+w RUN_DIR && rm -rf RUN_DIR`.

## How a run is isolated

All parts of a run belong to one Slurm job on one node (a task's endpoints may be on other nodes of the job, or
elsewhere). Sandboxes run in the foreground (`apptainer exec --userns --containall`), so each can be its own Slurm
step.

```
launcher (host, the user's own process; holds the evidence directory and every key)
 ├─ gateways                                 host processes on unix sockets in a private directory
 ├─ author sandbox                           own user, mount, PID, IPC and network namespaces: the agent CLI
 │    agent CLI (read-only), workspace (rw)   (read-only); network = loopback + a relay to the author gateway
 ├─ command sandbox                          a sibling: workspace (rw), fresh home and /tmp, GPUs; every agent
 │    every agent command and its children    command; network = loopback (+ relays the task declares)
 ├─ exec broker                              receives each command from the author sandbox over a unix socket and
 │                                            runs it in the command sandbox under strace, teeing its output
 └─ relays                                   join only a sandbox's network namespace: reachable from it,
                                              invisible to its processes
```

The two authoring sandboxes are siblings, not nested, so sites that forbid nested user namespaces still work, and
neither sandbox can create or enter the other. Only the broker and the relays, which run outside both, cross between
them. Sandboxes ignore site-wide binds from `apptainer.conf` (keeping read-only `/etc/hosts`, `/etc/resolv.conf` and
`/etc/localtime`), so a site that binds a shared filesystem everywhere cannot expose the run directory; each sandbox
has its own hostname mapped to its loopback address.

| Safeguard | Docker backend | Apptainer backend |
|---|---|---|
| Separate identities | author UID 1100, worker UID 1101, verifier UID 1102 | one UID; the authoring sandboxes, the evaluation sandboxes and the verifier are separate namespace sets. Commands cannot see the CLI's files, processes or network. |
| Environment fingerprinting | hash the container filesystem export before, after and in a fresh container | the SIF is immutable: its SHA-256 before authoring, after authoring and after evaluation; the workspace keeps the import-shadow check. |
| API routing | gateway container plus per-UID iptables | the CLI's sandbox has no network but a relay to the gateway's unix socket; the command sandbox has its own network namespace with no route to the gateway or the internet. |
| Process wall | stop the container, prove the drain with a host-PID observer | stopping a sandbox kills its PID namespace's init, and the kernel kills every member, daemons included; the launcher proves the namespace empty from `/proc`. |
| External journal | observer container with host PID view and ptrace, root-owned stdio supervisor | the broker starts every command under `strace -f` from outside the sandbox and tees its streams into the evidence directory, in a hash-chained journal; the gateway journals as with Docker. Nothing in the evidence directory is bound into a sandbox. |
| Restricted intermediate stage | fresh container, coordinator root, submission as worker | a fresh coordinator sandbox with the private inputs, and a sibling submission sandbox without them, where a second broker runs the submitted code (`worker-exec`). |
| Verifier | separate read-only container, UID 1102 | a separate sandbox started after evaluation, with read-only evidence, its own reports directory, the host network and the judge key; no GPU. It proves it is the recorded verifier sandbox by its PID namespace. |

**Authoring.** The agent CLI (the pinned Claude Code, or Codex) runs in the author sandbox; Claude Code's shell prefix
and Codex's executor send each command to the broker ([client.py](client.py); for Codex, [codex_forward.py](codex_forward.py)
runs `codex exec-server` beside the CLI and turns each process it is asked to start into a broker command, refusing
any other method). Command traces use the observer's strace options (`-f -ttt -T -v -xx -y`, one file per command),
which the review index parses; their overhead on GPU training is small. The CLI's native output is
copied by the host into the evidence and into the agent's local logs. At the deadline the host stops the CLI and
proves both sandboxes empty, then checks the image and seals the workspace.

**Evaluation.** The coordinator sandbox runs the task's trusted evaluation with the private inputs; submitted code runs
only in the submission sandbox, through the worker broker ([worker_client.py](worker_client.py)). A port the submitted
code serves (`evaluation.ports`) is relayed from the submission sandbox's loopback to the coordinator's, and an
operator service named by a loopback URL in `evaluation.env` from the coordinator's loopback to the host's. When the
sealed submission is mounted apart, the two sandboxes share a writable workdir and the task's `evaluation.shared`
directories. The coordinator's `kill_workers` stops every submitted process in the submission sandbox. Every fresh
worker home starts as a copy of the image's `/home/worker` plus the caches the Docker author image moves there from
`/root` (Hugging Face, Torch, Triton).

**Verification.** The verifier sandbox runs the unchanged `verifier.py`. For an Apptainer authority it checks each
stage's sandboxes and their binds against the kernel's mount tables, the process-wall proofs, the stage order,
environments without keys, the image digests and both broker journals. The judge reads [checklist.md](checklist.md),
the Docker checklist's rows worded for sandboxes, together with `established`, a statement per row of what the host and
the verifier checked mechanically.

**Network paths a task can declare.** Sandboxes have no network of their own; every path is a relay from the host.
An unprivileged launcher cannot give a sandbox another address, so other machines are reached by name:
[endpoints.py](endpoints.py) maps each host to a loopback address of its own in the sandbox's hosts file, binds a
listener for each allowed port there and joins each connection to the real host.

- `[breakwater.model_access]`: the development gateway runs on the host and is relayed into every command sandbox,
  replacements included, at the task's host and port (`gateway:8000` by default).
- `[[breakwater.endpoints]]` and the endpoints a task's host hooks declare: each name's declared ports, relayed into the
  sandboxes of the roles that reach it (the agent's commands, the submitted code, the coordinator), with a port map
  where the real service listens on another port.
- `[breakwater.authoring] inbound_ports`: relayed from the host's loopback into the command sandbox (each replacement
  gets its own relays), where the inbound host name is the sandbox's own loopback.
- A custom launcher's services (the `services=` of `agent_stage.AuthoringStage`), such as a driver gateway for a
  submitted agent, relayed in the same way.

## What is weaker, and what is stronger

- **Weaker:** all parts share one kernel UID and the per-user limits. A fork bomb or memory exhaustion in the command
  sandbox can starve the launcher; that voids the run (evidence capture fails) but cannot fake a pass. Per-stage CPU
  and memory limits need Slurm; otherwise only rlimits and CPU affinity are available. The isolation relies on
  unprivileged user namespaces, which expose more of the kernel than Docker's root-managed namespaces. GPU devices
  are not confined per sandbox (`--nv` exposes the node's GPUs; a Slurm job's device cgroup confines them).
- **Stronger:** the runtime image cannot change at all (read-only SIF), there are no setuid helpers, and the process
  wall is enforced by the kernel's PID namespace teardown rather than by scanning.
- **Same as Docker:** a sandbox's mount table (`/proc/self/mountinfo`) shows each bind's host source path, so name run
  and asset directories neutrally.

## Tests

```bash
apptainer build word-count.sif docker://python:3.12-slim
BREAKWATER_APPTAINER_IMAGE=word-count.sif python3 -m unittest discover -s breakwater/apptainer/tests -v
```

The suite runs the sandboxes, the broker, both stages, the launcher, the verifier's checks, Codex (with
`BREAKWATER_CODEX_CLI_DIR` set to an `install-cli --agent codex` directory), the example task end to end, the
endpoint and inbound-port relays and a task with host hooks on that image, with stand-in models. One test needs a
Slurm allocation: a real out-of-memory kill of a command-sandbox step, which the stage replaces and survives.

## Files

| File | Role |
|---|---|
| [run.py](run.py) | the launcher: contract, bindings, seeding, the three stages, seals |
| [sandbox.py](sandbox.py), [spawner.py](spawner.py) | stage sandboxes, the per-sandbox spawner, the process-wall proof |
| [agent_stage.py](agent_stage.py) | authoring: the gateways, the author and command sandboxes, the broker, relays, replacements |
| [evaluation_stage.py](evaluation_stage.py) | evaluation: the coordinator and submission sandboxes and their relays |
| [broker.py](broker.py), [client.py](client.py), [worker_client.py](worker_client.py) | the exec brokers and their clients |
| [codex_forward.py](codex_forward.py) | Codex's executor, forwarding each command to the broker |
| [relay.py](relay.py), [endpoints.py](endpoints.py) | relays into sandboxes' network namespaces |
| [checks.py](checks.py), [checklist.md](checklist.md) | the verifier's Apptainer checks and the judge's checklist |
| [install_cli.py](install_cli.py), [install_judge.py](install_judge.py) | the pinned agent CLI and judge runner |
| [reverify.py](reverify.py) | the verifier again on a finished run |
| [probe.py](probe.py) | a site's readiness |
