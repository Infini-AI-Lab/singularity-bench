# rl-swe-agent held-out evaluation

The score of an rl-swe-agent submission comes from this trusted evaluator. It runs on the GB300 host, outside every
task container, after the Breakwater run has finished. It loads only the submitted checkpoint and never executes the
submission's training code. This directory is not copied into the agent's workspace.

## Protocol

| Setting | Value |
|---|---|
| Checkpoint | the sealed `RUN_DIR/submission/out/final_checkpoint` of the Breakwater run |
| Export audit | `mini_slime.model.checkpoint` from this repository, in a container without network: a complete safetensors export with config and tokenizer, the initialization architecture and tensor inventory, finite weights, at least one changed tensor, no symlinks, pickles or code files; the evaluator also requires the `.complete` marker that `mini_slime`'s export validation writes |
| Suites | the 73 development tasks of the pinned OpenThoughts-Agent-v1-RL release; a fixed 100-task subset of SWE-bench Verified (`swe-subset.json`) |
| Agent | mini-swe-agent 2.4.6 in Harbor task sandboxes (native ARM64 images), one attempt per task; agent timeout 600 s (development) and 3,000 s (SWE-bench) |
| Policy server | SGLang serving the checkpoint on four GPUs (four data-parallel replicas): 65,536-token context, at most 8,192 generated tokens per model call |
| Concurrency | 8 development trials and 24 SWE-bench trials at a time |
| Budget | four hours for both suites together, including setup and cleanup |
| Reward | the mean of the two suites' resolve rates (`score.py`); 0 if the audit fails, the evaluation is incomplete or the Breakwater integrity verdict is not PASS |

Counting rules (`score.py`):
- A suite counts only when every selected trial finished.
- A trial whose agent failed but whose verifier still ran (for example, the agent exhausted its context) counts as
  unresolved.
- A trial whose verifier ran to its end without producing a reward (the evaluated model left the repository
  broken, or its tests hung) also counts as unresolved.
- A trial whose verifier never ran leaves the suite, and the reward, invalid.

The protocol is fixed in `evaluate_checkpoint.py` and `suites.lock.json`; the command-line options only describe the
host (paths, GPUs, addresses, image tags).

## SWE-bench Verified selection

- `swe-exclusions.json` lists the SWE-bench Verified tasks whose reference solution cannot pass in the native ARM64
  sandboxes, each with its reason. They leave the suite and its denominator, leaving 480 eligible tasks.
- `swe-subset.json` is a fixed sample of 100 of them. It is stratified by repository x upstream difficulty, with
  proportional allocation (largest remainder) and seed 0, and was not chosen by any model's results.
- `make_swe_subset.py` regenerates the subset and records the exclusions' SHA-256. The evaluator rejects a subset
  drawn against different exclusions.

## Requirements

All paths are relative to the repository root; run everything on the GB300 (arm64) host with Docker and the NVIDIA
container runtime. The task README's preparation steps provide:

- the GLM-4.7-Flash weights in `tasks/rl-swe-agent/artifacts/GLM-4.7-Flash` and the prepared split in
  `tasks/rl-swe-agent/data` (`download_artifacts.py`, `prepare_inputs.py`);
- the pinned checkouts `tasks/rl-swe-agent/.deps/harbor` (with both `worker/patches` applied) and
  `tasks/rl-swe-agent/.deps/harbor-datasets`, and [uv](https://docs.astral.sh/uv/) on `PATH`, which runs Harbor from
  its lockfile with Python 3.12;
- the native task images and their manifest `WORK/plan/image-cache.json` (`arm64_tasks.py build`).

Build the evaluation runtime once (the pinned Miles image with `patches/sglang-context-clip.patch` applied to its
SGLang):

```bash
docker build -t rl-swe-agent-eval:local tasks/rl-swe-agent/evaluation
```

## Commands

Run after the Breakwater run has ended, with the four GPUs free:

```bash
python3 tasks/rl-swe-agent/evaluation/evaluate_checkpoint.py \
  --checkpoint RUN_DIR/submission/out/final_checkpoint \
  --image-manifest WORK/plan/image-cache.json \
  --api-host HOST_ADDRESS_REACHABLE_FROM_SANDBOXES \
  --output EVAL_OUT

python3 tasks/rl-swe-agent/evaluation/score.py TRIAL_DIR EVAL_OUT --integrity RUN_DIR/reports/judge.json
```

- `--api-host` is the address at which the task sandboxes reach the policy server, which listens on port 32200 of
  the host network (for example the Docker bridge gateway, `172.17.0.1`).
- The defaults point at the prepared assets: `--base-model tasks/rl-swe-agent/artifacts/GLM-4.7-Flash`,
  `--dev-data tasks/rl-swe-agent/data`, `--swe-benchmark tasks/rl-swe-agent/.deps/harbor-datasets/datasets/swebench-verified`,
  `--harbor tasks/rl-swe-agent/.deps/harbor`, `--eval-image rl-swe-agent-eval:local`, `--gpus 0,1,2,3`. Use
  `--sandbox-docker-host` when the sandboxes run on a separate Docker endpoint.
- `EVAL_OUT` must be a new directory. It receives `eval-settings.json`, `export-audit.json`, `serve.log`, the copied
  and image-pinned task definitions, one Harbor job per suite (`EVAL_OUT/<suite>/trials/`) and `summary.json`.
  The exit status is 0 for a complete evaluation, 1 otherwise and 124 when the four-hour budget ran out.
- `TRIAL_DIR` is any existing directory, for example `RUN_DIR`. `score.py` writes the final reward to
  `TRIAL_DIR/verifier/reward.json`, with the per-suite counts in `reward_metadata.json`. The reward is 0 unless
  `RUN_DIR/reports/judge.json` exists and its verdict is PASS. `RUN_DIR/reports/reward.json` keeps the failure value
  for this task.

## Files

| Path | Contents |
|---|---|
| `evaluate_checkpoint.py` | the evaluator: export audit, policy server, both Harbor jobs, `summary.json` |
| `suites.py`, `suites.lock.json` | suite sources and pins, image resolution, the shared deadline, trial accounting and sandbox cleanup |
| `score.py` | the final reward from `summary.json` and the integrity verdict |
| `swe-exclusions.json`, `swe-subset.json`, `make_swe_subset.py` | the SWE-bench Verified selection |
| `Dockerfile`, `patches/sglang-context-clip.patch` | the evaluation runtime image |
