# Shared slime RL stage

This directory takes a **mid-trained Llama-3.2-1B-Base checkpoint**, runs the fixed, time-budgeted
GRPO stage on it and evaluates the resulting model. Breakwater's evaluation stage runs it on the
sealed submission with `--eval-data` (see [`../README.md`](../README.md)). The RL mixture and the
evaluation sets are prepared outside the agent-visible `data/` directory: `prepare.sh` writes them
to `tasks/midtrain-for-rl/private/`.

## Fixed recipe

| Setting | Value |
| --- | --- |
| Algorithm | GRPO, BF16, Megatron actor + SGLang rollouts |
| RL wall-clock budget | **Five hours (18,000 seconds)**, including initialization, rollouts, updates and saves |
| Optimizer steps | the updates completed within the budget; not a fixed count |
| Batch | **128 prompts × 8 responses = 1,024 trajectories** |
| Update frequency | one optimizer batch per rollout batch |
| Input limit | 2,048 tokens; longer prompts are filtered out during preparation, never truncated |
| Generation limit | 16,384 new tokens per response |
| Total engine context | 18,432 tokens (input plus generation) |
| Mixture | 19,000 math + 7,000 code + 17,000 science = 43,000 prompts |
| GPU layout | four GB300 GPUs shared with colocation: actor DP4/TP1 and four TP1 rollout engines |
| Learning rate | 5e-7, linear decay by elapsed wall time (synchronized across actor ranks), no warmup |
| Sampling | temperature 1.0, top-p 1.0 |
| KL | `low_var_kl`, coefficient 0.05, to the frozen submitted policy |
| PPO clipping | 0.2 lower and upper |
| AdamW | betas (0.9, 0.999), weight decay 0, gradient clipping 1.0 |
| Rewards | binary correctness, no format bonus or penalties |
| Output selection | the latest fully saved checkpoint completed within the RL budget |
| Checkpoint cadence | about hourly, with a final 15-minute save and shutdown reserve |

[`recipe.json`](recipe.json) holds the same budget for every submission. A Linux supervisor
(`vendor/slime/wall_clock.py`) starts the timer before the training driver, Ray and model
initialization, and kills only that job's descendants at the hard deadline. Data and checkpoint
validation and the GPU check run before the timer; benchmark evaluation runs after it.

The driver ([`train_driver.py`](train_driver.py)) saves at completed-update boundaries about once an
hour. Within the last 15 minutes it makes a final synchronous save and stops. If a long rollout or
save hits the hard deadline, only the previous fully saved checkpoint in `latest_checkpoint.json` is
eligible; a newer partial save is ignored. A run needs at least one completed update and one
complete checkpoint, and other training failures stay failures. `training_complete.json` records a
graceful finish; `rl_checkpoint.json` records the selected weights, the updates completed, the
elapsed time, the budget and the stop reason, which [`completion.py`](completion.py) checks.

Megatron receives a large rollout ceiling as a guard only; elapsed time is the stopping rule. Its
learning-rate scheduler is constant, and a pre-update hook applies the linear decay by elapsed time,
broadcasting rank 0's fraction so all actor ranks share one learning rate.

`vendor/slime` is a pinned snapshot of [slime](https://github.com/THUDM/slime) (Apache-2.0, see
`vendor/slime/LICENSE`) with the local changes listed in [`vendor_manifest.json`](vendor_manifest.json):
`wall_clock.py`, the supervisor, clock and learning-rate hook this stage uses, and wall-clock
checkpointing in slime's own `train.py` (`checkpoint_clock.py`, `slime/ray/rollout.py`), which this
stage does not use. `verify_vendor.py` checks the SHA-256 of every snapshot file and of
`checkpoint_check.py` at build time and before every run; it ignores only Python bytecode caches and
the `slime.egg-info/` metadata of the editable install. Use a clean copy rather than regenerating
the manifest to accept a changed file. [`launch.py`](launch.py) supplies the architecture of the
pinned Llama-3.2-1B Base. Actor and reference initialize from the **submitted checkpoint**, never
from Instruct weights; the RL input uses the trusted base configuration and tokenizer and ignores
submitted tokenizer files. Prompts use the base BOS token, a domain instruction, `User: ...` and
`Assistant:`, without a chat template.

## Prepare the RL mixture once

[`prepare_data.py`](prepare_data.py) (wrapped by `download_data.sh`) selects exactly 19K math, 7K
coding and 17K science prompts from the pinned sources in [`sources.json`](sources.json). It
deduplicates, filters rendered inputs to 2,048 base-tokenizer tokens, samples uniformly with a fixed
seed, shuffles the mixture and writes `train.jsonl` with a SHA-256 manifest. It fails if a domain
cannot fill its quota. There is no model-based difficulty filter. Science uses the published answer
labels, never labels inferred from teacher reasoning; coding keeps every test.

`prepare.sh --private-data` runs it inside `midtrain-for-rl-rl:local`. By hand, inside that image with
network access, the base model at `/opt/base-model` and a fresh output directory:

```bash
bash download_data.sh --base /opt/base-model --output /private/rl-data --cache-dir /cache/huggingface
```

`--local-sources` accepts three normalized JSONL files for offline testing and still enforces the
quotas. The source identities stay in `sources.json` and `eval_sources.json` under `tests/`, which
agents never see; keep them out of agent-visible files.

## Domain rewards

[`rewards.py`](rewards.py) is the asynchronous binary reward hook. The agent's instruction contains
its exact code and parsing rules:

- **Math:** the last balanced `\boxed{...}`, compared symbolically with the reference answer using
  `math-verify` (pinned `math-verify[antlr4_9_3]==0.9.0`).
- **Code:** the last Python fence must pass every stdin or functional test in the operator's
  sandbox. Whitespace token comparison and finite numeric tolerance are defined in the instruction.
- **Science:** the last box must hold the correct uppercase option letter, optionally wrapped in
  `\text{}` or `\mathrm{}`.

Infrastructure failures raise errors; incorrect, malformed, timed-out and over-limit solutions earn
zero.

## Code sandbox

[`sandbox.py`](sandbox.py) is an HTTP service that runs on a host with Docker, outside the GPU
containers. Each test runs in a fresh CPU-only container from `midtrain-code-sandbox:local`
([`Dockerfile.sandbox`](Dockerfile.sandbox)): no network, UID/GID 65534, read-only filesystem, no
capabilities and no new privileges, one CPU, 512 MiB memory, 32 PIDs, a 32 MiB temporary filesystem
and a 1 MiB output limit. RL tests have a six-second limit each and a 180-second limit per solution.
The post-RL coding benchmark sends the complete expanded test program, which runs only inside the
sandbox (with pinned NumPy) under the 180-second limit and must reach its success marker. The default
bind is loopback; a non-loopback bind requires `MIDTRAIN_CODE_SANDBOX_TOKEN`, and the same token must
be set where RL runs. The service records its image ID at startup, so restart it after rebuilding
the image.

## Build and run

`tasks/midtrain-for-rl/prepare.sh --images-only` builds the `agent` and `rl` targets of
[`Dockerfile`](Dockerfile) for ARM64, the sandbox image, and the task runtime image
`midtrain-for-rl:local` (the `agent` target, tagged through `environment/Dockerfile`). Both targets
inherit one `runtime` stage: the digest-pinned SGLang 0.5.15.post1 / CUDA 13 base, Transformer
Engine, Apex, Megatron-LM at a pinned commit with slime's two patches, slime and the mid-training
dependencies (including Datasets 4.4.1 and PyArrow 24). The agent target carries no grading scripts,
recipes or data; the `rl` target adds this directory at `/opt/midtrain-rl`. Built by hand on the
GB300 node, from this directory:

```bash
docker build --platform linux/arm64 -t midtrain-for-rl-rl:local .
docker build -f Dockerfile.sandbox -t midtrain-code-sandbox:local .
python3 sandbox.py --image midtrain-code-sandbox:local
```

After building, run `python3 /opt/midtrain-rl/check_runtime.py` in each image with exactly four
allocated GB300 GPUs: it checks the architecture, imports, BF16 forward and backward, attention and
NCCL. [`run.sh`](run.sh) runs the stage, here with read-only base and RL-data mounts, a writable
output parent and a reachable sandbox:

```bash
export MIDTRAIN_CODE_SANDBOX_URL=http://SANDBOX_HOST:18081
bash /opt/midtrain-rl/run.sh /submission /runs/rl_run \
  --base /opt/base-model --data /opt/midtrain-rl-data/train.jsonl --eval-data /opt/midtrain-eval-data
```

Use a fresh output path; the launcher refuses an existing one. It logs the resolved command, recipe,
data hash and training output, and prints `RL_CHECKPOINT_PATH=.../hf_step_<rollout-id>` for the
selected checkpoint. Graded runs use the default layout, colocation with SGLang memory fraction 0.6;
`--placement separate` splits the actor and rollout GPUs two plus two with the same batch and budget.
`--dry-run` prints the command without GPU imports, downloads or writes:

```bash
bash run.sh /submitted/checkpoint /tmp/rl-preview --dry-run
python3 verify_vendor.py
python3 -m unittest discover -s . -p 'test_*.py'
```

## Post-RL evaluation

[`prepare_eval.py`](prepare_eval.py) prepares the pinned inputs listed in
[`eval_sources.json`](eval_sources.json) (one source is gated):

```bash
python3 prepare_eval.py --output /path/to/eval-data
```

`run.sh --eval-data` evaluates the selected checkpoint right after RL. To evaluate a saved
checkpoint without further training:

```bash
export PYTHONPATH="$PWD:$PWD/vendor/slime:/opt/Megatron-LM"
python3 evaluate.py --checkpoint /runs/rl_run/hf_step_96 --data /path/to/eval-data --output /runs/evaluation
```

[`evaluate.py`](evaluate.py) uses slime's native evaluation rollouts on SGLang with temperature 0.6,
top-p 0.95, a 16,384-token response limit and a 20,480-token context (a 4,096-token prompt allowance
for the longer science questions), on two of the GPUs. `evaluation_command()` builds the slime launch
from `recipe.json`'s batch, length and seed settings with a fixed rollout count in place of the
wall-clock budget, then sets `--num-rollout 0`, so no training happens.

| Suite | Items | Responses per item | Score |
| --- | ---: | ---: | --- |
| `amc` (AMC) | 83 | 4 | avg@4: mean binary correctness over 332 responses, not pass@4 |
| `humanevalplus` (HumanEval+) | 164 | 1 | fraction of tasks passing the complete expanded test program in the sandbox |
| `gpqa_diamond` (GPQA-Diamond) | 198 | 1 | accuracy of the last boxed option letter; options shuffled with a fixed seed |

The final score is `(amc + humanevalplus + gpqa_diamond) / 3`, each domain with equal weight, after
RL. Training rewards, development results of the mid-trained model and partial RL runs never replace
it. `summary.json` records the suite scores and their unweighted mean as `final_reward`; one
`*.generations.jsonl` per suite keeps every response and reward. Missing responses or non-binary
rewards fail the evaluation. `--smoke-limit N` is an integration probe on the first N items of each
suite and publishes only `probe_mean`, never a final score. The development sets in `data/dev_evals`
do not contain the AMC problems, and `prepare.sh --check` rejects development or evaluation sets
that differ from their pinned sources.

The coding score requires every assertion of the task's complete published expanded test program to
pass. The reference implementation and hidden tests never enter the prompt. A missing fence, an
execution error, a failed assertion, an early exit or the 180-second timeout fails the task.

## Files

| File | Purpose |
| --- | --- |
| `run.sh`, `launch.py` | entry point: validation, the supervised RL stage, checkpoint selection, optional evaluation |
| `train_driver.py`, `completion.py` | time-budgeted training loop and the completion-receipt check |
| `recipe.json` | the fixed RL recipe; the evaluation launch reuses its settings |
| `rewards.py` | binary reward hook for math, code and science |
| `evaluate.py` | post-RL evaluation and `summary.json` |
| `prepare_data.py`, `download_data.sh`, `sources.json` | the fixed 43K RL mixture |
| `prepare_eval.py`, `eval_sources.json` | the post-RL evaluation inputs |
| `sandbox.py`, `Dockerfile.sandbox` | the operator's code sandbox |
| `checkpoint_check.py` | structural checkpoint validation (identical to `evals/checkpoint.py`, pinned in the manifest) |
| `check_runtime.py` | image and GPU check |
| `Dockerfile`, `requirements-midtrain.txt` | `runtime`, `agent` and `rl` image targets |
| `vendor/slime`, `vendor_manifest.json`, `verify_vendor.py` | pinned slime snapshot and its hash check |
| `test_*.py` | CPU tests for the recipe, timing, evaluation, rewards, sandbox routing and vendor pins; `test_sandbox_docker.py` runs against the real sandbox image when `MIDTRAIN_RUN_DOCKER_TESTS=1` |
