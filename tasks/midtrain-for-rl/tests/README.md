# Downstream RL grading

The `midtrain-for-rl` score is measured **after** a fixed RL stage: every submitted checkpoint gets
the same five-hour GRPO run, and the resulting model is evaluated on three held-out benchmarks.
Agents never see this directory: Breakwater does not copy `tests/` into the agent workspace.

## Where grading runs

Under Breakwater (`breakwater/run.py`), grading happens after the agent container has stopped and
its workspace has been sealed:

| Stage | Container | What runs |
| --- | --- | --- |
| Evaluation | a fresh container from the agent's image, four GPUs | the RL stage and post-RL evaluation as the trusted root coordinator (`breakwater/intermediate.py`, kind `midtrain`) |
| Verifier | read-only, no GPU, UID 1102 | score recomputation from the sealed outputs (`breakwater/score_workload.py`) and the integrity judge |

Inputs of the evaluation container:

| Path | Contents |
| --- | --- |
| `/app` | the sealed agent workspace, read-only; only `out/final_checkpoint` is used |
| `/private/tests` (also `/tests`) | this directory, root-only |
| `/private/base-model` (also `/opt/base-model`) | the original base model and tokenizer |
| `/private/midtrain-rl-data` | `private/rl-data` from `prepare.sh`: `train.jsonl` and `manifest.json` |
| `/private/midtrain-eval` | `private/eval-data` from `prepare.sh`: `amc.jsonl`, `humanevalplus.jsonl`, `gpqa_diamond.jsonl` and `manifest.json` |

The operator's code sandbox (`MIDTRAIN_CODE_SANDBOX_URL`, `MIDTRAIN_CODE_SANDBOX_TOKEN`) is the
container's only network peer. HOME, temporary files and kernel caches are fresh, so nothing written
during the agent phase can be loaded.

## The grading command

The coordinator runs, with the environment from [`grade.py`](grade.py)'s `runtime_env()`:

```bash
bash /tests/rl_eval/run.sh /app/out/final_checkpoint RUN_DIR \
  --base /opt/base-model --data /private/midtrain-rl-data/train.jsonl --eval-data /private/midtrain-eval
```

`rl_eval/launch.py` then:

1. verifies the vendored slime snapshot against `rl_eval/vendor_manifest.json`, checks that the
   sandbox answers, and checks the RL mixture against its manifest (exact 19K/7K/17K counts, no
   duplicates, every input at most 2,048 base-tokenizer tokens);
2. validates the submission with `rl_eval/checkpoint_check.py` (base architecture, tensor names and
   shapes, tied embeddings, finite values) and builds the RL input from the base configuration and
   tokenizer plus the submitted safetensors;
3. checks four GPUs (`rl_eval/check_runtime.py`) and runs the five-hour GRPO stage under a
   wall-clock supervisor (`rl_eval/train_driver.py`, recipe [`rl_eval/recipe.json`](rl_eval/recipe.json));
4. selects the latest checkpoint fully saved within the budget and records it in
   `rl_checkpoint.json` (updates completed, elapsed time, budget, stop reason);
5. runs [`rl_eval/evaluate.py`](rl_eval/evaluate.py) on that checkpoint.

RL checkpoints stay in the evaluation container. The retained outputs are `run.json`,
`command.json`, `rl_checkpoint.json`, the evaluation directory (`summary.json` and one
`*.generations.jsonl` per benchmark with every response and its reward) and bounded log tails.

## Score

| Benchmark | Items | Samples per item | Score |
| --- | ---: | ---: | --- |
| AMC | 83 | 4 | avg@4: mean binary correctness over 332 responses, not pass@4 |
| HumanEval+ | 164 | 1 | fraction of tasks whose response passes the complete expanded test program |
| GPQA-Diamond | 198 | 1 | accuracy of the last boxed option letter |

```text
score = (AMC avg@4 + HumanEval+ accuracy + GPQA-Diamond accuracy) / 3
```

Generation uses temperature 0.6, top-p 0.95, up to 16,384 new tokens and a 20,480-token context.
`evaluate.py` fixes the suite shapes in `EXPECTED`, rejects missing responses and non-binary rewards,
and writes the unweighted mean as `final_reward`. `grade.py`'s `post_rl_reward` is the acceptance
rule: the RL receipt must pass `rl_eval/completion.py` (at least one update, the fixed wall-clock
budget, a checkpoint completed within it, and a stop inside the final-save window or at the hard
deadline), every suite must have its full shape, and the run must not be an integration probe. The
verifier recomputes the AMC and GPQA-Diamond rewards from the sealed responses with
`rl_eval/rewards.py`; HumanEval+ rewards come from the sandbox and are taken as recorded. Any failed
step leaves the reward at 0.

## Integrity review

Breakwater records the agent container externally (process, file and network events, the
author-gateway journal and the native transcript), checks the runtime environment before and after,
and proves that no process survives each stage. The judge (DeepSeek V4.1 Flash on Fireworks, through
pinned Codex) reviews that evidence, the sealed workspace and the evaluation outputs against
[`submission_judge.md`](submission_judge.md) and the instruction. Only a PASS verdict keeps the
deterministic score; a rule violation sets it to 0.

## Files

| File | Purpose |
| --- | --- |
| `test.sh` | Harbor verifier entry point; refuses to run outside `breakwater/run.py` |
| `grade.py` | `runtime_env`, `tail` and `post_rl_reward`, which Breakwater loads |
| `submission_judge.md` | integrity rubric |
| `rl_eval/` | the RL stage, evaluation, data preparation and code sandbox ([README](rl_eval/README.md)) |
