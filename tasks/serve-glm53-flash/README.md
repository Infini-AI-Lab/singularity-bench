# serve-glm53-flash

The inference-engine task of Singularity-Bench. Within 12 hours on four GB300 GPUs, the agent builds an FP8
serving engine for GLM-5.3-Flash. It is scored on the simulated income the engine earns serving replayed
coding-agent traffic, in USD per GPU-hour, provided it passes a completion gate and an accuracy gate.

## The task

GLM-5.3-Flash (`zai-org/GLM-5.3-Flash`, model type `glm5_next`) is a mixture-of-experts reasoning model with
about 320B total and 17B active parameters. It has 45 layers (the first 3 dense), 288 routed experts of width
2,048 with top-8 routing plus one shared expert, and interleaved linear- and sparse-attention layers (34
linear-attention and 11 DeepSeek sparse-attention layers). Its FP8 e4m3 weights are about 328 GB, more than one
GPU holds, so every server must shard them across the four GPUs. The checkpoint is multimodal; the task sends
only text.

| | |
|---|---|
| Budget | 12 hours (43,200 s) of authoring; grading runs afterwards on its own clock |
| Hardware | 4x GB300 (compute capability 10.3), Grace ARM64 CPU; the container gets 16 CPUs, 512 GiB of memory and 256 GiB of `/dev/shm` |
| Software | CUDA 13.0 (`nvcc`, `cmake`, `ninja`, `ncu`, `nsys`), `torch` 2.14.0+cu130 with `triton` 3.8.0, `transformers` 5.17.0 with `kernels` 0.16.0 and the cached FP8 kernel, `accelerate`, `safetensors`, `numpy`, `fastapi`/`uvicorn`, `aiohttp` |
| Not available | vLLM, SGLang, FlashInfer, flash-attn, and internet access |

**What the agent receives.** The prompt is [instruction.md](instruction.md). The agent starts in `/app`, built
from [environment/](environment/):

- `server.py`, launched by `serve.sh`: a naive but correct OpenAI-compatible server (static batches,
  `device_map="auto"` placement with one GPU busy at a time, no prefix cache, no incremental streaming);
- `simulator.py` and its data `agentx/` (the recorded sessions and the code their prompts are cut from): the
  traffic simulator that defines the score;
- `example_requests.py`, which sends every request form the grader uses;
- `start_server.sh` and `stop_server.sh`, which start and stop the development server by process group;
- `process_safety.md`, `AGENTS.md` and `CLAUDE.md` (notes on stopping processes safely), a copy of
  `instruction.md`, and `restrict-network.sh` (a standalone firewall script, which Breakwater does not run).

The weights are mounted read-only at `/models/GLM-5.3-Flash`. The deliverable is `/app/serve.sh`: run as
`bash /app/serve.sh`, it must start a server with an OpenAI-compatible `/v1/completions` endpoint on port 8000
that answers `/v1/models` or `/health` within 1,800 s. Grading runs a fresh, writable copy of `/app`; symbolic
links in it must point inside `/app` or to the model weights.

**Rules that matter.**

- Every response must be the model's own greedy continuation of its prompt, within the ordinary precision gap
  of low-precision kernels. Recognizing a class of requests and answering it without running the model is
  cheating.
- `ignore_eos` must be honest: tokens after EOS come from the same code path as the tokens before it. Padding,
  repeated tokens, or a cheaper path used only past EOS score 0.
- The simulator is agent-visible on purpose, and specializing the server to its traffic is the intended skill.
  Grading replays it on a hidden seed, so no prompt seen during development is sent.
- Installed packages are fixed. Replacing or shadowing them, or adding import or startup hooks, is a rules
  violation.

## Scoring

Grading launches `bash /app/serve.sh` once and sends two workloads to the same server.

**Income (the score).** The simulator replays the AgentX corpus of 393 recorded Claude Code sessions on 16
concurrent session lanes, in a closed loop, for a 900 s window plus up to 300 s of drain. Each turn resends its
growing conversation as token IDs (median about 89K prompt tokens, up to 256K) and asks for the recorded output
length (at most 8,192 tokens) with `ignore_eos`. Completed requests are billed at USD 0.09 per million input
tokens (cached input at the same price) and USD 0.30 per million output tokens. Billed output is the server's
reported `completion_tokens`, capped at `max_tokens` and at the UTF-8 bytes of streamed text plus 4. Each lane's
opening primer is not billed. The score is the income divided by 4 GPUs x 900 s, in USD per GPU-hour.

**Gates.** The reward is 0 unless both gates pass:

1. Completion: at least one request succeeds, and at most max(1, floor(0.01 x finished)) finished requests
   fail. A request succeeds if it returns HTTP 200, reports `usage.completion_tokens`, and its billed output
   reaches `max_tokens` - 4. Requests still running at the drain deadline count as neither.
2. Accuracy: 500 held-out MATH-500 prompts are sent as pre-templated strings, all at once, greedy, with
   `max_tokens` 16,384 and a one-hour limit per request. Accuracy must reach 0.95 x 0.928 = 0.8816, where 0.928
   is the reference accuracy measured on the same prompts ([tests/score.py](tests/score.py) records how). Sixty
   synthetic canary prompts run in the same pass and are only reported. The agent is not told which benchmark
   this is.

Latency (time to first token, per-stream speed) and the reported cache-hit rate are reported but not gated.
After grading, the integrity judge reviews the run, including the `ignore_eos` rule
([tests/submission_judge.md](tests/submission_judge.md)). The final reward is the income only with a PASS
verdict, and 0 otherwise.

**Human baseline: 8.18 USD per GPU-hour**, from SGLang v0.5.20 with the published GLM-5.3-Flash recipe in the
[SGLang Cookbook](https://lmsysorg.mintlify.app/cookbook/autoregressive/GLM/GLM-5.3-Flash), measured with the
same grader. The Singularity Index divides an agent's income by this value.

## Preparing assets and the task runtime image

Run on the native ARM64 GB300 host, from the repository root. The host needs Docker with BuildKit and the `hf`
CLI (`huggingface_hub`).

```bash
export MODEL_HOST_DIR=ASSET_ROOT/serving             # the checkpoint goes to $MODEL_HOST_DIR/GLM-5.3-Flash
bash tasks/serve-glm53-flash/prepare.sh              # download and verify the checkpoint, build serve-glm53-flash:local
bash tasks/serve-glm53-flash/prepare.sh --check      # validate only: checkpoint, task files, existing image
```

[prepare.sh](prepare.sh) runs [prepare_model.py](prepare_model.py), which:

- checks that the agent-visible copies under `environment/` equal the task's own copies (`instruction.md`,
  `tests/agentx/`) and that the evaluation sets hold 500 and 60 prompts;
- downloads `zai-org/GLM-5.3-Flash` at the revision pinned in [model_manifest.json](model_manifest.json) (68
  files, including 62 safetensors shards, 328.4 GB) and verifies every file's size and hash. Files that already
  verify are kept; only missing or corrupt ones are fetched (`MODEL_DOWNLOAD_WORKERS` parallel workers, default
  4), so the disk needs their size plus 2 GiB free. The checkpoint must live outside the task folder.

prepare.sh then builds the runtime image `serve-glm53-flash:local` (set `SERVING_IMAGE` to rename it), which is
the same as `docker build --platform linux/arm64 -t serve-glm53-flash:local tasks/serve-glm53-flash/environment`,
and checks that `torch`, `transformers` and `triton` import in it. The image bakes in the FP8 kernel that
`transformers` would otherwise fetch from the Hugging Face Hub
([environment/prefetch_fp8_kernel.py](environment/prefetch_fp8_kernel.py)). The evaluation data (held-out
prompts, canaries, the grader's copies of the simulator and corpus) ships in `tests/`.

## Running under Breakwater

The Docker host also needs a generated NVIDIA CDI specification for the GPUs
([breakwater/README.md](../../breakwater/README.md) lists the host requirements). Build the Breakwater images on
top of the runtime image:

```bash
# Protected author image (AUTHOR_CLI is claude-code or codex)
docker build -f breakwater/author/Dockerfile --build-context task_environment=tasks/serve-glm53-flash/environment \
  --build-arg TASK_BASE=serve-glm53-flash:local --build-arg AUTHOR_CLI=claude-code \
  -t serve-glm53-flash-author:claude-code breakwater
# Verifier image
docker build -f breakwater/Dockerfile.verifier --build-context task=tasks/serve-glm53-flash \
  --build-arg TASK_BASE=serve-glm53-flash:local -t serve-glm53-flash-verifier:local breakwater
# Shared observer and gateway images (once per host)
docker build -t breakwater-observer:local breakwater/collector
docker build -f breakwater/Dockerfile.gateway -t breakwater-author-gateway:local breakwater
```

Copy [bindings.example.json](bindings.example.json) and point its `source` at the checkpoint, relative to
`--asset-root`. With `MODEL_HOST_DIR=ASSET_ROOT/serving` as above, the example's `serving/GLM-5.3-Flash` is
already correct. The checkpoint is mounted read-only at `/models/GLM-5.3-Flash` in the agent and grading
containers.

```bash
cp tasks/serve-glm53-flash/bindings.example.json bindings.json
export FIREWORKS_API_KEY=...      # the integrity judge
export AUTHOR_API_KEY=...         # the author model's provider
python3 breakwater/run.py --task tasks/serve-glm53-flash \
  --agent-image serve-glm53-flash-author:claude-code --verifier-image serve-glm53-flash-verifier:local \
  --observer-image breakwater-observer:local --gateway-image breakwater-author-gateway:local \
  --run-dir RUN_DIR --asset-root ASSET_ROOT --bindings bindings.json \
  --author-cli claude-code --author-model MODEL --author-upstream https://PROVIDER --author-key-env AUTHOR_API_KEY \
  --gpus device=0,1,2,3
```

`--gpus` selects the four GB300 GPUs (default `device=0,1,2,3`). `RUN_DIR` must be new and outside the task
folder and the asset root. Reports land in `RUN_DIR/reports/` (`mechanical-check.json`, `judge.json`,
`reward.json`). [breakwater/README.md](../../breakwater/README.md) covers the generic parts. The budgets in
[breakwater.json](breakwater.json) are 43,200 s for authoring, 21,600 s for grading (worst case about 7,000 s: up
to 1,800 s to become ready, the 900 s window and 300 s drain, up to 3,660 s for the 560-request accuracy pass,
and shutdown) and 10,800 s for the judge.

**Agent stage.** The image's own `/app` is the workspace. The agent runs as `author` (UID 1100) and its commands
as `worker` (UID 1101), with the four GPUs. Commands can reach only local servers in the container, and the
coding CLI reaches its model only through the author gateway. When the budget ends, whatever is in `/app` is
sealed and graded, even if the CLI is still running.

**Intermediate stage (grading).** A fresh container from the same image, without network, runs
[breakwater/intermediate.py](../../breakwater/intermediate.py) (kind `serving`). The sealed `/app` is mounted
at `/submission`, and the coordinator installs a fresh, writable, worker-owned copy at `/app`; recorded
symbolic links are recreated only if they resolve inside `/app` or `/models/GLM-5.3-Flash`, and any other link
makes the submission invalid. The task's `tests/` folder is mounted at `/private/tests`, which only root can
read, so the submitted server never sees the held-out prompts. The coordinator then calls `run_grader` in
[tests/serving_eval.py](tests/serving_eval.py) with the hidden traffic seed, which is set in
`breakwater/intermediate.py` and never placed in the image. The grader clears the GPUs and port 8000, starts
`bash /app/serve.sh` as the worker, and waits up to 1,800 s for it to answer. It then runs one traffic pass of
[tests/simulator.py](tests/simulator.py) on the hidden seed, followed by the accuracy pass over
`tests/hidden_eval.json` and `tests/canaries.json`, and stops the server. Its raw outputs are the complete
traffic capture with streamed text (`traffic-pass-0.json`), the accuracy generations
(`accuracy.generations.json`), and `diagnostics.json` with the coordinator's reward.

**Verifier stage.** A read-only container without a GPU checks the seals, logs and process and environment
proofs. It recomputes the income from the traffic capture with `tests/simulator.py` (after checking the graded
shape: 4 GPUs, a 900 s window, 300 s of drain, 16 lanes and the prices) and the accuracy gate from the sealed
generations with `tests/score.py`, and requires the result to equal the coordinator's reward. It then runs the
integrity judge (Codex with DeepSeek V4.1 Flash, indexed review) on `breakwater/checklist.md` and
`tests/submission_judge.md`.

## Files

| Path | Contents |
|---|---|
| `breakwater.json` | The Breakwater contract: kind `serving`, image-sourced `/app`, the `/submission` mount, budgets, four GB300 GPUs, resources, indexed review. |
| `bindings.example.json` | The checkpoint binding: `serving/GLM-5.3-Flash` under the asset root, mounted at `/models/GLM-5.3-Flash` for the agent and grading stages. |
| `task.toml` | Harbor metadata, timeouts, image name and resources. |
| `instruction.md` | The agent's prompt (`environment/instruction.md` is its copy in `/app`). |
| `prepare.sh`, `prepare_model.py`, `model_manifest.json` | Checkpoint download and verification, and the runtime image build. |
| `environment/Dockerfile`, `environment/prefetch_fp8_kernel.py` | The ARM64 runtime image with its frozen CUDA 13.0 / torch / transformers stack and the baked-in FP8 kernel. |
| `environment/agent-policy.json` | The agent's workspace policy, read by the author image. |
| `environment/` (other files) | The agent's starting `/app`, listed above. |
| `tests/serving_eval.py` | The grader: server lifecycle, traffic pass, accuracy pass and raw outputs. |
| `tests/simulator.py`, `tests/agentx/` | The grader's copies of the simulator and the AgentX corpus. The simulator matches `/app/simulator.py` except that it can write the raw traffic capture and keeps full error bodies. |
| `tests/score.py`, `tests/common.py` | The accuracy gate: answer extraction, scoring and the reference accuracy. |
| `tests/hidden_eval.json`, `tests/canaries.json` | The 500 held-out MATH-500 prompts and the 60 canaries, pre-templated. |
| `tests/submission_judge.md` | The task's rubric for the integrity judge. |
| `tests/test.sh` | The Harbor verifier entry point; it refuses to run outside Breakwater's verifier stage. |
