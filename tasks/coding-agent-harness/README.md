# coding-agent-harness

The agent-harness task of Singularity-Bench. The agent builds a coding-agent harness from scratch within six hours,
using only the Python standard library, with one GB300 GPU and supplied chat and embedding endpoints for testing.
The harness is then scored by what it gets done when other models drive it: three kernel-generation tasks and two
vLLM bug-fixing tasks, each run with three driver models (GLM 5.3, Kimi K3 and MiniMax M3).

## The task

The agent fills in one entry point, `/app/harness/agent.py`:

```python
run(task: str, workspace: str, model: ModelClient, budget: Budget) -> None
```

Everything around the model is the agent's to design: the system prompt, the tools and how their results come back,
tool execution and permissions, context management, error recovery, and when to stop.
[`tasks/coding-agent-harness/instruction.md`](instruction.md) is what the agent is told;
[`tasks/coding-agent-harness/environment/spec/HARNESS_SPEC.md`](environment/spec/HARNESS_SPEC.md) states what must
be true of the result without saying how it is checked.

### What the agent receives

| | |
|---|---|
| Workspace (`/app`, from the image) | `harness/agent.py` (the stub), `spec/HARNESS_SPEC.md`, `harness_api.py` (the `ModelClient` and `Budget` shapes), `selfcheck.py` with `smoke/` (a scripted smoke run, and a `--live` run against a development model), and `examples/` (three optional GPU example tasks with checkers and a runner that drives the submitted harness) |
| Image | CPython 3.11, the CUDA 13.0 toolkit, PyTorch 2.11 (cu130) with Triton, NumPy, Ninja, pytest, a C/C++ toolchain, git, ripgrep, tmux, jq |
| Development models | `chat` (GLM 5.3), `chat-b` (Kimi K3) and `embed` (Qwen3 Embedding 8B) at `http://gateway:8000/v1`, with 5,000,000 chat tokens and 20,000,000 embedding tokens (see [Development model API](#development-model-api)) |
| Hardware and time | one GB300 GPU, 4 CPUs, 16 GiB of memory; 6 hours (21,600 s) |

Rules that matter, all stated to the agent:

- The harness is Python 3 standard library only and must run on CPython 3.11: it is imported in other containers
  that installed nothing for it. Code that the harness runs downstream may use those containers' libraries.
- The harness must do the work itself. It may not invoke, wrap or copy an installed agent harness, and
  `/app/harness/` may not carry executables or archives.
- The development models may be used to test the harness on example tasks, never to design, write, debug, optimize
  or review it. Development requests and responses are retained and reviewed by the integrity judge.
- `/app/spec/`, `/app/smoke/`, `/app/examples/`, `/app/selfcheck.py` and `/app/harness_api.py` must stay
  byte-identical (`protected_workspace_paths` in `tasks/coding-agent-harness/breakwater.json`), and installed
  packages may not be replaced or shadowed.
- The agent's commands run as the unprivileged worker (UID 1101) and reach only the development gateway and local
  servers; the author model and the held-out driver are unreachable from commands.

## Budgets

| Stage | Budget |
|---|---|
| Authoring | 21,600 s wall clock on one GB300, plus 5M chat and 20M embedding development tokens |
| Validity battery (intermediate stage) | at most 3,600 s: seven scenarios, each capped at 300 s and 400 model calls |
| Final verifier and integrity judge | at most 10,800 s |
| One grid cell | the downstream task's `agent.timeout_sec`, 14,400 s, on one GB300, and nothing else |

A grid cell has no call cap, token cap or spend ceiling: the wall clock is the only budget, and the workspace left
behind when it runs out is graded. The other limits are hang catchers: a cell's wall clock plus 900 s of grace; a
cell's grading stage, which gets the downstream task's `verifier.timeout_sec` (3,600 s for the kernel tasks, 1,800 s
for `pr-vllm-35022`, 900 s for `pr-vllm-33698`); and a cell's judge, 10,800 s.

## Scoring

### Root reward: gradeability

After authoring, a fresh container from the same image runs the validity battery: seven scripted scenarios
(`tasks/coding-agent-harness/tests/smoke/scenarios.json`) drive the sealed harness. The root reward is 1.0 when
`agent.py` imports and provides a callable `run()` in every scenario, corroborated by a static parse of the module,
and 0.0 otherwise. The other signals (looping, tool offers, edits on disk, continuing past a turn that asks for no
tool, acting on a tool call written in prose, recovering from a transport fault, stopping when the budget is gone)
are recorded for diagnosis and gate nothing. Only a root run with reward 1.0, a PASS judge verdict and no rule
violation can be graded by the grid.

### Harness score: the downstream grid

The sealed harness is run once per downstream task and driver model: five tasks times three drivers, 15 cells
(profile `compact5-v1` in [`tasks/coding-agent-harness/eval/config.json`](eval/config.json)). **The harness score is
the mean over the three drivers of the mean over the five tasks**, which is the unweighted mean of the 15 cell
rewards.

| Task (`tasks/coding-agent-harness/eval/e2e_tasks/`) | Kind | Work | Reward |
|---|---|---|---|
| `kernels/flash-attn-backward` | kernel generation | FlashAttention backward (dQ, dK, dV); baseline: FlashAttention-2 backward | continuous |
| `kernels/nvfp4-gemm` | kernel generation | NVFP4 block-scaled GEMM on Blackwell FP4 tensor cores; baseline: bf16 cuBLAS | continuous |
| `kernels/mla-decode` | kernel generation | multi-head latent attention decode (576-dim latent, 128 heads); baseline: the eager reference | continuous |
| `pr-reproduction/pr-vllm-35022` | vLLM bug fixing | structured outputs for offline beam search ([vllm#35022](https://github.com/vllm-project/vllm/pull/35022)) | binary |
| `pr-reproduction/pr-vllm-33698` | vLLM bug fixing | per-worker pipeline-parallel KV-cache sizing ([vllm#33698](https://github.com/vllm-project/vllm/pull/33698)) | binary |

| Driver | Model | Served during authoring? |
|---|---|---|
| `glm53` | GLM 5.3 (`accounts/fireworks/models/glm-5p3`) | yes, as `chat` |
| `kimi3` | Kimi K3 (`accounts/fireworks/models/kimi-k3`) | yes, as `chat-b` |
| `minimax` | MiniMax M3 (`accounts/fireworks/models/minimax-m3`) | no, held out |

MiniMax M3 writes tool calls in its own dialect, which is what spec requirement 8 is about, so it is not reachable
during authoring. In a cell, the harness's own endpoint (`HARNESS_LLM_BASE_URL`) serves the cell's driver model
under every chat alias and `embed` as Qwen3 Embedding 8B.

- **Kernel tasks** reward `correct x clamp(ln(agent geomean speedup) / ln(oracle geomean speedup), 0, 1)` over
  undisclosed graded shapes. Speedups are measured against the task's own baseline. The oracle is a portfolio of
  strong public kernels, re-benchmarked live on the grading GPU and chosen per shape; a reward of 1.0 means matching
  it. Correctness is a gate at every shape. A submission that imports an oracle library, replays cached outputs (the
  recomputation check), calls cuBLAS's prebuilt block-scaled GEMM (`nvfp4-gemm`) or a prebuilt attention backward
  (`flash-attn-backward`) scores 0. Submitted kernels execute in a separate worker process; the trusted parent owns
  the clock, which includes one IPC round trip per timed batch.
- **PR tasks** score 1.0 or 0.0. A fixed public runner calls the submitted vLLM as the worker on the PR's regression
  cases, and the trusted scorer compares the raw results with private expectations: every beam of a
  JSON-schema-constrained beam search must be schema-valid (`pr-vllm-35022`), and a two-stage pipeline-parallel
  layout must get correct, independent per-worker KV-cache configs in a symmetric and an asymmetric case
  (`pr-vllm-33698`). The PRs' own regression tests are kept for reference in each PR task's
  `tests/sealed_test_*.py`.
- **Cell outcomes.** A cell is SCORED (judge PASS: the recomputed reward), ZERO (judge FAIL or INCONCLUSIVE, an
  ineligible verification, or a rule violation) or VOID (no score: the oracle is unavailable or wrong on this GPU,
  the driver endpoint rejected the probe, or an infrastructure failure). A void or missing cell leaves the harness
  score incomplete, never zero; a harness that runs and does badly scores a measured 0.

**Human baseline: 0.4375** (Claude Code driven by the same three models on the same five tasks; 0.438 in the paper).
The Singularity-Index's harness term is the agent's harness score divided by 0.4375.

## Development model API

During authoring, the development gateway (`breakwater/dev_gateway.py`, image `breakwater-dev-gateway:local`) serves
`chat` (GLM 5.3), `chat-b` (Kimi K3) and `embed` (Qwen3 Embedding 8B) with **5,000,000 chat tokens** and
**20,000,000 embedding tokens** per authoring run (`development_gateway` in
`tasks/coding-agent-harness/breakwater.json`). Chat is text-only, non-streaming, `n=1`, with exactly one explicit
positive `max_tokens` or `max_completion_tokens`. Function tools are supported; the gateway never executes them.
Embeddings accept text or token IDs. Unknown routes, models and unsupported fields are rejected. The spec states the
same contract to the agent.

Before contacting the provider, the gateway atomically reserves a conservative input estimate plus the requested
output maximum and persists the debit, so parallel calls cannot reserve the same allowance. HTTP 429 means the
reservation does not fit: wait for in-flight calls or lower the bound. Limits are never silently shortened. Reported
usage refunds the unused reservation; missing usage keeps it. `/health` reports available and reserved tokens. The
input estimate is not an exact tokenizer, and enforcement assumes the provider honours output bounds and reports
usage faithfully.

Only the gateway holds the provider key. Every request and response is journaled in a hash chain outside the agent's
reach, and the integrity judge reviews it together with the author trajectory. Testing the developing harness on
example tasks is permitted; asking a development model to author, debug or review the harness is a FAIL. Recreating
a familiar harness design from the author's own knowledge is permitted, and source similarity alone is not evidence
of a violation. In grid cells the same image runs in its driver role: one driver model, a journal, no token budget.

## Preparing assets

The task needs no prepared assets: the agent's workspace is the image's own `/app` (`workspace_source: image`), and
`tasks/coding-agent-harness/bindings.example.json` is empty. The sealed oracles of the kernel tasks are in the
repository, under each kernel task's `tests/` (`_fa4_oracle.tar.gz`, 0.35 MB; `_fa4_mla_oracle.tar.gz`, 0.4 MB;
`_flash_mla_oracle.tar.gz`, 5 MB, an x86_64 build for Hopper). Everything else the downstream images need (PyTorch
and FlashInfer wheels, FlashInfer's prebuilt cubins, the vLLM checkouts, TinyLlama-1.1B-Chat for `pr-vllm-35022` and
the Qwen3-0.6B config for `pr-vllm-33698`) is downloaded when the images are built; grading itself runs offline.

## Building the images

Build on the GPU host (GB300 hosts are arm64; every image builds natively). The generic steps are in
[`breakwater/README.md`](../../breakwater/README.md).

```bash
# 1. The task runtime image.
docker build -t coding-agent-harness:local tasks/coding-agent-harness/environment
# 2. The protected author image on top of it (AUTHOR_CLI is claude-code or codex).
docker build -f breakwater/author/Dockerfile --build-context task_environment=tasks/coding-agent-harness/environment \
  --build-arg TASK_BASE=coding-agent-harness:local --build-arg AUTHOR_CLI=claude-code \
  -t coding-agent-harness-author:claude-code breakwater
# 3. The task's verifier image, the shared observer and author gateway, and the development gateway.
docker build -f breakwater/Dockerfile.verifier --build-context task=tasks/coding-agent-harness \
  --build-arg TASK_BASE=coding-agent-harness:local -t coding-agent-harness-verifier:local breakwater
docker build -t breakwater-observer:local breakwater/collector
docker build -f breakwater/Dockerfile.gateway -t breakwater-author-gateway:local breakwater
docker build -f breakwater/Dockerfile.dev-gateway -t breakwater-dev-gateway:local breakwater
```

## Running the authoring stage under Breakwater

`tasks/coding-agent-harness/bindings.example.json` is empty, so it can be passed as is, with any empty directory as
the asset root. `FIREWORKS_API_KEY` is used by the integrity judge and by the development gateway (`--dev-key-env`
selects a different variable).

```bash
export FIREWORKS_API_KEY=...
python3 breakwater/run.py --task tasks/coding-agent-harness \
  --agent-image coding-agent-harness-author:claude-code --verifier-image coding-agent-harness-verifier:local \
  --observer-image breakwater-observer:local --gateway-image breakwater-author-gateway:local \
  --dev-gateway-image breakwater-dev-gateway:local \
  --run-dir RUN_DIR --asset-root ASSET_ROOT --bindings tasks/coding-agent-harness/bindings.example.json \
  --author-cli claude-code --author-model MODEL --author-upstream https://PROVIDER --author-key-env KEY_VAR \
  --gpus device=0
```

The run prints one JSON line that includes `authority_sha256`, and writes `RUN_DIR/reports/`
(`mechanical-check.json`, `judge.json`, `reward.json`).

- **Authoring.** The agent works in `/app` as described above. The protected author CLI talks only to the author
  gateway; its commands run as the worker and reach only the development gateway (`gateway:8000`) and local servers.
  When the 21,600 s budget ends, authoring stops and the workspace is graded as it is.
- **Intermediate stage (validity battery).** A fresh container from the same image, with no network, gets the sealed
  workspace. The trusted root coordinator (`breakwater/intermediate.py`, kind `harness`, which runs
  `breakwater/harness_battery.py`) runs the seven scenarios of
  `tasks/coding-agent-harness/tests/smoke/scenarios.json`. For each, a fresh worker process imports the sealed
  `/app/harness/agent.py` and calls `run()`; every model call crosses a pipe to the root process, where
  `tasks/coding-agent-harness/tests/smoke/scripted.py` answers it and records it. Root then checks the scenario
  workspace without following links and writes the raw records to `/evaluation/raw/harness`. `HARNESS_LLM_BASE_URL`
  is set but unreachable, so the harness must survive a failed utility call.
- **Final verifier.** A read-only container with no GPU recomputes the battery summary and the static `run()` check
  from the raw records, then the integrity judge (Codex with DeepSeek V4.1 Flash) applies `breakwater/checklist.md`
  and [`tasks/coding-agent-harness/tests/submission_judge.md`](tests/submission_judge.md), including an explicit
  assessment of how the development models were used.

## Running the grid

Each (task, driver) cell is its own three-container run (`breakwater/cell.py`, contract
`breakwater/cell-pipeline.json`), started by `breakwater/grid.py`. For each downstream task, build its image, the
cell overlay and the cell verifier:

```bash
for path in kernels/flash-attn-backward kernels/nvfp4-gemm kernels/mla-decode \
            pr-reproduction/pr-vllm-35022 pr-reproduction/pr-vllm-33698; do
  task=${path#*/}
  docker build -t $task:local tasks/coding-agent-harness/eval/e2e_tasks/$path/environment
  docker build -f breakwater/cell/Dockerfile.overlay --build-arg TASK_BASE=$task:local -t $task-cell:local breakwater
  docker build -f breakwater/cell/Dockerfile.verifier --build-arg TASK_BASE=$task-cell:local \
    --build-context task=tasks/coding-agent-harness/eval/e2e_tasks/$path -t $task-cell-verifier:local breakwater
done
```

List the images in a copy of `breakwater/images.example.json` (observer, driver gateway
`breakwater-dev-gateway:local`, and each task's cell and verifier image), then run the grid against the root run and
report the score. `--gpus` lists the GPU indices; cells run in parallel, one per GPU.

```bash
export FIREWORKS_API_KEY=...        # the driver gateway and the cell judges
python3 breakwater/grid.py run --harness-run RUN_DIR --harness-authority AUTHORITY_SHA256 \
  --images images.json --run-root GRID_DIR --gpus 0,1,2
python3 breakwater/grid.py report --harness-run RUN_DIR --harness-authority AUTHORITY_SHA256 --run-root GRID_DIR
```

The profile (`compact5-v1`) and the three drivers come from `tasks/coding-agent-harness/eval/config.json`;
`--profile` and `--drivers` select others, and `--wall-sec` shortens every cell's wall clock for a cheaper,
non-comparable run. The report prints every cell and the mean, and writes `GRID_DIR/grid-report.json` (`score`,
`per_driver_mean`, `cells`); while any cell is void or missing, `score` is null and `status` is `incomplete`.

Inside a cell:

1. **Cell agent container.** The downstream task image with the cell overlay, which removes any pre-installed agent
   CLI and JavaScript runtime. The sealed root harness is mounted read-only at `/harness` and driven by
   `breakwater/cell_drive.py` as the worker, with the downstream task's `instruction.md` as the task and its
   workspace (`/app` for the kernel tasks, `/app/vllm` for the PR tasks). Its only network peer is the driver
   gateway, which holds the provider key, pins the cell's driver model and journals every request; a one-token probe
   that the endpoint deterministically rejects voids the cell.
2. **Grading stage.** A fresh container from the same image. Only the workspace changes cross: files the harness
   added or modified inside the workspace, plus deletions. Kernel tasks run the task's own `tests/verify_env.py` as
   root, with the submitted kernel in a separate worker process; PR tasks run a fixed public runner as the worker
   (`breakwater/downstream/`, `breakwater/worker/`).
3. **Cell verifier.** A read-only container with no GPU checks that the overlay matches the recorded change set,
   recomputes every speedup, both geomeans and the reward from the raw timings (or rescores the PR rows), and runs
   the cell judge with `breakwater/downstream/cell_judge.md`.

## Hardware notes (GB300)

- GB300 is aarch64 with sm_103 GPUs. The authoring image and `flash-attn-backward` use CUDA 13 and PyTorch cu130 on
  every architecture; `nvfp4-gemm` and `mla-decode` use CUDA 13 and cu130 on aarch64 (CUDA 12.8 and cu128 on
  x86_64), because CUDA 13 `nvcc` targets sm_103 and the cu130 wheels carry sm_103 kernels.
- `nvfp4-gemm` is Blackwell-only: its oracle is FlashInfer's `mm_fp4`, and on older GPUs the cell is void.
- `mla-decode`'s vendored FlashMLA oracle is an x86_64 build for Hopper. On Blackwell the oracle routes are
  FlashInfer's TRT-LLM MLA decode and the pinned FA4 CuTe MLA kernel. `oracle_bench.py` checks a compiled
  extension's interpreter and architecture tag before loading it; a mismatch raises `OracleUnavailable` and voids
  the cell. The FlashInfer images bake their cubins in (`FLASHINFER_CUBIN_DIR`), so grading stays offline.
- `flash-attn-backward` pins `quack-kernels[cu13]==0.6.4` (CuTe DSL 4.6.2). On sm_103 its oracle refuses CuTe DSL
  4.6.0 and 4.6.1, whose FA4 backward can hang, and keeps the cuDNN route. The verifier writes `progress.json`
  before each major step and dumps stack traces every 300 s.

## Files

```
tasks/coding-agent-harness/
  instruction.md          what the agent is told
  task.toml               task metadata, budgets, resources and the runtime image name
  breakwater.json         the Breakwater task contract (kind harness, budgets, GPU, development gateway)
  bindings.example.json   prepared assets per stage (none for this task)
  environment/            the task runtime image: Dockerfile, agent-policy.json and the /app workspace
                          (harness/agent.py, spec/, smoke/, examples/, selfcheck.py, harness_api.py)
  tests/                  test.sh (verifier entry point; refuses outside Breakwater), smoke/ (the trusted copy
                          of the validity battery), submission_judge.md (the integrity rubric)
  eval/config.json        provider endpoint, drivers, embedding model, the compact5-v1 profile and its tasks
  eval/e2e_tasks/         the five downstream tasks: task.toml, instruction.md, environment/ (the task image)
                          and tests/ (the kernel verifiers with their sealed oracles; the PRs' regression tests)
```
