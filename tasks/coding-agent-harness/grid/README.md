# The coding-agent-harness grid

The grid scores a submitted harness by running it as the agent on the downstream tasks in
[`../eval/e2e_tasks/`](../eval/e2e_tasks/), with the driver models of [`../eval/config.json`](../eval/config.json).
Only a root run of the harness task whose own verification passed (reward 1.0, verdict PASS) is graded. Each
(task, driver) cell is its own three-stage Breakwater run, on either backend; the harness score is the mean over
drivers of the mean over tasks.

| File | What it is |
|---|---|
| [`grid.py`](grid.py) | runs every cell (`run`) and reports the score (`report`) |
| [`cell.py`](cell.py), [`apptainer_cell.py`](apptainer_cell.py) | one cell on Docker, one cell on Apptainer |
| [`cell-pipeline.json`](cell-pipeline.json) | the cell's contract (budgets, evidence roles) |
| [`cell_drive.py`](cell_drive.py), [`cell_setup.py`](cell_setup.py) | the trusted driver of the sealed harness inside the cell |
| [`cell_intermediate.py`](cell_intermediate.py), [`cell_observed.py`](cell_observed.py) | the cell's grading coordinator |
| [`verify/tests/`](verify/tests/), [`verification.py`](verification.py) | the cell's verification task: the task code Breakwater's verifier runs for a cell (scorer, overlay checks, review views) and the cell judge's rubric |
| [`downstream/`](downstream/) | per-task grading adapters and the data-only cell scorer |
| [`worker/`](worker/) | fixed public runners for submitted kernels and vLLM code |
| [`images/`](images/) | the cell overlay (Docker and Apptainer) and the Docker cell verifier |
| [`images.example.json`](images.example.json), [`images.apptainer.example.json`](images.apptainer.example.json) | the images a grid run uses |
| [`tests/`](tests/) | the grid's tests |

## Building the cell images

Docker: for each downstream task, build its image, the cell overlay and the cell verifier. The overlay and the
verifier use Breakwater's directory as their build context; the verifier also takes this directory and the task's.

```bash
for path in kernels/flash-attn-backward kernels/nvfp4-gemm kernels/mla-decode \
            pr-reproduction/pr-vllm-35022 pr-reproduction/pr-vllm-33698; do
  task=${path#*/}
  docker build -t $task:local tasks/coding-agent-harness/eval/e2e_tasks/$path/environment
  docker build -f tasks/coding-agent-harness/grid/images/Dockerfile.overlay --build-arg TASK_BASE=$task:local \
    -t $task-cell:local breakwater
  docker build -f tasks/coding-agent-harness/grid/images/Dockerfile.verifier --build-arg TASK_BASE=$task-cell:local \
    --build-context grid=tasks/coding-agent-harness/grid \
    --build-context task=tasks/coding-agent-harness/eval/e2e_tasks/$path -t $task-cell-verifier:local breakwater
done
```

Apptainer: build each downstream task's SIF with its `environment/build-apptainer.sh`, then the cell image from it
with the overlay, from the repository root. Build on a GPU node with `APPTAINER_TMPDIR` on local disk: the PR images
pick their PyTorch wheels from the build host's NVIDIA driver, as their Docker builds do, and would get CPU-only
wheels without one.

```bash
for path in kernels/flash-attn-backward kernels/nvfp4-gemm kernels/mla-decode \
            pr-reproduction/pr-vllm-35022 pr-reproduction/pr-vllm-33698; do
  task=${path#*/}
  TASK_SIF=$SIFS/$task.sif tasks/coding-agent-harness/eval/e2e_tasks/$path/environment/build-apptainer.sh
  apptainer build --fakeroot --build-arg TASK_SIF=$SIFS/$task.sif $SIFS/$task-cell.sif \
    tasks/coding-agent-harness/grid/images/apptainer.def
done
```

## Running the grid

List the images in a copy of [`images.example.json`](images.example.json) (Docker: the observer, the driver gateway
`breakwater-dev-gateway:local`, and each task's cell and verifier image) or of
[`images.apptainer.example.json`](images.apptainer.example.json) (Apptainer: each task's cell SIF). Then run the grid
against the root run and report the score. `--gpus` lists the GPU indices; cells run in parallel, one per GPU.

```bash
export FIREWORKS_API_KEY=...        # the driver gateway and the cell judges
python3 tasks/coding-agent-harness/grid/grid.py run --harness-run RUN_DIR --harness-authority AUTHORITY_SHA256 \
  --images images.json --run-root GRID_DIR --gpus 0,1,2
python3 tasks/coding-agent-harness/grid/grid.py report --harness-run RUN_DIR --harness-authority AUTHORITY_SHA256 \
  --run-root GRID_DIR
```

On Apptainer, install the judge runner (`python3 -m breakwater install-judge --dest JUDGE_DIR`) and add
`--backend apptainer --judge-dir JUDGE_DIR` to `run`. Docker gives each cell only its GPU's device; Apptainer exposes
every GPU of the node and assigns the cell its GPU with `CUDA_VISIBLE_DEVICES`. To confine each cell to its GPU on
Slurm, run each cell as its own one-GPU job instead (`apptainer_cell.py` takes the GPU Slurm assigns), writing each
run directory where `grid.py run` would put it (`GRID_DIR/<task>__<driver>/attempt-N`), then run `grid.py report`:

```bash
sbatch --gpus=1 --wrap "python3 tasks/coding-agent-harness/grid/apptainer_cell.py --task nvfp4-gemm --driver glm53 \
  --harness-run RUN_DIR --harness-authority AUTHORITY_SHA256 --image /path/to/nvfp4-gemm-cell.sif \
  --judge-dir JUDGE_DIR --run-dir GRID_DIR/nvfp4-gemm__glm53/attempt-1"
```

The profile (`compact5-v1`) and the three drivers come from [`../eval/config.json`](../eval/config.json); `--profile`
and `--drivers` select others, `--tasks` runs only some of the profile's tasks (the report still covers all of them),
and `--wall-sec` shortens every cell's wall clock for a cheaper, non-comparable run. The report prints every cell and
the mean, and writes `GRID_DIR/grid-report.json` (`score`, `per_driver_mean`, `cells`). A cell is SCORED (judge
PASS: its recomputed reward), ZERO (judge FAIL or INCONCLUSIVE, or a rule violation) or VOID (no score: the oracle is
unavailable, the driver endpoint refused, or an infrastructure failure); while any cell is void or missing, `score`
is null and `status` is `incomplete`.

## Inside a cell

1. **Authoring.** The downstream task image with the cell overlay, which removes any pre-installed agent CLI and
   JavaScript runtime. The sealed root harness is mounted read-only at `/harness` and driven by
   [`cell_drive.py`](cell_drive.py) as the worker, with the downstream task's `instruction.md` as the task and its
   workspace (`/app` for the kernel tasks, `/app/vllm` for the PR tasks). Its only network peer is the driver gateway
   (Breakwater's development gateway in its driver role), which holds the provider key, pins the cell's driver model
   and journals every request; a one-token probe that the endpoint deterministically rejects voids the cell.
2. **Grading.** A fresh sandbox from the same image. Only the workspace changes cross: files the harness added or
   modified inside the workspace, plus deletions. Kernel tasks run the task's own `tests/verify_env.py` as root, with
   the submitted kernel in a separate worker process; PR tasks run a fixed public runner as the worker
   ([`downstream/`](downstream/), [`worker/`](worker/)).
3. **Verification.** Breakwater's verifier, read-only and without a GPU, with the cell's verification task
   ([`verify/tests/breakwater_task.py`](verify/tests/breakwater_task.py)) as its task code and the cell record sealed
   as the run's task metadata. Its mechanical facts check that the overlay matches the recorded change set; its scorer
   recomputes every speedup, both geomeans and the reward from the raw timings (or rescores the PR rows) with
   [`downstream/cell_score.py`](downstream/cell_score.py), keeping the recomputation in
   `reports/deterministic/cell-score.json`; the cell judge applies
   [`verify/tests/submission_judge.md`](verify/tests/submission_judge.md). A cell the grader could not measure is void
   (`reports/void.json`).

## Tests

```bash
python3 -m pytest tasks/coding-agent-harness/grid/tests/test_grid.py    # scoring, overlays, attempts; no GPU or Docker
BREAKWATER_APPTAINER_IMAGE=IMAGE.sif python3 -m unittest discover -s tasks/coding-agent-harness/grid/tests \
  -p test_apptainer_cell.py                                               # one Apptainer cell end to end, toy task
```
