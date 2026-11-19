# Singularity-Bench: Can Agents Build Agents?

<div align="center">

Yang Zhou<sup>\*,1</sup>, Krish Agarwal<sup>\*,1</sup>, Zhuoming Chen<sup>\*,1</sup>, Zerui Wang<sup>2</sup>, Zhenyu Zhang<sup>3</sup>, Wenshuang Li<sup>4</sup>, Xinpeng Wei<sup>5</sup>, Zhichen Zeng<sup>6</sup>, Jingyu Liu<sup>7</sup>, Jue Wang<sup>8</sup>, Sen Wu<sup>9</sup>, Ziheng Jiang<sup>10</sup>, Zhen Dong<sup>2,11</sup>, Han Cai<sup>2</sup>, Binhang Yuan<sup>4</sup>, Ying Sheng<sup>6</sup>, Ce Zhang<sup>7,8</sup>, Bo Li<sup>12</sup>, Beidi Chen<sup>1</sup>

<sup>1</sup> Carnegie Mellon University &nbsp;&nbsp; <sup>2</sup> NVIDIA &nbsp;&nbsp; <sup>3</sup> The University of Texas at Austin &nbsp;&nbsp; <sup>4</sup> HKUST &nbsp;&nbsp; <sup>5</sup> Amazon AGI &nbsp;&nbsp; <sup>6</sup> RadixArk &nbsp;&nbsp; <sup>7</sup> University of Chicago &nbsp;&nbsp; <sup>8</sup> Together AI &nbsp;&nbsp; <sup>9</sup> Alation &nbsp;&nbsp; <sup>10</sup> Meta &nbsp;&nbsp; <sup>11</sup> University of California, Santa Barbara &nbsp;&nbsp; <sup>12</sup> University of Illinois Urbana-Champaign

<sup>\*</sup> Equal contribution

</div>

<div align="center">
[<a href="https://arxiv.org/abs/2611.16591">Paper</a>] | [<a href="https://infini-ai-lab.github.io/singularity-bench/">Website</a>]
</div>
<br>

----------

Singularity-Bench is an open-ended benchmark for autonomous AI development: can frontier agents build the systems that
produce the next generation of agents? Its six end-to-end tasks span pre-training, mid-training, post-training,
inference, the agent harness and cluster orchestration. Each task gives the agent a minimal starting point, a fixed time
budget and a single objective on NVIDIA GB300 GPUs, leaves the core design decisions to the agent, and compares the
result with a mature human-engineered baseline on identical hardware. Every task runs under Breakwater, our extension of
Harbor that blocks all the reward-hacking attacks tested in our red-teaming study. Across 11 agents built from 7
frontier models and 2 harnesses, **Opus 5.5 with Claude Code matches or outperforms the human baseline on all six tasks
with a Singularity-Index of 1.25**.

<p align="center">
  <img src="assets/lifecycle.gif" alt="The agent development lifecycle: each agent generation builds its successor through pre-training, mid-training, post-training, inference, cluster orchestration and the agent harness" width="100%">
</p>

## Quick Start

You need Linux hosts with NVIDIA GB300 GPUs (four per node; cluster-serverless uses four nodes), Python 3.11+, a
provider key for the agent's model, and a Fireworks key for the integrity judge (DeepSeek V4.1 Flash). Breakwater runs
every task on either of two backends:

- **Apptainer** (the default), for GPU clusters with Slurm and no Docker, such as most GB300 clusters: Apptainer 1.3
  or newer (a user-level, non-setuid install suffices), no root. rl-swe-agent's training and evaluation sandboxes also
  need rootless podman.
- **Docker**, on a host with a Docker daemon: Docker with BuildKit and the NVIDIA Container Toolkit with a CDI
  specification. It isolates the agent and its commands as separate users.

Each task README lists the assets it downloads and the images it builds. Run everything from the repository root
(`pip install -e .` also provides the commands as `breakwater ...`):

1. Prepare the task's assets and build its runtime image (`tasks/<name>/README.md`): for Apptainer with
   `tasks/<name>/environment/build-apptainer.sh` (the preparation scripts then take `PREPARE_RUNTIME=apptainer`), or
   with Docker.
2. Apptainer: install the pinned agent CLI and judge runner once (`python3 -m breakwater install-cli --dest CLI_DIR`,
   `python3 -m breakwater install-judge --dest JUDGE_DIR`). Docker: build the protected author image for your agent
   CLI (Claude Code or Codex), the task's verifier image and the shared observer and gateway images
   ([breakwater/README.md](breakwater/README.md#on-docker)).
3. Launch, with the asset root that holds the prepared assets:
   ```bash
   # Apptainer (the default backend)
   python3 -m breakwater run --task tasks/<name> --image <name>.sif --cli-dir CLI_DIR --judge-dir JUDGE_DIR ... \
     --asset-root ASSET_ROOT --run-dir RUN_DIR
   # Docker
   python3 -m breakwater run --backend docker --task tasks/<name> --agent-image <name>-author:claude-code ... \
     --asset-root ASSET_ROOT --run-dir RUN_DIR
   ```

Reports land in `RUN_DIR/reports/`: the mechanical checks, the integrity verdict and the final `reward.json`. Two tasks
add a scoring step after the run: coding-agent-harness grades the sealed harness on its grid of downstream tasks and
driver models (`tasks/coding-agent-harness/grid/`), and rl-swe-agent's held-out suites are scored on the host by
`tasks/rl-swe-agent/evaluation/`, which keeps the score only if the integrity review passed.

## Tasks

| Task | Phase | The agent builds | Budget and hardware | Metric | Human baseline |
|---|---|---|---|---|---|
| [open-ended-nano-gpt](tasks/open-ended-nano-gpt/README.md) | Pre-training | a language model, designed and trained from scratch | 12 h, 4x GB300 | validation perplexity on a private set (lower is better) | 14.22 (Megatron-LM, OLMo 2 recipe) |
| [midtrain-for-rl](tasks/midtrain-for-rl/README.md) | Mid-training | a mid-trained Llama-3.2-1B-Base that gains the most from a fixed RL stage | 12 h, 4x GB300 | mean post-RL accuracy | 0.175 (Megatron-LM, PRISM mixture) |
| [rl-swe-agent](tasks/rl-swe-agent/README.md) | Post-training (RL only) | a coding agent trained from GLM-4.7-Flash with multi-turn RL | 12 h, 4x GB300 | mean resolve rate (dev tasks + SWE-bench Verified) | 0.539 (Miles, Harbor RL recipe) |
| [serve-glm53-flash](tasks/serve-glm53-flash/README.md) | Inference engine | an FP8 serving engine for GLM-5.3-Flash under coding-agent traffic | 12 h, 4x GB300 | simulated income (USD per GPU-hour) | 8.18 (SGLang) |
| [coding-agent-harness](tasks/coding-agent-harness/README.md) | Agent harness | a coding harness for kernel generation and bug fixing, standard library only | 6 h, 1x GB300 | mean downstream task score | 0.438 (Claude Code) |
| [cluster-serverless](tasks/cluster-serverless/README.md) | Cluster management | a serverless inference orchestrator for 28 models | 4 h, 16x GB300 on four nodes | SLO attainment + completion | 0.460 (Dynamo with KAI Scheduler) |

## Singularity-Index

An agent's Singularity-Index is the mean, over the six tasks, of its score relative to the human baseline. For
pre-training the perplexity ratio is inverted, because lower perplexity is better:

```
I = ( PPL_human / PPL_agent  +  Acc_agent / Acc_human  +  Resolve_agent / Resolve_human
      + Income_agent / Income_human  +  Harness_agent / Harness_human  +  SLO_agent / SLO_human ) / 6
```

The baseline inputs are 14.216, 0.175, 0.5388, 8.18, 0.4375 and 0.4599, in the order of the table. If an agent fails a
task or produces no valid score, that task's term is zero, and the sum is still divided by six. An index of 1 matches
the human references on average; ratios are not capped, so agents can score above the baselines.

## Breakwater

Every task runs under [Breakwater](breakwater/README.md), the benchmark's evaluation and safety framework. It extends
[Harbor](https://github.com/harbor-framework/harbor), whose task format the tasks use, with five safeguards for
open-ended tasks whose evaluation may run the agent's own code:

1. **Environment fingerprinting**: the runtime's binaries, libraries and packages are inventoried before the agent
   phase and checked after it.
2. **API routing**: a gateway forwards only the agent's assigned model and blocks every other model access.
3. **Process wall**: every live process is stopped when the agent phase ends, before anything else starts.
4. **External journal**: the agent's activity and model traffic are recorded outside its reach; it can neither read
   nor write the record.
5. **Restricted intermediate stage**: a fresh, unprivileged container runs the submission before the read-only
   verifier recomputes the score and the LLM integrity judge reviews the evidence.

Breakwater is a standalone framework: any task written to its specification runs under it.
[breakwater/WRITING_TASKS.md](breakwater/WRITING_TASKS.md) shows how to write one, starting from the CPU-only
[examples/word-count](examples/word-count), and [breakwater/SPEC.md](breakwater/SPEC.md) is the full specification.

Each of the six tasks declares its contract in the `[breakwater]` tables of its `task.toml` (`python3 -m breakwater
check tasks/<name>` prints it), and its README describes its evaluation in full. Beyond a workspace, an evaluation
stage and a scorer, they use:

| Task | Also uses |
|---|---|
| open-ended-nano-gpt | trusted runners for the submitted generation adapter; a failure reward of -1, since lower perplexity is better |
| midtrain-for-rl | an operator's code-execution service that only the evaluation reaches; a checked record of the training data |
| rl-swe-agent | an endpoint (the operator's Harbor sandbox worker) and inbound ports for its callbacks; a score deferred to a host evaluator |
| serve-glm53-flash | a submitted server, graded from a fresh writable copy of the submission |
| coding-agent-harness | budgeted model access for the agent's software; protected files; its own launcher for the downstream grid |
| cluster-serverless | a remote GPU fleet: operator inputs and secrets, host hooks, and endpoints for each sandbox |

## Repository layout

```
breakwater/                     the evaluation and safety framework (one copy, used by every task)
  apptainer/                    its Apptainer backend
examples/word-count/            a minimal CPU-only task to start your own from
tasks/
  open-ended-nano-gpt/          pre-training
  midtrain-for-rl/              mid-training
  rl-swe-agent/                 post-training
  serve-glm53-flash/            inference engine
  coding-agent-harness/         agent harness
  cluster-serverless/           cluster management
```

Each task folder holds `instruction.md` (what the agent sees), `task.toml` (a Harbor task with the Breakwater contract
in its `[breakwater]` tables), `environment/` (the agent's runtime: its Dockerfile and Apptainer definition), `tests/`
(the task's trusted code and grading), its asset preparation scripts and a README with the full protocol.

The tasks' own tests live with them (`tasks/<name>/tests/unit/`, and the coding-agent-harness grid's in
`tasks/coding-agent-harness/grid/tests/`); the repository's `tests/` checks every task's contract against the spec:

```bash
python3 -m pytest tests examples/word-count/tests/unit tasks/cluster-serverless/tests/unit \
  tasks/coding-agent-harness/tests/unit tasks/coding-agent-harness/grid/tests tasks/serve-glm53-flash/tests/unit
```

rl-swe-agent's unit tests (`tasks/rl-swe-agent/tests/unit`) are described in its README; some need torch. The
framework's own tests are described in [breakwater/README.md](breakwater/README.md#tests).

## Acknowledgements

Breakwater extends [Harbor](https://github.com/harbor-framework/harbor), whose task format the tasks use. The
pre-training task starts from [nanoGPT](https://github.com/karpathy/nanoGPT) and includes
[Megatron-LM](https://github.com/NVIDIA/Megatron-LM) in its runtime, and the mid-training evaluation uses a pinned
snapshot of [slime](https://github.com/THUDM/slime).

## License

This repository is released under the [MIT License](LICENSE). Bundled third-party code and files keep their own
licenses: nanoGPT ([LICENSE](tasks/open-ended-nano-gpt/nanoGPT/LICENSE)), the Qwen tokenizer and model configurations
([LICENSE-QWEN](tasks/open-ended-nano-gpt/models/LICENSE-QWEN)) and slime
([LICENSE](tasks/midtrain-for-rl/tests/rl_eval/vendor/slime/LICENSE)).

## Citation
```
@misc{zhou2026singularitybenchagentsbuildagents,
      title={Singularity-Bench: Can Agents Build Agents?}, 
      author={Yang Zhou and Krish Agarwal and Zhuoming Chen and Zerui Wang and Zhenyu Zhang and Wenshuang Li and Xinpeng Wei and Zhichen Zeng and Jingyu Liu and Jue Wang and Sen Wu and Ziheng Jiang and Zhen Dong and Han Cai and Binhang Yuan and Ying Sheng and Ce Zhang and Bo Li and Beidi Chen},
      year={2026},
      eprint={2611.16591},
      archivePrefix={arXiv},
      primaryClass={cs.AI},
      url={https://arxiv.org/abs/2611.16591}, 
}
```
