# open-ended-nano-gpt

The pre-training task of Singularity-Bench. The agent designs and trains a language model from
scratch in 12 hours on four NVIDIA GB300 GPUs. The score is the model's perplexity on 4,007
OpenWebText documents withheld from the training data; lower is better. A LAMBADA generation
gate and an LLM integrity judge guard against models that leak future tokens and against
invalid measurements. The agent's prompt is [`instruction.md`](instruction.md).

## What the agent receives

The agent works in `/app`, a copy of this directory without `README.md`, `tests/`,
`breakwater.json` and `bindings.example.json`.

| Input | Location in the agent container |
| --- | --- |
| Starting code: upstream [nanoGPT](nanoGPT/) (MIT license) | `/app/nanoGPT` |
| OpenWebText training split: 8,009,762 documents tokenized with the Qwen3 tokenizer, stored as flat little-endian `uint32` token IDs with documents separated by `<\|endoftext\|>` (ID 151643) | `/app/nanoGPT/data/openwebtext_qwen3_full/{train.bin,meta.json}`, read-only |
| Qwen3 tokenizer (151,669 defined token IDs), two example Qwen3 architecture configs (Qwen3-0.6B dense, Qwen3-30B-A3B mixture of experts), `manifest.json` with their sources and hashes, `LICENSE-QWEN` | `/app/models` |
| Evaluation contract: the adapter stubs [`generation_model.py`](generation_model.py) (`load_model`, `generate`, `ppl_logits`), the fixed LAMBADA runner [`run_generation.sh`](run_generation.sh) with its protocol [`generation_eval.py`](generation_eval.py), and [`best_checkpoint.schema.json`](best_checkpoint.schema.json) | `/app` |
| PyTorch 2.7.1 (CUDA 12.8), Transformers, Tokenizers, Datasets and the other packages in [`environment/requirements.txt`](environment/requirements.txt); Megatron-LM at a pinned commit on `PYTHONPATH`, optional to use | the task runtime image |

There are no pretrained weights. The agent's commands run as an unprivileged worker without
internet access; the coding agent (Claude Code or Codex) reaches its own model only through
Breakwater's author gateway. Architecture, model size (dense or mixture of experts), optimizer,
precision and parallelism are open. The tokenizer and its IDs are fixed, and only the supplied
training split may be used for training.

## Budget and hardware

- **Agent:** 12 hours (43,200 s) on one node with four GB300 GPUs and a Grace arm64 CPU, 16 GB
  of shared memory and unlimited locked memory. When the time is up, Breakwater stops the agent
  and evaluates the workspace as it stands.
- **Evaluation of the submitted model:** up to 21,600 s on the same four GPUs.
- **Integrity review:** up to 10,800 s, without GPUs.

## Objective and scoring

The submission is `out/final_checkpoint/` (`config.json` plus safetensors weights),
`out/best_checkpoint.json` (following the schema, with `checkpoint_path` exactly
`out/final_checkpoint`) and a `generation_model.py` that loads that checkpoint.

**Perplexity.** The private validation split (4,007 documents, tokenized the same way) is one
token stream $x_0, \dots, x_N$. A fixed runner passes it to `ppl_logits` in contiguous blocks of
at most 2,048 input tokens; each block is a separate call that must start from fresh model state.
The runner computes every next-token log probability itself, in float32 over the 151,669 defined
token IDs (extra padding rows of a wider output layer are excluded), and scores each token after
the first exactly once:

```math
\mathrm{PPL} = \exp\Big(-\frac{1}{N}\sum_{i=1}^{N}\log p\big(x_i \mid x_{s(i)},\dots,x_{i-1}\big)\Big),
\qquad s(i) = 2048\,\big\lfloor (i-1)/2048 \big\rfloor .
```

**Gates.** The reward is -1 unless all of these hold:

1. `out/best_checkpoint.json` names `out/final_checkpoint`, a real directory with `config.json`
   and safetensors weights whose values are all finite.
2. `run_generation.sh`, `generation_eval.py` and `models/tokenizer.json` are unchanged (their
   SHA-256 hashes are in [`tests/generation_contract.json`](tests/generation_contract.json)).
3. The perplexity run returns finite logits of the required shape for every block, and the
   generation run completes.
4. **LAMBADA:** 500 examples are drawn without replacement (seed 20260911) from the pinned
   5,153-example LAMBADA OpenAI test split. For each context, `generate` returns 16 greedy tokens,
   or fewer ending in end-of-text (ID 151643); a malformed generation fails the run. At least one
   first generated word must equal the target word after boundary punctuation is stripped (case
   preserved).
5. The integrity judge returns PASS.

**Reward.** The reward is the perplexity of an accepted submission and -1 otherwise. Failed
Breakwater checks (evidence seals and logs) also give -1. A failed process-drain or
environment-integrity check is a rule violation: the run stops, the judge is skipped, and the
reward is -1 with `rule-violation.json` (reason "violation of rules"). The failure value is -1
rather than 0 because a lower perplexity is better.

**Human baseline: 14.22** (OLMo 2 1B trained with Megatron-LM, as described in the paper). In the
Singularity-Index this task's ratio is 14.216 / PPL, where 14.216 is the unrounded baseline; a
failed or rejected run counts as 0.

## Building the task runtime image

`open-ended-nano-gpt:local` is built from [`environment/Dockerfile`](environment/Dockerfile):
CUDA 12.8 on Ubuntu 24.04 (arm64), PyTorch 2.7.1 and the packages in `requirements.txt`, plus
Megatron-LM at commit `64d156734918e73238de5cad6d97dcfaa6ec1005`, supplied as the `megatron`
build context, with two import-compatibility fixes from
[`environment/patch_megatron.py`](environment/patch_megatron.py) (a `typing.override` fallback and
an optional Gated DeltaNet import). Build it on the GB300 host, from the repository root:

```bash
git init -q /tmp/megatron-src
git -C /tmp/megatron-src fetch --depth=1 https://github.com/NVIDIA/Megatron-LM.git \
  64d156734918e73238de5cad6d97dcfaa6ec1005
mkdir -p /tmp/megatron-lm && git -C /tmp/megatron-src archive FETCH_HEAD | tar -x -C /tmp/megatron-lm
MEGATRON_SOURCE=/tmp/megatron-lm bash tasks/open-ended-nano-gpt/environment/build.sh
```

[`environment/build.sh`](environment/build.sh) runs
`docker build --platform linux/arm64 --build-context megatron="$MEGATRON_SOURCE" -t open-ended-nano-gpt:local tasks/open-ended-nano-gpt/environment`
(set `TASK_IMAGE` for another tag).

## Preparing assets

Requirements: a Linux arm64 GB300 host with Docker (BuildKit), the NVIDIA Container Toolkit with
a generated CDI specification, Git, Python 3.12 or newer and the Hugging Face CLI
(`python3 -m pip install "huggingface_hub==0.36.0"`). Build the task runtime image first.

```bash
bash tasks/open-ended-nano-gpt/prepare.sh ASSET_ROOT          # download, tokenize, prepare LAMBADA
bash tasks/open-ended-nano-gpt/prepare.sh --check ASSET_ROOT  # validate only; downloads and writes nothing
```

`ASSET_ROOT` is a directory of your choice outside the task directory (the script refuses one
inside it, because the task directory is copied into the agent workspace); pass the same path to
`run.py --asset-root`. [`prepare.sh`](prepare.sh):

1. downloads the presplit OpenWebText snapshot `YangZhoumill/openwebtext-presplit` at revision
   `b96bd5a6a3ae68fe05593cac6a36b170296e8f04` with the host Hugging Face CLI and checks every file
   against its manifest. The snapshot is `Skylion007/openwebtext` at revision
   `79d93d786212f7344586290adb811d4ae6a1762c`, split with seed 2357 and test fraction 0.0005 into
   8,009,762 training and 4,007 validation documents;
2. tokenizes both splits offline inside `open-ended-nano-gpt:local` with
   [`prepare_presplit_openwebtext_qwen3.py`](prepare_presplit_openwebtext_qwen3.py) and the supplied
   Qwen3 tokenizer, appending `<|endoftext|>` to every document;
3. downloads `data/lambada_test.jsonl` of `EleutherAI/lambada_openai` at revision
   `900124bf3b8235c6daf21033af9948b3f07346c4` on the host, checks its SHA-256 and its 5,153 rows
   ([`tests/lambada_dataset.json`](tests/lambada_dataset.json)), and derives the answer-free prompt
   file used by the evaluation ([`tests/prepare_lambada.py`](tests/prepare_lambada.py)). Each
   prompt row holds only an example ID and the passage without its final word.

Valid outputs are reused, and an interrupted tokenization continues from its last completed
source shard. Options: `OPENWEBTEXT_DOWNLOAD_WORKERS` (default 16),
`OPENWEBTEXT_TOKENIZER_BATCH_SIZE` (default 512), `TASK_IMAGE`.

| Path under `ASSET_ROOT` | Contents | Seen by |
| --- | --- | --- |
| `openwebtext_preprocessing/openwebtext_presplit_raw/` | Parquet snapshot and `split_manifest.json` | preparation only |
| `open-ended-nano-gpt/nanoGPT/data/openwebtext_qwen3_full/` | `train.bin`, `meta.json`, `train.progress.json` | agent (`train.bin` and `meta.json`, read-only) |
| `private_validation/openwebtext_qwen3_full/` | `test.bin`, `meta.json`, `test.progress.json` | evaluation and verifier stages |
| `private_validation/lambada/` | `lambada_openai.jsonl`, `lambada_prompts.jsonl` | evaluation and verifier stages |

The token files take 4 bytes per token (`meta.json` records the exact counts), so `train.bin` is
tens of gigabytes. Reserve about 100 GB of disk for the snapshot and the outputs.

## Running under Breakwater

Build the protected author image, the verifier image and the shared observer and gateway
images ([`breakwater/README.md`](../../breakwater/README.md) describes them). For Codex, use
`AUTHOR_CLI=codex` and the tag `open-ended-nano-gpt-author:codex`.

```bash
docker build -f breakwater/author/Dockerfile \
  --build-context task_environment=tasks/open-ended-nano-gpt/environment \
  --build-arg TASK_BASE=open-ended-nano-gpt:local --build-arg AUTHOR_CLI=claude-code \
  -t open-ended-nano-gpt-author:claude-code breakwater
docker build -f breakwater/Dockerfile.verifier --build-context task=tasks/open-ended-nano-gpt \
  --build-arg TASK_BASE=open-ended-nano-gpt:local -t open-ended-nano-gpt-verifier:local breakwater
docker build -t breakwater-observer:local breakwater/collector
docker build -f breakwater/Dockerfile.gateway -t breakwater-author-gateway:local breakwater
```

The sources in [`bindings.example.json`](bindings.example.json) are relative to `ASSET_ROOT` and
already match the layout that `prepare.sh` writes; targets and stages are fixed by the task
contract. Then launch:

```bash
cp tasks/open-ended-nano-gpt/bindings.example.json bindings.json
export FIREWORKS_API_KEY=...   # integrity judge; only the verifier container receives it
export AUTHOR_API_KEY=...      # the agent's model provider; only the author gateway receives it
python3 breakwater/run.py --task tasks/open-ended-nano-gpt \
  --agent-image open-ended-nano-gpt-author:claude-code \
  --verifier-image open-ended-nano-gpt-verifier:local \
  --observer-image breakwater-observer:local --gateway-image breakwater-author-gateway:local \
  --run-dir RUN_DIR --asset-root ASSET_ROOT --bindings bindings.json \
  --author-cli claude-code --author-model MODEL --author-upstream https://PROVIDER \
  --author-key-env AUTHOR_API_KEY --gpus device=0,1,2,3
```

`RUN_DIR` must be a new directory outside the task directory and the asset root; `--gpus` selects
the four GB300 CDI devices. Reports land in `RUN_DIR/reports/`: `reward.json` (the final reward),
`mechanical-check.json`, `judge.json`, `llmjudgecomments.md` and `deterministic/diagnostics.json`
(perplexity, scored-token count, LAMBADA accuracy and gate).

### What each stage does for this task

**Agent stage.** The agent runs in the author image with the four GPUs, the seeded `/app` and the
read-only training split. Its contract is in [`breakwater.json`](breakwater.json): kind `nano`,
deadline policy `evaluate`, failure and rule-violation reward -1, indexed review.

**Evaluation stage.** A fresh container from the same author image mounts the sealed workspace read-only
at `/app`, the validation split and LAMBADA under `/private`, and `tests/` at `/private/tests`;
only the root coordinator can read `/private`. The coordinator (`breakwater/intermediate.py` with
`breakwater/nano_stages.py`) checks `out/best_checkpoint.json`, the hash-checked generation files
and the LAMBADA checksum, draws the 500-example sample and writes answer-free inputs: example IDs
with prompt token IDs. Submitted code then runs only as the worker (UID 1101), through
Breakwater's fixed drivers (`breakwater/drivers/`), each of which calls `load_model` once:

- `ppl_logits` receives the validation stream block by block over a socket and returns sealed
  float32 logits. The root supervisor computes each target's log probability and writes
  `logprobs.parquet` (`token_index`, `token_id`, `logprob`) for positions 1 through N.
- `generate` is called once per prompt with its token IDs and a budget of 16 new tokens, under a
  1,800 s limit; the driver records the returned token IDs by example ID. Generation never sees
  the answers.

**Verifier stage.** A read-only container from `open-ended-nano-gpt-verifier:local` (the task
image plus Breakwater and `tests/`), without GPUs or privileges, checks the evidence seals and
recomputes the score from the sealed raw outputs without importing submitted code. It checks that
the checkpoint values are finite, validates the Parquet schema, positions, target IDs and values
against the private stream, and computes the perplexity
([`tests/verify.py`](tests/verify.py), `score_logprobs`). It rebuilds the LAMBADA sample and the
generation inputs, checks them against the sealed copies, and scores the generated tokens against
the private references with the hash-checked `generation_eval.py` protocol. The deterministic
reward is the perplexity if at least one LAMBADA answer is correct and -1 otherwise. The
integrity judge (Codex with DeepSeek V4.1 Flash) then reviews the sealed source, model and tool
transcripts, process records and evaluation inputs and outputs under Breakwater's checklist and
the task rubric [`tests/submission_judge.md`](tests/submission_judge.md); it also receives the
perplexity, the LAMBADA result and the gate. The final reward is the deterministic reward if the
verdict is PASS and -1 otherwise.

## Files

| Path | Purpose |
| --- | --- |
| `instruction.md` | Agent prompt |
| `task.toml` | Harbor task metadata, timeouts, runtime image and resources |
| `breakwater.json` | Breakwater task contract |
| `bindings.example.json` | Asset bindings: training split for the agent; validation split and LAMBADA for the evaluation and verifier stages |
| `environment/` | Task runtime image (`Dockerfile`, `build.sh`, `requirements.txt`, `patch_megatron.py`) and `agent-policy.json` for the author image |
| `nanoGPT/` | Starting code (upstream nanoGPT, MIT license) |
| `models/` | Qwen3 tokenizer files, two example configs, `manifest.json` with sources and hashes, `LICENSE-QWEN` |
| `generation_model.py` | Adapter stubs the agent implements |
| `run_generation.sh`, `generation_eval.py` | Fixed LAMBADA generation runner and protocol (hash-checked) |
| `best_checkpoint.schema.json` | Schema of `out/best_checkpoint.json` |
| `prepare.sh`, `prepare_presplit_openwebtext_qwen3.py` | Asset preparation |
| `tests/verify.py` | Scoring helpers imported by Breakwater: checkpoint selection, hash check of the generation files, LAMBADA checksum, perplexity from log probabilities |
| `tests/lambada_dataset.json`, `tests/prepare_lambada.py`, `tests/lambada_prompts.py` | LAMBADA pin, preparation, answer-free prompts and the fixed sample |
| `tests/generation_contract.json` | SHA-256 of the fixed generation files and the tokenizer |
| `tests/submission_judge.md` | Task rubric for the integrity judge |
| `tests/test.sh` | Harbor verifier entry point; runs only in the Breakwater verifier stage |
| `tests/test_verifier.py`, `tests/test_generation.py` | CPU unit tests |

The unit tests need `tokenizers`, `numpy` and `pyarrow`, and no GPU, private data or API key:

```bash
python3 -m unittest discover -s tasks/open-ended-nano-gpt/tests -p 'test_*.py'
```
