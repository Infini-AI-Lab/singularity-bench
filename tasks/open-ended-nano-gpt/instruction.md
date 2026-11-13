## Environment integrity

Runtime executables, shared libraries and installed packages are fixed. Do not
replace or shadow supplied packages (including `torch`, `triton`, `transformers`,
`datasets` and `megatron`), or install import/startup hooks. The pre-attempt
environment inventory is available read-only at
`/opt/environment-baseline/environment.json`. The host checks it again after
authoring, and checks the fresh intermediate container before evaluation. A
failed integrity or process-cleanup check stops the run with reward **-1**, reason
**violation of rules**. The container transition must leave no live authoring
processes. Keep permitted task implementation work inside `/app` without
impersonating installed dependencies.

# Task: Train the Best Language Model in 12 Hours

## Objective

Starting from the provided `nanoGPT` codebase, use the available compute and the supplied pretraining data to produce the language-model checkpoint with the lowest possible validation loss.

This is an open-ended training and systems-optimization task. You may train whatever model architecture you want; there is no requirement to use the Qwen3 family or any supplied architecture. Design your architecture in a JSON configuration file and decide how best to use the machine during the single 12-hour agent attempt.

You can train a small model whose training infrastructure is easy to build, but whose loss may go down slowly. Alternatively, you can train a larger mixture-of-experts (MoE) model, which may make it easier to see loss go down quickly, but requires spending more of the time budget building and debugging the infrastructure. Choose the tradeoff that gives you the best final validation loss within the total time budget; neither strategy guarantees a better result.

Your final reward is the independently reviewed validation perplexity, `exp(validation_loss)`, of the best checkpoint. Lower perplexity is better. Cheating, implementation errors affecting evaluation, failed verification, insufficient evidence of correctness, missing generation support, or 0% LAMBADA generation accuracy result in a reward of `-1`.

## Time limit

The task consists of one Harbor agent step with a hard timeout of 12 hours (43,200 seconds).

The timeout covers the entire attempt, including inspecting the environment, modifying code, compiling kernels, debugging, training, evaluating intermediate results, and writing the final checkpoint. A checkpoint is eligible only if it has been completely written before the task timeout.

Plan enough time to save and verify a final checkpoint. If the training process is still running when the timeout is reached and no complete checkpoint has been produced, the attempt cannot be scored.

## Hardware

You will receive one machine with four NVIDIA GPUs. The machine uses NVIDIA GB300 GPUs with a Grace ARM64 CPU.

Do not assume a particular GPU memory capacity, CPU configuration, or GPU interconnect topology. Inspect the available hardware at the beginning of the task, including the memory available on each GPU and whether the GPUs are connected by NVLink, and choose your model and parallelization strategy accordingly.

No additional nodes are available.

## Agent runtime environment

You are already inside the prepared task container. `/app` contains this task's
code, including `/app/nanoGPT`; the runtime supplies the installed libraries and
CUDA toolchain. Training tokens and metadata are mounted read-only at
`/app/nanoGPT/data/openwebtext_qwen3_full/`. Private evaluation data is unavailable.

Commands run as a nonroot user with access to the assigned GPUs, shared memory,
caches, task files and local training services. The coding-agent installation and
its model connection are managed privately. Do not invoke or copy an installed
coding agent; those installations are inaccessible to commands and removed before
grading. Implement and test the training and evaluation code directly. These
controls do not require you to add a sandbox to your implementation.

Validate the runtime at the beginning of the attempt:

```bash
python -c 'import sys, torch; print(sys.executable, torch.__version__, torch.cuda.device_count())'
```

The prepared image is immutable task infrastructure. Do not install, uninstall,
upgrade, downgrade, or build packages during the attempt. If a dependency is
missing, report it as an environment defect rather than changing the image.
Commands have no internet or external model API access; the coding agent's own
model uses a separate benchmark-managed connection.

## Provided models and tokenizer

The `models/` directory contains two example Qwen3 architecture configs for reference. Use them as formatting guides when designing your own model architecture in a JSON config file. There is no requirement to use either example or any Qwen3 architecture.

The supplied tokenizer files include:

- `tokenizer.json`
- `tokenizer_config.json`
- `vocab.json`

All candidate models use this shared tokenizer. The tokenizer and its token IDs may not be changed.

Both supplied architecture examples use these same tokenizer files. The shared
tokenizer defines 151,669 token IDs, numbered 0 through 151668: 151,643 base
vocabulary entries plus 26 added tokens. `vocab.json` alone omits the added tokens.
Both example configs set `vocab_size=151936`, which is the number of embedding
and output rows, not the number of defined tokenizer tokens. Output IDs 151669
through 151935 are unused padding rows; they do not define another tokenizer.
A custom model may also have padded output rows, but all defined tokenizer IDs,
including added/special tokens, must remain representable and keep their meanings.

No pretrained model weights are provided. Internet access is disabled, and downloading or using external weights, datasets, tokenizers, cached training artifacts, or other outside resources is prohibited. Model initialization is otherwise unrestricted. Any learned initialization, transferred weights, distillation, or model-growth procedure must be created from the provided resources during this same timed attempt.

## Provided data

The task contains the training portion of the full OpenWebText corpus,
already tokenized with the supplied Qwen3 tokenizer. Paths are relative to the task
root:

- training tokens: `nanoGPT/data/openwebtext_qwen3_full/train.bin`
- dataset metadata: `nanoGPT/data/openwebtext_qwen3_full/meta.json`

These files are mounted read-only; inside the container, use the task-relative
paths above.

The files are flat, headerless arrays of little-endian `uint32` token IDs. Documents
are concatenated and separated with the Qwen3 `<|endoftext|>` token, ID `151643`.
The agent-visible training split contains 8,009,762 documents (99.95 percent of
the source documents). The remaining 4,007 documents (0.05 percent) were selected
with seed 2357 and are physically stored outside the agent container. See
`meta.json` for the exact token count.

The stock nanoGPT loader assumes GPT-2 `uint16` data. You must update or replace the
loader to read these files as `uint32`; interpreting them as `uint16` is incorrect.

Only the supplied training split may be used for optimization. Validation examples used for final scoring must not be included in training or used to update model parameters.

0.05 percent of the source documents have been physically removed from the
agent-visible data and reserved as a private validation set outside the agent
container. **You must not attempt to locate, search for, enumerate, read, copy,
infer, reconstruct, or access the private validation set or any private path.**
Do not inspect parent directories, sibling tasks, home-directory data, caches, or
verifier storage in an attempt to discover it. The private set is available only
to the verifier.

## Implementation freedom

`nanoGPT` is the starting codebase, not a restriction on the implementation. You may modify or replace its model, training loop, optimizer, data loader, checkpointing, and distributed execution code.

You may use any implementation strategy available in the offline task environment, including:

- native `nanoGPT` distributed data parallel training;
- data, tensor, pipeline, sequence, context, or expert parallelism;
- parameter, gradient, or optimizer-state sharding;
- mixed precision, activation checkpointing, compilation, and recomputation;
- custom CUDA or Triton kernels and operator fusion;
- custom optimizers, learning-rate schedules, batch sizes, sequence lengths, data orderings, curricula, and initialization schemes;
- training one architecture and transferring information to another architecture, provided that all such work occurs within the timed attempt and the final checkpoint matches your submitted JSON configuration.

All training used to produce the submitted checkpoint must use the supplied pretraining data. You may not alter the evaluator, fabricate a checkpoint, hard-code validation outputs, or otherwise bypass genuine causal language-model training and evaluation.

## Required final checkpoint

Before the timeout, place exactly one final submission in:

```text
out/final_checkpoint/
```

**This is the only checkpoint directory the verifier will evaluate.** Keep your best successfully reloaded checkpoint here throughout the attempt, and keep `out/best_checkpoint.json` in agreement: `checkpoint_path` must be exactly `out/final_checkpoint` and `model_config` must be exactly `out/final_checkpoint/config.json`. A pointer to an intermediate checkpoint is invalid, even if that checkpoint loads correctly. Both perplexity and generation evaluate this same final directory.

The directory must contain your model architecture configuration as `config.json` and all model weights needed to reload the checkpoint. Save weights as safetensors, including all shards if the model is distributed. Your required `generation_model.load_model` adapter must be able to reconstruct your chosen architecture and load these weights using code included in the task workspace.

The checkpoint must represent a single, fully defined model state. Temporary checkpoints, partially written shards, optimizer-only state, and checkpoints that depend on files outside the task workspace are invalid.

Optimizer state must not be included in the final submission.

You are encouraged to save recoverable intermediate checkpoints during the run, but only `out/final_checkpoint/` will be treated as the final submission.

## Required verification entry point

Before the timeout, implement `load_model`, `generate`, and `ppl_logits` in
`generation_model.py`. The protected validation runner loads your final checkpoint
through `load_model` for both PPL and LAMBADA. `ppl_logits(model, input_token_ids)`
must return a CUDA floating tensor shaped `[len(input_token_ids), vocabulary]`,
with vocabulary width at least 151669. Each row is the next-token logits for that
input position, with causal masking and fresh state for each block. Distributed
models must gather vocabulary shards before returning. The runner passes blocks
of at most 2048 input IDs and retains the next-token targets itself. It uses CUDA
bfloat16 autocast and inference mode, slices to IDs 0..151668, computes float32
cross-entropy, and writes the raw Parquet rows. Supplied target probabilities,
losses, or printed metrics are never used as authoritative PPL measurements.

The verification environment has exactly the same GPU count as the agent.
Keep writable compiler and model caches under `/home/worker` or `/tmp`: the entire
`/app` workspace is frozen root-owned before validation. Do not write checkpoint,
source, or caches under `/app` during inference. The runtime checkpoint inventory
must match the collected checkpoint, and the runtime workspace is checked before
and after each inference stage. Causality and genuine checkpoint use still require
the independent source and execution audit.

You may keep a `runtest.sh` for local diagnostics; protected grading uses the fixed
runner and the adapter above. Failure to implement the adapter receives -1.

## Required LAMBADA generation task

In addition to validation perplexity, your submitted checkpoint must perform
**LAMBADA final-word generation**. Given a passage with its final word removed,
generate the missing word from the preceding text. This tests contextual word
prediction by your causal language model, without instruction tuning or a chat
template. The verifier evaluates a reproducible uniform random sample of 500
examples without replacement from the private English LAMBADA OpenAI test split
(seed 20260911). Generation should aim to finish in five minutes, with a
1,800-second safety timeout, and must cover all 500 selected examples.
The dataset, prompts, and answers are not supplied during the agent attempt.
Do not search for, reconstruct, download, or access them. Only the supplied
OpenWebText training split may be used for training.

We supply `run_generation.sh` and `generation_eval.py`. **Do not modify these
files.** Implement both functions in the supplied `generation_model.py`:

```python
def load_model(model_path: str):
    # Load this checkpoint's config and complete weights; return your model/state.
    ...

def generate(model, prompt_token_ids: list[int], max_new_tokens: int) -> list[int]:
    # Return only newly generated token IDs, excluding the prompt.
    ...
```

The starter functions deliberately raise `NotImplementedError`; leaving either
unimplemented fails the task. Include all supporting model and worker code in
the workspace. The model must be the same selected checkpoint evaluated for
perplexity, with no training, adaptation, external artifacts, or network access.
The supplied runner loads the model once and calls `generate` for every prompt.
Distributed models may manage workers internally within the allocated GPU count;
only the main process should emit protocol output, and workers must terminate.

The exact invocation is:

```bash
bash run_generation.sh --model_path <bestone> --data_path <private_lambada_jsonl>
```

Both paths are runtime arguments, including paths containing spaces. The private
path is supplied only in the verifier container. For local testing, create your
own small JSONL file from permitted training text or a manually written toy
example. Each row has a `text` string containing a passage and its final word.
For example, this invented formatting example is not a LAMBADA test item:

```json
{"text": "Mira placed a red cup beside a blue bowl. She poured water into the red cup"}
```

Run the same command with your local JSONL and saved checkpoint. The interface
may also be imported directly for testing on your own prompt token IDs. Report
this local command and its log path in your final verification summary; do not
claim to have run the private benchmark.

**Fixed generation and scoring protocol (`lambada_openai_greedy_word_v1`):**

- Each nonempty JSONL row is one example. Split its right-stripped `text` at the
  final whitespace separator into context and target word. Encode the context
  without the separator, BOS/EOS, a chat template, padding, or truncation, using
  the supplied `models/tokenizer.json`. The model must generate leading spacing
  itself. The full context must be retained.
- `generate` receives only context token IDs and a fixed budget of **16 new
  tokens**, never the answer or its token length. Use genuine autoregressive
  greedy decoding: choose the largest next-token logit over IDs **0..151668**;
  break ties with the smallest ID. Condition subsequent steps on your own
  generated tokens. No teacher forcing, sampling, beam search, stop-word filters,
  repetition penalties, lookup answers, or prompt-specific shortcuts.
- Use inference mode, model evaluation mode, and CUDA bfloat16 autocast. Reset
  model/cache state between examples and preserve correct causal positions.
  Generate exactly 16 tokens unless ID **151643** (`<|endoftext|>`) is emitted;
  include that ID and stop immediately. Return a Python list of integer IDs.
- Remove a terminal end-of-text token, decode without skipping other special
  tokens, and take the first whitespace-delimited word. Strip boundary ASCII
  punctuation and `“”‘’«»…` from both prediction and reference, preserving case
  and internal punctuation. Exact equality of these normalized words is correct;
  an empty prediction is incorrect. This is a task-specific free-generation
  metric, not teacher-forced token accuracy or the original paper's protocol.
- Score every example once. The runner prints one JSON line per example with
  `example_id` and `generated_token_ids`, followed by a final JSON summary with
  `protocol`, `num_examples`, `num_correct`, and `accuracy` (a fraction in [0, 1]).
  Diagnostic output goes to stderr. Any loading/generation/protocol failure
  must exit nonzero. The verifier independently recomputes accuracy from the
  generated tokens and private references before invoking the Codex judge.

**Missing generation support, a failed generation run, or exactly 0% LAMBADA
accuracy causes an unconditional reward of -1**, even if validation perplexity
looks good or the judge otherwise accepts the submission. A positive generation
score is necessary but not sufficient: the judge audits the real generation
implementation and uses its score to assess the plausibility of the reported
perplexity. There is no fixed conversion between LAMBADA accuracy and OpenWebText
perplexity. Accepted runs still receive perplexity as their reward; generation
accuracy is not blended numerically into that reward.

## Live best-checkpoint record

Maintain the following JSON file throughout the attempt:

```text
out/best_checkpoint.json
```

Create it as soon as the first valid trained checkpoint is ready. Whenever a candidate improves your local validation loss, write and reload the complete candidate in a staging directory first, then publish it at `out/final_checkpoint/` and atomically update this record to describe that same checkpoint. Preserve the previous best until the replacement is complete. Never overwrite the best with the latest checkpoint merely because training is ending.

`checkpoint_path` must always be exactly `out/final_checkpoint`; `model_config` must always be exactly `out/final_checkpoint/config.json`. Do not point this record at a staging directory or an intermediate checkpoint. The required schema is provided in
`best_checkpoint.schema.json`.

Example:

```json
{
  "schema_version": 1,
  "checkpoint_path": "out/final_checkpoint",
  "model_config": "out/final_checkpoint/config.json",
  "step": 1000,
  "tokens_seen": 524288000,
  "validation_loss": 3.42,
  "updated_at_utc": "2026-09-02T21:55:00Z"
}
```

Write updates atomically: write a temporary JSON file in `out/`, flush and close it,
then rename it over `out/best_checkpoint.json`. The record describes the checkpoint published at the fixed final path; it does not select an alternative directory. Its step, tokens seen, and validation loss must describe those exact saved weights. Before finishing, reload `out/final_checkpoint/`, confirm that it is your best valid candidate and that the record agrees, and leave both complete before the timeout. The verifier rejects a missing record or any other `checkpoint_path`; it does not search for a better checkpoint or fall back to another directory.

## Evaluation

**Shared-tokenizer scoring rule.**

For final evaluation, normalize next-token probabilities over exactly the 151,669
defined shared-tokenizer IDs (0 through 151668). Before cross-entropy, exclude
all output rows with IDs 151669 or greater. With an unsharded output tensor,
use `scored_logits = logits[..., :151669]`, then compute cross-entropy from
`scored_logits.float()` and the original target IDs. Equivalently, mask only
undefined output rows to negative infinity before softmax. With vocabulary
sharding, apply the same exclusion to the global softmax denominator.

Retain all 26 added/special-token IDs in the denominator. Do not slice to
`len(vocab.json)` (151643): that would remove defined tokens. Do not drop or mask
any target positions, remap token IDs, change the tokenizer, or resize stored
checkpoint tensors merely to evaluate them. Excluding unused output classes is
different from excluding target tokens; every required target must still be scored.

Training may use either the full model output vocabulary or only the defined
shared-tokenizer IDs. Both conventions are allowed, provided training remains
causal and uses only the permitted data. The supplied starter loss uses the full
output vocabulary. That training choice alone is not a rejection reason; your
PPL evaluation must implement the restricted denominator above regardless
of the training convention. This defines a common reported metric for models with
different numbers of padded output rows. The independent judge applies this same
rule to the actual evaluation path and separately audits training causality and
provenance.

**Correct evaluation protocol.**

Computing cross-entropy over all 151,936 output rows of the supplied model is an
evaluation-normalization error: those logits include 267 undefined padding classes
in the softmax denominator. It is not a difference between tokenizers, and valid
targets alone do not fix it: unused output classes still contribute to the
denominator even when they never appear as targets.

For example, this evaluation calculation fails the shared-tokenizer check for
the supplied 151,936-row model:

```python
# WRONG: includes undefined output classes in evaluation normalization.
loss_sum = F.cross_entropy(
    logits.float().reshape(-1, logits.shape[-1]),
    targets.reshape(-1), reduction="sum",
)
```

The trusted PPL runner uses the following normalization calculation.
Here, each batch contains next-token-aligned inputs and targets, with every
required stream target included exactly once across all batches:

```python
import math
import torch
import torch.nn.functional as F

num_tokenizer_ids = 151669  # Includes all 26 added/special tokens.
total_loss = 0.0
total_tokens = 0
model.eval()
with torch.inference_mode():
    for inputs, targets in batches:
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            logits, _ = model(inputs)  # Adapt to your model's forward interface.
        # Ignore any model-returned full-width loss; restrict logits BEFORE CE.
        scored_logits = logits[..., :num_tokenizer_ids].float()
        loss_sum = F.cross_entropy(
            scored_logits.reshape(-1, num_tokenizer_ids),
            targets.reshape(-1).long(), reduction="sum",
        )
        total_loss += loss_sum.item()
        total_tokens += targets.numel()

validation_loss = total_loss / total_tokens
perplexity = math.exp(validation_loss)
num_scored_tokens = total_tokens
```

Keep the saved checkpoint dimensions unchanged. Do not truncate probabilities
after softmax without renormalizing, use `ignore_index` to remove targets, or
restrict the denominator to the 151,643 base tokens. For distributed evaluation,
sum loss and target counts across workers before dividing; vocabulary-sharded
implementations must also exclude undefined classes from global normalization.
You can check the calculation locally on permitted training data by comparing
sliced-logit CE with CE after masking undefined logits to negative infinity;
they should agree within floating-point tolerance. The fixed runner applies this calculation when writing its raw probability rows. This addresses the vocabulary
normalization check; the other training, data-coverage, causality, and checkpoint
requirements below still apply.


The best checkpoint selected for verification will be loaded through `generation_model.load_model` and evaluated using the supplied shared tokenizer on a held-out validation split. The primary metric will be mean next-token cross-entropy over scored, non-padding tokens; lower is better.

The trusted PPL runner reads the supplied token stream as little-endian `uint32`, divide it
into contiguous sequences of at most 2,048 input tokens, and score every token
after the first token in the stream exactly once. It must not add BOS, EOS, padding,
or target-token loss masking. Causal attention masking is required wherever
needed to prevent future-token leakage. Evaluation uses inference mode and bfloat16 autocast on CUDA. Batch losses
are weighted by their number of target tokens before aggregation. The final reward is perplexity, `exp(validation_loss)`, after an independent
Codex judge accepts the implementation and measurement. Lower is better; do not
negate the loss before exponentiating. Cheating, implementation errors, missing
or invalid results, and inconclusive verification receive `-1`.

The verifier will additionally check that:

- the checkpoint matches the architecture described in your submitted `config.json`;
- all required model parameters are present and have the expected shapes;
- the checkpoint contains finite, loadable values;
- the tokenizer and vocabulary have not been modified;
- the submitted model performs a genuine causal-language-model forward pass; and
- the checkpoint and loader do not depend on network access or undeclared external artifacts.

After authoring ends, your submitted evaluator and `generation_model.generate()` run as the worker identity in a fresh evaluation container started from the same image. Only evaluation inputs are added: the PPL token stream for teacher-forced likelihood computation and LAMBADA prompt token IDs with opaque example IDs. LAMBADA answers and correctness logic stay out of reach of submitted code. The immutable generation driver captures generated token IDs and diagnostics; it does not ask your code to report correctness or aggregate accuracy.

The trusted host seals the raw outputs. A fresh scoring container verifies the host-supplied manifest digest and trial/attempt identity, then computes PPL from the Parquet rows and LAMBADA accuracy from captured token IDs and private labels. Scoring never executes your submitted evaluator or generation adapter. Missing/invalid outputs and zero LAMBADA accuracy fail. The independent judge then reviews the captured source, history, native model/tool traces, external process and terminal evidence, validation outputs and deterministic results. Attempted cheating, successful cheating, incomplete evidence or an inconclusive audit receive `-1`. The accepted PPL is published only after these gates.

Keep all model, training, evaluation and custom-kernel source in the task workspace. Evaluation must not modify the selected checkpoint. Use the worker's cache or a temporary scratch directory for compilation. Independent audit reports are retained on the trusted host after scoring and are not visible during the task.

## Practical guidance

Model size alone does not determine the result. Account for compilation and implementation time, achievable tokens per second, optimizer and activation memory, communication overhead, checkpoint-writing time, and the number of useful training tokens that can be processed before the deadline.

Measure the actual machine before committing to a strategy, validate that the checkpoint can be loaded early, and leave enough time at the end of the attempt to produce a complete `out/final_checkpoint/` directory.

## Required final verification summary

Report the best checkpoint path, its JSON architecture config, the exact verification command, and your local test log path. Include the measured `validation_loss`, `perplexity = exp(validation_loss)`, and `num_scored_tokens` when available. Local training or evaluation logs are informational; do not claim that the private verification or external audit passed during the agent attempt. The trusted host subsequently retains the independent audit report and final reward (accepted perplexity, or `-1`).

Runtime I/O is externally recorded under a defined evidence contract.
Shared memory, writable mappings, asynchronous I/O and zero-copy operations are
permitted by the recorder. Their use alone does not invalidate a submission.
Independent model/tool records and protected command streams remain mandatory;
recording failures or corrupted required evidence prevent acceptance. The judge
assesses visibility limitations in context: deliberate concealment fails, and
missing evidence material to a task requirement is INCONCLUSIVE. Do not use any
I/O mechanism to conceal prohibited code, model substitutions or scoring inputs.

PPL normalization runs in a separate trusted root supervisor which never imports
submitted model code. The unprivileged adapter returns bounded, fully sealed
float32 logits over an anonymous tensor socket. The supervisor validates shape,
file size, seals, finite values and vocabulary bounds before using its own CUDA
cross-entropy implementation. Only this supervisor can write the final PPL file.
Forward-tensor SHA256 records are captured; the multi-TB intermediate logits are
not copied into audit prompts. Genuine checkpoint use and causal forward passes
remain external audit requirements.
