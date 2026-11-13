## Environment integrity

Runtime executables, shared libraries and installed packages are fixed. Do not
replace or shadow supplied packages (including `torch`, `transformers`, `sglang`,
`transformer_engine` and the Megatron-LM install at `/opt/Megatron-LM`), or install
import/startup hooks. The pre-attempt environment inventory is available read-only at
`/opt/environment-baseline/environment.json`. The host checks it again after
authoring, and checks the fresh intermediate container before evaluation. A failed
integrity or process-cleanup check stops the run with reward **0**, reason
**violation of rules**. The container transition must leave no live authoring
processes. Keep permitted task implementation work, including a modified copy of
Megatron-LM, inside `/app` without impersonating installed dependencies.

# Mid-train a base model for reinforcement learning

## Objective

You have one 12-hour attempt to mid-train **Llama-3.2-1B-Base**, the
pretrained base checkpoint published as `meta-llama/Llama-3.2-1B` (not the
Instruct model). The goal is to prepare this specific 1B model to gain the most
from subsequent reinforcement learning (RL) on math, coding and science.

We will evaluate your provided checkpoint by running RL on it using **43,000
training examples: 19,000 math, 7,000 coding and 17,000 science**. Each task uses
reward computation tailored to that task. Every submission and the human
reference baseline receive the same RL procedure and **five-hour wall-clock
budget**, including initialization, rollouts, updates and saves. Completed update
counts depend on throughput; the budget is not a fixed number of steps.

After that fixed RL training, we will evaluate the resulting model on held-out
competitive math, coding and science benchmarks. **The final score is the
unweighted average of the three domain scores:**

```text
final_score = (math_score + coding_score + science_score) / 3
```

Each domain contributes equally, regardless of its number of RL training
examples. Optimize your mid-training for the strongest performance after this
shared RL stage. Direct evaluations of your mid-trained checkpoint are useful
research signals; the final score comes from the model after RL.

This is an open-ended research and systems task. You decide which of the
provided data to use, in what proportions and order, how to filter, deduplicate
and format it, which training methods to apply, the hyperparameters, and the
infrastructure. Continued pretraining, supervised fine-tuning, reinforcement
learning, multi-stage curricula, checkpoint averaging or merging, and any
combination of them are all allowed within the rules below.

## Hard 12-hour limit

The attempt has a hard wall-clock limit of **12 hours**, covering inspection,
data processing, implementation, debugging, training, evaluation, checkpoint
export and verification.

**Providing a complete, trained checkpoint at `/app/out/final_checkpoint` before
the deadline is the primary deliverable and is essential to completing this
task. Ablation results, code and research notes cannot substitute for that
checkpoint. You must allocate enough time for a substantial main training run,
not only ablations.** A strong mid-training run usually takes many hours, so
reserve those hours before scheduling experiments. Set an explicit budget early: cap data
processing, infrastructure work and exploratory runs; measure end-to-end training
throughput; compute the latest start time for your main run; and keep a separate
buffer for export, evaluation and checkpoint validation. Reassess as measurements
improve. **Do not spend the attempt on many small trials and then submit a barely
trained checkpoint.** Publish an early valid checkpoint as a safeguard, then
replace it whenever a better one is validated.

A process still running at the deadline does not make an incomplete checkpoint
eligible. Stop every process you launched before you finish.

## Hardware

You have one machine with four NVIDIA GB300 GPUs and a Grace ARM64 CPU. No other
machines are available. Inspect GPU
memory, interconnect, CPU cores, RAM and the disk free under `/app` at the start,
and choose parallelism, sequence lengths and batch sizes accordingly.

## Runtime environment

You are inside the prepared task container. Commands run as a non-root user.

| Path | Contents |
| --- | --- |
| `/app` | Writable workspace: this task's files, `evals/`, and your code and outputs |
| `/app/models/Llama-3.2-1B` | The base model (read-only, Hugging Face format) |
| `/app/data` | The provided training data and development evaluation sets (read-only) |
| `/app/scratch` | Create it for tokenized data, caches and intermediate checkpoints; it is not collected |
| `/opt/Megatron-LM` | Megatron-LM source, on `PYTHONPATH`; copy it into `/app` if you need to modify it |

The Python environment shares the downstream RL runtime: CUDA 13, the pinned
SGLang 0.5.15.post1 image's PyTorch and Transformers, Slime, Datasets, PyArrow,
math-verify and patched Megatron-LM. Transformer Engine and Apex are installed;
Megatron's local layer implementations are also available. Megatron-LM converts
Llama checkpoints to and from Hugging Face format through Megatron-Bridge, which
is not installed; if you train with Megatron, you also own that conversion. You
may use any other training stack available in the environment.

Validate the runtime first:

```bash
python -c 'import sys, torch; print(sys.executable, torch.__version__, torch.cuda.device_count())'
```

The image is immutable task infrastructure. Do not install, upgrade or remove
packages; report a missing dependency as an environment defect. Commands have no
internet or external model API access. The coding agent's own model connection
is managed separately; do not invoke or copy the installed coding agent.

## Provided model

`/app/models/Llama-3.2-1B` holds **Llama-3.2-1B-Base**, the pretrained
(not instruction-tuned) `meta-llama/Llama-3.2-1B` checkpoint:
16 layers, hidden size 2,048, 32 query and 8 key-value heads, vocabulary 128,256
with tied input and output embeddings, RoPE with Llama 3 scaling, and a
131,072-token maximum position. Its tokenizer, including every special token, is
fixed. You may use any special tokens the tokenizer already defines; you may not
add tokens or change the vocabulary.

## Provided data

`/app/data` holds read-only copies of pinned public Hugging Face datasets in
their upstream formats (Parquet or JSON Lines) and columns. About 1.1 TB in
total; `/app/data/README.md` and `/app/data/SOURCES.json` list each directory's
source revision, license, file count, size and, where computed, row counts.

| Directory | Contents |
| --- | --- |
| `dclm-edu/sample/` | 100 randomly selected, unfiltered shards of DCLM-Edu: web documents annotated with FineWeb-Edu educational scores (`edu_score`, `edu_int_score` 2 to 5) |
| `dclm-edu/edu_int_score_ge4/` | Every DCLM-Edu document with `edu_int_score` of 4 or 5, from all 1,769 shards |
| `nemotron-post-training-v1/` | Synthetic conversations with reasoning traces; file prefixes `chat`, `code`, `math`, `stem`, `tool` |
| `megamath-web-pro/` | High-quality mathematical web documents |
| `mixture-of-thoughts/{math,code,science}/` | Verified long reasoning traces for math, code and science problems |
| `opencodereasoning-2/{python,cpp}/` | Competitive-programming solutions with reasoning traces, critiques, judgements and test pass rates |
| `openthoughts3/` | Reasoning traces for math, code and science questions (`domain` column) |
| `wildchat-1m/` | Real user conversations with ChatGPT models (`model` column) |
| `tulu-3-sft-personas-*/` | Persona-driven synthetic conversations: instruction following, math, grade-school math, algebra and code |
| `ultrachat-200k/` | Filtered multi-turn synthetic assistant dialogues |
| `starcoderdata/` | Permissively licensed GitHub source code in 86 languages, plus GitHub issues, Jupyter notebooks and commits |
| `opc-annealing-algorithmic-corpus/` | Algorithm-related source files from a refined GitHub code corpus |
| `dev_evals/` | Development evaluation sets (see [Development evaluations](#development-evaluations)) |

You may use any subset of the training data in any proportion, with any
filtering, deduplication, formatting, packing, ordering or repetition.

## Data mixture and feasible token budget

**You cannot train this 1B model on the entire 700B tokens within the 12-hour
limit. That is not possible with the assigned compute.** Treat the provided
corpora as a pool from which to build an effective training mixture, rather than
as a requirement to consume every token.

The recommended starter approach is to choose the composition and proportions of the
different datasets we provide, and mix them as you see fit. Measure actual token
proportions rather than assuming document counts reflect them.

Work on data loading, sampling, tokenization, packing and mixture proportions;
build the training infrastructure for data loading you need within the provided environment.
Measure throughput to set a realistic token budget. Run targeted ablations or
small training and evaluation experiments on this specific 1B base model to
compare mixtures and methods. **Ablations must fit within a capped exploratory
budget and leave enough time for the main training run, checkpoint export and
validation.** Stop exploratory work at your planned cutoff and start the main
run; producing the trained checkpoint takes priority over additional ablations.
Record planned and achieved token proportions, the evidence behind your choices,
and how they should improve the model's ability to benefit from the shared RL
stage.

## Rules

1. Initialize from exactly the provided base model. No other model weights are
   available or allowed.
2. Train only on the provided training data (everything under `/app/data` except
   `dev_evals/`) and on data generated by the model you are training, such as RL
   rollouts or self-generated samples. Do not write training examples yourself,
   whether by hand or from your own knowledge.
3. The development evaluation sets are for measurement only. Never train on
   them or use them to select individual training examples.
4. Do not reconstruct, guess, solicit or otherwise target the questions or
   answers of any evaluation benchmark, and do not attempt to locate or read
   private grading data, verifier files or anything outside `/app`, `/opt` and
   the installed environment.
5. Do not modify the provided evaluation tools in place to change their results,
   fabricate measurements, or leave processes running at the end.

## Required submission

Before the deadline, place exactly one final checkpoint at:

```text
/app/out/final_checkpoint/
```

This is the only checkpoint the graders use. It must be a complete Hugging Face
`LlamaForCausalLM` directory:

- `config.json` with the base model's architecture fields unchanged;
- weights in `.safetensors` (a single file, or shards with
  `model.safetensors.index.json`) holding exactly the base model's tensor names
  and shapes, with the output projection still tied to the input embedding (no
  separate `lm_head.weight`);
- finite values in bfloat16, float16 or float32 (bfloat16 recommended);
- no symbolic links, adapters, optimizer state or partial shards.

Tokenizer files may be included but are ignored: graders always use the base
tokenizer and load your weights into the base configuration. Check a candidate
with:

```bash
python evals/checkpoint.py out/final_checkpoint
```

Publish atomically: write and validate a candidate in a staging directory under
`/app/out`, then rename it to `final_checkpoint`, keeping the previous best until
the replacement is complete. Never replace a better checkpoint with a worse one
merely because training is ending.

## Downstream RL evaluation (Very Important)

The graders initialize RL from your submitted checkpoint and train on 43K
examples: **19K math + 7K coding + 17K science**, with reward computation tailored
to each task. All submissions, including the human reference baseline, use the same RL
procedure and a **five-hour wall-clock RL budget (18,000 seconds)**. The number
of completed optimizer updates depends on throughput; there is no fixed update
target. Configured rollout ceilings are safety bounds, not the training budget.
The model after RL is evaluated on held-out competitive math,
coding and science benchmarks; the final score is the unweighted average of the
three domain scores. The shared Slime recipe uses **GRPO within that five-hour
budget**, with **128 prompts and 8 responses per prompt** in each batch (1,024
trajectories per optimizer step). Inputs are limited to **2,048 tokens**, and each response may generate up
to **16,384 tokens**. Math rewards check the final boxed answer symbolically;
coding rewards execute the generated Python solution against the problem's
tests in a separate sandbox; science rewards check the final boxed option letter
against the provided expected answer. RL uses the same fixed sampled dataset
for every submission. Additional hyperparameters and the held-out evaluation
configuration are documented with the grading setup.

The RL timer includes initialization, rollouts, updates and checkpoint saves. The
driver saves approximately hourly and reserves the final 15 minutes for a final
synchronous save and shutdown. If the hard deadline interrupts a rollout or
save, grading uses only the latest fully saved checkpoint completed within the
budget; partial saves and runs without any complete update/checkpoint are
ineligible. Linear learning-rate decay follows elapsed wall time. Held-out
benchmark scoring runs afterward, outside the five-hour RL timer.

## Exact downstream RL reward implementation

The following rules apply to the shared RL stage after your mid-training. The
private training sources, prompts, labels and test cases are not disclosed. The
43K domain counts and reward algorithms are public. These functions are shown
so you can choose compatible output formatting without accessing private inputs.
They operate on the generated **response only**, not the prompt or conversation
history. Every correctness reward is binary, `0.0` or `1.0`; there is no format
bonus, partial test credit, length bonus, or required `<think>` delimiter.

### Math and science answer extraction

Only the **last literal** `\boxed{` starts the answer. Braces are balanced from
that point; nested LaTeX braces are supported and the extracted contents are
stripped. `\fbox{}`, unboxed text, a spaced `\boxed {`, and missing closing braces
do not qualify. If the last box is malformed, extraction fails even if an earlier
box was valid. Text after a valid last box does not change its extracted value.

For math, both the private reference answer and extracted prediction are wrapped
in `\boxed{...}`, parsed with `math_verify.parse` using `LatexExtractionConfig`,
and compared with `verify(gold, prediction)`. Both parsed results must be nonempty.
The runtime pins `math-verify[antlr4_9_3]==0.9.0`. This is symbolic equivalence,
not string matching or a model-based judge.

For science, the stripped box must fully match a single uppercase letter, or
`\text{A}` / `\mathrm{A}` (with optional whitespace immediately inside those
braces). Lowercase letters, `Answer: A`, `(A)`, `\boxed{A.}`, and multiple letters
are rejected. The letter must belong to that question's valid-option list and
exactly match its private reference label. A correct letter elsewhere in the
reasoning gives no credit. For example, `\boxed{A} ... \boxed{B}` is scored as B.

### Code extraction and service result

Only fenced blocks matching the regular expression below qualify: an opening
triple backtick immediately followed by `python` or `py` (case-insensitive),
optional whitespace, a newline, then the code and a closing triple backtick.
The **last matching block** is stripped and used verbatim. An untagged fence,
plain code, or a response without a matching block earns zero. Include the full
program, necessary imports and, for function tasks, the required function or
`Solution` method. Do not rely on helpers defined in another answer block.

The extracted code and private test label are sent to an operator-owned HTTP
sandbox. The service must return `status == "ok"` and a boolean `passed`.
All tests passing gives 1; any ordinary test failure gives 0. A missing sandbox,
HTTP/transport failure, malformed service reply or container-launch failure is
an infrastructure error, not a zero-quality sample to silently train on. This
service is available to the downstream verifier, not to authoring processes.

Here is the exact reward module, including parsing, service calls and dispatch:

```python
"""Slime async binary rewards: symbolic math, option-label science, sandboxed code."""
import asyncio
from concurrent.futures import ProcessPoolExecutor
import multiprocessing
import json
import os
import re
import urllib.request

_math_pool = None


def math_pool():
    global _math_pool
    if _math_pool is None:
        # Slime's rollout loop can run in a thread. Math-Verify's signal-based
        # timeouts must execute in the main thread of an independent process.
        _math_pool = ProcessPoolExecutor(max_workers=4,
                                        mp_context=multiprocessing.get_context('spawn'))
    return _math_pool


def final_boxed(text):
    start = text.rfind('\\boxed{')
    if start < 0:
        return None
    start += len('\\boxed{')
    depth = 1
    for end in range(start, len(text)):
        if text[end] == '{':
            depth += 1
        elif text[end] == '}':
            depth -= 1
            if depth == 0:
                return text[start:end].strip()
    return None


def math_reward(response, label):
    from math_verify import LatexExtractionConfig, parse, verify
    answer = final_boxed(response)
    if answer is None:
        return 0.0
    gold = parse('\\boxed{' + str(label) + '}', extraction_config=[LatexExtractionConfig()])
    prediction = parse('\\boxed{' + answer + '}', extraction_config=[LatexExtractionConfig()])
    return float(bool(gold) and bool(prediction) and verify(gold, prediction))


def science_reward(response, label, valid_letters):
    answer = final_boxed(response)
    if answer is None:
        return 0.0
    match = re.fullmatch(r'\\(?:text|mathrm)\{\s*([A-Z])\s*\}|([A-Z])', answer)
    if not match:
        return 0.0
    letter = match.group(1) or match.group(2)
    return float(letter in valid_letters and letter == label)


def extract_code(response):
    blocks = re.findall(r'```(?:python|py)\s*\n(.*?)```', response, re.DOTALL | re.IGNORECASE)
    return blocks[-1].strip() if blocks else None


def code_reward(response, label):
    code = extract_code(response)
    if not code:
        return 0.0
    url = os.environ.get('MIDTRAIN_CODE_SANDBOX_URL')
    if not url:
        raise RuntimeError('MIDTRAIN_CODE_SANDBOX_URL is required; never execute generated code in the grader')
    headers = {'Content-Type': 'application/json'}
    token = os.environ.get('MIDTRAIN_CODE_SANDBOX_TOKEN')
    if token:
        headers['Authorization'] = 'Bearer ' + token
    payload = json.dumps({'code': code, 'label': label}).encode()
    request = urllib.request.Request(url.rstrip('/') + '/score', data=payload, headers=headers)
    with urllib.request.urlopen(request, timeout=600) as result:
        report = json.load(result)
    if report.get('status') != 'ok' or not isinstance(report.get('passed'), bool):
        raise RuntimeError(f'code sandbox infrastructure failed: {report}')
    return float(report['passed'])


async def reward(args, sample, **kwargs):
    domain = sample.metadata['domain']
    if domain == 'math':
        return await asyncio.get_running_loop().run_in_executor(
            math_pool(), math_reward, sample.response or '', sample.label)
    if domain == 'science':
        return science_reward(sample.response or '', sample.label, sample.metadata['valid_letters'])
    if domain == 'code':
        return await asyncio.to_thread(code_reward, sample.response or '', sample.label)
    raise ValueError(f'unknown reward domain: {domain}')
```

### Exact RL code-test comparison and invocation

The private label has `testtype` equal to `stdin` or `functional`, a nonempty
`tests` list of `{input, output}` pairs, and `fn_name` for functional tests.
Every supplied test must pass, in order, with no subsampling. Each test starts a
fresh CPU Docker container; code never runs in the GPU grader or HTTP server.
Containers have no network, run as UID/GID 65534, use a read-only filesystem,
no capabilities and no new privileges, one CPU, a 512 MiB memory cap, 32 PIDs,
a 32 MiB temporary filesystem, and a 1 MiB stdout limit. Each ordinary test has
a six-second wall-clock execution limit (including container startup), and the
whole solution has a 180-second test-loop deadline checked before each test.
Nonzero exit, timeout, excessive output or invalid functional JSON fails the
solution. Docker launch errors stop grading as infrastructure failures.

For stdin tests the program receives the supplied input verbatim. Output strings
are stripped and split on whitespace; token counts must agree. Equal tokens
pass directly. Otherwise both tokens must parse as finite floats and satisfy
`math.isclose(..., rel_tol=1e-6, abs_tol=1e-6)`. Functional inputs are either a
JSON-encoded argument on each nonempty input line or an argument list. If the
program defines `Solution`, the named method on a fresh instance is called;
otherwise the named top-level function is called. Its return value must be JSON
serializable and stdout must contain only that JSON. `List` and `Optional` are
provided in its namespace; other imports are the solution's responsibility.
Lists and dictionaries are compared recursively, numeric values use the same
tolerance, and other values use equality. The implementation below is authoritative
for type handling, including Python's numeric/bool relationships and nonfinite
values; it is not a custom checker chosen per problem.

```python
import json
import math

def equivalent(actual, expected):
    if isinstance(expected, str):
        if not isinstance(actual, str):
            return False
        expected = expected.strip().split()
        actual = actual.strip().split()
        if len(actual) != len(expected):
            return False
        for got, want in zip(actual, expected):
            if got == want:
                continue
            try:
                if not math.isfinite(float(got)) or not math.isfinite(float(want)):
                    return False
                if not math.isclose(float(got), float(want), rel_tol=1e-6, abs_tol=1e-6):
                    return False
            except ValueError:
                return False
        return True
    if type(actual) is not type(expected) and not (isinstance(actual, (int, float)) and isinstance(expected, (int, float))):
        return False
    if isinstance(expected, list):
        return len(actual) == len(expected) and all(equivalent(a, b) for a, b in zip(actual, expected))
    if isinstance(expected, dict):
        return actual.keys() == expected.keys() and all(equivalent(actual[k], v) for k, v in expected.items())
    if isinstance(expected, (int, float)) and not isinstance(expected, bool):
        return math.isfinite(actual) and math.isclose(actual, expected, rel_tol=1e-6, abs_tol=1e-6)
    return actual == expected


def functional_program(code, fn_name, test_input):
    if isinstance(test_input, str):
        arguments = [json.loads(line) for line in test_input.splitlines() if line.strip()]
    elif isinstance(test_input, list):
        arguments = test_input
    else:
        raise ValueError('functional input must encode a list of arguments')
    wrapper = '''
import json as _judge_json
import sys as _judge_sys
_judge_args = _judge_json.loads(_judge_sys.stdin.read())
_judge_namespace = {"__name__": "candidate", "List": list, "Optional": __import__("typing").Optional}
exec(compile(SOURCE, "candidate.py", "exec"), _judge_namespace)
_judge_target = _judge_namespace["Solution"]() if "Solution" in _judge_namespace else None
_judge_fn = getattr(_judge_target, FUNCTION) if _judge_target is not None else _judge_namespace[FUNCTION]
_judge_result = _judge_fn(*_judge_args)
print(_judge_json.dumps(_judge_result))
'''
    wrapper = wrapper.replace('SOURCE', '_judge_source').replace('FUNCTION', '_judge_function')
    return '_judge_source = ' + repr(code) + '\n_judge_function = ' + repr(fn_name) + '\n' + wrapper, json.dumps(arguments)
```

The service uses this exact method for RL tests. `self.execute(program, stdin)`
is the isolated container operation described above and returns stdout, or `None`
on an ordinary execution failure. Its timeout defaults to six seconds.

```python
import time

class Judge:
    def score_rl(self, code, label):
        if not isinstance(code, str) or len(code) > (1 << 20):
            raise ValueError('invalid code')
        tests = label['tests']
        if not tests or label['testtype'] not in ('stdin', 'functional'):
            raise ValueError('invalid test suite')
        # Evaluate every supplied test, with an upper bound per generated solution.
        deadline = time.monotonic() + 180
        for test in tests:
            if time.monotonic() >= deadline:
                return False
            if label['testtype'] == 'stdin':
                result = self.execute(code, test['input'])
                if result is None or not equivalent(result, test['output']):
                    return False
            else:
                program, input_text = functional_program(code, label['fn_name'], test['input'])
                result = self.execute(program, input_text)
                if result is None:
                    return False
                try:
                    actual = json.loads(result)
                    expected = json.loads(test['output']) if isinstance(test['output'], str) else test['output']
                except (ValueError, TypeError):
                    return False
                if not equivalent(actual, expected):
                    return False
        return True
```

The code-test runner for final post-RL benchmarks can use a different private
test format; that does not change these RL reward rules. Training reward and
final benchmark score remain separate measurements.

## Development evaluations

`evals/README.md` describes the tools and sets. In brief:

- `evals/reasoning.py`: generated-answer accuracy on math (MATH problems outside
  MATH-500, AIME 2022-2023), science (SuperGPQA Science) and code
  (LiveCodeBench release_v1), with a configurable prompt template;
- `evals/checkpoint.py`: the submission's structural check.

They measure your checkpoint directly, not after the graders' RL, and their
scores are informational only.

## Research log

Publish `/app/out/training_dataset.json` alongside your final checkpoint. It
identifies the complete processed dataset used by the run that produced that
checkpoint, including every shard, index and mixture stage consumed by that run.
Preserve those files until collection finishes. Do not select discarded ablation
datasets or intermediate preprocessing versions. Use `/app/out/processed_dataset/`
for materialized training data so it stays out of hourly source snapshots.

```json
{
  "schema_version": 1,
  "training_run": "selected-main-run",
  "files": [
    {"path": "/app/out/processed_dataset/math.bin"},
    {"path": "/app/out/processed_dataset/math.idx"}
  ]
}
```

List absolute paths to regular, agent-produced files under
`/app/out/processed_dataset/`; symlinks and provided
read-only corpora cannot be exported. Each entry may include `sha256`, which the
collector verifies. Write this manifest atomically and keep it aligned with the
selected checkpoint. The collector archives the dataset once after authoring
stops, recording its actual hashes and sizes. It does not archive intermediate
checkpoint or dataset writes. If you consume the provided inputs directly or
stream without producing dataset files, use `files: []`, `mode: "provided_inputs"`
or `"streaming"`, and a nonempty `reason`; document the loader, sampling recipe,
seeds and data revisions in the research log. Dataset selection remains subject
to review against your code and terminal history.

Keep `/app/RESEARCH_LOG.md` current. Record the data you used and how you
processed and mixed it, planned and achieved mixture proportions, token counts
per source and stage, ablations on the 1B model, training methods,
hyperparameters and exact commands, throughput, evaluations and their results,
failures and abandoned approaches, and why the final checkpoint was selected.

## Recommended execution order

1. Inspect the hardware, environment, base model, `/app/data/README.md` and
   `evals/README.md`. Score the base model on the development suites you plan to
   use.
2. Reserve enough time for the main training run, export and validation first;
   then cap the ablation budget and set the latest start of the main run.
3. Build and validate the data pipeline and training stack on small runs;
   measure throughput.
4. Publish an early valid checkpoint and check it with `evals/checkpoint.py`
   and the reasoning development suites.
5. Launch the main run before your cutoff; monitor it and evaluate candidates.
6. Publish the best validated checkpoint at `out/final_checkpoint`, confirm the
   checkpoint structure and development results, update the research log, and stop
   all processes before the deadline.
