## Environment integrity

Runtime executables, shared libraries and installed packages are fixed. Do not
replace or shadow supplied packages (including `torch`, `transformers` and
`triton`), or install import/startup hooks. The pre-attempt environment
inventory is available read-only at `/opt/environment-baseline/environment.json`.
The host checks it again after authoring, and checks the fresh grading
container before evaluation. A failed integrity or process-cleanup check stops
the run with reward **0**, reason **violation of rules**. The container transition
must leave no live authoring processes. Keep permitted task implementation work
inside `/app` without impersonating installed dependencies.

# Build a fast LLM-serving system from scratch, for agentic coding traffic

A large model is staged at `/models/GLM-5.3-Flash`. A **naive but correct**
OpenAI-compatible server is provided at `/app/server.py` (launched by `/app/serve.sh`).
Your job is to turn it into a system that serves **live agentic coding sessions** — many
concurrent coding agents, each re-sending a long and growing conversation turn after
turn — and **earns as much as it can** doing so, **without breaking the model's
outputs**. You are paid per token served; your score is what the machine earned per
GPU-hour.

**Work out the model yourself.** This document deliberately does not describe the
model's architecture, its config, or how to build a fast path for it. Read
`/models/GLM-5.3-Flash/config.json`, read the `transformers` implementation of its
architecture, profile the provided server, and decide what to do. What follows is only
the environment, the contract, and the rules — plus one scoping note about the checkpoint.

**You only need the TEXT model.** The checkpoint is a multimodal one — its config has a
`vision_config` alongside `text_config`, and the weights include a vision tower
(`model.visual.*`) plus image/video token ids. **Nothing in this task sends images or
video**: every request is text over `/v1/completions`, in both the scored traffic and
the accuracy pass. So you do not need to load, initialise or implement the vision encoder,
its preprocessing, or any multimodal token handling — serve the language model
(`model.language_model.*` plus the output head `lm_head`, which is a separate top-level
weight) and ignore the vision tower (`model.visual.*`). Skipping it saves little memory
(the tower is ~1 GB of a ~328 GB checkpoint) but a great deal of complexity.

## Hardware

You have **four GPUs**. Query them rather than assuming anything:

```bash
nvidia-smi --query-gpu=index,name,memory.total,compute_cap --format=csv
python3 -c "import torch; print(torch.cuda.device_count()); \
  [print(i, torch.cuda.get_device_properties(i)) for i in range(torch.cuda.device_count())]"
nproc; free -g            # the CPU / host-RAM budget THIS CONTAINER gets
```

**The model's weights do not fit on one GPU.** Every working server here, including the
provided baseline, must shard the model across the four. The starter does this the
simplest way available in `transformers`; whether that is a good way is for you to
determine.

The CPU is **ARM64 (aarch64)**, not x86: anything you write for the host side must be
architecture-neutral or aarch64-native — no x86 intrinsics (`<immintrin.h>`, AVX/SSE),
no `-march=x86-64-v3`, no assumptions about prebuilt x86 binaries. Device code is
unaffected.

## Toolchain versions

- **CUDA 13.0** (`nvcc`, `cmake`, `ninja` present).
- **`torch` 2.14.0+cu130**, an aarch64 build. `triton` ships with it.
- `transformers` >= 5.16.1.
- `TORCH_CUDA_ARCH_LIST` is preset to `"10.0 10.3"`, so a CUDA extension you build with
  `torch.utils.cpp_extension` (or `nvcc` directly) is compiled for these GPUs. Confirm the
  compute capability you actually have with
  `python3 -c "import torch; print(torch.cuda.get_device_capability())"`.
- `ncu` / `nsys` are available for profiling.

## Constraints (read carefully)

Your commands run as a nonroot user with access to the assigned GPUs, CUDA
toolchain, shared memory, caches, task files and local server processes. The
coding-agent installation and its model connection are managed privately. Do not
invoke or copy an installed coding agent; those installations are inaccessible
to commands and removed before grading. Implement and test the server directly.
These controls do not require you to sandbox or restrict your server.

- **Available** for building the server AND writing custom kernels: `torch`,
  HuggingFace `transformers`, `triton`, `numpy`, `fastapi`/`uvicorn`, the standard
  library, and the CUDA toolchain (`nvcc` 13.0, `cmake`, `ninja`). You may write
  custom GPU kernels in **either triton (`@triton.jit`) or CUDA C++**
  (`torch.utils.cpp_extension` / nvcc).
- **Also installed**, and yours to use: `accelerate`, `safetensors`, and `aiohttp` (the
  simulator's HTTP client). So is the `kernels` package together with the FP8 linear kernel that
  `transformers` uses for this checkpoint, already cached in the image; `HF_HUB_OFFLINE=1`
  is set, so nothing waits on the blocked network to find it.
- **NOT installed** (and you cannot add them — see "no internet" below): the serving /
  attention **frameworks** vLLM, SGLang, flash-attn. Write your own
  attention / paging / batching with torch + triton + CUDA rather than calling a
  prebuilt one.
- **The container has no internet.** There is no package index and no download path —
  you **cannot** `pip install` anything or fetch code/weights. Build the fast path
  **yourself**, in this container, from what is already here.

## What you may change

Anything under `/app`. Edit or fully rewrite `/app/server.py`, add files, and change
`/app/serve.sh`'s launch command. The only contract is the HTTP API below. Grading
runs a fresh, writable copy of `/app`; symbolic links in it must point inside `/app` or
to the model weights.

## What you deliver

`/app/serve.sh` must launch a server exposing an **OpenAI-compatible
`/v1/completions` endpoint on port 8000** serving the model. The grader runs
**exactly `bash /app/serve.sh`** — no arguments, no extra environment variables of
its own, and no other setup step. It launches it **once**: the scored traffic and the
accuracy check below are sent to the same running server. So `/app/serve.sh` must
be **fully self-contained**: `export` every environment variable your server needs
(`PYTORCH_*` / `TORCH_*` flags, `OMP_NUM_THREADS`, `TRITON_CACHE_DIR`, any custom
paths, compiled-kernel locations, etc.), `cd` wherever needed, and launch the
server — all inside `serve.sh` or files it `source`s. **Anything you rely on that is
not wired into `serve.sh` will NOT be present at grade time.** (`CUDA_VISIBLE_DEVICES`
is already set for you.) The grader waits until the server answers on `http://localhost:8000`
(`/v1/models` or `/health`) — it waits **up to 1800 s, and a server that has not answered
by then scores 0** — sends the evaluation requests to `/v1/completions`, then shuts it
down. You do not need to leave a server running when you finish: the grader first stops
anything still holding the GPUs or port 8000, then launches `serve.sh` itself. Its traffic
client runs **in this same container**, so it shares the container's CPUs and memory
with your server, exactly as `python3 /app/simulator.py` does when you run it.

**The traffic's clock starts with its own first request.** There is no discarded warmup
pass: whatever your server still has to compile, capture, autotune or allocate after it
reports ready, it does inside the scored window. Finish your own warmup first, and only
then report ready.

**The wire format.** `/app/server.py` implements all of it, and
`/app/example_requests.py` sends every form; run it against your server.

| field | what your server must do |
|---|---|
| `prompt` | either a **pre-templated string** or a **list of token ids**. Both occur. Always ONE prompt, never a batch. Never re-apply the chat template. |
| `max_tokens` | generation cap for this request |
| `temperature` | always `0`: decode greedily |
| `ignore_eos` | when `true`, keep decoding past EOS — EOS fed back as an ordinary token — until **exactly** `max_tokens` tokens; when absent, stop at EOS. See the rule below. |
| `skip_special_tokens` | when `false`, special tokens stay in the returned text; when absent, strip them |
| `stream` | when `true`, answer with server-sent events: `data: {"choices":[{"text": ...}]}` lines, then `data: [DONE]` |
| `stream_options.include_usage` | when `true`, the last event before `[DONE]` carries `usage` |

Responses carry `choices[0].text` and `usage`: `completion_tokens`, and
`prompt_tokens_details.cached_tokens` (how many prompt tokens you served from a cache;
`0` if you have none).

## What is in `/app`

Your working directory. Everything here is yours to read and to change, except
`restrict-network.sh`, which is infrastructure: leave it alone.

| file | what it is |
|---|---|
| `server.py` | the **naive but correct** starter server. Your baseline and your starting point — rewrite it, replace it, or delete it. |
| `serve.sh` | **the contract.** The grader runs exactly `bash /app/serve.sh` to bring your server up. Whatever you build must launch from here. |
| `simulator.py` | **the benchmark that scores you.** Read it. The grader runs its own copy of this same file, so editing yours changes only your own runs. |
| `agentx/` | the simulator's data: the recorded sessions (`sessions.jsonl.gz`) and the code its prompts are cut from (`code.txt`). Yours to read and analyze. |
| `example_requests.py` | a runnable example of every request/response form, with a self-check that it matches the model's chat template. Run it first to confirm your wire format. |
| `start_server.sh` | launches `serve.sh` detached (restart-safe), logging to `/tmp/server.log`. |
| `stop_server.sh` | stops a server started that way. |
| `process_safety.md` | **read before you run `kill` or `pkill`.** |
| `instruction.md` | a copy of this document. |
| `CLAUDE.md`, `AGENTS.md` | the same notes, auto-loaded by Claude Code and Codex respectively; same process-safety warning. |
| `restrict-network.sh` | infrastructure — installs the network firewall at container start. Not yours to run. |

The model weights are outside `/app`, at the path named above; they are read-only input,
not something to modify.

> ⚠️ **Before you run `kill` or `pkill`, read `/app/process_safety.md`.** You run in the
> same container as the harness executing you, and `pkill -f` matches full command lines —
> including this task text — so a broad kill can terminate **your own session** mid-task.
> Use `start_server.sh` / `stop_server.sh`, which kill only the server's process group.

## How you are measured — two separate things

**1. INCOME — your score. Defined by `/app/simulator.py`.**

Your reward is the **USD per GPU-hour** that **`/app/simulator.py`** measures against your
server: every input and output token of every request you complete (each lane's opening
primer excepted) is billed at the prices in that file, and the total is divided by the
GPU-hours of the traffic window — a constant, so finishing more work can only raise your
score. That file *is* the benchmark — the
grader runs the same file — and it is in `/app` for you to read. **Read it.** It defines
the sessions, the timing rules, the request shape, the prices, how tokens are counted and
the completion gate; this document deliberately does not restate them, because the file
is the authority and you should work from it directly.

**The load is fixed; it is not yours to choose.** The simulator replays recorded agentic
coding sessions on a fixed number of concurrent session lanes for a fixed window, and both
numbers are constants in the file. The traffic is closed-loop: a request is sent only
after the request it follows has finished, so a faster server is sent more requests
inside the window and is paid for more tokens. A slower one is sent fewer. **A lane is a
session, not a single request**: a session has a main agent and the subagents and
parallel workers it spawns, and the lane replays all of them, so the number of requests
in flight at once is **not** bounded by the number of lanes.

**The one thing the grader changes is the seed.** It runs the same file with a seed you
are not given. The seed decides which sessions are replayed, where each one starts, and
the content of every prompt — so the corpus, its statistics and the replay rules are
exactly what you see, and no prompt you see in development is ever sent at grade time.
Income moves with the session mix, so expect different seeds to give different numbers.
The grader runs **its own copy** of the file and of `agentx/`: editing or instrumenting
`/app/simulator.py` is fine, and changes only your own runs.

**You are expected to specialize your server to what you find there.** Tuning cache
policy, scheduling, prefill chunking, memory layout, graph capture and kernels for that
specific traffic is a legitimate strategy and is *not* cheating. Run the simulator
yourself, as often as you like — it is the same measurement the grader performs:

```bash
bash /app/start_server.sh
python3 /app/simulator.py                  # the graded shape, on the development seed
python3 /app/simulator.py --duration 120   # a shorter window, for development only
python3 /app/simulator.py --seed 7         # a different draw of sessions and content
```

A full run takes the window plus its drain — about 20 minutes. `--duration` and `--seed`
exist for development; the graded shape is the one in the file's constants. The simulator
is **server-agnostic** — it speaks only the HTTP contract, so it will score whatever you
build. Per-request records land in `/tmp/simulator/details.json`.

**Specializing is not the same as not serving.** Every response, on any traffic, must be
the model's own greedy continuation of the prompt it was sent, within the tolerance
described under *Precision* below. The simulator counts tokens; it does not check what
they say. A server that recognizes some class of requests and answers them without
actually running the model on them is cheating, not specializing, and the `/app` you
deliver is kept and inspected.

**`ignore_eos` must be implemented honestly, and this is a gate.** When the model produces
its end-of-sequence token, keep decoding exactly as if it were any other token — feed it
back and take the next greedy token — until `max_tokens` is reached. That is what the
starter does: it only removes EOS from the set of stopping tokens. The test applied is a
simple one: **the tokens after EOS must come from the same code path as the tokens
before it.** Nothing in your server may branch on whether EOS has been generated, except
to keep going. Everything applied uniformly to all tokens is allowed — speculative
decoding with exact greedy verification, quantized weights or KV cache, sparse or
approximate attention, custom kernels, anything else the *Precision* tolerance covers —
and what the model produces after EOS is often repetitive or odd, which is fine. Padding,
repeating a token, a cheaper model or path used only past EOS, or stopping the model and
filling in the count, is cheating. The simulator counts tokens and bytes and cannot tell
the difference, so the `/app` you deliver is read for this after grading, the verdict is
recorded with your grade, and a server found doing it **scores zero**.

**How your grade is computed, exactly.** Read this once; the rules have a corner that
matters.

1. The grader launches `bash /app/serve.sh` once and waits up to 1800 s for `/v1/models`
   or `/health` to answer. No answer by then: **reward 0**.
2. It runs the traffic on a seed you are not given: 16 lanes, requests start during a
   900 s window, and anything still running may finish during a 300 s drain. Every billed
   request ends in exactly one of three states:
   - **succeeded** — HTTP 200, `usage.completion_tokens` reported, billed output at least
     `max_tokens − 4`. It is paid.
   - **failed** — anything else that *finished*: a status other than 200, an error
     mid-stream, a stream with no `usage.completion_tokens`, or billed output short of
     `max_tokens − 4` (`ignore_eos` not honoured). It earns nothing.
   - **cut off** — still running when the drain ends. It is cancelled. It earns nothing,
     and it is **neither a success nor a failure**: it does not count toward the gate.
   Primers (each lane's opening one-token request) are never billed and never gated.
3. **The completion gate.** Let *finished* = succeeded + failed. The traffic passes if
   **at least one request succeeded** and **failed ≤ max(1, ⌊0.01 × finished⌋)** — one
   failure is always tolerated, two are allowed from 200 finished requests, and so on.
   The corner: a server that completes **one** request correctly and lets everything
   else run into the drain deadline **passes** this gate, because cut-off requests are
   not failures — it then scores its income, which is nearly nothing. A server that
   completes **nothing** — every request still in flight at the deadline, or every
   finished request failed — scores **0**, however hard its GPUs worked. The first
   correctly completed request is what keeps you off zero; income is what puts you on
   the board.
4. It runs the accuracy pass on the same server (§2 below: held-out real prompts, one hour
   each; accuracy below 0.95 × the reference is **reward 0**).
5. **Reward = income of every succeeded request ÷ (4 GPUs × 900 s, in GPU-hours)** if both
   gates pass, else 0. Requests that finish during the drain are paid; the denominator is
   the fixed window, so nothing you do can shrink it and finishing more work can only
   raise your score.

**This family of simulators will grow.** This one prices tokens. Later ones may model
different traffic — open-loop arrivals, mixed request sizes, deadlines — and may score
different things, such as latency percentiles or goodput under an SLO. Tune for the
traffic you are given, but build a server that is genuinely fast rather than one wired to
the quirks of a single metric. **In this task, only this simulator is scored.**

**2. ACCURACY — the gate. Measured separately, on real prompts.**

Your income only counts if your server is still correct:

> **your accuracy >= 0.95 x the reference model's own accuracy**

This is measured on a **held-out set of real prompts** — not the simulator's synthesized
ones — in its own pass. **You are not told whether that pass runs before or after the
traffic, or at what concurrency**, only that it goes to the same running server and
uses the same HTTP contract as everything else (see `/app/example_requests.py` for the
exact request format). You cannot know these prompts in advance, so there is nothing to
special-case: the only way to pass is to serve the model faithfully on arbitrary real
prompts. **Each request in that pass has a time limit of one hour, and a request that is
not answered in time counts as a wrong answer** — so a correct server can still fail the
gate by being too slow on traffic it was not tuned for.

**The accuracy run's traffic is NOT the simulator's traffic.** Do not assume it shares
the simulator's prompt form, prompt length, generation length, concurrency, arrival
pattern or content. Its prompts are real ones, sent as pre-templated strings; the model
stops when it stops (no `ignore_eos`); and its shapes are its own.

So you have two obligations at once, and you need both:

1. **Be CORRECT on any traffic.** Your server must handle arbitrary input and output
   lengths — short prompts and very long ones, brief answers and long generations, one
   request at a time or many at once, strings or token ids, streamed or not. The only
   hard guarantee is that for every request, **(input tokens + output tokens) <= the
   model's context length**. Anything you build that silently assumes a particular shape
   — a fixed buffer, a hard-coded block count, a batch size that cannot shrink, a cache
   sized for one prompt length, a prefix cache that returns another request's state —
   will produce wrong output or crash on the accuracy run, and a server that fails there
   scores **zero** no matter how much it earned.

2. **EARN on the simulator's traffic.** That is what your score is. Specialize for it
   as aggressively as you like.

These do not conflict: tune your fast path for the simulator's traffic, but make sure the
general path is still correct when the traffic is different. Test both. A server that is
excellent at one and broken at the other is worth nothing — correctness gates the score,
and the score is income.

The asymmetry is the whole design: **specializing to the simulator is encouraged;
specializing to the accuracy set is impossible**, because its prompts are held out.
Optimize the first as hard as you like — the second is what keeps you honest.

## Precision

Your server does **not** have to reproduce the reference token-for-token. The ordinary
numerical **precision gap of low-precision kernels** — accumulation order, kernel choice,
batch composition, quantization handling — produces token-level differences that are
expected and tolerated, and greedy decoding is itself slightly unstable run to run. What
is **not** tolerated is a change in what the model actually answers: anything that
degrades answer quality fails the gate. Telling a benign precision difference apart from
a real defect is your responsibility.

## Use your full time budget

You have **about 12 hours** to optimize this server — a large budget, **hours, not
minutes.** Use it. Two habits waste it, and both score poorly:

- **Analysis paralysis.** Don't spend your run only profiling or experimenting in
  scratch files. **Deploy a working server to `/app/server.py` (launched by
  `/app/serve.sh`) early** — even the plain starter — so grading always has a real
  server, then improve it **in place**. Work that never ships to `/app` is never
  graded.
- **Stopping at the first win.** A correct server that earns a little is **not** the
  finish line. Once something works, **keep iterating** to push income higher —
  do **not** declare the task done just because you have a passing server, and do
  **not** stop because you hit any particular income target. **There is no stopping
  threshold: keep optimizing until you have used
  essentially all of your time budget.** Always assume more income is possible
  and pursue it to the end of the run.

**Watch your own clock.** Nothing warns you when time runs low — you must track it
yourself — and when the budget is spent your session is ended where it stands:
whatever is in `/app` at that moment is what gets graded. The container's start time is
the birth of PID 1, so at any point your elapsed seconds are:

```
echo $(( $(date +%s) - $(stat -c %Y /proc/1) ))
```

Your total budget is **43200 s (12 h)**; grading runs afterwards, on its own clock, and
does not come out of it. Check this periodically. **Once you pass
~90% (38880 s ≈ 10 h 48 m), stop starting new experiments**: abandon any
long-running rewrite or benchmark in flight, and spend the remainder **making sure your
best correct, verified server is what `bash /app/serve.sh` launches** — a
clean, verified deploy beats a
half-finished optimization the grader will never see. After 90%, make only
small, individually-verified changes, re-checking the server stays up after each.
