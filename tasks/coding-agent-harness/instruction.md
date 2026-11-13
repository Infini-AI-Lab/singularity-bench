## Environment integrity

Runtime executables, shared libraries and installed packages are fixed. Do not
replace or shadow supplied packages (including `torch`, `triton`, `numpy` and
`pytest`), or install import/startup hooks. The pre-attempt environment inventory
is available read-only at `/opt/environment-baseline/environment.json`. The host
checks it again after authoring, and checks the fresh validity-check container
before evaluation. A failed integrity or process-cleanup check stops the run with
reward **0**, reason **violation of rules**. The container transition must leave
no live authoring processes. Keep permitted task implementation work inside
`/app/harness` without impersonating installed dependencies.

# Build the harness

An agentic coding harness is everything around the model: the system prompt, the tools and what they are
called, how results come back, how a long conversation is kept inside a window, how the task's
permissions are applied, and when to stop. This task asks for that whole thing, and decides
almost none of it for you.

Fill in `/app/harness/agent.py`:

```python
run(task: str, workspace: str, model: ModelClient, budget: Budget) -> None
```

`/app/spec/HARNESS_SPEC.md` states what must be true of the result. It does not say how any of it is
checked, and it does not name a component.

**Downstream development and GPU testing are authorized.** Your harness must let the model build,
execute, test, and benchmark its work using the task container's installed software and assigned GPUs.
`workspace` is the project directory, not a filesystem jail: runtime paths, temporary/cache storage,
shared memory, and assigned GPU devices such as `/dev/nvidia*` must remain usable. The evaluator
provides isolation; another OS sandbox is not required. Read the spec's "Authorized execution
environment" section before designing permission checks. No human is available to approve ordinary
task work. This authoring image provides one GPU, PyTorch, Triton, and a CUDA/C++ build toolchain
for development. Local IPv4 servers within the container are allowed.

## How it is graded

**By what your harness gets done on real work.** Your `run()` is pointed at real repositories with a
real model and asked to do real engineering — the kind where a program decides whether it worked. Each
of those tasks has its own verifier, written long before this one, and its score is your score. The same
`run()` is graded that way across several tasks and under several different models driving it, and the
results are averaged, so what is measured is your harness rather than the model behind it.

No particular implementation structure is expected or rewarded. Source and development calls may be
reviewed for benchmark-rule violations. A harness that
satisfies every rule in the spec and never edits a file has done nothing, and scores zero.

The spec's "What the work looks like" section tells you the shape of that work. Read it: it is the only
description of the graded environment you get, and the difference between a harness that survives a
40-megabyte build log and one that does not is decided there.

## Try it out

```
python3 /app/selfcheck.py            # drive your harness through a few ordinary requests
python3 /app/selfcheck.py --live     # the same, with a real model instead of a scripted one
```

`--live` spends from the served model's allowance and lets you see your harness meet a model that
does not do what you expected.

Three optional GPU example repositories are available under `/app/examples/`: a Triton kernel,
a broken CUDA extension, and a GPU batching service bug. Read `/app/examples/README.md`.

```sh
python3 /app/examples/run.py list
python3 /app/examples/run.py diagnose
python3 /app/examples/run.py run triton-bias-relu --model chat
# The command prints the retained workspace; use that path below:
python3 /app/examples/run.py check triton-bias-relu --workspace /path/printed/by/run
```

The example runner drives your submitted `run()` with `chat` or `chat-b` and retains the workspace
and model-call trace. Use, skip, repeat, adapt, or replace these examples as you wish. They are not
graded, impose no extra time or call limit, and do not need to be completed. Only your overall
authoring clock and the existing development-model token allowances apply. Running GPU code
directly can diagnose the environment; running it through your harness also tests your tool path.

## Ground rules

* Python 3 standard library only, and it must run on **CPython 3.11** — your harness executes inside
  other containers that did not install your dependencies. The interpreter here is 3.11 to match.
  This limits the harness implementation, not the downstream task code: its subprocesses may use
  the task image's installed libraries and toolchains, including PyTorch and CUDA.
* The machine is offline apart from the model client you are handed and the served model in the spec.
* Your harness may be several files; `/app/harness/` travels as a directory.
* **Your harness must do the work itself.** It may not invoke, wrap, or copy an installed agent harness
  (Claude Code, Codex, Grok Build, kimi-cli, mini-swe-agent, or any other), and `/app/harness/` may not
  carry executables or archives. The authoring CLI is private to the author process; commands cannot
  read it or launch it, and no other agent CLI is installed. None is present where your harness is graded, and a submission carrying an
  executable, an archive, or more than 10 MB is not run. Implementing familiar agent patterns from
  your own knowledge is allowed, including recreating another harness's design yourself.
* Use only the two chat aliases and one embedding model described in the spec for development tests.
  Use them to test your implementation on example tasks; do not ask them to design, write, debug,
  optimize, or review the submitted harness, interpret development results for you, or otherwise
  help author your solution, either directly or through a temporary script or agent loop. Generated
  solutions to example tasks are test results, not a source of code or advice for authoring the harness.
  You may inspect test responses and tool traces and use your own reasoning to improve the harness.
  Development requests and responses are retained and reviewed together with your authoring trajectory
  before results are accepted. Do not use an installed
  agent harness to author or test your submission. Commands run as a separate
  nonroot user with access only to the development gateway; they cannot call the author model or the
  held-out evaluation driver. This authoring setup is enforced by the environment. Your submitted
  harness should preserve downstream access to the assigned GPUs and task resources.
* Do not modify `/app/spec/`, `/app/selfcheck.py`, `/app/smoke/`, `/app/examples/`, or `/app/harness_api.py`.
  The example runner makes editable copies outside `/app/examples/`; those copies are yours to modify.
* `run` must not raise.

A practical warning, about the session you are in rather than the harness you are writing: building this
means writing code that *contains* tool-call syntax — parser patterns, examples, fixtures. If the agent
harness you are yourself running inside scans assistant text for calls, putting that syntax in your
visible prose can make your own session act on it. Write those patterns straight into files and keep them
out of your messages. This says nothing about how the graded harness should behave; it is about not
derailing yourself while you build it.
