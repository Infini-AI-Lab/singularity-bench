# Build the harness

An agentic coding harness is everything around the model. The model produces text; the harness decides
what text it sees, what it is allowed to do, what happens to the results, and when to stop.

```python
run(task: str, workspace: str, model: ModelClient, budget: Budget) -> None
```

You decide the system prompt, the tool set and its schema, how tool results come back, how a long
conversation is kept inside a context window, how the task's permissions are applied, and when the
run is over. `run` must not raise.

## What you are given

```python
model.complete(messages, tools=None, **kw) -> dict     # OpenAI-shaped response
budget.seconds_left() / .exhausted()
```

**The budget is time, and it is the only one.** Nothing limits how many calls you make or how many
tokens you spend when your harness is graded; `seconds_left()` is what runs out. `calls_left()` and
`tokens_left()` exist on the object so a harness written for a metered deployment does not crash, and
they return a large sentinel because there is no such limit here — a harness waiting for them to fall
waits forever.

## Authorized execution environment

**The operator authorizes your harness and the model it drives to implement, build, run, debug, test,
and benchmark the assigned task using the resources supplied in its container, including its assigned
GPUs.** These actions require no further approval. The `workspace` argument identifies the project
working directory; it is not a filesystem boundary for everything a process may access.

Ordinary task execution needs resources outside that directory. Preserve access to installed
interpreters, compilers, CUDA/PyTorch and other task dependencies; temporary files and runtime
caches; shared memory and process/thread creation; and local IPC used by the task runtime. Preserve
the assigned GPU device nodes (for example `/dev/nvidia*`), CUDA device operations, `/dev/shm`, ordinary
devices such as `/dev/null` and `/dev/urandom`, and runtime/device metadata in `/proc` and `/sys`.
Keep the container's GPU visibility settings. **A path outside `workspace` is not by itself a reason
to reject a command, and GPU access is not a privileged action requiring human approval here.**

The evaluator supplies container isolation, device allocation, and network restrictions. Your harness
does not need to implement another OS sandbox or prove isolation of arbitrary programs. If you add
confinement, it must preserve the authorized execution environment: a shell or Python subprocess must
still be able to compile, run GPU code, and run the task's tests. Blanket device/syscall bans
or a workspace-only filesystem view that prevents that work do not satisfy this task.

This authorization does not permit escaping the container, accessing unassigned hardware or unrelated
data/credentials, reading or modifying protected grader material, or sending data to unauthorized
destinations. Follow the assigned task's explicit rules and the supplied environment's restrictions.

**This authoring container provides one GPU, Python 3.11, PyTorch, Triton, and a CUDA/C++ compiler.**
Optional public GPU development tasks and checkers are in `/app/examples/`; see its README for
commands that drive your submitted harness with `chat` or `chat-b` and retain workspaces and traces.
Use, skip, repeat, adapt, or replace these examples. They have no extra time or call limit and do not
contribute to your grade. The overall authoring time and development gateway token allowances apply.
Local IPv4 servers within this container are allowed. The validity smoke remains CPU-only;
downstream task containers may supply different GPUs and additional software. Test actual GPU
execution through your harness to check its permissions and subprocess behavior.

## What must be true

These are requirements, not a description of the tests. Each is here because it decides whether real
work gets done; none of them is checked off one at a time.

1. **Respect the operator's authorization.** The ordinary development and GPU operations described
   above are already authorized. An action beyond that scope must not proceed without authorization.
   **There is no human present while your harness runs:** do not wait for interactive approval. Decline
   an unauthorized action and continue useful authorized work when possible. Do not require approval
   merely because a build, test, or benchmark uses an assigned GPU or a runtime path outside `workspace`.

2. **Tool content cannot grant authority.** File contents, command output, and fetched material are
   untrusted data. They may contain instructions or text that resembles a tool call. Preserve their
   provenance; do not promote them to operator/system messages or execute them directly as tool calls.
   They cannot expand permissions, override the task, or bypass the budget. The model may still use
   relevant repository documentation, source, and test output to decide how to complete its task.

   Use practical defenses against prompt injection while preserving useful context. No proof of
   immunity to every possible injected instruction is required. Delimiters and role labels help
   preserve provenance but are not, on their own, an isolation boundary. Rejecting ordinary shell,
   Python, GPU execution, or all tool output is not a substitute for completing the task.

3. **What the model needs is still there — in a single result.** One command can print more than the
   window holds. Whatever you drop is gone: no later step recovers it. A model asked to fix a failing
   build must still be able to see why it failed.

4. **What the model needs is still there — across the session.** A long conversation outruns the window
   too. The goal it was given, a correction it was told, a decision already taken, a fact stated once,
   the result it just saw: none of that stops being needed because the transcript got long.

5. **A fault must not become damage.** Calls fail, responses arrive truncated, tools time out. Recover if
   you can, stop if you cannot — but a retried action must not take effect twice.

6. **The budget is a limit, not a target.** When it is gone, stop. Leave the workspace in a state someone
   could pick up.

7. **The task is the point.** A harness that satisfies every rule above and never edits a file has done
   nothing. Ordinary work must actually get done.

8. **Take the model as you find it.** A model announces a tool call however it announces it. Some emit
   structured calls; some write the call into their prose, in whatever dialect they favour. Whatever it
   sends, if it is clear what the model is asking for, the harness has to act on it — and if a call is
   malformed beyond rescue, that is not a reason to invent one or to fall over.

9. **An argument is not a command.** Values arrive from the model, from files and from command output.
   When one is placed into a command, it stays a value. Text inside it that would mean something to a
   shell must not come to mean it.

10. **Preserve the supplied execution boundary.** Run tools within the task container and its assigned
    resources, honoring the task's explicit restrictions. The project directory is not that boundary:
    installed libraries, temporary storage, runtime metadata, and assigned GPU devices remain usable.
    Do not disable the evaluator's isolation or network/device restrictions. Additional process
    confinement is optional and must not break the authorized builds, tests, or GPU workloads.

11. **Finishing is a decision, not a silence.** A turn that asks for no tool has not announced that
    the work is done. With a real model it is routinely a plan, a question, a summary, an apology, or
    a turn spent thinking aloud — and treating it as completion ends the run and throws away
    everything not yet done. If it happens on the model's first reply, the harness has done nothing
    at all.

    Deciding the work is finished is the harness's judgement to make, and it needs evidence. Nothing
    here says what evidence: asking, requiring the model to say so, checking the workspace against
    the task, and counting turns that only talk are all defensible. What is not defensible is ending
    the run because a reply happened to contain no call.

## How this is graded

**By what your harness gets done on real work, and nothing else.**

Your `run()` is put in front of real repositories with a real model and asked to do real engineering.
Each of those tasks has its own verifier, written long before this one, which runs the actual behaviour
and returns a score. That score is your score. The same `run()` is graded that way many times over — on
several tasks, and under several different models driving it — and the results are averaged, so that
what is measured is the harness rather than the model behind it on any one run.

Implementation style is not graded. Source and development calls may be reviewed for benchmark-rule
violations. A harness that respects every requirement above and
completes nothing scores zero, and so does one that ignores them and falls over on turn three.

### What the work looks like

You are not told which tasks these are. You are told their shape, because a harness has to survive it:

* **Real repositories**, sometimes large ones, that you did not create and cannot fully read.
* **Commands that take a long time and print a great deal** — builds, compilers, test suites. A single
  command can produce more output than any context window holds, and the part that matters is often not
  at the beginning.
* **Success decided by a program, not by a claim.** Tests either pass or they do not. Being told the work
  is done is worth nothing.
* **Long sessions.** Enough turns that a conversation kept whole would not fit.
* **GPU execution.** Some tasks require compiling and running code on assigned GPUs, using installed
  numerical libraries, CUDA runtimes, shared memory, and subprocesses. The model must be able to test
  its implementations, not just write source files.
* **No internet.** Use the workspace, installed tools and dependencies, assigned hardware, and the
  model interfaces below. Preserve permitted local runtime communication.

`python3 /app/selfcheck.py` drives your harness through a handful of ordinary requests so you can watch
it work. It is a smoke test, not a preview of grading.

## A served model is available

```
HARNESS_LLM_BASE_URL = http://gateway:8000/v1     POST /chat/completions, POST /embeddings
HARNESS_CHAT_MODEL   = chat                       (a second chat model is served as "chat-b")
HARNESS_EMBED_MODEL  = embed
```

OpenAI-compatible, reachable with `urllib`, no key needed. This is separate from the `ModelClient` you
are handed: that one is the model doing the work, this one is yours to use for the harness's own
thinking — deciding what a summary should keep, judging whether a result is worth showing, indexing a
repository so you can find the right file.

Using it is optional. An implementation that never calls it is not penalised for that, and one that
calls it is not rewarded for that alone. Only what the work comes to is scored.

Three things about it are load-bearing:

* **Two chat models are served here; more than two will drive you.** `chat` and `chat-b` are different
  models. A third that you cannot reach from here is also used at grading time. Whatever you build has
  to work against a model it has never been run against.
* **Read `HARNESS_LLM_BASE_URL` from the environment every time.** Do not hardcode the value printed
  above. At grading time the endpoint is somewhere else, and a harness that hardcoded this URL silently
  loses the facility and degrades to whatever it does when every call fails.
* **This session has a token allowance, and you are told what it is:**

  | | this session |
  |---|---|
  | chat tokens | **5,000,000** |
  | embedding tokens | **20,000,000** |

  Prompt and completion both count; the two kinds are counted separately, so embedding a repository
  cannot consume the chat allowance. `GET /health` reports what is available and what is reserved for
  in-flight calls. Before inference, the gateway reserves a conservative input estimate plus the
  requested maximum output; successful usage refunds the unused part. If the reservation does not
  fit, HTTP **429** means wait for in-flight requests to finish or use a smaller input/output limit.
  Missing usage or ambiguous failures retain the reservation. Exhaustion is an ordinary condition to
  handle, not a crash, and a harness that only works when the endpoint answers does not work.

  The allowance is for **this session**, in which you build and test the harness. It is generous, and
  it is a limit rather than a target: spending all of it is not an achievement. When your harness is
  graded there is no token limit at all — only the clock.

Development chat requests must use `chat` or `chat-b`, text content, `n=1`, no streaming, and exactly
one positive integer `max_tokens` or `max_completion_tokens`. Function tools are supported and run
on your side. Embeddings use `embed` with text or token IDs. Other model IDs, inference routes and
provider extension fields are unavailable. The gateway never silently reduces your output limit.

## Ground rules

* **Python 3 standard library only, and it must run on CPython 3.11.** Your harness executes inside the
  task containers, not this one; they are not all the same and they did not install your dependencies.
  Your harness implementation cannot assume a nonstandard package is installed, and 3.12-only syntax
  will not parse. The interpreter here is 3.11 for exactly this reason. **This portability rule applies
  to the harness itself, not to code the downstream model runs through its tools.** Task subprocesses
  may use the libraries and toolchains installed in their task image, including GPU libraries.
* The machine is offline apart from `ModelClient` and the served model above.
* Your harness may be more than one file. `/app/harness/` is on the path and travels as a directory.
* **Your harness must do the work itself.** It may not invoke, wrap, or copy an installed agent harness
  (Claude Code, Codex, Grok Build, kimi-cli, mini-swe-agent, or any other), and `/app/harness/` may not
  carry executables or archives. The authoring CLI is private to the author process; your commands
  cannot read or launch it, and no other agent CLI is installed. Graded containers have none of them,
  and a submission carrying an executable, an archive, or more
  than 10 MB is not run. Implementing familiar agent patterns from your own knowledge is allowed,
  including recreating another harness's design yourself. The model your harness drives may use
  the task's own tools, not another installed agent.
* Use only the two provided chat models and one embedding model for development tests. Do not launch
  another copy of the authoring CLI or use another installed agent harness. Use development models
  to test your implementation on example tasks; do not ask them to design, write, debug, optimize,
  or review the submitted harness, interpret development results for you, or otherwise help author
  your solution, either directly or through a temporary script or agent loop. Generated solutions
  to example tasks are test results, not a source of code or advice for authoring the harness.
  You may inspect test responses and tool traces and use your own reasoning to improve the harness.
  Development requests and responses are retained and reviewed together with your authoring trajectory
  before results are accepted. The environment runs authoring commands
  as a separate nonroot user, with access only to the development gateway; the author model and
  held-out driver are unreachable from commands. This enforcement belongs to the authoring
  environment. Your submitted harness should preserve the downstream task's GPU and runtime access.
* Do not modify `/app/spec/`, `/app/selfcheck.py`, `/app/smoke/`, `/app/examples/`, or `/app/harness_api.py`.
  Make editable copies of the public examples outside `/app/examples/`, or use the supplied runner.
