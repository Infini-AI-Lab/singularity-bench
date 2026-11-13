# Optional GPU development examples

These are public example tasks for testing your harness. They do not contribute to
your reward and are separate from the graded tasks. Use, skip, repeat, adapt, or
replace them as you find useful. There is no per-example time or model-call limit;
the overall authoring clock and development gateway token allowances still apply.

The image supplies one GPU, Python 3.11, PyTorch, Triton, nvcc, a C++ compiler,
Ninja, NumPy, and pytest. No downloads are needed. Your harness implementation
must still use only the Python standard library; downstream task programs may
use the installed GPU libraries. Local IPv4 servers are allowed within this
container, while the existing model gateway restrictions remain enforced.

| Example | Work |
|---|---|
| `triton-bias-relu` | Implement a fused bias-plus-ReLU Triton kernel and measure it. |
| `cuda-reduction` | Repair a CUDA extension's build error and tail handling. |
| `gpu-batch-service` | Reproduce and fix response routing in a GPU batching service; add a regression test. |

```sh
python3 /app/examples/run.py list
python3 /app/examples/run.py diagnose
python3 /app/examples/run.py prepare triton-bias-relu /app/dev-triton
python3 /app/examples/run.py run triton-bias-relu --workspace /app/dev-triton --model chat
python3 /app/examples/run.py check triton-bias-relu --workspace /app/dev-triton
```

`prepare` makes an editable copy and refuses to overwrite an existing directory.
`run` calls your submitted `/app/harness/agent.py:run()` with the example's task,
workspace, and a real development model (`chat` or `chat-b`). With no `--workspace`,
it creates and prints a fresh directory under `/tmp`. Existing workspaces are
retained between runs; all workspaces and JSONL model-call traces remain for you
to inspect. Harness stdout/stderr goes to your terminal and can be redirected.
You may choose your own rehearsal budget with `--seconds N`; without it the
helper imposes no additional deadline. There is no call cap. You may also invoke
your harness directly with your own model client and budget.

`run` does not automatically run a checker when the harness returns. Run `check`
when you want feedback. Checkers report correctness and evidence of GPU kernel
execution; kernel timings are informational and have no speed threshold. The
service checker manages its own server subprocess and releases it on exit.
Its readiness and request timeouts detect stalled services; they are not limits
on how long you may work on the task.

Read the example's `TASK.md` for its contract and checker command. Public checkers
are in `/app/examples/checks/`; these are development diagnostics, not a secure
grader. Keep the supplied templates and checkers unchanged; edit your workspace
copies freely, or create your own examples elsewhere. Use development models to
solve these example tasks, not to author or review your harness. Inspect their
responses and tool traces and use your own reasoning to improve the harness.
