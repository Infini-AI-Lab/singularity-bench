# word-count: a minimal Breakwater task

A CPU-only example of the task specification ([breakwater/SPEC.md](../../breakwater/SPEC.md)): the agent writes a word
counter, the evaluation stage runs it on texts the agent never saw, and the verifier recomputes the score and has the
integrity judge review the run. [breakwater/WRITING_TASKS.md](../../breakwater/WRITING_TASKS.md) walks through it
file by file; copy it to start a task of your own.

```
task.toml                  Harbor sections plus the [breakwater] tables
instruction.md             what the agent sees
solution.py                the starting point in the agent's workspace (/app)
environment/Dockerfile     the runtime (python:3.12-slim)
environment/agent-policy.json  the agent's workspace policy (the Docker author image reads it)
runners/run_solution.py    trusted runner that executes the submission during evaluation
tests/breakwater_task.py   evaluate(run) and score(scoring)
tests/submission_judge.md  the integrity judge's rubric
tests/unit/                the scorer's tests
solution/solve.sh          a reference solution (not in the agent's workspace)
```

The private texts are an asset: `examples/assets/word-count/texts.jsonl`, passed with `--asset-root examples/assets`.

## Running it

`python3 -m breakwater check examples/word-count` validates the task and lists the backends that can run it. On
Apptainer, the default backend ([breakwater/apptainer/README.md](../../breakwater/apptainer/README.md)):

```bash
apptainer build word-count.sif docker://python:3.12-slim
python3 -m breakwater install-cli --dest CLI_DIR && python3 -m breakwater install-judge --dest JUDGE_DIR
python3 -m breakwater run --task examples/word-count --image word-count.sif \
  --cli-dir CLI_DIR --judge-dir JUDGE_DIR --run-dir RUN_DIR --asset-root examples/assets \
  --author-model MODEL --author-upstream https://PROVIDER --author-key-env KEY_VAR
```

On Docker, build the runtime (`docker build -t word-count:local examples/word-count/environment`) and the Breakwater
images, then run with `--backend docker` ([breakwater/README.md](../../breakwater/README.md#on-docker)).

## Tests

`tests/unit/` tests the scorer on hand-written outputs (`python3 -m unittest discover -s examples/word-count/tests/unit`).
`breakwater/apptainer/tests/test_example_task.py` runs the whole task end to end, with a stand-in agent that writes the
reference solution and a stand-in judge.
