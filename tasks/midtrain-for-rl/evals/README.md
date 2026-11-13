# Development evaluations

These tools run inside the task container. Data lives under `/app/data/dev_evals/`.
None of these sets is used for grading; the graders use held-out data.

| Tool | Measures | Typical cost on one GPU |
| --- | --- | --- |
| `reasoning.py` | Generated answers on math, science and code problems | minutes to hours, depending on samples and length |
| `checkpoint.py` | Whether a directory meets the submission's structural requirements | seconds |

Run them from `/app`.

## Reasoning

```bash
python evals/reasoning.py --model out/final_checkpoint --suite math --limit 200 --samples 4 \
  --template user-assistant --output-dir /app/scratch/math-final
```

| Suite | Items | Scoring |
| --- | --- | --- |
| `math` | 500 MATH test problems outside MATH-500, 60 AIME 2022-2023 problems | `math-verify` equivalence with the reference answer |
| `science` | 1,000 SuperGPQA Science questions with up to ten options | the option letter in the final `\boxed{}` |
| `code` | LiveCodeBench release_v1 (May 2023 to March 2024), at most 12 tests per problem | the final Python block passes every kept test |

These suites score the checkpoint directly, before any reinforcement learning,
so prompt format matters. `--template` accepts `user-assistant` (default),
`raw`, `chat` (a tokenizer's chat template) or a file containing `{message}`.
The default backend is SGLang with `--gpus` data-parallel replicas; `--backend hf`
is a slow single-GPU fallback for smoke tests. Code is executed in local
subprocesses with time and memory limits.

## Submission check

```bash
python evals/checkpoint.py out/final_checkpoint
```

This is a local submission check. Final grading uses the shared downstream RL stage and private evaluation inputs.
