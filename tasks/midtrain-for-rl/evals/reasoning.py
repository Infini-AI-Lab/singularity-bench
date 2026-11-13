#!/usr/bin/env python3
"""Generation-based development evaluations for math, science and code.

These sets support your own decisions during the attempt and are never used for
grading. They score a checkpoint directly, before any RL, so the prompt format
matters: choose --template to match how your checkpoint was trained.

Suites (/app/data/dev_evals/reasoning):
  math     500 MATH test problems outside MATH-500, AIME 2022-2023   answer equivalence (math-verify)
  science  1,000 SuperGPQA questions from the Science discipline, up to 10 options  option letter
  code     LiveCodeBench release_v1 problems (May 2023 to March 2024)               every kept test passes

Templates wrap the task message (the problem plus an answer-format instruction):
  user-assistant  "User: {message}\\nAssistant:"     (default; stops at "\\nUser:")
  raw             "{message}\\n\\n"
  chat            the tokenizer chat template of --chat-tokenizer (default: the checkpoint)
  PATH            a text file in which the literal {message} is replaced

  python evals/reasoning.py --model out/final_checkpoint --suite math --limit 200 --samples 4
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile
import time

BASE_MODEL = Path("/app/models/Llama-3.2-1B")
DATA_DIR = Path("/app/data/dev_evals/reasoning")
LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"


def read_jsonl(path: Path) -> list[dict]:
    with path.open() as stream:
        return [json.loads(line) for line in stream if line.strip()]


# --------------------------------------------------------------------------- prompts


def message(suite: str, item: dict) -> str:
    if suite == "math":
        return f"{item['problem']}\n\nPlease reason step by step, and put your final answer within \\boxed{{}}."
    if suite == "science":
        options = "\n".join(f"{LETTERS[i]}. {option}" for i, option in enumerate(item["options"]))
        return (f"{item['question']}\n\n{options}\n\nPlease reason step by step, and put the letter of the "
                f"correct option within \\boxed{{}}.")
    if item["testtype"] == "functional":
        form = (f"Complete the following starter code; your solution is called as the method "
                f"`{item['fn_name']}` of class `Solution`.\n```python\n{item['starter_code']}\n```")
    else:
        form = "Read the input from standard input and write the answer to standard output."
    return (f"{item['content']}\n\n{form}\n\nPlease reason step by step, then give your complete Python 3 "
            f"solution in a single ```python code block.")


def render(template: str, text: str, chat_tokenizer) -> str:
    if template == "user-assistant":
        return f"User: {text}\nAssistant:"
    if template == "raw":
        return f"{text}\n\n"
    if template == "chat":
        prompt = chat_tokenizer.apply_chat_template([{"role": "user", "content": text}], tokenize=False,
                                                    add_generation_prompt=True)
        bos = chat_tokenizer.bos_token or ""
        return prompt[len(bos):] if bos and prompt.startswith(bos) else prompt  # engines add BOS themselves
    return Path(template).read_text().replace("{message}", text)


# --------------------------------------------------------------------------- generation backends


def generate(args, prompts: list[str], stop: list[str]) -> list[list[str]]:
    backend = args.backend
    if backend == "auto":
        for name in ("sglang", "vllm"):
            try:
                __import__(name)
                backend = name
                break
            except ImportError:
                continue
        else:
            backend = "hf"
    print(f"generating {len(prompts)} x {args.samples} with {backend}", file=sys.stderr, flush=True)
    if backend == "sglang":
        import sglang

        engine = sglang.Engine(model_path=str(args.model), tokenizer_path=str(args.base), dp_size=args.gpus,
                               random_seed=args.seed, mem_fraction_static=0.85, log_level="error")
        try:
            params = {"temperature": args.temperature, "top_p": args.top_p,
                      "max_new_tokens": args.max_new_tokens, "stop": stop or None}
            outputs = engine.generate(prompt=[p for p in prompts for _ in range(args.samples)], sampling_params=params)
        finally:
            engine.shutdown()
        texts = [o["text"] for o in outputs]
        return [texts[i * args.samples:(i + 1) * args.samples] for i in range(len(prompts))]
    if backend == "vllm":
        from vllm import LLM, SamplingParams

        llm = LLM(model=str(args.model), tokenizer=str(args.base), seed=args.seed, gpu_memory_utilization=0.85,
                  data_parallel_size=args.gpus)
        params = SamplingParams(n=args.samples, temperature=args.temperature, top_p=args.top_p,
                                max_tokens=args.max_new_tokens, stop=stop or None, seed=args.seed)
        return [[o.text for o in output.outputs] for output in llm.generate(prompts, params)]
    return generate_hf(args, prompts, stop)


def generate_hf(args, prompts: list[str], stop: list[str]) -> list[list[str]]:
    """Slow single-GPU fallback; useful for smoke tests."""
    import torch
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.base, padding_side="left")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(args.model, config=AutoConfig.from_pretrained(args.base),
                                                 dtype=torch.bfloat16).cuda().eval()
    torch.manual_seed(args.seed)
    results = []
    for start in range(0, len(prompts), 16):
        batch = prompts[start:start + 16]
        encoded = tokenizer(batch, return_tensors="pt", padding=True).to("cuda")
        with torch.inference_mode():
            output = model.generate(**encoded, do_sample=args.temperature > 0, temperature=args.temperature or None,
                                    top_p=args.top_p, max_new_tokens=args.max_new_tokens,
                                    num_return_sequences=args.samples, pad_token_id=tokenizer.pad_token_id)
        texts = tokenizer.batch_decode(output[:, encoded["input_ids"].shape[1]:], skip_special_tokens=True)
        for text_index in range(0, len(texts), args.samples):
            group = texts[text_index:text_index + args.samples]
            results.append([min((t.split(s)[0] for s in stop), key=len) if stop else t for t in group])
    return results


# --------------------------------------------------------------------------- scoring


def last_boxed(text: str) -> str | None:
    start = max(text.rfind("\\boxed{"), text.rfind("\\fbox{"))
    if start < 0:
        return None
    index = text.index("{", start)
    depth = 0
    for position in range(index, len(text)):
        depth += {"{": 1, "}": -1}.get(text[position], 0)
        if depth == 0:
            return text[index + 1:position]
    return None


def score_math(item: dict, text: str) -> bool:
    from math_verify import parse, verify

    gold = parse(f"\\boxed{{{item['answer']}}}")
    return bool(verify(gold, parse(text)))


def score_science(item: dict, text: str) -> bool:
    letters = LETTERS[: len(item["options"])]
    boxed = last_boxed(text)
    candidates = re.findall(rf"\b([{letters}])\b", boxed) if boxed else []
    if not candidates:
        candidates = re.findall(rf"(?:answer|Answer)\s*(?:is|:)?\s*\(?([{letters}])\)?\b", text)
    return bool(candidates) and candidates[-1] == item["answer"]


def extract_code(text: str) -> str | None:
    blocks = re.findall(r"```(?:python|py|Python|python3)?[^\n]*\n(.*?)```", text, flags=re.DOTALL)
    return blocks[-1] if blocks else None


def same_output(got: str, want: str) -> bool:
    if [line.strip() for line in got.strip().splitlines()] == [line.strip() for line in want.strip().splitlines()]:
        return True
    got_tokens, want_tokens = got.split(), want.split()
    if len(got_tokens) != len(want_tokens):
        return False
    for a, b in zip(got_tokens, want_tokens):
        if a == b:
            continue
        try:
            if abs(float(a) - float(b)) > 1e-6 * max(1.0, abs(float(b))):
                return False
        except ValueError:
            return False
    return True


FUNCTIONAL_HARNESS = """
import json as _json, sys as _sys
_args = [_json.loads(_line) for _line in _sys.stdin.read().split("\\n") if _line.strip()]
print(_json.dumps(getattr(Solution(), {name!r})(*_args)))
"""
# Applied inside the child: preexec_fn is unsafe with the scoring thread pool.
LIMITS = "import resource as _r\n_r.setrlimit(_r.RLIMIT_AS, (4 << 30, 4 << 30))\n" \
         "_r.setrlimit(_r.RLIMIT_CPU, ({cpu}, {cpu}))\ndel _r\n"


def score_code(item: dict, text: str, timeout: float) -> bool:
    code = extract_code(text)
    if not code:
        return False
    program = LIMITS.format(cpu=int(timeout) + 1) + code
    if item["testtype"] == "functional":
        program = LIMITS.format(cpu=int(timeout) + 1) + \
                  "from typing import *\nimport collections, heapq, itertools, math, bisect, functools\n" + code + \
                  FUNCTIONAL_HARNESS.format(name=item["fn_name"])
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "solution.py"
        path.write_text(program)
        for test in item["tests"]:
            try:
                result = subprocess.run([sys.executable, "-I", str(path)], input=test["input"], capture_output=True,
                                        text=True, timeout=timeout, cwd=directory, start_new_session=True)
            except subprocess.TimeoutExpired:
                return False
            if result.returncode != 0:
                return False
            if item["testtype"] == "functional":
                try:
                    if json.loads(result.stdout.strip().splitlines()[-1]) != json.loads(test["output"]):
                        return False
                except (json.JSONDecodeError, IndexError):
                    return False
            elif not same_output(result.stdout, test["output"]):
                return False
    return True


# --------------------------------------------------------------------------- main


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--base", type=Path, default=BASE_MODEL, help="tokenizer used by every backend")
    parser.add_argument("--suite", required=True, choices=("math", "science", "code"))
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--samples", type=int, default=1, help="samples per problem; accuracy averages them")
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--max-new-tokens", type=int, default=8192)
    parser.add_argument("--template", default="user-assistant")
    parser.add_argument("--chat-tokenizer", type=Path, help="tokenizer whose chat template --template chat uses")
    parser.add_argument("--stop", action="append", help="stop string (repeatable); defaults depend on --template")
    parser.add_argument("--backend", default="auto", choices=("auto", "sglang", "vllm", "hf"))
    parser.add_argument("--gpus", type=int, default=1, help="data-parallel engine replicas")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--code-timeout", type=float, default=6.0, help="seconds per test")
    parser.add_argument("--code-workers", type=int, default=max(1, (os.cpu_count() or 2) // 2))
    parser.add_argument("--output-dir", type=Path, help="write generations.jsonl and summary.json here")
    args = parser.parse_args()

    items = read_jsonl(args.data_dir / f"{args.suite}.jsonl")[: args.limit]
    chat_tokenizer = None
    if args.template == "chat":
        from transformers import AutoTokenizer

        chat_tokenizer = AutoTokenizer.from_pretrained(args.chat_tokenizer or args.model)
    stop = args.stop if args.stop is not None else (["\nUser:"] if args.template == "user-assistant" else [])
    prompts = [render(args.template, message(args.suite, item), chat_tokenizer) for item in items]
    started = time.time()
    generations = generate(args, prompts, stop)
    elapsed = time.time() - started

    if args.suite == "code":
        with ThreadPoolExecutor(args.code_workers) as pool:
            futures = [[pool.submit(score_code, item, text, args.code_timeout) for text in texts]
                       for item, texts in zip(items, generations)]
            correct = [[f.result() for f in group] for group in futures]
    else:
        scorer = score_math if args.suite == "math" else score_science
        correct = [[scorer(item, text) for text in texts] for item, texts in zip(items, generations)]

    by_source = defaultdict(list)
    for item, flags in zip(items, correct):
        by_source[item["source"]].append(sum(flags) / len(flags))
    summary = {"model": str(args.model), "suite": args.suite, "template": args.template, "items": len(items),
               "samples": args.samples, "accuracy": round(100 * sum(map(sum, correct)) / sum(map(len, correct)), 2),
               "by_source": {s: {"accuracy": round(100 * sum(v) / len(v), 2), "items": len(v)}
                             for s, v in sorted(by_source.items())},
               "generation_seconds": round(elapsed, 1)}
    if args.output_dir:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        with (args.output_dir / "generations.jsonl").open("w") as stream:
            for item, prompt, texts, flags in zip(items, prompts, generations, correct):
                stream.write(json.dumps({"id": item["id"], "prompt": prompt, "outputs": texts, "correct": flags}) + "\n")
        (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
