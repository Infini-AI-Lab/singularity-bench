"""Validated scheduling and placement options for the pinned vLLM engine.

This is an allowlist, not a list of forbidden spellings. Loading, model structure,
numerics, endpoints and executable extensions belong to the platform. Unknown
flags, abbreviations, positional arguments and duplicate options are errors.
"""
from __future__ import annotations

import math

SHAPE_ARGS = {"--tensor-parallel-size": "tp", "--pipeline-parallel-size": "pp",
              "--data-parallel-size": "dp"}
ALIASES = {"-tp": "--tensor-parallel-size", "-pp": "--pipeline-parallel-size",
           "-dp": "--data-parallel-size"}
BOOL_ARGS = {"--enable-sleep-mode", "--enable-prefix-caching",
             "--no-enable-prefix-caching", "--enable-chunked-prefill",
             "--no-enable-chunked-prefill", "--enable-expert-parallel", "--enforce-eager"}
INT_ARGS = set(SHAPE_ARGS) | {"--max-num-seqs", "--max-num-batched-tokens", "--block-size"}
FLOAT_ARGS = {"--gpu-memory-utilization", "--swap-space", "--cpu-offload-gb"}
VALUE_ARGS = INT_ARGS | FLOAT_ARGS | {"--scheduling-policy"}
# Platform-owned settings, for callers that display them. Validation uses the
# allowlist above, so new vLLM flags do not become permitted automatically.
RESERVED = {"--model", "--served-model-name", "--host", "--port", "--max-model-len",
            "--quantization", "--dtype", "--kv-cache-dtype", "--load-format", "--hf-overrides"}


def validate(engine_args: list[str] | None) -> tuple[list[str], dict, list[str]]:
    shape = {"tp": 0, "pp": 1, "dp": 1}
    out, problems, seen = [], [], set()
    if engine_args is None:
        return out, shape, problems
    if not isinstance(engine_args, list) or not all(isinstance(a, str) for a in engine_args):
        return out, shape, ["engine_args must be a list of strings"]
    i = 0
    while i < len(engine_args):
        arg = engine_args[i]
        head, eq, val = arg.partition("=")
        head = ALIASES.get(head, head)
        i += 1
        if head not in BOOL_ARGS | VALUE_ARGS:
            problems.append(f"{head!r} is not an allowed engine option")
            continue
        identity = head.replace("--no-enable-", "--enable-")
        if identity in seen:
            problems.append(f"duplicate or conflicting option {head}")
        seen.add(identity)
        if head in BOOL_ARGS:
            if eq:
                problems.append(f"{head} takes no value; omit it to disable it")
            out.append(head)
            continue
        if not eq:
            if i >= len(engine_args) or engine_args[i].startswith("-"):
                problems.append(f"{head} needs a value")
                continue
            val = engine_args[i]
            i += 1
        try:
            if head in INT_ARGS:
                number = int(val)
                if number < 1:
                    raise ValueError("must be positive")
                val = str(number)
                if head in SHAPE_ARGS:
                    shape[SHAPE_ARGS[head]] = number
                if head == "--data-parallel-size" and number != 1:
                    raise ValueError("data parallel replicas must be deployed as separate workers")
            elif head in FLOAT_ARGS:
                number = float(val)
                if not math.isfinite(number) or number < 0:
                    raise ValueError("must be finite and nonnegative")
                if head == "--gpu-memory-utilization" and not 0 < number <= 1:
                    raise ValueError("must be in (0, 1]")
                val = str(number)
            elif val not in {"fcfs", "priority"}:
                raise ValueError("must be fcfs or priority")
        except ValueError as exc:
            problems.append(f"{head}: invalid value {val!r}: {exc}")
        out.extend((head, val))
    return out, shape, problems


def parse(engine_args: list[str] | None) -> tuple[dict, list[str]]:
    _, shape, problems = validate(engine_args)
    return shape, problems


def sanitise(engine_args: list[str] | None) -> list[str]:
    """Canonical arguments, or an error. Never silently drop forbidden settings."""
    args, _, problems = validate(engine_args)
    if problems:
        raise ValueError("; ".join(problems))
    return args


def world_size(shape: dict, min_tp: int) -> int:
    return (shape.get("tp") or min_tp) * shape.get("pp", 1) * shape.get("dp", 1)


def has(args, flag: str) -> bool:
    return flag in (args or ())


def util_of(args, default: float) -> float:
    for i, arg in enumerate(args or ()):
        if arg == "--gpu-memory-utilization" and i + 1 < len(args):
            return float(args[i + 1])
        if arg.startswith("--gpu-memory-utilization="):
            return float(arg.split("=", 1)[1])
    return default


def int_of(args, flag: str, default: int) -> int:
    """Read an integer scheduling option after allowlist validation."""
    for i, arg in enumerate(args or ()):
        if arg == flag and i + 1 < len(args):
            return int(args[i + 1])
        if arg.startswith(flag + '='):
            return int(arg.split('=', 1)[1])
    return default
