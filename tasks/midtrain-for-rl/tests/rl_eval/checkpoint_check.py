#!/usr/bin/env python3
"""Structural validation of a submitted checkpoint against the base model.

A valid submission is a complete Hugging Face directory whose weights load into
the base model's exact architecture (meta-llama/Llama-3.2-1B, text only, tied
input/output embeddings): config.json with the same architecture fields, the
same tensor names and shapes in .safetensors files, and only finite values.
There is no constraint on how far the weights moved from the base model.
Tokenizer files are not inspected; scoring always uses the base tokenizer.

  python evals/checkpoint.py out/final_checkpoint
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import struct
import sys

BASE_MODEL = Path("/app/models/Llama-3.2-1B")
# Fields that determine tensor shapes or the forward computation.
ARCHITECTURE = ("model_type", "hidden_size", "intermediate_size", "num_hidden_layers", "num_attention_heads",
                "num_key_value_heads", "head_dim", "vocab_size", "max_position_embeddings", "rms_norm_eps",
                "rope_theta", "rope_scaling", "tie_word_embeddings", "hidden_act", "attention_bias", "mlp_bias")
DTYPES = {"BF16", "F16", "F32"}


def header(path: Path) -> dict:
    with path.open("rb") as stream:
        (length,) = struct.unpack("<Q", stream.read(8))
        if length > 100_000_000:
            raise ValueError(f"{path.name}: implausible safetensors header")
        data = json.loads(stream.read(length))
    data.pop("__metadata__", None)
    return data


def inventory(directory: Path) -> tuple[dict, list[str]]:
    """Tensor name -> (dtype, shape, file) for the shards a loader would read."""
    errors = []
    index = directory / "model.safetensors.index.json"
    if index.is_file():
        weight_map = json.loads(index.read_text())["weight_map"]
        files = sorted(set(weight_map.values()))
    else:
        weight_map, files = None, ["model.safetensors"]
    tensors = {}
    for name in files:
        path = directory / name
        if not path.is_file():
            errors.append(f"missing weight file {name}")
            continue
        for tensor, meta in header(path).items():
            if tensor in tensors:
                errors.append(f"tensor {tensor} appears in more than one file")
            tensors[tensor] = (meta["dtype"], tuple(meta["shape"]), name)
    if weight_map is not None:
        for tensor, name in weight_map.items():
            if tensors.get(tensor, (None, None, name))[2] != name:
                errors.append(f"index places {tensor} in {name}, which does not contain it")
    return tensors, errors


def finite(directory: Path, tensors: dict) -> list[str]:
    import torch
    from safetensors import safe_open

    errors = []
    by_file = {}
    for tensor, (_, _, name) in tensors.items():
        by_file.setdefault(name, []).append(tensor)
    for name, names in by_file.items():
        with safe_open(directory / name, framework="pt") as handle:
            for tensor in names:
                value = handle.get_tensor(tensor)
                if not torch.isfinite(value).all():
                    errors.append(f"{tensor} contains NaN or Inf")
    return errors


def validate(directory: Path, base: Path, check_values: bool = True) -> dict:
    errors = []
    if not directory.is_dir():
        return {"valid": False, "errors": [f"{directory} is not a directory"]}
    links = [str(p.relative_to(directory)) for p in directory.rglob("*") if p.is_symlink()]
    if links:
        errors.append(f"symbolic links are not allowed: {links[:5]}")
    try:
        config = json.loads((directory / "config.json").read_text())
    except (OSError, json.JSONDecodeError) as error:
        return {"valid": False, "errors": errors + [f"config.json: {error}"]}
    reference = json.loads((base / "config.json").read_text())
    for field in ARCHITECTURE:
        if config.get(field) != reference.get(field):
            errors.append(f"config {field}={config.get(field)!r}, expected {reference.get(field)!r}")
    if "LlamaForCausalLM" not in config.get("architectures", []):
        errors.append(f"architectures={config.get('architectures')!r}, expected LlamaForCausalLM")
    expected, base_errors = inventory(base)
    if base_errors:
        raise RuntimeError(f"base model inventory is invalid: {base_errors}")
    tensors, inventory_errors = inventory(directory)
    errors += inventory_errors
    missing = sorted(set(expected) - set(tensors))
    extra = sorted(set(tensors) - set(expected))
    if missing:
        errors.append(f"{len(missing)} missing tensors, e.g. {missing[:5]}")
    if extra:
        hint = " (the output projection must stay tied to the input embedding)" if "lm_head.weight" in extra else ""
        errors.append(f"{len(extra)} unexpected tensors{hint}, e.g. {extra[:5]}")
    for tensor in sorted(set(expected) & set(tensors)):
        dtype, shape, _ = tensors[tensor]
        if shape != expected[tensor][1]:
            errors.append(f"{tensor} has shape {list(shape)}, expected {list(expected[tensor][1])}")
        if dtype not in DTYPES:
            errors.append(f"{tensor} has dtype {dtype}; use one of {sorted(DTYPES)}")
    if check_values and not errors:
        errors += finite(directory, tensors)
    parameters = sum(math.prod(shape) for _, shape, _ in tensors.values())
    dtypes = sorted({dtype for dtype, _, _ in tensors.values()})
    return {"valid": not errors, "errors": errors, "tensors": len(tensors), "parameters": parameters,
            "dtypes": dtypes, "path": str(directory)}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--base", type=Path, default=BASE_MODEL)
    parser.add_argument("--skip-values", action="store_true", help="skip the NaN/Inf scan")
    args = parser.parse_args()
    result = validate(args.checkpoint, args.base, not args.skip_values)
    print(json.dumps(result, indent=2))
    return 0 if result["valid"] else 1


if __name__ == "__main__":
    sys.exit(main())
