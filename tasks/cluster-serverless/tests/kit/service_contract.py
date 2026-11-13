"""Resolve the published model/tier/input contract; no private workload information.

Input length means the complete native-token wire prompt, including history and
the protocol's control tokens. A missing tier/bin is an unadvertised product.
"""
from __future__ import annotations

import math


class UnsupportedProduct(ValueError):
    pass


def resolve(model: dict, tiers: dict, service_tier: str, input_tokens: int) -> tuple[float, float]:
    if isinstance(input_tokens, bool) or int(input_tokens) != input_tokens or input_tokens < 0:
        raise ValueError("input_tokens must be a nonnegative integer")
    table = model.get("service_contracts")
    if table is not None:
        for row in table:
            if input_tokens <= row["max_input_tokens"]:
                target = row["tiers"].get(service_tier)
                if target is None:
                    raise UnsupportedProduct(f"{model['id']}: {service_tier} is not offered at {input_tokens} input tokens")
                return float(target["ttft_ms"]), float(target["tpot_ms"])
        raise UnsupportedProduct(f"{model['id']}: no contract covers {input_tokens} input tokens")
    # Small test fixtures without a contract table remain readable. Generated
    # catalogues carry the explicit table, and the world builder validates it.
    if service_tier not in tiers:
        raise UnsupportedProduct(f"unknown service tier: {service_tier}")
    tier = tiers[service_tier]
    ttft = tier.get("ttft_ms", model.get("slo_ttft_ms", 2000.) * tier.get("ttft_mult", 1.))
    return float(ttft), float(model["slo_tpot_ms"]) * float(tier["tpot_mult"])


def available(model: dict, tiers: dict, input_lengths: list[int]) -> list[str]:
    out = []
    for tier in tiers:
        try:
            for length in input_lengths:
                resolve(model, tiers, tier, length)
        except UnsupportedProduct:
            continue
        out.append(tier)
    return out


def validate(catalogue: dict) -> None:
    names = set(catalogue["service_tiers"])
    for model in catalogue["models"]:
        previous = 0
        for row in model["service_contracts"]:
            upper = row["max_input_tokens"]
            if not isinstance(upper, int) or upper <= previous or upper > model["context_len"]:
                raise ValueError(f"invalid input bins for {model['id']}")
            if set(row["tiers"]) - names:
                raise ValueError("contract advertises an unknown tier")
            for target in row["tiers"].values():
                for field in ("ttft_ms", "tpot_ms"):
                    value = target[field]
                    if isinstance(value, bool) or not math.isfinite(value) or value <= 0:
                        raise ValueError(f"invalid {field} for {model['id']}")
            previous = upper
        if not model["service_contracts"] or not any(r["tiers"] for r in model["service_contracts"]):
            raise ValueError(f"{model['id']} has no advertised product")
