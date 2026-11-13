"""Build fixed public service contracts from frozen market and calibration inputs.

The default is explicitly an unqualified development profile (modeled prefill
estimates). A target-qualified profile must supply complete hardware measurements;
grading never profiles an agent.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import physics
import service_contract

ROOT = Path(__file__).resolve().parent
BINS = (1024, 4096, 8192, 16384, 32768)
TIERS = {"priority": {"ttft_mult": .5, "tpot_mult": .75, "share": .15},
         "standard": {"ttft_mult": 1., "tpot_mult": 1., "share": .70},
         "bulk": {"ttft_mult": 3., "tpot_mult": 2., "share": .15}}


def input_bins(model):
    limit = int(model["context_len"])
    return sorted({min(n, limit) for n in BINS})


def development_prefill(model, size):
    architecture=model.get('architecture',{})
    geometry=(architecture.get('num_hidden_layers') or 32)*(architecture.get('hidden_size') or 4096)
    return physics.prefill_ms(size,model.get('serving_active_gib',model.get('active_gib',model['weights_gib'])),
                             is_moe=bool(model.get('is_moe')),geometry=geometry,tp=int(model['min_tp']),
                             architecture=architecture)


def apply(catalogue: dict, profile: dict | None = None, *, market_path: Path | None = None) -> dict:
    path = market_path or ROOT / "market-anchors.json"
    market = json.loads(path.read_text())
    source = profile or {"schema_version": 1, "status": "development", "models": {}}
    if source.get("schema_version") != 1 or source.get("status") not in ("development", "target-qualified"):
        raise ValueError("unknown catalogue calibration profile")
    qualified = source["status"] == "target-qualified"
    model_slos = catalogue.get("slo_scope") == "model"
    tiers = {"standard": dict(TIERS["standard"], share=1.)} if model_slos else TIERS
    if qualified:
        if not source.get("evidence_sha256") or not source.get("hardware"):
            raise ValueError("target qualification requires hardware and raw evidence fingerprints")
        hardware = source["hardware"]
        if (hardware.get("gpu") != "B300" or catalogue["n_gpus"] % 4
                or hardware.get("nodes") != catalogue["n_gpus"] // 4
                or hardware.get("gpus_per_node") != 4):
            raise ValueError("target profile must identify the catalogue's four-B300 trays")
        if any(len(v) != 64 or any(c not in "0123456789abcdef" for c in v)
               for v in source["evidence_sha256"].values()):
            raise ValueError("invalid raw evidence fingerprint")
        if set(source["models"]) != {m["id"] for m in catalogue["models"]}:
            raise ValueError("target profile must qualify the entire selected catalogue")
    catalogue["contract_version"] = 4 if model_slos else 3
    catalogue["service_tiers"] = {k: dict(v) for k, v in tiers.items()}
    catalogue["calibration"] = {
        "status": source["status"], "hardware": source.get("hardware"),
        "profile_sha256": hashlib.sha256(json.dumps(source, sort_keys=True, separators=(",", ":")).encode()).hexdigest(),
        "market_anchors_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "prefill_basis": "measured warm p95" if qualified else "modeled development estimates; no hardware qualification",
    }
    for model in catalogue["models"]:
        mid = model["id"]
        anchor = market["models"].get(mid)
        base_ttft, base_tpot = float(model["slo_ttft_ms"]), float(model["slo_tpot_ms"])
        if anchor:
            if anchor["checkpoint_revision"] != model["checkpoint_revision"]:
                raise ValueError(f"market tokenizer revision mismatch for {mid}")
            base_ttft = max(base_ttft, math.ceil(anchor["median_ttft_ms"] / 250) * 250)
            ratio = float(anchor["native_per_standardized_token"])
            speed = float(anchor["median_output_standardized_tokens_per_second"])
            if not math.isfinite(ratio * speed) or ratio <= 0 or speed <= 0:
                raise ValueError(f"invalid tokenizer ratio/market speed for {mid}")
            base_tpot = max(base_tpot, math.ceil(1000 / (ratio * speed)))
        model["slo_ttft_ms"], model["slo_tpot_ms"] = base_ttft, base_tpot
        model["slo_source"] = {"kind": "market-informed" if anchor else "active-size-band",
                               "aa_slug": anchor["aa_slug"] if anchor else None,
                               "tokenizer_ratio": anchor["native_per_standardized_token"] if anchor else None}
        bins = input_bins(model)
        measurements = source["models"].get(mid)
        if measurements:
            if measurements["checkpoint_revision"] != model["checkpoint_revision"]:
                raise ValueError(f"calibration checkpoint mismatch for {mid}")
            curve = {n: float(measurements["prefill_p95_ms"][str(n)]) for n in bins}
            if any(not math.isfinite(v) or v < 0 for v in curve.values()):
                raise ValueError("invalid measured prefill curve")
            allowed = measurements["allowed_tiers_by_input"]
            if qualified and not measurements.get("deployment"):
                raise ValueError("qualification lacks a tested deployment")
            if measurements.get("deployment"):
                deployment = measurements["deployment"]
                tp, util = deployment["tp"], deployment["util"]
                if (tp not in (1, 2, 4, 8) or tp < model["min_tp"] or not math.isfinite(util)
                        or not 0 < util <= catalogue["usable_util_per_gpu"]):
                    raise ValueError("qualified placement exceeds the supported reservation limits")
                import engine_args
                canonical, shape, errors = engine_args.validate(deployment["engine_args"])
                if errors or shape["tp"] != tp or shape["pp"] != 1 or shape["dp"] != 1 or abs(engine_args.util_of(canonical, -1.) - util) > 1e-9:
                    raise ValueError("qualified placement and engine arguments disagree")
                model["min_tp"] = model["tp"] = tp
                model["min_util"] = util
                model["qualified_deployment"] = deployment
        else:
            if qualified:
                raise ValueError(f"missing calibration for {mid}")
            curve = {n: development_prefill(model, n) for n in bins}
            allowed = {str(n): list(tiers) for n in bins}
        reference = curve[bins[0]]
        # A larger prompt never receives a smaller prefill allowance due to noise.
        extra = 0.
        table = []
        for n in bins:
            extra = 0. if model_slos else max(extra, curve[n] - reference)
            names = allowed[str(n)]
            if model_slos and names != ['standard']:
                raise ValueError('per-model SLOs may not remove input shapes or service classes')
            if len(names) != len(set(names)) or set(names) - set(tiers):
                raise ValueError("invalid advertised product list")
            table.append({"max_input_tokens": n, "tiers": {
                tier: {"ttft_ms": round(base_ttft * tiers[tier]["ttft_mult"] + extra, 3),
                       "tpot_ms": round(base_tpot * tiers[tier]["tpot_mult"], 3)} for tier in names}})
        model["service_contracts"] = table
    catalogue["demanded_units"] = round(sum(m["min_tp"] * m["min_util"] for m in catalogue["models"]), 3)
    catalogue["oversubscription"] = round(catalogue["demanded_units"] / catalogue["fleet_capacity_units"], 3)
    service_contract.validate(catalogue)
    return catalogue
