"""How likely each model is to be asked for. ONE definition, read by both sides.

`catalogue.py` publishes `expected_share` from this and `tracegen.py` samples from it, so the published
expectation predicts the traffic. Two independent definitions would disagree: a Zipf curve laid over a
shuffled model order produces a ranking essentially unrelated to a published provider-count share.

That matters because of WHEN a plane has to decide. The setup window carries no traffic, so staging is
committed before a single request arrives, and `expected_share` is the only information there is. A
wrong hint is worse than no hint: it actively points at the wrong models, and the cost of staging one is
minutes of card time.

The curve is skewed on purpose -- a real serverless fleet is heavily concentrated -- and it is applied in
POPULARITY ORDER. `RANK_JITTER` keeps it from being exact, and the workload's demand shifts perturb it
further, so a plane still has to notice what is actually being asked for; it is just not misled at the
only moment it cannot measure anything.
"""
from __future__ import annotations

import random

# Concentration: a plain Zipf over the popularity ranking. Over the 28-model catalogue, s=1.0 gives the
# top model about 25% of expected traffic and the top ten about 75% (28x from first rank to last). A much
# steeper curve would make the head trivially stageable and the tail irrelevant.
ZIPF_S = 1.0          # 28 models: top-10 = 74.6% of expected traffic, top model 25.5%
# How far the realised ranking wanders from the published one: enough that a plane which only reads the
# hint loses something, little enough that the hint is worth reading.
RANK_JITTER = 0.03    # rank noise, as a fraction of the number of models
DRIFT_JITTER = 0.02   # additional perturbation after the drift point, so a fitted placement rots


def zipf_weights(n: int, s: float | None = None) -> list[float]:
    # `s=None`, not `s=ZIPF_S`: a default argument is bound once at import, so a caller that sets
    # `popularity.ZIPF_S` would otherwise be silently ignored.
    s = ZIPF_S if s is None else s
    raw = [1.0 / (i + 1) ** s for i in range(n)]
    total = sum(raw)
    return [x / total for x in raw]


def base_order(models: list[dict]) -> list[dict]:
    """Most popular first. Ties broken by id, so the order is a pure function of the catalogue and
    two processes reading the same file cannot disagree about it."""
    return sorted(models, key=lambda m: (-float(m.get("n_providers") or m.get("popularity_weight") or 0.0),
                                         m["id"]))


def shaped_weights(models: list[dict]) -> dict[str, float]:
    """The Zipf curve laid on the popularity ranking: {model_id: share}, summing to 1."""
    order = base_order(models)
    zw = zipf_weights(len(order))
    return {m["id"]: zw[i] for i, m in enumerate(order)}


def jittered_weights(models: list[dict], rng: random.Random, strength: float) -> dict[str, float]:
    """`shaped_weights` with the ranking perturbed by a bounded amount.

    Gaussian noise on the RANK, not on the weight: the Zipf curve is steep, so perturbing weights
    directly would move the head around far more violently than a provider's popularity ever drifts.
    """
    order = base_order(models)
    n = len(order)
    moved = sorted(((i + rng.gauss(0.0, strength * n), m["id"]) for i, m in enumerate(order)),
                   key=lambda t: t[0])
    zw = zipf_weights(n)
    return {mid: zw[i] for i, (_, mid) in enumerate(moved)}
