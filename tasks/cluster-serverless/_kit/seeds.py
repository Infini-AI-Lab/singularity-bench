"""One master seed, and every other stream derived from it.

Everything descends from CLUSTER_MASTER_SEED through `numpy.random.SeedSequence`, which exists for exactly
this: it takes an entropy vector, mixes it properly, and hands out independent child streams that cannot
accidentally correlate. Two streams differ if any label differs; the same labels always give the same
stream; and the whole corpus -- public and private, every rate and every SLA -- is a pure function of one
integer plus the labels recorded alongside it. Seeding each stream from an ad-hoc string instead would
leave no way to say what the dataset is a function of, and would need a hand-rolled bridge between
Python's `random.Random` and ServeGen's numpy `RandomState`, which takes a 32-bit int.

The trace-level label (`graded-1`, `public-1`) stays in the vector. It is what separates the agent's
practice data from the data it is graded on, and it must keep doing so under a shared master seed.
"""
from __future__ import annotations

import hashlib
import os
import random

import numpy as np

MASTER_SEED = int(os.environ.get("CLUSTER_MASTER_SEED", "42"))


def _label_int(label: object) -> int:
    """A stable 64-bit integer for a label. NOT `hash()`: that is randomised per process unless
    PYTHONHASHSEED is set, which would make the corpus depend on an environment variable nobody sets."""
    return int.from_bytes(hashlib.sha256(str(label).encode()).digest()[:8], "big")


def entropy(*labels: object) -> list[int]:
    """The entropy vector for a stream: the master seed, then one integer per label."""
    return [MASTER_SEED, *(_label_int(x) for x in labels)]


def seq(*labels: object) -> np.random.SeedSequence:
    return np.random.SeedSequence(entropy(*labels))


def rng(*labels: object) -> random.Random:
    """A `random.Random` for this stream. 128 bits of state, so two streams do not collide."""
    state = seq(*labels).generate_state(4, dtype=np.uint32)
    return random.Random(int.from_bytes(state.tobytes(), "little"))


def np_rng(*labels: object) -> np.random.Generator:
    return np.random.default_rng(seq(*labels))


def legacy_int(*labels: object) -> int:
    """A 32-bit int, for APIs that want a raw seed -- ServeGen's `RandomState` in particular."""
    return int(seq(*labels).generate_state(1, dtype=np.uint32)[0])
