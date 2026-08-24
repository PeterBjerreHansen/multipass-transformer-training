from __future__ import annotations

import math
import random
from typing import Sequence

from model_factory import supports_pass_override


class PassMixtureSampler:
    """Checkpointable random pass-count sampler for one training batch."""

    def __init__(
        self,
        weights: Sequence[float],
        *,
        max_passes: int,
        seed: int,
    ):
        if len(weights) != max_passes:
            raise ValueError("--pass-mixture must contain exactly --max-passes values")
        values = tuple(float(weight) for weight in weights)
        if any(not math.isfinite(weight) or weight < 0 for weight in values):
            raise ValueError("--pass-mixture weights must be finite and non-negative")
        total = sum(values)
        if total <= 0:
            raise ValueError("--pass-mixture must contain positive probability mass")
        self.probabilities = tuple(weight / total for weight in values)
        self.rng = random.Random(seed)
        self.sample_count = 0
        self.histogram: dict[int, int] = {}

    def sample(self) -> int:
        draw = self.rng.random()
        cumulative = 0.0
        selected = len(self.probabilities)
        for passes, probability in enumerate(self.probabilities, start=1):
            cumulative += probability
            if draw < cumulative:
                selected = passes
                break
        self.sample_count += 1
        self.histogram[selected] = self.histogram.get(selected, 0) + 1
        return selected

    def state_dict(self) -> dict:
        return {
            "probabilities": self.probabilities,
            "rng_state": self.rng.getstate(),
            "sample_count": self.sample_count,
            "histogram": dict(self.histogram),
        }

    def load_state_dict(self, state: dict) -> None:
        if tuple(state["probabilities"]) != self.probabilities:
            raise ValueError("pass mixture changed across resume")
        self.rng.setstate(state["rng_state"])
        self.sample_count = int(state["sample_count"])
        self.histogram = {
            int(key): int(value) for key, value in state["histogram"].items()
        }

    def stats(self) -> dict:
        return {
            "samples": self.sample_count,
            "probabilities": list(self.probabilities),
            "histogram": dict(sorted(self.histogram.items())),
        }


def build_pass_mixture(args, *, seed: int) -> PassMixtureSampler | None:
    weights = getattr(args, "pass_mixture", None)
    if weights is None:
        return None
    if not supports_pass_override(args.architecture):
        raise ValueError("--pass-mixture requires a multi-pass architecture")
    return PassMixtureSampler(
        weights,
        max_passes=args.max_passes,
        seed=seed,
    )
