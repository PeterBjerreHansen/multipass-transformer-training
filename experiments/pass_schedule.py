from __future__ import annotations

from dataclasses import dataclass
import math
import random
from typing import Sequence


@dataclass(frozen=True)
class PassSchedulePhase:
    start_step: int
    probabilities: tuple[tuple[int, float], ...]


def parse_pass_schedule(specifications: Sequence[str]) -> tuple[PassSchedulePhase, ...]:
    phases: list[PassSchedulePhase] = []
    for specification in specifications:
        try:
            start_text, mixture_text = specification.split("=", maxsplit=1)
            start_step = int(start_text)
        except (TypeError, ValueError) as error:
            raise ValueError(
                "pass schedule phases must use START=PASS:WEIGHT,..."
            ) from error
        if start_step < 1:
            raise ValueError("pass schedule start steps must be positive")

        weights: dict[int, float] = {}
        for entry in mixture_text.split(","):
            try:
                pass_text, weight_text = entry.split(":", maxsplit=1)
                passes = int(pass_text)
                weight = float(weight_text)
            except (TypeError, ValueError) as error:
                raise ValueError(
                    "pass schedule mixtures must use PASS:WEIGHT,..."
                ) from error
            if passes in weights:
                raise ValueError(f"duplicate pass count in phase: {passes}")
            weights[passes] = weight

        if not weights:
            raise ValueError("pass schedule mixtures must not be empty")
        if any(passes < 1 for passes in weights):
            raise ValueError("scheduled pass counts must be positive")
        if any(not math.isfinite(weight) or weight <= 0 for weight in weights.values()):
            raise ValueError("pass schedule weights must be finite and positive")
        total = sum(weights.values())
        probabilities = tuple(
            (passes, weight / total)
            for passes, weight in sorted(weights.items())
        )
        phases.append(PassSchedulePhase(start_step, probabilities))

    if not phases:
        raise ValueError("pass schedule must contain at least one phase")
    starts = [phase.start_step for phase in phases]
    if starts[0] != 1:
        raise ValueError("pass schedule must start at step 1")
    if starts != sorted(set(starts)):
        raise ValueError("pass schedule start steps must be strictly increasing")
    return tuple(phases)


class ProbabilisticPassScheduler:
    def __init__(self, specifications: Sequence[str], *, seed: int):
        self.phases = parse_pass_schedule(specifications)
        self.rng = random.Random(seed)
        self.sample_count = 0
        self.histogram: dict[int, int] = {}

    @property
    def maximum_passes(self) -> int:
        return max(passes for phase in self.phases for passes, _ in phase.probabilities)

    def phase_at(self, step: int) -> PassSchedulePhase:
        if step < 1:
            raise ValueError("step must be positive")
        active = self.phases[0]
        for phase in self.phases[1:]:
            if step < phase.start_step:
                break
            active = phase
        return active

    def sample(self, step: int) -> int:
        phase = self.phase_at(step)
        draw = self.rng.random()
        cumulative = 0.0
        selected = phase.probabilities[-1][0]
        for passes, probability in phase.probabilities:
            cumulative += probability
            if draw < cumulative:
                selected = passes
                break
        self.sample_count += 1
        self.histogram[selected] = self.histogram.get(selected, 0) + 1
        return selected

    def state_dict(self) -> dict:
        return {
            "rng_state": self.rng.getstate(),
            "sample_count": self.sample_count,
            "histogram": dict(self.histogram),
        }

    def load_state_dict(self, state: dict) -> None:
        self.rng.setstate(state["rng_state"])
        self.sample_count = int(state["sample_count"])
        self.histogram = {int(key): int(value) for key, value in state["histogram"].items()}

    def stats(self) -> dict:
        return {
            "samples": self.sample_count,
            "histogram": dict(sorted(self.histogram.items())),
        }


def build_pass_scheduler(args, *, seed: int) -> ProbabilisticPassScheduler | None:
    specifications = getattr(args, "train_pass_schedule", None)
    if not specifications:
        return None
    if args.architecture == "transformer":
        raise ValueError("--train-pass-schedule requires a multi-pass architecture")
    scheduler = ProbabilisticPassScheduler(specifications, seed=seed)
    if scheduler.maximum_passes > args.max_passes:
        raise ValueError("scheduled pass count cannot exceed --max-passes")
    return scheduler
