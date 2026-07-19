"""Validate state-dependent Metropolis-Hastings proposals on a toy chain."""

from __future__ import annotations

import argparse
import json
import math
import random
from dataclasses import dataclass


TARGET = (0.15, 0.35, 0.50)
PROPOSAL = (
    (0.10, 0.60, 0.30),
    (0.20, 0.20, 0.60),
    (0.50, 0.25, 0.25),
)


@dataclass(frozen=True)
class ToyValidationResult:
    expected: tuple[float, ...]
    observed: tuple[float, ...]
    total_variation_distance: float
    corrected_proposal_ratio: bool

    def to_dict(self) -> dict[str, object]:
        return {
            "expected": self.expected,
            "observed": self.observed,
            "total_variation_distance": self.total_variation_distance,
            "corrected_proposal_ratio": self.corrected_proposal_ratio,
        }


def _draw_index(probabilities: tuple[float, ...], rng: random.Random) -> int:
    draw = rng.random()
    cumulative = 0.0
    for index, probability in enumerate(probabilities):
        cumulative += probability
        if draw < cumulative:
            return index
    return len(probabilities) - 1


def metropolis_hastings_step(
    current: int,
    rng: random.Random,
    correct_proposal_ratio: bool = True,
) -> int:
    proposed = _draw_index(PROPOSAL[current], rng)
    log_acceptance = math.log(TARGET[proposed]) - math.log(TARGET[current])
    if correct_proposal_ratio:
        reverse = PROPOSAL[proposed][current]
        forward = PROPOSAL[current][proposed]
        log_acceptance += math.log(reverse) - math.log(forward)
    if math.log(max(rng.random(), 1e-300)) < min(0.0, log_acceptance):
        return proposed
    return current


def run_toy_validation(
    samples: int = 50_000,
    burn_in: int = 1_000,
    seed: int = 42,
    correct_proposal_ratio: bool = True,
) -> ToyValidationResult:
    if samples < 1 or burn_in < 0:
        raise ValueError("samples must be positive and burn_in non-negative")
    rng = random.Random(seed)
    state = 0
    counts = [0] * len(TARGET)
    for step in range(samples + burn_in):
        state = metropolis_hastings_step(state, rng, correct_proposal_ratio)
        if step >= burn_in:
            counts[state] += 1

    observed = tuple(count / samples for count in counts)
    distance = 0.5 * sum(
        abs(expected - actual)
        for expected, actual in zip(TARGET, observed)
    )
    return ToyValidationResult(
        expected=TARGET,
        observed=observed,
        total_variation_distance=distance,
        corrected_proposal_ratio=correct_proposal_ratio,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=int, default=50_000)
    parser.add_argument("--burn-in", type=int, default=1_000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    corrected = run_toy_validation(args.samples, args.burn_in, args.seed, True)
    uncorrected = run_toy_validation(args.samples, args.burn_in, args.seed, False)
    print(
        json.dumps(
            {
                "with_proposal_ratio": corrected.to_dict(),
                "without_proposal_ratio": uncorrected.to_dict(),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

