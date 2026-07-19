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


POWER_ALPHABET = ("a", "b")
POWER_MIN_LENGTH = 2
POWER_MAX_LENGTH = 5
_EOS = ""


@dataclass(frozen=True)
class PowerToyValidationResult:
    alpha: float
    expected: dict[str, float]
    observed: dict[str, float]
    total_variation_distance: float
    corrected_proposal_ratio: bool

    def to_dict(self) -> dict[str, object]:
        return {
            "alpha": self.alpha,
            "expected": self.expected,
            "observed": self.observed,
            "total_variation_distance": self.total_variation_distance,
            "corrected_proposal_ratio": self.corrected_proposal_ratio,
        }


def _toy_next_distribution(sequence: tuple[str, ...]) -> dict[str, float]:
    """Return the toy model's next-symbol distribution; _EOS ends a sequence."""
    last = sequence[-1] if sequence else None
    if last == "a":
        base = {"a": 0.2, "b": 0.2, _EOS: 0.6}
    elif last == "b":
        base = {"a": 0.4, "b": 0.3, _EOS: 0.3}
    else:
        base = {"a": 0.6, "b": 0.4, _EOS: 0.0}
    if len(sequence) < POWER_MIN_LENGTH:
        continue_mass = base["a"] + base["b"]
        return {"a": base["a"] / continue_mass, "b": base["b"] / continue_mass, _EOS: 0.0}
    if len(sequence) >= POWER_MAX_LENGTH:
        return {"a": 0.0, "b": 0.0, _EOS: 1.0}
    return base


def _toy_sequence_log_probability(sequence: tuple[str, ...]) -> float:
    log_probability = 0.0
    for position, symbol in enumerate(sequence):
        log_probability += math.log(_toy_next_distribution(sequence[:position])[symbol])
    return log_probability + math.log(_toy_next_distribution(sequence)[_EOS])


def _enumerate_toy_sequences() -> list[tuple[str, ...]]:
    sequences: list[tuple[str, ...]] = []

    def expand(sequence: tuple[str, ...]) -> None:
        distribution = _toy_next_distribution(sequence)
        if distribution[_EOS] > 0.0:
            sequences.append(sequence)
        if len(sequence) < POWER_MAX_LENGTH:
            for symbol in POWER_ALPHABET:
                if distribution[symbol] > 0.0:
                    expand(sequence + (symbol,))

    expand(())
    return sequences


def _toy_sample_continuation(
    prefix: tuple[str, ...],
    rng: random.Random,
) -> tuple[str, ...]:
    sequence = prefix
    while True:
        distribution = _toy_next_distribution(sequence)
        draw = rng.random()
        cumulative = 0.0
        chosen = _EOS
        for symbol in ("a", "b", _EOS):
            cumulative += distribution[symbol]
            if draw < cumulative:
                chosen = symbol
                break
        if chosen == _EOS:
            return sequence
        sequence = sequence + (chosen,)


def run_power_toy_validation(
    alpha: float = 4.0,
    samples: int = 50_000,
    burn_in: int = 1_000,
    seed: int = 42,
    correct_proposal_ratio: bool = True,
) -> PowerToyValidationResult:
    """Check suffix-resampling MH against the exact power target p^alpha.

    Sequences have variable length, so this also validates that scoring
    natural termination (EOS) removes the need for equal-length suffixes.
    The corrected rule uses exponent alpha - 1 because the proposal is the
    model itself; the naive rule keeps exponent alpha and is biased.
    """
    if samples < 1 or burn_in < 0:
        raise ValueError("samples must be positive and burn_in non-negative")
    if alpha < 1.0:
        raise ValueError("alpha must be at least 1")
    rng = random.Random(seed)
    log_probabilities = {
        sequence: _toy_sequence_log_probability(sequence)
        for sequence in _enumerate_toy_sequences()
    }
    weights = {
        sequence: math.exp(alpha * log_probability)
        for sequence, log_probability in log_probabilities.items()
    }
    normalizer = sum(weights.values())
    expected = {
        "".join(sequence): weight / normalizer
        for sequence, weight in weights.items()
    }

    exponent = alpha - 1.0 if correct_proposal_ratio else alpha
    state = _toy_sample_continuation((), rng)
    counts: dict[tuple[str, ...], int] = {}
    for step in range(samples + burn_in):
        split = rng.randrange(POWER_MIN_LENGTH)
        proposal = _toy_sample_continuation(state[:split], rng)
        delta = log_probabilities[proposal] - log_probabilities[state]
        if math.log(max(rng.random(), 1e-300)) < min(0.0, exponent * delta):
            state = proposal
        if step >= burn_in:
            counts[state] = counts.get(state, 0) + 1

    observed = {
        "".join(sequence): count / samples for sequence, count in counts.items()
    }
    distance = 0.5 * sum(
        abs(probability - observed.get(sequence, 0.0))
        for sequence, probability in expected.items()
    )
    return PowerToyValidationResult(
        alpha=alpha,
        expected=expected,
        observed=observed,
        total_variation_distance=distance,
        corrected_proposal_ratio=correct_proposal_ratio,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=int, default=50_000)
    parser.add_argument("--burn-in", type=int, default=1_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--alpha", type=float, default=4.0)
    args = parser.parse_args()

    corrected = run_toy_validation(args.samples, args.burn_in, args.seed, True)
    uncorrected = run_toy_validation(args.samples, args.burn_in, args.seed, False)
    power_corrected = run_power_toy_validation(
        args.alpha, args.samples, args.burn_in, args.seed, True
    )
    power_uncorrected = run_power_toy_validation(
        args.alpha, args.samples, args.burn_in, args.seed, False
    )
    print(
        json.dumps(
            {
                "with_proposal_ratio": corrected.to_dict(),
                "without_proposal_ratio": uncorrected.to_dict(),
                "power_with_proposal_ratio": power_corrected.to_dict(),
                "power_without_proposal_ratio": power_uncorrected.to_dict(),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

