"""Utilities for identifying uncertain token positions."""

from __future__ import annotations

import math


def criticality_scores(token_log_probabilities: list[float]) -> list[float]:
    """Use token surprise (negative log-probability) as a first-pass score."""
    if any(not math.isfinite(value) for value in token_log_probabilities):
        raise ValueError("token log-probabilities must be finite")
    return [-value for value in token_log_probabilities]


def top_critical_positions(
    token_log_probabilities: list[float],
    count: int = 1,
    minimum_position: int = 0,
) -> list[int]:
    """Return the most uncertain token indices, with stable tie-breaking."""
    if count < 1:
        raise ValueError("count must be positive")
    if minimum_position < 0:
        raise ValueError("minimum_position must be non-negative")

    scores = criticality_scores(token_log_probabilities)
    candidates = range(minimum_position, len(scores))
    ranked = sorted(candidates, key=lambda position: (-scores[position], position))
    return ranked[:count]

