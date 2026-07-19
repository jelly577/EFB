"""Difficulty-based problem selection for MATH-500 runs."""

from __future__ import annotations

from typing import Iterable, Sequence


def parse_levels(spec: str | None) -> tuple[int, ...]:
    """Parse a --levels argument like "4,5" into a sorted tuple of ints."""
    if not spec:
        return ()
    levels = sorted({int(part) for part in spec.split(",") if part.strip()})
    for level in levels:
        if not 1 <= level <= 5:
            raise ValueError(f"MATH-500 levels are 1-5, got {level}")
    return tuple(levels)


def select_problems(
    items: Iterable[dict],
    levels: Sequence[int],
    limit: int,
) -> list[tuple[int, dict]]:
    """Return up to *limit* (dataset_index, item) pairs, filtered by level.

    Items keep dataset order so the selection is deterministic and the
    dataset index stays traceable in the result records.
    """
    if limit < 1:
        raise ValueError("limit must be positive")
    selected: list[tuple[int, dict]] = []
    for index, item in enumerate(items):
        if levels and item.get("level") not in levels:
            continue
        selected.append((index, item))
        if len(selected) == limit:
            break
    return selected
