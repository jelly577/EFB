"""Small, deliberately conservative helpers for math-answer evaluation."""

from __future__ import annotations

import re


def extract_boxed_answer(text: str) -> str | None:
    """Return the final balanced ``\\boxed{...}`` value in *text*."""
    marker = r"\boxed{"
    answers: list[str] = []
    search_from = 0

    while (start := text.find(marker, search_from)) != -1:
        content_start = start + len(marker)
        depth = 1
        cursor = content_start
        while cursor < len(text) and depth:
            if text[cursor] == "{":
                depth += 1
            elif text[cursor] == "}":
                depth -= 1
            cursor += 1
        if depth == 0:
            answers.append(text[content_start : cursor - 1].strip())
            search_from = cursor
        else:
            break

    return answers[-1] if answers else None


def normalize_answer(answer: str) -> str:
    """Apply only safe textual normalizations; no symbolic equivalence yet."""
    normalized = answer.strip().replace("$", "")
    normalized = normalized.replace(r"\left", "").replace(r"\right", "")
    normalized = re.sub(r"\s+", "", normalized)
    return normalized.rstrip(".")


def is_correct(prediction: str | None, ground_truth: str) -> bool:
    if prediction is None:
        return False
    gold = extract_boxed_answer(ground_truth) or ground_truth
    return normalize_answer(prediction) == normalize_answer(gold)

