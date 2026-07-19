"""Compute accounting for Power Sampling experiments."""

from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass
class SamplingMetrics:
    initial_tokens: int = 0
    proposal_tokens: int = 0
    rejected_tokens: int = 0
    attempts: int = 0
    accepted: int = 0

    @property
    def total_generated_tokens(self) -> int:
        return self.initial_tokens + self.proposal_tokens

    @property
    def acceptance_rate(self) -> float:
        return self.accepted / self.attempts if self.attempts else 0.0

    @property
    def rejected_token_fraction(self) -> float:
        total = self.total_generated_tokens
        return self.rejected_tokens / total if total else 0.0

    def to_dict(self) -> dict[str, int | float]:
        values = asdict(self)
        values.update(
            total_generated_tokens=self.total_generated_tokens,
            acceptance_rate=self.acceptance_rate,
            rejected_token_fraction=self.rejected_token_fraction,
        )
        return values

