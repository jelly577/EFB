"""A minimal fixed-step Power Sampling baseline."""

from __future__ import annotations

import math
import random
from dataclasses import asdict, dataclass

from .backend import SamplingBackend
from .metrics import SamplingMetrics


@dataclass(frozen=True)
class PowerSamplingConfig:
    steps: int = 8
    initial_max_new_tokens: int = 1024
    suffix_max_new_tokens: int = 128
    seed: int = 42

    def __post_init__(self) -> None:
        if self.steps < 0:
            raise ValueError("steps must be non-negative")
        if self.initial_max_new_tokens < 1 or self.suffix_max_new_tokens < 1:
            raise ValueError("token limits must be positive")


@dataclass(frozen=True)
class SamplingResult:
    initial_text: str
    final_text: str
    initial_log_likelihood: float
    final_log_likelihood: float
    metrics: SamplingMetrics
    config: PowerSamplingConfig

    def to_dict(self) -> dict[str, object]:
        return {
            "initial_text": self.initial_text,
            "final_text": self.final_text,
            "initial_log_likelihood": self.initial_log_likelihood,
            "final_log_likelihood": self.final_log_likelihood,
            "metrics": self.metrics.to_dict(),
            "config": asdict(self.config),
        }


def accepts_metropolis(
    current_log_likelihood: float,
    proposal_log_likelihood: float,
    uniform_draw: float,
) -> bool:
    """Apply the handbook's introductory likelihood-ratio baseline rule."""
    if not 0.0 <= uniform_draw < 1.0:
        raise ValueError("uniform_draw must be in [0, 1)")
    log_acceptance = min(0.0, proposal_log_likelihood - current_log_likelihood)
    return uniform_draw == 0.0 or math.log(uniform_draw) < log_acceptance


class FixedPowerSampler:
    def __init__(
        self,
        backend: SamplingBackend,
        config: PowerSamplingConfig | None = None,
    ) -> None:
        self.backend = backend
        self.config = config or PowerSamplingConfig()
        self.random = random.Random(self.config.seed)

    def run(self, prompt: str, initial_text: str | None = None) -> SamplingResult:
        if initial_text is None:
            initial = self.backend.generate_initial(
                prompt,
                self.config.initial_max_new_tokens,
            )
            initial_text = initial.text
            initial_tokens = initial.token_count
        else:
            initial_tokens = self.backend.token_count(initial_text)

        if not initial_text:
            raise ValueError("initial continuation must not be empty")

        current_text = initial_text
        current_score = self.backend.score(prompt, current_text)
        initial_score = current_score
        metrics = SamplingMetrics(initial_tokens=initial_tokens)

        for _ in range(self.config.steps):
            current_token_count = self.backend.token_count(current_text)
            if current_token_count == 0:
                break
            split_index = self.random.randrange(current_token_count)
            proposal = self.backend.resample_suffix(
                prompt,
                current_text,
                split_index,
                self.config.suffix_max_new_tokens,
            )
            proposal_score = self.backend.score(prompt, proposal.text)

            metrics.attempts += 1
            metrics.proposal_tokens += proposal.token_count
            if accepts_metropolis(current_score, proposal_score, self.random.random()):
                current_text = proposal.text
                current_score = proposal_score
                metrics.accepted += 1
            else:
                metrics.rejected_tokens += proposal.token_count

        return SamplingResult(
            initial_text=initial_text,
            final_text=current_text,
            initial_log_likelihood=initial_score,
            final_log_likelihood=current_score,
            metrics=metrics,
            config=self.config,
        )
