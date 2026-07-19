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
class AdaptivePowerSamplingConfig(PowerSamplingConfig):
    min_steps: int = 2
    gain_threshold: float = 0.01
    patience: int = 2

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.min_steps < 1:
            raise ValueError("min_steps must be positive")
        if self.gain_threshold < 0:
            raise ValueError("gain_threshold must be non-negative")
        if self.patience < 1:
            raise ValueError("patience must be positive")


@dataclass(frozen=True)
class SamplingStep:
    step: int
    split_token_index: int
    current_token_count: int
    current_log_likelihood: float
    proposal_log_likelihood: float
    accepted: bool
    resulting_log_likelihood: float
    proposal_tokens: int


@dataclass(frozen=True)
class SamplingResult:
    initial_text: str
    final_text: str
    initial_log_likelihood: float
    final_log_likelihood: float
    metrics: SamplingMetrics
    config: PowerSamplingConfig
    trace: tuple[SamplingStep, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "initial_text": self.initial_text,
            "final_text": self.final_text,
            "initial_log_likelihood": self.initial_log_likelihood,
            "final_log_likelihood": self.final_log_likelihood,
            "metrics": self.metrics.to_dict(),
            "config": asdict(self.config),
            "trace": [asdict(step) for step in self.trace],
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

    def reseed(self, seed: int) -> None:
        self.random.seed(seed)

    def should_stop(self, score_history: list[float]) -> bool:
        return False

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
        metrics = SamplingMetrics(
            initial_tokens=initial_tokens,
            planned_attempts=self.config.steps,
        )
        score_history = [current_score]
        trace: list[SamplingStep] = []

        for step_index in range(self.config.steps):
            current_token_count = self.backend.token_count(current_text)
            if current_token_count == 0:
                break
            earliest_split = max(
                0,
                current_token_count - self.config.suffix_max_new_tokens,
            )
            split_index = self.random.randrange(earliest_split, current_token_count)
            proposal = self.backend.resample_suffix(
                prompt,
                current_text,
                split_index,
                self.config.suffix_max_new_tokens,
            )
            proposal_score = self.backend.score(prompt, proposal.text)

            metrics.attempts += 1
            metrics.proposal_tokens += proposal.token_count
            previous_score = current_score
            accepted = accepts_metropolis(
                current_score,
                proposal_score,
                self.random.random(),
            )
            if accepted:
                current_text = proposal.text
                current_score = proposal_score
                metrics.accepted += 1
            else:
                metrics.rejected_tokens += proposal.token_count
            score_history.append(current_score)
            trace.append(
                SamplingStep(
                    step=step_index + 1,
                    split_token_index=split_index,
                    current_token_count=current_token_count,
                    current_log_likelihood=previous_score,
                    proposal_log_likelihood=proposal_score,
                    accepted=accepted,
                    resulting_log_likelihood=current_score,
                    proposal_tokens=proposal.token_count,
                )
            )
            if self.should_stop(score_history):
                metrics.stopped_early = metrics.attempts < self.config.steps
                break

        return SamplingResult(
            initial_text=initial_text,
            final_text=current_text,
            initial_log_likelihood=initial_score,
            final_log_likelihood=current_score,
            metrics=metrics,
            config=self.config,
            trace=tuple(trace),
        )


class AdaptivePowerSampler(FixedPowerSampler):
    def __init__(
        self,
        backend: SamplingBackend,
        config: AdaptivePowerSamplingConfig | None = None,
    ) -> None:
        super().__init__(backend, config or AdaptivePowerSamplingConfig())
        self.config: AdaptivePowerSamplingConfig

    def should_stop(self, score_history: list[float]) -> bool:
        attempts = len(score_history) - 1
        if attempts < self.config.min_steps:
            return False
        history_start = -self.config.patience - 1
        recent_gains = [
            abs(current - previous)
            for previous, current in zip(
                score_history[history_start:-1],
                score_history[history_start + 1 :],
            )
        ]
        return len(recent_gains) == self.config.patience and all(
            gain <= self.config.gain_threshold for gain in recent_gains
        )
