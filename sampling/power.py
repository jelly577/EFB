"""A minimal fixed-step Power Sampling baseline."""

from __future__ import annotations

import math
import random
from dataclasses import asdict, dataclass

from evaluation.answers import extract_boxed_answer

from .backend import SamplingBackend
from .metrics import SamplingMetrics


@dataclass(frozen=True)
class PowerSamplingConfig:
    steps: int = 8
    initial_max_new_tokens: int = 2048
    suffix_max_new_tokens: int = 128
    alpha: float = 4.0
    seed: int = 42

    def __post_init__(self) -> None:
        if self.steps < 0:
            raise ValueError("steps must be non-negative")
        if self.initial_max_new_tokens < 1 or self.suffix_max_new_tokens < 1:
            raise ValueError("token limits must be positive")
        if self.alpha < 1.0:
            raise ValueError("alpha must be at least 1")


@dataclass(frozen=True)
class AdaptivePowerSamplingConfig(PowerSamplingConfig):
    min_steps: int = 2
    gain_threshold: float = 0.01
    patience: int = 2
    rejection_patience: int = 4

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.min_steps < 1:
            raise ValueError("min_steps must be positive")
        if self.gain_threshold < 0:
            raise ValueError("gain_threshold must be non-negative")
        if self.patience < 1:
            raise ValueError("patience must be positive")
        if self.rejection_patience < 1:
            raise ValueError("rejection_patience must be positive")


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
    answer_guard_rejected: bool = False
    incomplete_rejected: bool = False


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
    alpha: float = 4.0,
) -> bool:
    """Metropolis-Hastings acceptance for the power target p^alpha.

    The proposal resamples a suffix from the model itself, so the proposal
    ratio cancels one factor of the likelihood ratio and the acceptance
    reduces to (alpha - 1) * (log p' - log p). With alpha = 1 the chain
    accepts everything, which is correct: it then samples from p directly.
    """
    if not 0.0 <= uniform_draw < 1.0:
        raise ValueError("uniform_draw must be in [0, 1)")
    if alpha < 1.0:
        raise ValueError("alpha must be at least 1")
    log_acceptance = min(
        0.0,
        (alpha - 1.0) * (proposal_log_likelihood - current_log_likelihood),
    )
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

    def should_stop(self, trace: list[SamplingStep]) -> bool:
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
            uniform_draw = self.random.random()
            incomplete_rejected = not proposal.ended_naturally
            guard_rejected = (
                not incomplete_rejected
                and extract_boxed_answer(current_text) is not None
                and extract_boxed_answer(proposal.text) is None
            )
            accepted = (
                not incomplete_rejected
                and not guard_rejected
                and accepts_metropolis(
                    current_score,
                    proposal_score,
                    uniform_draw,
                    self.config.alpha,
                )
            )
            if accepted:
                current_text = proposal.text
                current_score = proposal_score
                metrics.accepted += 1
            else:
                metrics.rejected_tokens += proposal.token_count
                if incomplete_rejected:
                    metrics.incomplete_rejections += 1
                elif guard_rejected:
                    metrics.answer_guard_rejections += 1
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
                    answer_guard_rejected=guard_rejected,
                    incomplete_rejected=incomplete_rejected,
                )
            )
            if self.should_stop(trace):
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

    def should_stop(self, trace: list[SamplingStep]) -> bool:
        """Stop on converged accepted gains or a long rejection streak.

        Rejected steps leave the score unchanged, so counting them as
        zero-gain steps would conflate "the chain rejects proposals" with
        "accepted moves stopped improving". Rejections only trigger a stop
        through the separate, longer rejection_patience streak.
        """
        if len(trace) < self.config.min_steps:
            return False

        consecutive_rejections = 0
        for step in reversed(trace):
            if step.accepted:
                break
            consecutive_rejections += 1
        if consecutive_rejections >= self.config.rejection_patience:
            return True

        accepted_gains = [
            abs(step.resulting_log_likelihood - step.current_log_likelihood)
            for step in trace
            if step.accepted
        ]
        if len(accepted_gains) < self.config.patience:
            return False
        return all(
            gain <= self.config.gain_threshold
            for gain in accepted_gains[-self.config.patience :]
        )
