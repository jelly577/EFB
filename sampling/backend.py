"""Backend contract used by the sampler and lightweight tests."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True)
class GeneratedText:
    text: str
    token_count: int
    # False when generation hit the token budget without emitting EOS, in
    # which case the text's probability mass is incomplete and it must not
    # compete on likelihood with completed sequences.
    ended_naturally: bool = True


class SamplingBackend(Protocol):
    def reseed(self, seed: int) -> None:
        """Reset generation randomness for a reproducible problem run."""

    def generate_initial(self, prompt: str, max_new_tokens: int) -> GeneratedText:
        """Generate an initial continuation conditioned on *prompt*."""

    def build_chat_prompt(self, system: str, user: str) -> str:
        """Format a system+user conversation into this model's prompt string."""

    def token_count(self, text: str) -> int:
        """Return the backend tokenizer's token count for *text*."""

    def score(self, prompt: str, continuation: str) -> float:
        """Return log p(continuation | prompt)."""

    def token_log_probabilities(
        self,
        prompt: str,
        continuation: str,
    ) -> list[float]:
        """Return one conditional log-probability per continuation token."""

    def resample_suffix(
        self,
        prompt: str,
        continuation: str,
        split_token_index: int,
        max_new_tokens: int,
    ) -> GeneratedText:
        """Keep a continuation prefix and sample a replacement suffix."""
