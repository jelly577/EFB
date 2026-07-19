"""Backend contract used by the sampler and lightweight tests."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True)
class GeneratedText:
    text: str
    token_count: int


class SamplingBackend(Protocol):
    def generate_initial(self, prompt: str, max_new_tokens: int) -> GeneratedText:
        """Generate an initial continuation conditioned on *prompt*."""

    def token_count(self, text: str) -> int:
        """Return the backend tokenizer's token count for *text*."""

    def score(self, prompt: str, continuation: str) -> float:
        """Return log p(continuation | prompt)."""

    def resample_suffix(
        self,
        prompt: str,
        continuation: str,
        split_token_index: int,
        max_new_tokens: int,
    ) -> GeneratedText:
        """Keep a continuation prefix and sample a replacement suffix."""

