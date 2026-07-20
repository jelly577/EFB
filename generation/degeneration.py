"""Re-roll degenerate initial generations before the MH chain starts."""

from __future__ import annotations

from evaluation.answers import extract_boxed_answer
from sampling.backend import GeneratedText, SamplingBackend

# Stride between retry seeds; far larger than any problem count so retry
# streams never collide with the per-problem seeds of neighbouring problems.
RETRY_SEED_STRIDE = 100_003


def is_degenerate(generation: GeneratedText) -> bool:
    """A degenerate generation burned its whole budget without an answer."""
    return (
        not generation.ended_naturally
        and extract_boxed_answer(generation.text) is None
    )


def generate_initial_with_reroll(
    backend: SamplingBackend,
    prompt: str,
    max_new_tokens: int,
    problem_seed: int,
    max_retries: int = 3,
) -> tuple[GeneratedText, int]:
    """Generate an initial continuation, re-rolling degenerate ones.

    The initial generation is only the chain's starting point, not part of
    the MH kernel, so re-rolling it under a fresh seed does not bias the
    target distribution — unlike e.g. a repetition penalty, which would
    change the proposal distribution for every problem.

    Returns the generation and the number of retries used. Problems whose
    first generation is clean consume the exact same random stream as
    before this feature existed, keeping earlier paired runs comparable.
    """
    if max_retries < 0:
        raise ValueError("max_retries must be non-negative")
    generation = backend.generate_initial(prompt, max_new_tokens)
    retries = 0
    while is_degenerate(generation) and retries < max_retries:
        retries += 1
        backend.reseed(problem_seed + RETRY_SEED_STRIDE * retries)
        generation = backend.generate_initial(prompt, max_new_tokens)
    return generation, retries
