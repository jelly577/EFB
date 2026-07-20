import unittest

from generation.degeneration import (
    RETRY_SEED_STRIDE,
    generate_initial_with_reroll,
    is_degenerate,
)
from sampling.backend import GeneratedText

CLEAN = GeneratedText("the answer is \\boxed{7}", 10)
DEGENERATE = GeneratedText("Leone SEEK ponsored", 2048, ended_naturally=False)
CAPPED_WITH_ANSWER = GeneratedText("\\boxed{7} then rambling", 2048, ended_naturally=False)
SHORT_NO_ANSWER = GeneratedText("I cannot solve this.", 12)


class RerollBackend:
    def __init__(self, generations: list[GeneratedText]) -> None:
        self.generations = iter(generations)
        self.reseeds: list[int] = []

    def reseed(self, seed: int) -> None:
        self.reseeds.append(seed)

    def generate_initial(self, prompt: str, max_new_tokens: int) -> GeneratedText:
        return next(self.generations)


class IsDegenerateTests(unittest.TestCase):
    def test_capped_without_answer_is_degenerate(self) -> None:
        self.assertTrue(is_degenerate(DEGENERATE))

    def test_natural_ending_is_not_degenerate(self) -> None:
        self.assertFalse(is_degenerate(CLEAN))
        self.assertFalse(is_degenerate(SHORT_NO_ANSWER))

    def test_capped_with_answer_is_not_degenerate(self) -> None:
        self.assertFalse(is_degenerate(CAPPED_WITH_ANSWER))


class GenerateInitialWithRerollTests(unittest.TestCase):
    def test_clean_first_try_uses_original_stream(self) -> None:
        backend = RerollBackend([CLEAN])
        generation, retries = generate_initial_with_reroll(
            backend, "p", 2048, problem_seed=42
        )
        self.assertEqual(generation, CLEAN)
        self.assertEqual(retries, 0)
        self.assertEqual(backend.reseeds, [])

    def test_degenerate_then_clean_reseeds_once(self) -> None:
        backend = RerollBackend([DEGENERATE, CLEAN])
        generation, retries = generate_initial_with_reroll(
            backend, "p", 2048, problem_seed=42
        )
        self.assertEqual(generation, CLEAN)
        self.assertEqual(retries, 1)
        self.assertEqual(backend.reseeds, [42 + RETRY_SEED_STRIDE])

    def test_gives_up_after_max_retries(self) -> None:
        backend = RerollBackend([DEGENERATE] * 4)
        generation, retries = generate_initial_with_reroll(
            backend, "p", 2048, problem_seed=42, max_retries=3
        )
        self.assertTrue(is_degenerate(generation))
        self.assertEqual(retries, 3)
        self.assertEqual(
            backend.reseeds,
            [42 + RETRY_SEED_STRIDE * n for n in (1, 2, 3)],
        )

    def test_zero_retries_disables_reroll(self) -> None:
        backend = RerollBackend([DEGENERATE])
        generation, retries = generate_initial_with_reroll(
            backend, "p", 2048, problem_seed=42, max_retries=0
        )
        self.assertTrue(is_degenerate(generation))
        self.assertEqual(retries, 0)
        self.assertEqual(backend.reseeds, [])

    def test_negative_retries_rejected(self) -> None:
        with self.assertRaises(ValueError):
            generate_initial_with_reroll(
                RerollBackend([CLEAN]), "p", 2048, problem_seed=42, max_retries=-1
            )


if __name__ == "__main__":
    unittest.main()
