import unittest

from sampling.backend import GeneratedText
from sampling.power import (
    FixedPowerSampler,
    PowerSamplingConfig,
    accepts_metropolis,
)


class FakeBackend:
    def __init__(self) -> None:
        self.proposals = iter(
            [
                GeneratedText("better", 3),
                GeneratedText("worse", 4),
            ]
        )

    def generate_initial(self, prompt: str, max_new_tokens: int) -> GeneratedText:
        return GeneratedText("initial", 2)

    def token_count(self, text: str) -> int:
        return 2

    def score(self, prompt: str, continuation: str) -> float:
        return {"initial": -10.0, "better": -5.0, "worse": -100.0}[continuation]

    def resample_suffix(
        self,
        prompt: str,
        continuation: str,
        split_token_index: int,
        max_new_tokens: int,
    ) -> GeneratedText:
        return next(self.proposals)


class PowerSamplingTests(unittest.TestCase):
    def test_metropolis_always_accepts_higher_likelihood(self) -> None:
        self.assertTrue(accepts_metropolis(-10.0, -5.0, 0.999))

    def test_fixed_sampler_accounts_for_rejected_tokens(self) -> None:
        sampler = FixedPowerSampler(
            FakeBackend(),
            PowerSamplingConfig(steps=2, seed=7),
        )
        result = sampler.run("prompt")

        self.assertEqual(result.final_text, "better")
        self.assertEqual(result.metrics.attempts, 2)
        self.assertEqual(result.metrics.accepted, 1)
        self.assertEqual(result.metrics.proposal_tokens, 7)
        self.assertEqual(result.metrics.rejected_tokens, 4)
        self.assertEqual(result.metrics.total_generated_tokens, 9)
        self.assertEqual(result.metrics.acceptance_rate, 0.5)


if __name__ == "__main__":
    unittest.main()
