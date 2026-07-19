import unittest

from sampling.backend import GeneratedText
from sampling.power import (
    AdaptivePowerSampler,
    AdaptivePowerSamplingConfig,
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

    def token_log_probabilities(
        self,
        prompt: str,
        continuation: str,
    ) -> list[float]:
        return [self.score(prompt, continuation) / 2] * 2

    def resample_suffix(
        self,
        prompt: str,
        continuation: str,
        split_token_index: int,
        max_new_tokens: int,
    ) -> GeneratedText:
        return next(self.proposals)


class FlatBackend(FakeBackend):
    def __init__(self) -> None:
        self.counter = 0

    def score(self, prompt: str, continuation: str) -> float:
        return -10.0

    def resample_suffix(
        self,
        prompt: str,
        continuation: str,
        split_token_index: int,
        max_new_tokens: int,
    ) -> GeneratedText:
        self.counter += 1
        return GeneratedText(f"flat-{self.counter}", 3)


class WindowBackend(FlatBackend):
    def token_count(self, text: str) -> int:
        return 10


class RejectingBackend(FakeBackend):
    def __init__(self) -> None:
        self.counter = 0

    def score(self, prompt: str, continuation: str) -> float:
        return -10.0 if continuation == "initial" else -1000.0

    def resample_suffix(
        self,
        prompt: str,
        continuation: str,
        split_token_index: int,
        max_new_tokens: int,
    ) -> GeneratedText:
        self.counter += 1
        return GeneratedText(f"bad-{self.counter}", 3)


class TruncatingBackend(FakeBackend):
    """Every proposal is better-scored but hit the budget without EOS."""

    def __init__(self) -> None:
        self.counter = 0

    def score(self, prompt: str, continuation: str) -> float:
        return -10.0 if continuation == "initial" else -1.0

    def resample_suffix(
        self,
        prompt: str,
        continuation: str,
        split_token_index: int,
        max_new_tokens: int,
    ) -> GeneratedText:
        self.counter += 1
        return GeneratedText(f"cut-{self.counter}", 3, ended_naturally=False)


class BoxedBackend(FakeBackend):
    def __init__(self) -> None:
        pass

    def generate_initial(self, prompt: str, max_new_tokens: int) -> GeneratedText:
        return GeneratedText("answer \\boxed{7}", 4)

    def score(self, prompt: str, continuation: str) -> float:
        return -10.0 if "boxed" in continuation else -1.0

    def resample_suffix(
        self,
        prompt: str,
        continuation: str,
        split_token_index: int,
        max_new_tokens: int,
    ) -> GeneratedText:
        return GeneratedText("no final answer", 3)


class PowerSamplingTests(unittest.TestCase):
    def test_metropolis_always_accepts_higher_likelihood(self) -> None:
        self.assertTrue(accepts_metropolis(-10.0, -5.0, 0.999, alpha=4.0))

    def test_alpha_one_accepts_every_proposal(self) -> None:
        self.assertTrue(accepts_metropolis(-5.0, -1000.0, 0.999, alpha=1.0))

    def test_higher_alpha_rejects_worse_proposals(self) -> None:
        self.assertFalse(accepts_metropolis(-5.0, -10.0, 0.5, alpha=4.0))

    def test_alpha_below_one_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            accepts_metropolis(-5.0, -1.0, 0.5, alpha=0.5)
        with self.assertRaises(ValueError):
            PowerSamplingConfig(alpha=0.5)

    def test_completeness_guard_rejects_budget_truncated_proposals(self) -> None:
        sampler = FixedPowerSampler(
            TruncatingBackend(),
            PowerSamplingConfig(steps=3, seed=7),
        )
        result = sampler.run("prompt")

        self.assertEqual(result.final_text, "initial")
        self.assertEqual(result.metrics.accepted, 0)
        self.assertEqual(result.metrics.incomplete_rejections, 3)
        self.assertTrue(all(step.incomplete_rejected for step in result.trace))

    def test_answer_guard_rejects_proposals_losing_boxed_answer(self) -> None:
        sampler = FixedPowerSampler(
            BoxedBackend(),
            PowerSamplingConfig(steps=2, seed=7),
        )
        result = sampler.run("prompt")

        self.assertEqual(result.final_text, "answer \\boxed{7}")
        self.assertEqual(result.metrics.accepted, 0)
        self.assertEqual(result.metrics.answer_guard_rejections, 2)
        self.assertTrue(all(step.answer_guard_rejected for step in result.trace))

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
        self.assertEqual(len(result.trace), 2)

    def test_adaptive_sampler_stops_after_repeated_flat_accepted_gains(self) -> None:
        sampler = AdaptivePowerSampler(
            FlatBackend(),
            AdaptivePowerSamplingConfig(
                steps=8,
                min_steps=2,
                patience=2,
                gain_threshold=0.01,
                seed=7,
            ),
        )
        result = sampler.run("prompt")

        self.assertEqual(result.metrics.attempts, 2)
        self.assertEqual(result.metrics.accepted, 2)
        self.assertTrue(result.metrics.stopped_early)
        self.assertEqual(result.metrics.saved_attempts, 6)

    def test_adaptive_sampler_does_not_stop_on_short_rejection_streak(self) -> None:
        sampler = AdaptivePowerSampler(
            RejectingBackend(),
            AdaptivePowerSamplingConfig(
                steps=8,
                min_steps=2,
                patience=2,
                rejection_patience=4,
                gain_threshold=0.01,
                seed=7,
            ),
        )
        result = sampler.run("prompt")

        self.assertEqual(result.metrics.accepted, 0)
        self.assertEqual(result.metrics.attempts, 4)
        self.assertTrue(result.metrics.stopped_early)
        self.assertEqual(result.metrics.saved_attempts, 4)

    def test_split_positions_stay_inside_suffix_window(self) -> None:
        sampler = FixedPowerSampler(
            WindowBackend(),
            PowerSamplingConfig(steps=20, suffix_max_new_tokens=3, seed=7),
        )
        result = sampler.run("prompt")

        self.assertEqual(len(result.trace), 20)
        self.assertTrue(
            all(step.split_token_index >= 7 for step in result.trace)
        )

    def test_reseed_reproduces_split_positions(self) -> None:
        sampler = FixedPowerSampler(
            WindowBackend(),
            PowerSamplingConfig(steps=5, suffix_max_new_tokens=3, seed=7),
        )
        first = sampler.run("prompt")
        sampler.reseed(7)
        second = sampler.run("prompt")

        self.assertEqual(
            [step.split_token_index for step in first.trace],
            [step.split_token_index for step in second.trace],
        )


if __name__ == "__main__":
    unittest.main()
