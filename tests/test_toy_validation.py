import unittest

from sampling.toy_validation import run_power_toy_validation, run_toy_validation


class ToyValidationTests(unittest.TestCase):
    def test_proposal_ratio_recovers_target_distribution(self) -> None:
        corrected = run_toy_validation(samples=30_000, burn_in=1_000, seed=7)
        uncorrected = run_toy_validation(
            samples=30_000,
            burn_in=1_000,
            seed=7,
            correct_proposal_ratio=False,
        )

        self.assertLess(corrected.total_variation_distance, 0.03)
        self.assertLess(
            corrected.total_variation_distance,
            uncorrected.total_variation_distance,
        )

    def test_power_acceptance_recovers_power_target(self) -> None:
        corrected = run_power_toy_validation(
            alpha=4.0, samples=30_000, burn_in=1_000, seed=7
        )
        uncorrected = run_power_toy_validation(
            alpha=4.0,
            samples=30_000,
            burn_in=1_000,
            seed=7,
            correct_proposal_ratio=False,
        )

        self.assertLess(corrected.total_variation_distance, 0.03)
        self.assertLess(
            corrected.total_variation_distance,
            uncorrected.total_variation_distance,
        )


if __name__ == "__main__":
    unittest.main()
