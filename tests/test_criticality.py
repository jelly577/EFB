import unittest

from sampling.criticality import criticality_scores, top_critical_positions


class CriticalityTests(unittest.TestCase):
    def test_uses_negative_log_probability_as_surprise(self) -> None:
        self.assertEqual(criticality_scores([-0.1, -2.0, -0.5]), [0.1, 2.0, 0.5])

    def test_ranks_most_uncertain_positions_first(self) -> None:
        positions = top_critical_positions([-0.1, -2.0, -0.5], count=2)
        self.assertEqual(positions, [1, 2])


if __name__ == "__main__":
    unittest.main()

