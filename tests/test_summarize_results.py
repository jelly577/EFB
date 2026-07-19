import unittest

from evaluation.summarize_results import summarize_records


class SummarizeResultsTests(unittest.TestCase):
    def test_summarizes_compute_and_accuracy(self) -> None:
        records = [
            {
                "model": "toy",
                "mode": "adaptive",
                "is_correct": True,
                "runtime_seconds": 1.0,
                "metrics": {
                    "total_generated_tokens": 10,
                    "rejected_tokens": 2,
                    "rejected_token_fraction": 0.2,
                    "acceptance_rate": 0.5,
                    "saved_attempts": 3,
                    "stopped_early": True,
                },
            },
            {
                "model": "toy",
                "mode": "adaptive",
                "is_correct": False,
                "runtime_seconds": 3.0,
                "metrics": {
                    "total_generated_tokens": 20,
                    "rejected_tokens": 8,
                    "rejected_token_fraction": 0.4,
                    "acceptance_rate": 0.25,
                    "saved_attempts": 1,
                    "stopped_early": False,
                },
            },
        ]

        summary = summarize_records(records)

        self.assertEqual(summary["accuracy"], 0.5)
        self.assertEqual(summary["mean_total_tokens"], 15.0)
        self.assertEqual(summary["mean_saved_attempts"], 2.0)
        self.assertEqual(summary["mean_runtime_seconds"], 2.0)


if __name__ == "__main__":
    unittest.main()
