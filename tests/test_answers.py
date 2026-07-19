import unittest

from evaluation.answers import extract_boxed_answer, is_correct


class AnswerTests(unittest.TestCase):
    def test_extracts_last_balanced_boxed_answer(self) -> None:
        text = r"First \boxed{1}, then final \boxed{\frac{1}{2}}."
        self.assertEqual(extract_boxed_answer(text), r"\frac{1}{2}")

    def test_compares_basic_latex_formatting(self) -> None:
        self.assertTrue(is_correct(r"\frac{1}{2}", r"$\boxed{\frac{1}{2}}$"))


if __name__ == "__main__":
    unittest.main()
