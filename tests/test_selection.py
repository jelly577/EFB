import unittest

from generation.selection import parse_levels, select_problems


def make_items(levels: list[int]) -> list[dict]:
    return [{"problem": f"p{i}", "level": level} for i, level in enumerate(levels)]


class ParseLevelsTests(unittest.TestCase):
    def test_none_and_empty_mean_no_filter(self) -> None:
        self.assertEqual(parse_levels(None), ())
        self.assertEqual(parse_levels(""), ())

    def test_parses_sorted_unique_levels(self) -> None:
        self.assertEqual(parse_levels("5,4,4"), (4, 5))

    def test_rejects_out_of_range_levels(self) -> None:
        with self.assertRaises(ValueError):
            parse_levels("0,4")
        with self.assertRaises(ValueError):
            parse_levels("6")


class SelectProblemsTests(unittest.TestCase):
    def test_keeps_dataset_order_and_indices(self) -> None:
        items = make_items([1, 4, 2, 5, 4])
        selected = select_problems(items, (4, 5), limit=10)

        self.assertEqual([index for index, _ in selected], [1, 3, 4])
        self.assertEqual([item["problem"] for _, item in selected], ["p1", "p3", "p4"])

    def test_limit_truncates_selection(self) -> None:
        items = make_items([4] * 6)
        selected = select_problems(items, (4,), limit=2)

        self.assertEqual([index for index, _ in selected], [0, 1])

    def test_no_levels_means_first_n(self) -> None:
        items = make_items([1, 2, 3, 4, 5])
        selected = select_problems(items, (), limit=3)

        self.assertEqual([index for index, _ in selected], [0, 1, 2])

    def test_positive_limit_required(self) -> None:
        with self.assertRaises(ValueError):
            select_problems(make_items([4]), (4,), limit=0)


if __name__ == "__main__":
    unittest.main()
