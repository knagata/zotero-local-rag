from __future__ import annotations

import unittest

from scripts.build_gate3_audit import choose_rows


class Gate3AuditTests(unittest.TestCase):
    def test_seeded_sample_is_reproducible_and_sorted(self):
        rows = [{"item_key": "I", "section_id": f"w{index}"} for index in range(10)]
        first = choose_rows(rows, count=4, seed=42)
        second = choose_rows(rows, count=4, seed=42)
        self.assertEqual(first, second)
        self.assertEqual(first, sorted(first, key=lambda row: (row["item_key"], row["section_id"])))

    def test_rejects_sample_larger_than_population(self):
        with self.assertRaises(ValueError):
            choose_rows([], count=1, seed=1)


if __name__ == "__main__":
    unittest.main()
