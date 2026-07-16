from __future__ import annotations

import unittest

from scripts.eval_retrieval import score_ranked_items


class RetrievalEvaluationTests(unittest.TestCase):
    def test_hit_and_reciprocal_rank(self):
        score = score_ranked_items(["A", "B", "C"], {"B"}, 3)
        self.assertEqual(score, {"hit": 1.0, "rr": 0.5})

    def test_miss_outside_k(self):
        score = score_ranked_items(["A", "B", "C"], {"C"}, 2)
        self.assertEqual(score, {"hit": 0.0, "rr": 0.0})


if __name__ == "__main__":
    unittest.main()
