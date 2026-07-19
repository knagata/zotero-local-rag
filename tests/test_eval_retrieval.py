from __future__ import annotations

import os
import unittest

from scripts.eval_retrieval import _temporary_env, evaluate


class RetrievalEvaluationTests(unittest.TestCase):
    def test_temporary_env_restores_feature_flag(self):
        os.environ.pop("HIERARCHICAL_SEARCH_V2_ENABLE", None)
        with _temporary_env({"HIERARCHICAL_SEARCH_V2_ENABLE": "1"}):
            self.assertEqual(os.environ["HIERARCHICAL_SEARCH_V2_ENABLE"], "1")
        self.assertNotIn("HIERARCHICAL_SEARCH_V2_ENABLE", os.environ)

    def test_evaluation_scores_hierarchical_response_shape(self):
        report = evaluate(
            [{"id": "q1", "query": "query", "expected_item_keys": ["ITEM"]}],
            lambda *_args, **_kwargs: {"results": [{"meta": {"itemKey": "ITEM", "lang": "en"}}]},
            k=3, kwargs={"auto_expand": True},
        )
        self.assertEqual(report["hit_at_k"], 1.0)
        self.assertEqual(report["mrr_at_k"], 1.0)


if __name__ == "__main__":
    unittest.main()
