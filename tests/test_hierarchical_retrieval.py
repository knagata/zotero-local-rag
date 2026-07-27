from __future__ import annotations

import unittest

from src.hierarchical_retrieval import (
    explicit_note_intent, fuse_retrieval_paths, partition_leaf_ids, retrieval_policy_allowed,
)


class HierarchicalRetrievalTests(unittest.TestCase):
    def test_leaf_ids_are_deduplicated_and_partitioned_at_one_hundred(self):
        values = [f"n{index}" for index in range(205)] + ["n1"]
        batches = partition_leaf_ids(values)
        self.assertEqual([len(batch) for batch in batches], [100, 100, 5])

    def test_explicit_notes_are_only_allowed_for_matching_intent(self):
        self.assertTrue(explicit_note_intent(["この議論の脚注と出典を探す"]))
        self.assertFalse(explicit_note_intent("贈与交換について探す"))
        self.assertFalse(retrieval_policy_allowed({"retrieval_policy": "explicit_only"}, allow_explicit=False))
        self.assertTrue(retrieval_policy_allowed({"retrieval_policy": "explicit_only"}, allow_explicit=True))
        self.assertFalse(retrieval_policy_allowed({"retrieval_policy": "exclude"}, allow_explicit=True))

    def test_rrf_uses_specified_path_weights_and_retains_provenance(self):
        fused = fuse_retrieval_paths([
            ("leaf", [{"id": "a"}, {"id": "b"}]),
            ("same_item", [{"id": "b"}]),
            ("direct", [{"id": "c"}, {"id": "a"}]),
        ], routed_nodes_by_chunk={"a": ["node-1"]})
        self.assertEqual([row["id"] for row in fused], ["b", "a", "c"])
        self.assertEqual(fused[1]["retrieval_paths"], ["leaf", "direct"])
        self.assertEqual(fused[1]["routed_by_node_ids"], ["node-1"])


if __name__ == "__main__":
    unittest.main()
