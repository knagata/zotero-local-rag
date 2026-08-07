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

    def test_bibliography_is_reachable_when_a_query_asks_for_references(self):
        # A bibliography is where a reader sees what a work points at, so it
        # has to be reachable; "exclude" made 22,267 chunks unsearchable. It
        # stays explicit_only rather than normal because the content is
        # reference entries, not argument.
        from src.document_structure import ZONE_POLICIES

        policy = ZONE_POLICIES["bibliography"][1]
        self.assertEqual(policy, "explicit_only")
        metadata = {"zone": "bibliography", "retrieval_policy": policy}
        for query in ("この本の参考文献", "references on infrastructure", "出典を教えて"):
            with self.subTest(query=query):
                self.assertTrue(retrieval_policy_allowed(
                    metadata, allow_explicit=explicit_note_intent(query),
                ))
        self.assertFalse(retrieval_policy_allowed(
            metadata, allow_explicit=explicit_note_intent("ケアの倫理について"),
        ))

    def test_bibliography_still_never_feeds_a_summary(self):
        # Reachable in search is not the same as quotable evidence for a
        # summary: a section summary built from a reference list would describe
        # the list, not the work.
        from src.document_structure import ZONE_POLICIES

        self.assertEqual(ZONE_POLICIES["bibliography"][0], "exclude")
        self.assertEqual(ZONE_POLICIES["bibliography"][2], "extract")

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
