from __future__ import annotations

import unittest

from scripts.audit_v3_cutover import (
    compare_item, global_gate_failures, item_metrics, structure_failures,
)
from src.document_structure import STRUCTURE_VERSION, source_fingerprint
from src.source_coverage import make_source_coverage


def _complete_coverage():
    return make_source_coverage(
        unit_kind="document", expected_units=[1], attempted_units=[1], text_units=[1],
    )


class CutoverAuditTests(unittest.TestCase):
    def test_metrics_count_zone_assignment_and_duplicates(self):
        rows = [
            {"id": "a", "text": "abc", "metadata": {"zone": "body", "node_id": "n1"}},
            {"id": "a", "text": "def", "metadata": {"zone": "footnote"}},
        ]
        result = item_metrics(rows)
        self.assertEqual(result["duplicate_chunk_ids"], 1)
        self.assertEqual(result["node_coverage"], 0.5)
        self.assertEqual(result["zones"], {"body": 1, "footnote": 1})

    def test_compare_requires_v3_rows_unique_and_fully_assigned(self):
        old = [{"id": "old", "text": "x" * 100, "metadata": {}}]
        good = [{"id": "new", "text": "x" * 80, "metadata": {"node_id": "n", "zone": "body"}}]
        self.assertTrue(compare_item("ITEM", old, good)["passed"])
        bad = [{"id": "new", "text": "x", "metadata": {}}]
        self.assertEqual(compare_item("ITEM", old, bad)["failures"], ["incomplete_node_coverage"])

    def test_new_item_has_no_character_ratio_outlier(self):
        new = [{"id": "new", "text": "text", "metadata": {"node_id": "n"}}]
        result = compare_item("ITEM", [], new)
        self.assertTrue(result["delta"]["new_item"])
        self.assertIsNone(result["delta"]["character_ratio"])
        self.assertFalse(result["delta"]["character_ratio_outlier"])

    def test_notes_are_excluded_from_canonical_coverage(self):
        rows = [
            {"id": "source", "text": "body", "metadata": {"source_type": "pdf", "node_id": "n"}},
            {"id": "note", "text": "memo", "metadata": {"source_type": "note"}},
        ]
        self.assertEqual(item_metrics(rows)["node_coverage"], 1.0)

    def test_structure_version_and_fingerprint_are_required(self):
        rows = [{"id": "a", "text": "body", "metadata": {"source_type": "pdf"}}]
        good = {
            "structure_version": STRUCTURE_VERSION,
            "source_fingerprint": source_fingerprint(rows),
            "status": "exact",
        }
        self.assertEqual(structure_failures(good, rows), [])
        bad = {**good, "source_fingerprint": "sha256:stale"}
        self.assertEqual(structure_failures(bad, rows), ["stale_structure_fingerprint"])

    def test_structure_requires_every_attachment_root(self):
        rows = [
            {"id": "A:p1", "text": "one", "metadata": {"source_type": "pdf", "attachmentKey": "A"}},
            {"id": "B:p1", "text": "two", "metadata": {"source_type": "pdf", "attachmentKey": "B"}},
        ]
        structure = {
            "structure_version": STRUCTURE_VERSION,
            "source_fingerprint": source_fingerprint(rows),
            "status": "exact",
            "nodes": [
                {"node_type": "attachment_root", "attachment_key": "A"},
            ],
        }
        self.assertIn(
            "structure_attachment_coverage_mismatch",
            structure_failures(structure, rows),
        )

    def test_global_gate_requires_exact_indexes_and_completed_checkpoints(self):
        manifest = {
            "pipeline_fingerprint": "sha256:p", "hnsw_validated": True,
            "files": {"A": {"quality": {"source_coverage": _complete_coverage()}}},
        }
        self.assertEqual(global_gate_failures(
            manifest=manifest, manifest_attachment_keys={"A"}, chroma_attachment_keys={"A"},
            chroma_ids={"c"}, lexical_ids={"c"}, pipeline_config_exists=True,
        ), [])
        manifest["post_index_pending"] = ["ITEM"]
        self.assertIn("post_index_pending", global_gate_failures(
            manifest=manifest, manifest_attachment_keys={"A"}, chroma_attachment_keys={"A"},
            chroma_ids={"c"}, lexical_ids=set(), pipeline_config_exists=True,
        ))

    def test_global_gate_catches_chunks_no_per_item_comparison_can_reach(self):
        # P6-3 (2026-07-29): the per-item comparison iterates legacy item
        # keys, so a chunk with no itemKey is invisible to it no matter how
        # many items are compared. This is the one check in the gate that
        # looks at the whole collection instead of one item at a time.
        manifest = {
            "pipeline_fingerprint": "sha256:p", "hnsw_validated": True,
            "files": {"A": {"quality": {"source_coverage": _complete_coverage()}}},
        }
        self.assertEqual(global_gate_failures(
            manifest=manifest, manifest_attachment_keys={"A"}, chroma_attachment_keys={"A"},
            chroma_ids={"c"}, lexical_ids={"c"}, pipeline_config_exists=True,
            chunks_without_item_count=0,
        ), [])
        self.assertIn("chunks_without_item", global_gate_failures(
            manifest=manifest, manifest_attachment_keys={"A"}, chroma_attachment_keys={"A"},
            chroma_ids={"c"}, lexical_ids={"c"}, pipeline_config_exists=True,
            chunks_without_item_count=17,
        ))
        self.assertIn("chunks_without_attachment", global_gate_failures(
            manifest=manifest, manifest_attachment_keys={"A"}, chroma_attachment_keys={"A"},
            chroma_ids={"c"}, lexical_ids={"c"}, pipeline_config_exists=True,
            chunks_without_attachment_count=1,
        ))

    def test_global_gate_revalidates_manifest_source_coverage(self):
        manifest = {
            "pipeline_fingerprint": "sha256:p", "hnsw_validated": True,
            "files": {"A": {"quality": {
                "source_coverage": make_source_coverage(
                    unit_kind="page", expected_units=[1, 2],
                    attempted_units=[1, 2], text_units=[1],
                ),
            }}},
        }
        self.assertIn("incomplete_source_coverage", global_gate_failures(
            manifest=manifest, manifest_attachment_keys={"A"}, chroma_attachment_keys={"A"},
            chroma_ids={"c"}, lexical_ids={"c"}, pipeline_config_exists=True,
        ))

    def test_recorded_partial_coverage_adoption_does_not_block_cutover(self):
        # U5 (2026-07-30): a document indexed on purpose with a partial-coverage
        # tag is an accounted-for gap, not an unexplained one.
        manifest = {
            "pipeline_fingerprint": "sha256:p", "hnsw_validated": True,
            "files": {"A": {"quality": {
                "source_coverage": make_source_coverage(
                    unit_kind="page", expected_units=[1, 2],
                    attempted_units=[1, 2], text_units=[1],
                ),
                "source_coverage_adopted": True,
            }}},
        }
        self.assertEqual(global_gate_failures(
            manifest=manifest, manifest_attachment_keys={"A"}, chroma_attachment_keys={"A"},
            chroma_ids={"c"}, lexical_ids={"c"}, pipeline_config_exists=True,
        ), [])

    def test_adoption_flag_cannot_excuse_a_self_contradicting_coverage_record(self):
        manifest = {
            "pipeline_fingerprint": "sha256:p", "hnsw_validated": True,
            "files": {"A": {"quality": {
                "source_coverage": make_source_coverage(
                    unit_kind="page", expected_units=[1],
                    attempted_units=[1], text_units=[1], blank_units=[1],
                ),
                "source_coverage_adopted": True,
            }}},
        }
        self.assertIn("incomplete_source_coverage", global_gate_failures(
            manifest=manifest, manifest_attachment_keys={"A"}, chroma_attachment_keys={"A"},
            chroma_ids={"c"}, lexical_ids={"c"}, pipeline_config_exists=True,
        ))


if __name__ == "__main__":
    unittest.main()
