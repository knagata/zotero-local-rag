from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from scripts.audit_v3_cutover import (
    compare_item, get_document_nodes, get_document_structure, global_gate_failures,
    item_metrics, partial_coverage_adoptions, structure_failures,
)
from src import db_relations
from src.document_structure import STRUCTURE_VERSION, build_document_structure, source_fingerprint
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

    def test_get_document_structure_merged_with_nodes_passes_real_lookup(self):
        # get_document_structure() only returns the document_structures row --
        # it has no "nodes" key. main() must attach document_nodes itself
        # before calling structure_failures(), or the attachment-coverage
        # check always sees an empty node list and fails every item
        # regardless of the real structure (2026-08-03).
        tempdir = tempfile.TemporaryDirectory()
        self.addCleanup(tempdir.cleanup)
        db_patch = patch.object(db_relations, "DB_PATH", str(Path(tempdir.name) / "relations.db"))
        db_patch.start()
        self.addCleanup(db_patch.stop)
        db_relations._db_initialized = False
        self.addCleanup(lambda: setattr(db_relations, "_db_initialized", False))

        rows = [
            {"id": "A:p1", "text": "one", "metadata": {"source_type": "pdf", "attachmentKey": "A"}},
        ]
        built = build_document_structure("ITEM", rows)
        db_relations.replace_document_structure(
            "ITEM", source_fingerprint=built["source_fingerprint"],
            structure_version=STRUCTURE_VERSION, status=built["status"],
            confidence=built["confidence"], nodes=built["nodes"], diagnostics=built["diagnostics"],
        )

        structure = get_document_structure("ITEM")
        self.assertNotIn("nodes", structure)
        merged = {**structure, "nodes": get_document_nodes("ITEM")}
        self.assertEqual(structure_failures(merged, rows), [])

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


class PartialCoverageAdoptionListingTests(unittest.TestCase):
    def _entry(self, ratio, *, adopted=True, expected=10, accounted=5):
        return {"title": f"t{ratio}", "quality": {
            "source_coverage": _complete_coverage(),
            "source_coverage_adopted": adopted,
            "source_coverage_shortfall": {
                "unit_kind": "page", "expected_units": expected,
                "accounted_units": accounted, "covered_ratio": ratio,
                "reasons": ["source_units_unaccounted"], "unaccounted_sample": [7],
            },
        }}

    def test_worst_recovered_documents_are_listed_first(self):
        manifest = {"files": {
            "B": self._entry(0.9), "A": self._entry(0.2), "C": self._entry(0.5),
        }}
        rows = partial_coverage_adoptions(manifest)
        self.assertEqual([row["attachment_key"] for row in rows], ["A", "C", "B"])
        self.assertEqual(rows[0]["reasons"], ["source_units_unaccounted"])
        self.assertEqual(rows[0]["unaccounted_sample"], [7])

    def test_unmeasurable_coverage_sorts_after_measured_gaps(self):
        manifest = {"files": {"UNKNOWN": self._entry(None), "LOW": self._entry(0.1)}}
        self.assertEqual(
            [row["attachment_key"] for row in partial_coverage_adoptions(manifest)],
            ["LOW", "UNKNOWN"],
        )

    def test_documents_without_an_adoption_are_not_listed(self):
        manifest = {"files": {
            "A": self._entry(0.5, adopted=False),
            "B": {"quality": {"source_coverage": _complete_coverage()}},
            "C": {"no_quality": True},
        }}
        self.assertEqual(partial_coverage_adoptions(manifest), [])

    def test_missing_shortfall_still_lists_the_attachment(self):
        manifest = {"files": {"A": {"quality": {"source_coverage_adopted": True}}}}
        rows = partial_coverage_adoptions(manifest)
        self.assertEqual(len(rows), 1)
        self.assertIsNone(rows[0]["covered_ratio"])
        self.assertEqual(rows[0]["reasons"], [])


if __name__ == "__main__":
    unittest.main()
