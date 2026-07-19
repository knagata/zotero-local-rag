from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from src import db_relations
from src.build_structure_summaries import build_structure_summaries
from src.document_structure import build_document_structure


def _chunk(chunk_id: str, text: str, **metadata: str):
    return {"id": chunk_id, "text": text, "metadata": metadata}


class DocumentStructureTests(unittest.TestCase):
    def setUp(self):
        self.tempdir = tempfile.TemporaryDirectory()
        self.db_patch = patch.object(db_relations, "DB_PATH", str(Path(self.tempdir.name) / "relations.db"))
        self.db_patch.start()
        db_relations._db_initialized = False

    def tearDown(self):
        self.db_patch.stop()
        db_relations._db_initialized = False
        self.tempdir.cleanup()

    def test_preserves_nested_order_and_never_merges_noncontiguous_same_heading(self):
        chunks = [
            _chunk("A:p1", "first", attachmentKey="A", chapter="Part I", section="Opening"),
            _chunk("A:p2", "second", attachmentKey="A", chapter="Part I", section="Opening"),
            _chunk("A:p3", "third", attachmentKey="A", chapter="Part I", section="Methods"),
            _chunk("A:p4", "fourth", attachmentKey="A", chapter="Part II", section="Opening"),
        ]
        result = build_document_structure("ITEM", chunks)
        self.assertEqual(result["status"], "exact")
        self.assertTrue(result["diagnostics"]["valid"])
        headings = [
            (node["node_type"], node["title"], node["parent_node_id"])
            for node in result["nodes"] if node["title"]
        ]
        self.assertEqual([entry[1] for entry in headings], ["Part I", "Opening", "Methods", "Part II", "Opening"])
        opening_nodes = [node for node in result["nodes"] if node["title"] == "Opening"]
        self.assertEqual(len(opening_nodes), 2)
        self.assertNotEqual(opening_nodes[0]["node_id"], opening_nodes[1]["node_id"])
        leaves = [node for node in result["nodes"] if node["chunks"]]
        self.assertEqual(
            [[entry["chunk_id"] for entry in leaf["chunks"]] for leaf in leaves],
            [["A:p1", "A:p2"], ["A:p3"], ["A:p4"]],
        )

    def test_uses_contiguous_fallback_when_no_headings_exist(self):
        chunks = [
            _chunk("A:p1", "a" * 16_000, attachmentKey="A"),
            _chunk("A:p2", "b" * 16_000, attachmentKey="A"),
            _chunk("A:p3", "c" * 1_000, attachmentKey="A"),
        ]
        result = build_document_structure("ITEM", chunks)
        self.assertEqual(result["status"], "flat_fallback")
        leaves = [node for node in result["nodes"] if node["chunks"]]
        self.assertEqual(len(leaves), 2)
        self.assertEqual([entry["chunk_id"] for entry in leaves[0]["chunks"]], ["A:p1"])
        self.assertEqual([entry["chunk_id"] for entry in leaves[1]["chunks"]], ["A:p2", "A:p3"])

    def test_persists_tree_status_and_descendant_chunk_lookup(self):
        chunks = [
            _chunk("A:p1", "first", attachmentKey="A", chapter="One"),
            _chunk("A:p2", "second", attachmentKey="A", chapter="Two"),
        ]
        result = build_document_structure("ITEM", chunks)
        db_relations.replace_document_structure(
            "ITEM", source_fingerprint=result["source_fingerprint"],
            structure_version=result["structure_version"], status=result["status"],
            confidence=result["confidence"], nodes=result["nodes"], diagnostics=result["diagnostics"],
        )
        stored = db_relations.get_document_structure("ITEM")
        self.assertEqual(stored["leaf_count"], 2)
        nodes = db_relations.get_document_nodes("ITEM", include_chunks=True)
        self.assertEqual(nodes[0]["node_type"], "item_root")
        self.assertEqual(db_relations.get_node_descendant_chunks([nodes[0]["node_id"]]), ["A:p1", "A:p2"])

    def test_processing_state_keeps_empty_and_blocked_distinct(self):
        db_relations.mark_artifact_status("ITEM", "cases", "empty", reason_code="no_case_found")
        db_relations.mark_artifact_status(
            "ITEM", "summary", "blocked", reason_code="no_cloud", message="Excluded by policy.",
        )
        states = {row["artifact_type"]: row for row in db_relations.get_item_processing_status("ITEM")}
        self.assertEqual(states["cases"]["status"], "empty")
        self.assertEqual(states["summary"]["status"], "blocked")
        self.assertEqual(states["summary"]["reason_code"], "no_cloud")

    def test_extractive_node_summaries_are_saved_but_not_searchable(self):
        chunks = [
            _chunk("A:p1", "Fieldwork participants exchanged gifts in the village. " * 20,
                   attachmentKey="A", chapter="Findings"),
        ]
        result = build_document_structure("ITEM", chunks)
        db_relations.replace_document_structure(
            "ITEM", source_fingerprint=result["source_fingerprint"],
            structure_version=result["structure_version"], status=result["status"],
            confidence=result["confidence"], nodes=result["nodes"], diagnostics=result["diagnostics"],
        )
        with patch("src.build_structure_summaries.get_item_chunks", return_value=chunks):
            output = build_structure_summaries("ITEM", mode="extractive")
        self.assertEqual(output["status"], "degraded")
        rows = db_relations.get_document_node_summaries("ITEM")
        self.assertTrue(rows)
        self.assertTrue(all(row["summary_kind"] == "extractive" and row["searchable"] == 0 for row in rows))


if __name__ == "__main__":
    unittest.main()
