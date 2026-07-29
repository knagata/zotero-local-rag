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

    def test_zone_policies_are_persisted_on_leaf_nodes(self):
        result = build_document_structure(
            "ITEM", [_chunk("A:p1", "note", attachmentKey="A", zone="footnote")],
        )
        db_relations.replace_document_structure(
            "ITEM", source_fingerprint=result["source_fingerprint"],
            structure_version=result["structure_version"], status=result["status"],
            confidence=result["confidence"], nodes=result["nodes"], diagnostics=result["diagnostics"],
        )
        leaf = next(node for node in db_relations.get_document_nodes("ITEM", include_chunks=True) if node["chunks"])
        self.assertEqual((leaf["zone"], leaf["summary_policy"], leaf["retrieval_policy"]),
                         ("footnote", "exclude", "explicit_only"))

    def test_processing_state_keeps_empty_and_blocked_distinct(self):
        db_relations.mark_artifact_status("ITEM", "references", "empty", reason_code="no_references")
        db_relations.mark_artifact_status(
            "ITEM", "summary", "blocked", reason_code="no_cloud", message="Excluded by policy.",
        )
        states = {row["artifact_type"]: row for row in db_relations.get_item_processing_status("ITEM")}
        self.assertEqual(states["references"]["status"], "empty")
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

    def test_summary_reduction_parts_are_replaced_atomically(self):
        chunks = [_chunk("A:p1", "text", attachmentKey="A")]
        result = build_document_structure("ITEM", chunks)
        db_relations.replace_document_structure(
            "ITEM", source_fingerprint=result["source_fingerprint"],
            structure_version=result["structure_version"], status=result["status"],
            confidence=result["confidence"], nodes=result["nodes"], diagnostics=result["diagnostics"],
        )
        node_id = result["nodes"][0]["node_id"]
        db_relations.replace_document_node_summary_parts(
            node_id, [{"child_node_ids": ["child-a", "child-b"], "summary": "reduction", "model": "deepseek:test"}],
            prompt_version="test", source_fingerprint=result["source_fingerprint"],
        )
        self.assertEqual(db_relations.get_document_node_summary_parts(node_id)[0]["child_node_ids"], ["child-a", "child-b"])
        db_relations.replace_document_node_summary_parts(
            node_id, [], prompt_version="test", source_fingerprint=result["source_fingerprint"],
        )
        self.assertEqual(db_relations.get_document_node_summary_parts(node_id), [])

    def test_blank_node_summary_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "summary must not be empty"):
            db_relations.save_document_node_summary(
                "node", "ITEM", "   ", summary_kind="llm",
                source_fingerprint="fixture", source_chunk_count=1,
                source_chars=10,
            )

    def test_purge_removes_v3_structure_artifact_and_cascades(self):
        # R7: purge_removed_items must delete V3 document structure / node rows
        # (with their cascading summaries) and artifact status/event rows for
        # items no longer present in Zotero, while keeping surviving items intact.
        def _seed(item_key: str) -> str:
            # purge_removed_items derives the DB key universe from
            # item_citation_status, so each item must be registered there.
            db_relations.update_item_citation_status(item_key, "resolved")
            chunks = [_chunk(f"{item_key}:p1", "body", attachmentKey=item_key)]
            result = build_document_structure(item_key, chunks)
            db_relations.replace_document_structure(
                item_key, source_fingerprint=result["source_fingerprint"],
                structure_version=result["structure_version"], status=result["status"],
                confidence=result["confidence"], nodes=result["nodes"],
                diagnostics=result["diagnostics"],
            )
            leaf = next(node for node in result["nodes"] if node.get("chunks"))
            db_relations.save_document_node_summary(
                leaf["node_id"], item_key, "leaf summary", summary_kind="extractive",
                source_fingerprint=result["source_fingerprint"],
                source_chunk_count=1, source_chars=4, quality_status="degraded",
            )
            db_relations.mark_artifact_status(
                item_key, "structure", "success", attachment_key=item_key,
            )
            return leaf["node_id"]

        keep_node = _seed("KEEP")
        drop_node = _seed("DROP")

        counts = db_relations.purge_removed_items({"KEEP"})

        self.assertEqual(counts["document_structures"], 1)
        # document_nodes reported rowcount can under-count because the
        # parent_node_id self-referential ON DELETE CASCADE removes descendants
        # before the bulk statement reaches them; correctness is asserted below
        # via the actual post-purge row counts.
        self.assertGreaterEqual(counts["document_nodes"], 1)
        self.assertEqual(counts["artifact_processing_status"], 1)
        self.assertGreaterEqual(counts["artifact_processing_events"], 1)

        conn = db_relations.get_db_connection()
        try:
            def _count(sql: str, params: tuple) -> int:
                return conn.execute(sql, params).fetchone()[0]

            # Dropped item: all V3 rows gone, including cascaded summaries.
            self.assertEqual(_count("SELECT COUNT(*) FROM document_structures WHERE item_key = ?", ("DROP",)), 0)
            self.assertEqual(_count("SELECT COUNT(*) FROM document_nodes WHERE item_key = ?", ("DROP",)), 0)
            self.assertEqual(_count("SELECT COUNT(*) FROM document_node_summaries WHERE node_id = ?", (drop_node,)), 0)
            self.assertEqual(_count("SELECT COUNT(*) FROM document_node_chunks WHERE node_id = ?", (drop_node,)), 0)
            self.assertEqual(_count("SELECT COUNT(*) FROM artifact_processing_status WHERE item_key = ?", ("DROP",)), 0)
            self.assertEqual(_count("SELECT COUNT(*) FROM artifact_processing_events WHERE item_key = ?", ("DROP",)), 0)

            # Surviving item: everything retained (structure has multiple nodes).
            self.assertEqual(_count("SELECT COUNT(*) FROM document_structures WHERE item_key = ?", ("KEEP",)), 1)
            self.assertGreater(_count("SELECT COUNT(*) FROM document_nodes WHERE item_key = ?", ("KEEP",)), 0)
            self.assertEqual(_count("SELECT COUNT(*) FROM document_node_summaries WHERE node_id = ?", (keep_node,)), 1)
            self.assertEqual(_count("SELECT COUNT(*) FROM artifact_processing_status WHERE item_key = ?", ("KEEP",)), 1)
        finally:
            conn.close()


if __name__ == "__main__":
    unittest.main()
