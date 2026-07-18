from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from src import db_relations


class RelationReportTests(unittest.TestCase):
    def setUp(self):
        self.tempdir = tempfile.TemporaryDirectory()
        self.db_path = Path(self.tempdir.name) / "relations.db"
        self.db_patch = patch.object(db_relations, "DB_PATH", str(self.db_path))
        self.db_patch.start()
        db_relations._db_initialized = False

    def tearDown(self):
        self.db_patch.stop()
        db_relations._db_initialized = False
        self.tempdir.cleanup()

    def _insert_reference(self):
        db_relations.insert_reference(
            cited_paper_id="S2-WRONG", cited_title="Surprising reference",
            cited_year=2002, context_snippet="citation context",
            citing_item_key="ITEM", citing_chunk_id="ATT:p1:para1",
            similarity_distance=0.1, source="s2", s2_status="matched",
        )

    def test_report_stays_visible_until_review_then_disable_survives_refresh(self):
        self._insert_reference()
        relation = db_relations.get_reference_relations_for_item("ITEM")[0]
        self.assertEqual(relation["relation_key"], "references:ITEM:S2-WRONG")

        report = db_relations.submit_relation_report(
            direction="references", item_key="ITEM", external_paper_id="S2-WRONG",
            reason="not_in_source", details="Not present in the source bibliography.",
            reporter="mcp:claude",
        )
        self.assertEqual(report["status"], "pending")
        self.assertEqual(len(db_relations.get_reference_relations_for_item("ITEM")), 1)

        self.assertTrue(db_relations.review_relation_report(report["report_id"], "disable"))
        self.assertEqual(db_relations.get_reference_relations_for_item("ITEM"), [])
        self.assertEqual(db_relations.get_references_for_chunk("ATT:p1:para1"), [])

        # S2 refresh replaces the source row, but the stable disable decision remains.
        self._insert_reference()
        self.assertEqual(db_relations.get_reference_relations_for_item("ITEM"), [])
        audit = db_relations.get_reference_relations_for_item("ITEM", include_disabled=True)
        self.assertEqual(audit[0]["review_status"], "disabled")

    def test_keep_restores_relation_and_duplicate_report_is_idempotent(self):
        self._insert_reference()
        first = db_relations.submit_relation_report(
            direction="references", item_key="ITEM", external_paper_id="S2-WRONG",
            reason="other", details="Needs a manual check.", reporter="ui",
        )
        second = db_relations.submit_relation_report(
            direction="references", item_key="ITEM", external_paper_id="S2-WRONG",
            reason="wrong_work", details="Identifier appears to resolve to another work.",
            reporter="mcp:claude",
        )
        self.assertEqual(first["report_id"], second["report_id"])
        self.assertEqual(second["report_count"], 2)
        self.assertTrue(db_relations.review_relation_report(first["report_id"], "keep"))
        self.assertEqual(len(db_relations.get_reference_relations_for_item("ITEM")), 1)
        self.assertEqual(db_relations.get_relation_reports("kept")[0]["status"], "kept")

    def test_incoming_citation_can_be_disabled(self):
        db_relations.insert_citation(
            citing_paper_id="S2-CITER", citing_title="Citing paper", citing_year=2024,
            context_snippet="It cites the local paper.", cited_item_key="ITEM",
            cited_chunk_id="ATT:p2:para2", similarity_distance=0.2, page_hint="2",
        )
        report = db_relations.submit_relation_report(
            direction="citations", item_key="ITEM", external_paper_id="S2-CITER",
            reason="wrong_work", details="The identifier points to a different paper.",
        )
        self.assertEqual(len(db_relations.get_citation_relations_for_item("ITEM")), 1)
        db_relations.review_relation_report(report["report_id"], "disable")
        self.assertEqual(db_relations.get_citation_relations_for_item("ITEM"), [])
        self.assertEqual(db_relations.get_citations_for_chunk("ATT:p2:para2"), [])

    def test_unknown_relation_cannot_be_reported(self):
        with self.assertRaisesRegex(ValueError, "does not exist"):
            db_relations.submit_relation_report(
                direction="references", item_key="ITEM", external_paper_id="MISSING",
                reason="other", details="No such relation.",
            )

    def test_citation_graph_queries_hide_disabled_relation(self):
        from scripts import show_citation_graph

        self._insert_reference()
        connection = db_relations.get_db_connection()
        self.assertEqual(len(show_citation_graph.get_refs(connection, ["ITEM"], 10)), 1)
        connection.close()
        report = db_relations.submit_relation_report(
            direction="references", item_key="ITEM", external_paper_id="S2-WRONG",
            reason="not_in_source", details="Absent from the source bibliography.",
        )
        db_relations.review_relation_report(report["report_id"], "disable")
        connection = db_relations.get_db_connection()
        self.assertEqual(show_citation_graph.get_refs(connection, ["ITEM"], 10), [])
        connection.close()


if __name__ == "__main__":
    unittest.main()
