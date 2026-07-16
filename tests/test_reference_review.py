from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from scripts.audit_ambiguous_works import audit
from src import db_relations
from src.reference_agent import commit_approved_reference_candidates
from src.reference_quality_report import build_report


class ReferenceReviewTests(unittest.TestCase):
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

    def test_staging_is_idempotent_and_does_not_create_work_edges(self):
        references = [{"raw": "Author (2020). A title.", "title": "A title", "year": 2020}]
        first = db_relations.stage_reference_candidates("ITEM", "heuristic", references)
        second = db_relations.stage_reference_candidates("ITEM", "heuristic", references)
        self.assertEqual(first, {"staged": 1, "updated": 0})
        self.assertEqual(second, {"staged": 0, "updated": 1})
        rows = db_relations.get_reference_review_candidates("pending")
        self.assertEqual(len(rows), 1)
        self.assertTrue(db_relations.set_reference_review_status(rows[0]["review_id"], "rejected", "bad"))
        self.assertEqual(len(db_relations.get_reference_review_candidates("rejected")), 1)
        connection = db_relations.get_db_connection()
        try:
            self.assertEqual(connection.execute("SELECT COUNT(*) FROM work_edges").fetchone()[0], 0)
        finally:
            connection.close()

    def test_quality_report_marks_title_only_candidates(self):
        db_relations.stage_reference_candidates(
            "ITEM", "heuristic", [{"raw": "This is a sufficiently long raw reference", "title": "家族"}],
        )
        report = build_report(db_relations.get_reference_review_candidates())
        self.assertEqual(report["candidates"], 1)
        self.assertEqual(report["flags"]["title_only_unverified"], 1)

    def test_audit_finds_only_title_only_work_with_distinct_raw_references(self):
        citing = db_relations.resolve_work(zotero_item_key="OWNED", title="Owned")
        ambiguous = db_relations.resolve_work(title="家族")
        identified = db_relations.resolve_work(title="Gift", doi="10.1/gift")
        db_relations.save_work_edge(citing, ambiguous, source="test", raw_reference="ref one")
        db_relations.save_work_edge(citing, ambiguous, source="test", raw_reference="ref two")
        db_relations.save_work_edge(citing, identified, source="test", raw_reference="ref three")
        report = audit(self.db_path)
        self.assertEqual(report["candidate_count"], 1)
        self.assertEqual(report["ambiguous_title_only"][0]["work_id"], ambiguous)

    def test_only_approved_literal_identifier_is_committed(self):
        raw = "Author. Title. https://doi.org/10.1234/example"
        db_relations.stage_reference_candidates(
            "ITEM", "test", [{"raw": raw, "title": "Title", "doi": "10.1234/example"}],
        )
        row = db_relations.get_reference_review_candidates()[0]
        db_relations.set_reference_review_status(row["review_id"], "approved")
        result = commit_approved_reference_candidates()
        self.assertEqual(result["committed"], 1)
        committed = db_relations.get_reference_review_candidates("approved")[0]
        self.assertGreater(committed["committed_edge_id"], 0)

    def test_approved_hallucinated_identifier_is_not_committed(self):
        db_relations.stage_reference_candidates(
            "ITEM", "test", [{"raw": "Author. Title.", "title": "Title", "doi": "10.1234/missing"}],
        )
        row = db_relations.get_reference_review_candidates()[0]
        db_relations.set_reference_review_status(row["review_id"], "approved")
        result = commit_approved_reference_candidates()
        self.assertEqual(result["insufficient_evidence"], 1)
        self.assertEqual(result["committed"], 0)


if __name__ == "__main__":
    unittest.main()
