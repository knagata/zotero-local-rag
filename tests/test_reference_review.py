from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from scripts.audit_ambiguous_works import audit
from src import db_relations
from src.reference_agent import commit_approved_reference_candidates
from src.reference_quality_report import build_report
from scripts.stage_unverified_epub_refs import stage


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
        references = [{
            "raw": "Author (2020). A title.", "title": "A title", "year": 2020,
            "contributors": [{"name": "Author", "role": "author"}],
        }]
        first = db_relations.stage_reference_candidates("ITEM", "heuristic", references)
        second = db_relations.stage_reference_candidates("ITEM", "heuristic", references)
        self.assertEqual(first, {"staged": 1, "updated": 0})
        self.assertEqual(second, {"staged": 0, "updated": 1})
        rows = db_relations.get_reference_review_candidates("pending")
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["contributors"], [{"name": "Author", "role": "author"}])
        self.assertTrue(db_relations.set_reference_review_status(rows[0]["review_id"], "rejected", "bad"))
        self.assertEqual(len(db_relations.get_reference_review_candidates("rejected")), 1)
        connection = db_relations.get_db_connection()
        try:
            self.assertEqual(connection.execute("SELECT COUNT(*) FROM work_edges").fetchone()[0], 0)
        finally:
            connection.close()

    def test_unverified_epub_refs_stage_with_provenance_and_no_graph_write(self):
        connection = db_relations.get_db_connection()
        connection.execute('''
            INSERT INTO global_references
                (id, citing_item_key, source, raw_reference_text, context_snippet, s2_status)
            VALUES (42, 'ITEM', 'epub', 'Author (2020). A title.', 'citation context', 'unverified')
        ''')
        connection.commit()
        connection.close()
        dry_run = stage(self.db_path)
        self.assertEqual(dry_run["to_stage"], 1)
        committed = stage(self.db_path, commit=True)
        self.assertEqual(committed["staged"], 1)
        row = db_relations.get_reference_review_candidates("pending")[0]
        self.assertEqual(row["source_reference_id"], 42)
        self.assertEqual(row["source_context"], "citation context")
        self.assertEqual(row["source_kind"], "epub-unverified")
        self.assertEqual(row["year"], 2020)
        connection = db_relations.get_db_connection()
        self.assertEqual(connection.execute("SELECT COUNT(*) FROM work_edges").fetchone()[0], 0)
        connection.close()
        self.assertEqual(stage(self.db_path, commit=True)["to_stage"], 0)

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

    def test_claude_decision_requires_literal_identifier_for_approval(self):
        raw = "Author (2020). A sufficiently distinctive title. doi:10.1234/example"
        db_relations.stage_reference_candidates(
            "ITEM", "epub-unverified", [{"raw": raw, "title": None}],
        )
        review_id = db_relations.get_reference_review_candidates()[0]["review_id"]
        changed = db_relations.apply_reference_review_decision({
            "review_id": review_id, "status": "approved",
            "title": "A sufficiently distinctive title", "authors": ["Author"],
            "year": 2020, "doi": "10.1234/example", "note": "Claude: exact DOI",
        })
        self.assertTrue(changed)
        row = db_relations.get_reference_review_candidates("approved")[0]
        self.assertEqual(row["doi"], "10.1234/example")
        self.assertEqual(row["authors"], ["Author"])

    def test_claude_decision_rejects_unsupported_approval(self):
        db_relations.stage_reference_candidates(
            "ITEM", "epub-unverified", [{"raw": "Author (2020). A title.", "title": None}],
        )
        review_id = db_relations.get_reference_review_candidates()[0]["review_id"]
        with self.assertRaisesRegex(ValueError, "literal DOI or ISBN"):
            db_relations.apply_reference_review_decision({
                "review_id": review_id, "status": "approved", "year": 2020,
            })
        self.assertEqual(
            db_relations.get_reference_review_candidates("pending")[0]["status"], "pending"
        )


if __name__ == "__main__":
    unittest.main()
