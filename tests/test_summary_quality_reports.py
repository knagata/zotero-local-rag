from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from src import db_relations


class SummaryQualityReportTests(unittest.TestCase):
    def setUp(self):
        self.tempdir = tempfile.TemporaryDirectory()
        self.db_path = Path(self.tempdir.name) / "relations.db"
        self.db_patch = patch.object(db_relations, "DB_PATH", str(self.db_path))
        self.db_patch.start()
        db_relations._db_initialized = False
        db_relations.save_item_summary("ITEM", "item summary", "deepseek:flash")
        db_relations.save_section_summary(
            "ITEM", "w0", "section summary", model="deepseek:flash",
        )

    def tearDown(self):
        self.db_patch.stop()
        db_relations._db_initialized = False
        self.tempdir.cleanup()

    def test_report_is_idempotent_and_disable_is_fingerprint_scoped(self):
        first = db_relations.submit_summary_quality_report(
            item_key="ITEM", section_id="w0", reason="unsupported_claim",
            details="The section source contradicts this claim.",
            evidence_chunk_ids=["chunk-1"],
        )
        second = db_relations.submit_summary_quality_report(
            item_key="ITEM", section_id="w0", reason="wrong_number",
            details="The source says 12 rather than 21.",
            evidence_chunk_ids=["chunk-2"],
        )
        self.assertEqual(first["report_id"], second["report_id"])
        self.assertEqual(second["report_count"], 2)
        self.assertTrue(db_relations.resolve_summary_quality_report(
            first["report_id"], "disable", triage_model="deepseek:test",
            triage_evidence={"quote": "source says 12"},
        ))
        self.assertIn(("ITEM", "w0"), db_relations.get_disabled_summary_keys())

        db_relations.save_section_summary(
            "ITEM", "w0", "regenerated summary", model="deepseek:flash",
        )
        self.assertNotIn(("ITEM", "w0"), db_relations.get_disabled_summary_keys())

    def test_uncertain_remains_pending_for_exception_review(self):
        report = db_relations.submit_summary_quality_report(
            item_key="ITEM", section_id=None, reason="misleading_summary",
            details="The summary omits a material limitation in the source.",
        )
        db_relations.resolve_summary_quality_report(
            report["report_id"], "uncertain", triage_model="deepseek:test",
        )
        pending = db_relations.get_summary_quality_reports("pending")
        self.assertEqual(pending[0]["triage_status"], "uncertain")
        self.assertEqual(pending[0]["summary_key"].split(":", 1)[0], "item")

    def test_unknown_summary_or_short_details_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "at least 10"):
            db_relations.submit_summary_quality_report(
                item_key="ITEM", section_id="w0", reason="other", details="wrong",
            )
        with self.assertRaisesRegex(ValueError, "does not exist"):
            db_relations.submit_summary_quality_report(
                item_key="MISSING", section_id=None, reason="other",
                details="There is no current summary for this item.",
            )


if __name__ == "__main__":
    unittest.main()
