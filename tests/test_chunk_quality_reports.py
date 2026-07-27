"""Tests for reader-reported source-text damage (note 79, U0-b).

Covers the whole loop the feature exists for: a report is stored against the
exact text that was read, a re-extraction retires it, and a pending report
raises the document's standing as a re-OCR candidate. That last link is the
point of the feature -- degraded OCR is invisible to lexical search, so the
reader who hits it is often the only available signal.
"""
from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

import db_relations  # noqa: E402
from src.reocr_quality import candidate_assessment  # noqa: E402

DAMAGED = "informa- tion srorage and rerrieval wirhout permission in writin"


class ChunkQualityReportStoreTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tempdir = tempfile.TemporaryDirectory()
        self.db_patch = patch.object(
            db_relations, "DB_PATH", str(Path(self.tempdir.name) / "relations.db"),
        )
        self.db_patch.start()
        db_relations._db_initialized = False

    def tearDown(self) -> None:
        self.db_patch.stop()
        db_relations._db_initialized = False
        self.tempdir.cleanup()

    def _submit(self, **kwargs):
        payload = dict(
            item_key="ITEM", chunk_id="ATT:p12:para3:part0", chunk_text=DAMAGED,
            reason="ocr_garbled", details="srorage/rerrieval should be storage/retrieval",
            attachment_key="ATT", page=12,
        )
        payload.update(kwargs)
        return db_relations.submit_chunk_quality_report(**payload)

    def test_report_is_stored_and_listed(self):
        result = self._submit()
        self.assertEqual(result["status"], "pending")
        reports = db_relations.get_chunk_quality_reports("pending")
        self.assertEqual(len(reports), 1)
        self.assertEqual(reports[0]["chunk_id"], "ATT:p12:para3:part0")
        self.assertEqual(reports[0]["page"], 12)
        self.assertEqual(reports[0]["attachment_key"], "ATT")

    def test_repeat_report_of_the_same_text_increments_rather_than_duplicates(self):
        self._submit()
        second = self._submit(details="also wirhout -> without, writin -> writing")
        self.assertEqual(second["report_count"], 2)
        self.assertEqual(len(db_relations.get_chunk_quality_reports("pending")), 1)

    def test_reextraction_retires_the_report(self):
        # Scoping by the chunk's text hash is what makes this automatic: a
        # report describes text that no longer exists once the document is
        # re-extracted, so it must not keep accusing the fixed version.
        self._submit()
        repaired = self._submit(chunk_text="information storage and retrieval without permission")
        self.assertEqual(repaired["report_count"], 1)
        reports = db_relations.get_chunk_quality_reports("pending")
        self.assertEqual(len(reports), 2)
        self.assertEqual({row["report_count"] for row in reports}, {1})

    def test_unknown_reason_is_rejected(self):
        with self.assertRaises(ValueError):
            self._submit(reason="looks_odd")

    def test_details_must_carry_evidence(self):
        with self.assertRaises(ValueError):
            self._submit(details="bad")

    def test_item_key_and_chunk_id_are_required(self):
        with self.assertRaises(ValueError):
            self._submit(item_key="  ")
        with self.assertRaises(ValueError):
            self._submit(chunk_id="")

    def test_dismissed_reports_are_not_revived_by_a_repeat_report(self):
        self._submit()
        conn = db_relations.get_db_connection()
        try:
            conn.execute("UPDATE chunk_quality_reports SET status = 'dismissed'")
            conn.commit()
        finally:
            conn.close()
        again = self._submit()
        self.assertEqual(again["status"], "dismissed")
        self.assertEqual(db_relations.get_chunk_quality_reports("pending"), [])


class ChunkReportRaisesReocrPriorityTests(unittest.TestCase):
    """A pending report must actually reach re-OCR candidate selection."""

    def _assess(self, chunk_reports):
        # Deliberately healthy-looking deterministic metrics: this is the case
        # the reader signal exists for, where character counts and gibberish
        # ratios show nothing but the text is still unquotable.
        healthy = [
            {"id": f"ATT:p{page}:para0:part0",
             "text": "a perfectly ordinary sentence of running prose. " * 12,
             "metadata": {"attachmentKey": "ATT", "lang": "en"}}
            for page in range(1, 11)
        ]
        return candidate_assessment(
            quality={"total_pages": 10, "scanned_ratio": 0.0},
            chunks=healthy, structure_status="exact", summary_reports=[],
            current_engine="pymupdf", current_version="3",
            target_engine="docling", target_version="v3-adapter-1",
            chunk_reports=chunk_reports,
        )

    def test_healthy_document_without_reports_is_not_a_candidate(self):
        assessment = self._assess([])
        self.assertFalse(assessment["candidate"])
        self.assertNotIn("reported_source_text_damage", assessment["reasons"])

    def test_a_single_report_makes_it_a_candidate(self):
        assessment = self._assess([{"report_count": 1}])
        self.assertTrue(assessment["candidate"])
        self.assertIn("reported_source_text_damage", assessment["reasons"])

    def test_repeated_reports_score_higher(self):
        one = self._assess([{"report_count": 1}])["score"]
        many = self._assess([{"report_count": 3}])["score"]
        self.assertGreater(many, one)

    def test_report_weight_is_capped(self):
        # A single much-read passage must not outrank every other signal.
        capped = self._assess([{"report_count": 3}])["score"]
        runaway = self._assess([{"report_count": 99}])["score"]
        self.assertEqual(capped, runaway)


if __name__ == "__main__":
    unittest.main()
