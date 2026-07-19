from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from src import db_relations


class CaseQualityReportTests(unittest.TestCase):
    def setUp(self):
        self.tempdir = tempfile.TemporaryDirectory()
        self.db_path = Path(self.tempdir.name) / "relations.db"
        self.db_patch = patch.object(db_relations, "DB_PATH", str(self.db_path))
        self.db_patch.start()
        db_relations._db_initialized = False
        db_relations.replace_case_annotations("ITEM", "w0", [{
            "description": "A grounded practice occurred.",
            "chunk_id": "chunk-1", "evidence_quote": "grounded practice",
            "quality_status": "partial", "confidence": 0.8,
        }], model="deepseek:test")

    def tearDown(self):
        self.db_patch.stop()
        db_relations._db_initialized = False
        self.tempdir.cleanup()

    def test_case_status_evidence_and_reversible_report(self):
        case = db_relations.get_case_annotations("ITEM")[0]
        self.assertEqual(case["quality_status"], "partial")
        self.assertEqual(case["evidence"][0]["chunk_id"], "chunk-1")
        report = db_relations.submit_case_quality_report(
            case_id=case["case_id"], reason="unsupported_field",
            details="The stored period is not supported by the source chunk.",
            evidence_chunk_ids=["chunk-1"],
        )
        db_relations.resolve_case_quality_report(report["report_id"], "disable")
        self.assertIn(case["case_id"], db_relations.get_disabled_case_ids())

        connection = db_relations.get_db_connection()
        connection.execute(
            "UPDATE case_annotations SET description='regenerated' WHERE case_id=?",
            (case["case_id"],),
        )
        connection.commit()
        connection.close()
        self.assertNotIn(case["case_id"], db_relations.get_disabled_case_ids())

    def test_invalid_report_is_rejected(self):
        case_id = db_relations.get_case_annotations("ITEM")[0]["case_id"]
        with self.assertRaisesRegex(ValueError, "at least 10"):
            db_relations.submit_case_quality_report(
                case_id=case_id, reason="other", details="wrong",
            )

    def test_item_replacement_rolls_back_atomically(self):
        before = db_relations.get_case_annotations("ITEM")
        with self.assertRaises(Exception):
            db_relations.replace_item_case_annotations("ITEM", [{
                "section_id": "w1", "cases": [{
                    "description": "new case", "confidence": {"invalid": "sqlite value"},
                }],
            }], model="deepseek:test")
        after = db_relations.get_case_annotations("ITEM")
        self.assertEqual([row["description"] for row in after], [row["description"] for row in before])


if __name__ == "__main__":
    unittest.main()
