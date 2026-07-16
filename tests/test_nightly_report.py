from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from scripts.nightly_report import build_report
from src import db_relations


class NightlyReportTests(unittest.TestCase):
    def test_reports_recent_summaries_and_review_queue(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(
            db_relations, "DB_PATH", str(Path(directory) / "relations.db")
        ):
            db_relations._db_initialized = False
            db_relations.save_item_summary("ITEM", "summary", "codex:test")
            db_relations.stage_reference_candidates(
                "ITEM", "test", [{"raw": "Author (2020). Title.", "title": "Title"}],
            )
            report = build_report(Path(db_relations.DB_PATH), since_hours=24)
            self.assertEqual(report["item_summaries"][0]["model"], "codex:test")
            self.assertEqual(report["reference_queue"]["pending"], 1)
            db_relations._db_initialized = False


if __name__ == "__main__":
    unittest.main()
