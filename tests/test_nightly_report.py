from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from scripts.nightly_report import build_report, read_latest_run
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

    def test_reports_quota_skip_reason_from_latest_run(self):
        with tempfile.TemporaryDirectory() as directory:
            log = Path(directory) / "nightly.log"
            log.write_text(
                "[2026-07-18T03:30:00+09:00] nightly summaries start\n"
                '{"allowed": false, "reason": "weekly_quota_floor", '
                '"remaining_percent": 19}\n'
                "[2026-07-18T03:30:01+09:00] nightly summaries skipped: "
                "weekly quota floor or quota unavailable\n",
                encoding="utf-8",
            )
            run = read_latest_run(log)
            self.assertEqual(run["status"], "skipped")
            self.assertEqual(run["quota"]["remaining_percent"], 19)
            self.assertEqual(run["quota"]["reason"], "weekly_quota_floor")

    def test_reports_completed_latest_run(self):
        with tempfile.TemporaryDirectory() as directory:
            log = Path(directory) / "nightly.log"
            log.write_text(
                "[old] nightly summaries start\nTraceback (most recent call last):\n"
                "[new] nightly summaries start\n[new-end] nightly summaries end\n",
                encoding="utf-8",
            )
            run = read_latest_run(log)
            self.assertEqual(run["status"], "completed")
            self.assertEqual(run["started_at"], "new")


if __name__ == "__main__":
    unittest.main()
