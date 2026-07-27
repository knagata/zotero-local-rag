from __future__ import annotations

import sqlite3
import tempfile
import unittest
from pathlib import Path

from scripts.audit_legacy_retirement import LEGACY_COLLECTIONS, compare, fts_snapshot, sqlite_snapshot


class LegacyRetirementAuditTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.db = self.root / "relations.db"
        self.fts = self.root / "lexical.sqlite3"
        with sqlite3.connect(self.db) as conn:
            conn.executescript("""
                CREATE TABLE item_summaries (item_key TEXT PRIMARY KEY, summary TEXT, updated_at TEXT);
                CREATE TABLE section_summaries (item_key TEXT, section_id TEXT, summary TEXT, updated_at TEXT);
                CREATE TABLE insight_generation_status (item_key TEXT, kind TEXT, updated_at TEXT);
                INSERT INTO insight_generation_status VALUES ('A', 'sections', '2026-07-01');
            """)
        with sqlite3.connect(self.fts) as conn:
            conn.execute("CREATE VIRTUAL TABLE chunks_fts USING fts5(chunk_id UNINDEXED, item_key UNINDEXED, attachment_key UNINDEXED, body)")
            conn.execute("INSERT INTO chunks_fts VALUES ('c1', 'I1', 'A1', 'text')")

    def tearDown(self):
        self.temp.cleanup()

    def test_sqlite_snapshot_is_read_only_and_fingerprints_content(self):
        before = sqlite_snapshot(self.db)
        self.assertEqual(before["tables"]["item_summaries"]["count"], 0)
        self.assertEqual(before["tables"]["insight_generation_status"]["count"], 1)
        with sqlite3.connect(self.db) as conn:
            conn.execute("UPDATE insight_generation_status SET item_key='B'")
        after = sqlite_snapshot(self.db)
        report = compare(
            {"relations": before, "legacy_chroma": {"collections": {}}, "legacy_fts": {"mtime_ns": 1}, "legacy_manifest": {"sha256": "a"}},
            {"relations": after, "legacy_chroma": {"collections": {}}, "legacy_fts": {"mtime_ns": 1}, "legacy_manifest": {"sha256": "a"}},
        )
        self.assertIn("legacy_table_content_changed:insight_generation_status", report["failures"])

    def test_compare_reports_growth_and_collection_metadata_change(self):
        collections = {
            name: {"count": 1, "metadata_fingerprint": "one"}
            for name in LEGACY_COLLECTIONS
        }
        baseline = {
            "relations": {"tables": {name: {"count": 0, "fingerprint": "same"} for name in ("item_summaries", "section_summaries", "insight_generation_status")}},
            "legacy_chroma": {"collections": collections},
            "legacy_fts": {"chunks": 1, "mtime_ns": 1},
            "legacy_manifest": {"sha256": "same"},
        }
        changed = {
            **baseline,
            "relations": {"tables": {**baseline["relations"]["tables"], "item_summaries": {"count": 1, "fingerprint": "changed"}}},
            "legacy_chroma": {"collections": {**collections, "zotero_paragraphs": {"count": 2, "metadata_fingerprint": "changed"}}},
            "legacy_fts": {"chunks": 2, "mtime_ns": 2},
            "legacy_manifest": {"sha256": "changed"},
        }
        report = compare(baseline, changed)
        self.assertFalse(report["passed"])
        self.assertIn("legacy_table_row_growth:item_summaries", report["failures"])
        self.assertIn("legacy_collection_row_growth:zotero_paragraphs", report["failures"])
        self.assertIn("legacy_fts_row_growth", report["failures"])
        self.assertIn("legacy_manifest_changed", report["failures"])

    def test_fts_snapshot_counts_rows_without_opening_a_write_connection(self):
        result = fts_snapshot(self.fts)
        self.assertEqual(result["chunks"], 1)
        self.assertEqual(result["items"], 1)
        self.assertEqual(result["attachments"], 1)


if __name__ == "__main__":
    unittest.main()
