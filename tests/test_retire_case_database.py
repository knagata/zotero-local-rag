import json
from pathlib import Path
import sqlite3
import tempfile
import unittest
from unittest.mock import patch

from scripts import retire_case_database


class RetireCaseDatabaseTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.db = self.root / "relations.db"
        with sqlite3.connect(self.db) as conn:
            conn.executescript('''
                CREATE TABLE case_annotations (case_id INTEGER PRIMARY KEY, item_key TEXT);
                CREATE TABLE case_evidence (evidence_id INTEGER PRIMARY KEY, case_id INTEGER);
                CREATE TABLE case_quality_reports (report_id INTEGER PRIMARY KEY, case_id INTEGER);
                CREATE TABLE insight_generation_status (
                    item_key TEXT, kind TEXT CHECK(kind IN ('sections','cases'))
                );
                CREATE TABLE artifact_processing_status (
                    item_key TEXT NOT NULL, attachment_key TEXT NOT NULL DEFAULT '',
                    artifact_type TEXT NOT NULL CHECK(artifact_type IN
                        ('extraction','structure','summary','cases','references','embeddings')),
                    status TEXT NOT NULL CHECK(status IN
                        ('pending','running','success','empty','degraded','blocked','failed','stale','excluded')),
                    reason_code TEXT, message TEXT, retryable INTEGER NOT NULL DEFAULT 0,
                    attempt_count INTEGER NOT NULL DEFAULT 0, source_fingerprint TEXT,
                    processor_version TEXT, model TEXT, counts_json TEXT, fallback_kind TEXT,
                    started_at TIMESTAMP, finished_at TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    PRIMARY KEY(item_key, attachment_key, artifact_type)
                );
                CREATE TABLE artifact_processing_events (
                    event_id INTEGER PRIMARY KEY, item_key TEXT, attachment_key TEXT,
                    artifact_type TEXT, from_status TEXT, to_status TEXT, reason_code TEXT,
                    message TEXT, run_id TEXT, created_at TIMESTAMP
                );
                INSERT INTO case_annotations VALUES (1, 'ITEM');
                INSERT INTO case_evidence VALUES (2, 1);
                INSERT INTO case_quality_reports VALUES (3, 1);
                INSERT INTO insight_generation_status VALUES ('ITEM', 'cases');
                INSERT INTO artifact_processing_status
                    (item_key, artifact_type, status) VALUES ('ITEM', 'cases', 'success');
                INSERT INTO artifact_processing_status
                    (item_key, artifact_type, status) VALUES ('ITEM', 'summary', 'success');
            ''')

    def tearDown(self):
        self.temp.cleanup()

    def test_dry_run_is_default_and_writes_nothing(self):
        result = retire_case_database.run(
            db_path=self.db, chroma_path=self.root / "chroma",
            backups_path=self.root / "backups", apply=False,
        )
        self.assertEqual(result["mode"], "dry-run")
        self.assertEqual(result["sqlite"]["case_annotations"], 1)
        self.assertFalse((self.root / "backups").exists())
        with sqlite3.connect(self.db) as conn:
            self.assertEqual(conn.execute("SELECT COUNT(*) FROM case_annotations").fetchone()[0], 1)

    @patch.object(retire_case_database, "backup_chroma", return_value={"zotero__cases": 2})
    @patch.object(retire_case_database, "retire_chroma")
    def test_apply_exports_before_removal_and_preserves_other_statuses(self, retire_chroma, _backup):
        result = retire_case_database.run(
            db_path=self.db, chroma_path=self.root / "chroma",
            backups_path=self.root / "backups", apply=True,
        )
        backup_dir = Path(result["backup_dir"])
        self.assertTrue((backup_dir / "cases.sqlite.sql").exists())
        rows = [json.loads(line) for line in (backup_dir / "cases.sqlite.jsonl").read_text().splitlines()]
        self.assertTrue(any(row["table"] == "case_annotations" for row in rows))
        with sqlite3.connect(self.db) as conn:
            names = {row[0] for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
            self.assertNotIn("case_annotations", names)
            self.assertEqual(
                conn.execute("SELECT artifact_type FROM artifact_processing_status").fetchone()[0],
                "summary",
            )
            with self.assertRaises(sqlite3.IntegrityError):
                conn.execute(
                    "INSERT INTO artifact_processing_status (item_key, artifact_type, status) VALUES ('X','cases','success')"
                )
            with self.assertRaises(sqlite3.IntegrityError):
                conn.execute(
                    "INSERT INTO insight_generation_status (item_key, kind) VALUES ('X','cases')"
                )
        retire_chroma.assert_called_once_with(self.root / "chroma", {"zotero__cases": 2})


if __name__ == "__main__":
    unittest.main()
