from __future__ import annotations

import sqlite3
import tempfile
import unittest
from pathlib import Path

from scripts.check_chroma_health import (
    check_integrity, is_fts_only, orphaned_segment_dirs, repair_fts_index, run_check,
)


def _make_chroma_sqlite(db_path: Path, *, segment_ids: list[str]) -> None:
    connection = sqlite3.connect(str(db_path))
    connection.execute("CREATE TABLE segments (id TEXT PRIMARY KEY)")
    for segment_id in segment_ids:
        connection.execute("INSERT INTO segments VALUES (?)", (segment_id,))
    connection.execute("CREATE VIRTUAL TABLE embedding_fulltext_search USING fts5(string_value)")
    for i in range(1, 50):
        connection.execute(
            "INSERT INTO embedding_fulltext_search(rowid, string_value) VALUES (?, ?)",
            (i, f"chunk number {i} about anthropocene climate research"),
        )
    connection.commit()
    connection.close()


def _corrupt_fts_index(db_path: Path) -> None:
    # Deletes a row FTS5's own API would never remove this way, desyncing the
    # inverted-index shadow table from the content it is supposed to mirror --
    # the same shape of damage an interrupted write left in production
    # (2026-08-03).
    connection = sqlite3.connect(str(db_path))
    connection.execute(
        "DELETE FROM embedding_fulltext_search_data "
        "WHERE id = (SELECT max(id) FROM embedding_fulltext_search_data)"
    )
    connection.commit()
    connection.close()


class IsFtsOnlyTests(unittest.TestCase):
    def test_empty_issues_is_not_fts_only(self):
        self.assertFalse(is_fts_only([]))

    def test_all_fts_issues_is_fts_only(self):
        self.assertTrue(is_fts_only([
            "malformed inverted index for FTS5 table main.embedding_fulltext_search",
        ]))

    def test_a_non_fts_issue_is_not_fts_only(self):
        self.assertFalse(is_fts_only([
            "malformed inverted index for FTS5 table main.embedding_fulltext_search",
            "row 4 missing from index embeddings_pkey",
        ]))


class ChromaHealthCheckTests(unittest.TestCase):
    def setUp(self):
        self.tempdir = tempfile.TemporaryDirectory()
        self.addCleanup(self.tempdir.cleanup)
        self.chroma_dir = Path(self.tempdir.name)
        self.db_path = self.chroma_dir / "chroma.sqlite3"

    def test_clean_database_has_no_issues(self):
        _make_chroma_sqlite(self.db_path, segment_ids=["seg-a"])
        self.assertEqual(check_integrity(self.db_path), [])

    def test_corrupted_fts_index_is_detected_and_repaired(self):
        _make_chroma_sqlite(self.db_path, segment_ids=["seg-a"])
        _corrupt_fts_index(self.db_path)
        issues = check_integrity(self.db_path)
        self.assertTrue(issues)
        self.assertTrue(is_fts_only(issues))

        repair_fts_index(self.db_path)
        self.assertEqual(check_integrity(self.db_path), [])

    def test_run_check_repairs_fts_only_corruption_and_passes(self):
        _make_chroma_sqlite(self.db_path, segment_ids=["seg-a"])
        _corrupt_fts_index(self.db_path)
        report = run_check(self.chroma_dir, repair_fts=True)
        self.assertTrue(report["passed"])
        self.assertTrue(report["fts_repair_attempted"])
        self.assertEqual(report["integrity_issues"], [])

    def test_run_check_reports_without_repairing_when_disabled(self):
        _make_chroma_sqlite(self.db_path, segment_ids=["seg-a"])
        _corrupt_fts_index(self.db_path)
        report = run_check(self.chroma_dir, repair_fts=False)
        self.assertFalse(report["passed"])
        self.assertFalse(report["fts_repair_attempted"])
        self.assertTrue(report["integrity_issues"])

    def test_missing_database_is_reported_not_raised(self):
        report = run_check(self.chroma_dir, repair_fts=True)
        self.assertFalse(report["checked"])
        self.assertFalse(report["passed"])


class OrphanedSegmentDirsTests(unittest.TestCase):
    def setUp(self):
        self.tempdir = tempfile.TemporaryDirectory()
        self.addCleanup(self.tempdir.cleanup)
        self.chroma_dir = Path(self.tempdir.name)
        self.db_path = self.chroma_dir / "chroma.sqlite3"

    def test_referenced_segment_dir_is_not_orphaned(self):
        live = "93f04bbc-4e3e-412c-bac0-1940da60e0ad"
        _make_chroma_sqlite(self.db_path, segment_ids=[live])
        (self.chroma_dir / live).mkdir()
        self.assertEqual(orphaned_segment_dirs(self.chroma_dir, self.db_path), [])

    def test_unreferenced_segment_dir_is_orphaned(self):
        live = "93f04bbc-4e3e-412c-bac0-1940da60e0ad"
        orphan = "9233c541-99a7-4ce5-bc48-f3589ae01600"
        _make_chroma_sqlite(self.db_path, segment_ids=[live])
        (self.chroma_dir / live).mkdir()
        (self.chroma_dir / orphan).mkdir()
        self.assertEqual(orphaned_segment_dirs(self.chroma_dir, self.db_path), [orphan])

    def test_non_uuid_directories_are_ignored(self):
        _make_chroma_sqlite(self.db_path, segment_ids=[])
        (self.chroma_dir / "not-a-segment").mkdir()
        self.assertEqual(orphaned_segment_dirs(self.chroma_dir, self.db_path), [])


if __name__ == "__main__":
    unittest.main()
