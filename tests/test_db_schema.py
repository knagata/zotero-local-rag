from __future__ import annotations

import sqlite3
import tempfile
import threading
import time
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from unittest.mock import MagicMock, patch

from src import db_relations
from src.db_schema import add_column


class AddColumnTests(unittest.TestCase):
    def test_duplicate_column_is_the_only_ignored_operational_error(self):
        cursor = MagicMock()
        cursor.execute.side_effect = sqlite3.OperationalError(
            "duplicate column name: title"
        )
        add_column(cursor, "ALTER TABLE works ADD COLUMN title TEXT")

    def test_locked_database_is_not_misreported_as_an_applied_migration(self):
        cursor = MagicMock()
        cursor.execute.side_effect = sqlite3.OperationalError("database is locked")
        with self.assertRaisesRegex(sqlite3.OperationalError, "database is locked"):
            add_column(cursor, "ALTER TABLE works ADD COLUMN title TEXT")


class SchemaInitializationConcurrencyTests(unittest.TestCase):
    def test_failed_initialization_closes_connection(self):
        connection = MagicMock()
        connection.execute.return_value.fetchone.return_value = (0,)
        with tempfile.TemporaryDirectory() as directory, \
                patch.object(
                    db_relations, "DB_PATH", str(Path(directory) / "relations.db")
                ), \
                patch.object(db_relations, "_db_initialized", False), \
                patch.object(db_relations, "_initialized_db_path", None), \
                patch.object(db_relations.sqlite3, "connect", return_value=connection), \
                patch.object(
                    db_relations, "_init_db", side_effect=RuntimeError("migration failed")
                ):
            with self.assertRaisesRegex(RuntimeError, "migration failed"):
                db_relations.get_db_connection()

        connection.close.assert_called_once_with()

    def test_first_connection_initializes_each_database_once(self):
        with tempfile.TemporaryDirectory() as directory:
            db_path = str(Path(directory) / "relations.db")
            entered = 0
            entered_lock = threading.Lock()

            def slow_initialize(_connection):
                nonlocal entered
                with entered_lock:
                    entered += 1
                time.sleep(0.05)

            def connect_and_close(_value):
                connection = db_relations.get_db_connection()
                connection.close()

            with patch.object(db_relations, "DB_PATH", db_path), \
                    patch.object(db_relations, "_db_initialized", False), \
                    patch.object(db_relations, "_initialized_db_path", None), \
                    patch.object(db_relations, "_init_db", side_effect=slow_initialize):
                with ThreadPoolExecutor(max_workers=4) as executor:
                    list(executor.map(connect_and_close, range(4)))

            self.assertEqual(entered, 1)


class VersionedSchemaInitializationTests(unittest.TestCase):
    @staticmethod
    def _legacy_connection(path: Path) -> sqlite3.Connection:
        connection = sqlite3.connect(path)
        connection.executescript(
            """
            CREATE TABLE global_citations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                citing_paper_id TEXT,
                citing_title TEXT,
                citing_year INTEGER,
                context_snippet TEXT,
                cited_item_key TEXT,
                cited_chunk_id TEXT,
                similarity_distance REAL,
                page_hint TEXT,
                created_at TIMESTAMP
            );
            CREATE TABLE global_references (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                cited_paper_id TEXT,
                cited_title TEXT,
                cited_year INTEGER,
                context_snippet TEXT,
                citing_item_key TEXT,
                citing_chunk_id TEXT,
                similarity_distance REAL,
                page_hint TEXT,
                source TEXT,
                raw_reference_text TEXT,
                created_at TIMESTAMP
            );
            CREATE TABLE item_citation_status (
                item_key TEXT PRIMARY KEY,
                s2_status TEXT,
                last_checked_at TIMESTAMP
            );
            """
        )
        connection.execute(
            "INSERT INTO global_citations(citing_paper_id, cited_item_key) "
            "VALUES ('paper-1', 'item-1')"
        )
        connection.commit()
        return connection

    def test_new_database_is_initialized_and_versioned(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "relations.db"
            connection = sqlite3.connect(path)
            db_relations._init_db(connection)

            self.assertEqual(connection.execute("PRAGMA user_version").fetchone()[0], 1)
            self.assertIsNotNone(
                connection.execute(
                    "SELECT 1 FROM sqlite_master WHERE type='table' "
                    "AND name='document_structures'"
                ).fetchone()
            )
            connection.close()

    def test_legacy_database_keeps_data_while_ad_hoc_columns_migrate(self):
        with tempfile.TemporaryDirectory() as directory:
            connection = self._legacy_connection(Path(directory) / "relations.db")
            db_relations._init_db(connection)

            columns = {
                row[1]
                for row in connection.execute("PRAGMA table_info(global_citations)")
            }
            self.assertIn("citing_doi", columns)
            self.assertEqual(connection.execute("SELECT count(*) FROM global_citations").fetchone()[0], 1)
            self.assertEqual(connection.execute("PRAGMA user_version").fetchone()[0], 1)
            connection.close()

    def test_second_initialization_is_a_noop_for_version_one(self):
        with tempfile.TemporaryDirectory() as directory:
            connection = sqlite3.connect(Path(directory) / "relations.db")
            db_relations._init_db(connection)
            with patch.object(
                db_relations,
                "add_column",
                side_effect=AssertionError("versioned migration ran twice"),
            ):
                db_relations._init_db(connection)
            connection.close()

    def test_unknown_future_schema_version_fails_before_writing(self):
        with tempfile.TemporaryDirectory() as directory:
            connection = sqlite3.connect(Path(directory) / "relations.db")
            connection.execute("PRAGMA user_version = 2")
            connection.commit()

            with self.assertRaisesRegex(RuntimeError, "newer than supported"):
                db_relations._init_db(connection)

            self.assertIsNone(
                connection.execute(
                    "SELECT 1 FROM sqlite_master WHERE type='table' "
                    "AND name='global_citations'"
                ).fetchone()
            )
            self.assertEqual(connection.execute("PRAGMA user_version").fetchone()[0], 2)
            connection.close()

    def test_connection_rejects_future_version_before_changing_journal_mode(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "relations.db"
            connection = sqlite3.connect(path)
            connection.execute("PRAGMA user_version = 2")
            connection.commit()
            original_mode = connection.execute("PRAGMA journal_mode").fetchone()[0]
            connection.close()

            with patch.object(db_relations, "DB_PATH", str(path)), patch.object(
                db_relations, "_db_initialized", False,
            ), patch.object(db_relations, "_initialized_db_path", None):
                with self.assertRaisesRegex(RuntimeError, "newer than supported"):
                    db_relations.get_db_connection()

            connection = sqlite3.connect(path)
            self.assertEqual(connection.execute("PRAGMA user_version").fetchone()[0], 2)
            self.assertEqual(
                connection.execute("PRAGMA journal_mode").fetchone()[0],
                original_mode,
            )
            connection.close()

    def test_failed_migration_rolls_back_schema_and_version(self):
        with tempfile.TemporaryDirectory() as directory:
            connection = self._legacy_connection(Path(directory) / "relations.db")
            original_add_column = db_relations.add_column
            calls = 0

            def fail_after_first_column(cursor, sql):
                nonlocal calls
                calls += 1
                if calls == 2:
                    raise RuntimeError("migration interrupted")
                return original_add_column(cursor, sql)

            with patch.object(db_relations, "add_column", side_effect=fail_after_first_column):
                with self.assertRaisesRegex(RuntimeError, "migration interrupted"):
                    db_relations._init_db(connection)

            columns = {
                row[1]
                for row in connection.execute("PRAGMA table_info(global_citations)")
            }
            self.assertNotIn("citing_citation_count", columns)
            self.assertEqual(connection.execute("SELECT count(*) FROM global_citations").fetchone()[0], 1)
            self.assertEqual(connection.execute("PRAGMA user_version").fetchone()[0], 0)
            connection.close()

    def test_existing_transaction_is_not_committed_by_schema_savepoint(self):
        with tempfile.TemporaryDirectory() as directory:
            connection = sqlite3.connect(Path(directory) / "relations.db")
            connection.execute("CREATE TABLE caller_state (value TEXT)")
            connection.execute("INSERT INTO caller_state VALUES ('uncommitted')")
            self.assertTrue(connection.in_transaction)

            db_relations._init_db(connection)

            self.assertTrue(connection.in_transaction)
            self.assertEqual(
                connection.execute("SELECT value FROM caller_state").fetchone()[0],
                "uncommitted",
            )
            connection.commit()
            connection.close()

    def test_failed_schema_savepoint_preserves_the_callers_transaction(self):
        with tempfile.TemporaryDirectory() as directory:
            connection = sqlite3.connect(Path(directory) / "relations.db")
            connection.execute("CREATE TABLE caller_state (value TEXT)")
            connection.execute("INSERT INTO caller_state VALUES ('uncommitted')")

            with patch.object(
                db_relations,
                "_migrate_relation_identity",
                side_effect=RuntimeError("migration interrupted"),
            ):
                with self.assertRaisesRegex(RuntimeError, "migration interrupted"):
                    db_relations._init_db(connection)

            self.assertTrue(connection.in_transaction)
            self.assertEqual(
                connection.execute("SELECT value FROM caller_state").fetchone()[0],
                "uncommitted",
            )
            self.assertIsNone(
                connection.execute(
                    "SELECT 1 FROM sqlite_master WHERE type='table' "
                    "AND name='global_citations'"
                ).fetchone()
            )
            self.assertEqual(connection.execute("PRAGMA user_version").fetchone()[0], 0)
            connection.rollback()
            connection.close()


if __name__ == "__main__":
    unittest.main()
