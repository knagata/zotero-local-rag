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


if __name__ == "__main__":
    unittest.main()
