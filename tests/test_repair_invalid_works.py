from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from scripts.repair_invalid_works import repair
from src import db_relations


class RepairInvalidWorksTests(unittest.TestCase):
    def setUp(self):
        self.tempdir = tempfile.TemporaryDirectory()
        self.db_path = Path(self.tempdir.name) / "relations.db"
        self.db_patch = patch.object(db_relations, "DB_PATH", str(self.db_path))
        self.db_patch.start()
        db_relations._db_initialized = False

    def tearDown(self):
        self.db_patch.stop()
        db_relations._db_initialized = False
        self.tempdir.cleanup()

    def test_dry_run_and_committed_repair_with_backup(self):
        citing = db_relations.resolve_work(zotero_item_key="OWNED", title="Owned")
        connection = db_relations.get_db_connection()
        cursor = connection.execute("INSERT INTO works DEFAULT VALUES")
        invalid = int(cursor.lastrowid)
        connection.commit()
        connection.close()
        db_relations.save_work_edge(citing, invalid, source="test", raw_reference="raw")
        preview = repair(self.db_path, [invalid])
        self.assertEqual(preview["before"]["incoming_edges"], 1)
        backup = Path(self.tempdir.name) / "backup.sqlite3"
        result = repair(self.db_path, [invalid], commit=True, backup_path=backup)
        self.assertEqual(result["deleted_edges"], 1)
        self.assertEqual(result["deleted_works"], 1)
        self.assertTrue(backup.exists())

    def test_refuses_work_with_bibliographic_identity(self):
        identified = db_relations.resolve_work(title="Real work", doi="10.1/real")
        backup = Path(self.tempdir.name) / "backup.sqlite3"
        with self.assertRaises(RuntimeError):
            repair(self.db_path, [identified], commit=True, backup_path=backup)
        self.assertFalse(backup.exists())


if __name__ == "__main__":
    unittest.main()
