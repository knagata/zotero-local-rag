from __future__ import annotations

import sqlite3
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from scripts.repair_short_form_references import repair
from src import db_relations


class RepairShortFormReferencesTests(unittest.TestCase):
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

    def test_preserves_raw_reference_while_removing_false_mapping(self):
        citing = db_relations.resolve_work(zotero_item_key="OWN", title="Owned")
        cited = db_relations.resolve_work(s2_paper_id="FALSE", title="False")
        connection = db_relations.get_db_connection()
        connection.execute('''
            INSERT INTO global_references
                (cited_paper_id, cited_title, citing_item_key, citing_chunk_id,
                 source, raw_reference_text, s2_status)
            VALUES ('FALSE', 'False', 'OWN', 'chunk', 'epub', '21 Ibid., p. 46.', 'mapped')
        ''')
        connection.commit()
        connection.close()
        db_relations.save_work_edge(
            citing, cited, source="epub", raw_reference="21 Ibid., p. 46.",
            citing_chunk_id="chunk",
        )
        preview = repair(self.db_path)
        self.assertEqual(preview["false_edges"], 1)
        backup = Path(self.tempdir.name) / "backup.db"
        repair(self.db_path, commit=True, backup_path=backup)
        connection = sqlite3.connect(self.db_path)
        row = connection.execute(
            "SELECT raw_reference_text, cited_title, s2_status FROM global_references"
        ).fetchone()
        edge_count = connection.execute("SELECT COUNT(*) FROM work_edges").fetchone()[0]
        connection.close()
        self.assertEqual(row, ("21 Ibid., p. 46.", None, "short_form"))
        self.assertEqual(edge_count, 0)
        self.assertTrue(backup.exists())


if __name__ == "__main__":
    unittest.main()
