from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from src import db_relations, work_identity


class WorkIdentityTests(unittest.TestCase):
    def setUp(self):
        self.tempdir = tempfile.TemporaryDirectory()
        self.db_patch = patch.object(
            db_relations, "DB_PATH", str(Path(self.tempdir.name) / "relations.db")
        )
        self.db_patch.start()
        db_relations._db_initialized = False

    def tearDown(self):
        self.db_patch.stop()
        db_relations._db_initialized = False
        self.tempdir.cleanup()

    def test_promotes_only_distinct_authored_chapters(self):
        db_relations.resolve_work(zotero_item_key="ITEM", title="Volume")
        connection = db_relations.get_db_connection()
        connection.executemany(
            """
            INSERT INTO section_summaries
                (item_key, section_id, chapter, summary, chapter_authors)
            VALUES (?, ?, ?, ?, ?)
            """,
            [
                ("ITEM", "c1", "Chapter A", "a", "Author A"),
                ("ITEM", "c2", "Chapter B", "b", "Author B"),
            ],
        )
        connection.commit()
        connection.close()
        preview = work_identity.promote_chapters("ITEM")
        self.assertTrue(preview["eligible"])
        result = work_identity.promote_chapters("ITEM", dry_run=False)
        self.assertEqual(result["status"], "promoted")
        self.assertEqual(len(result["chapter_work_ids"]), 2)

    def test_explicit_original_title_creates_translation_link(self):
        item = {
            "title": "贈与論", "extra": "Original title: Essai sur le don",
            "language": "ja", "date": "2014", "creators": [{"lastName": "Mauss"}],
        }
        with patch.object(work_identity, "_zotero_item", return_value=item):
            preview = work_identity.detect_translation("ITEM")
            self.assertEqual(preview["original_title"], "Essai sur le don")
            result = work_identity.detect_translation("ITEM", dry_run=False)
        self.assertEqual(result["status"], "linked")
        self.assertEqual(
            set(db_relations.get_work_cluster(result["derived_work_id"])),
            {result["derived_work_id"], result["original_work_id"]},
        )
        self.assertEqual(
            db_relations.get_canonical_work_id(result["derived_work_id"]),
            result["original_work_id"],
        )
        candidates = db_relations.get_s2_lookup_candidates("ITEM")
        self.assertIn("Essai sur le don", {row["title"] for row in candidates})

    def test_ndl_translation_requires_matching_author(self):
        item = {
            "title": "贈与論", "extra": "", "language": "ja", "date": "2014",
            "creators": [{"firstName": "Marcel", "lastName": "Mauss"}],
        }
        record = {
            "title": "贈与論", "alternative_titles": ["Essai sur le don"],
            "authors": "Lewis Hyde", "ndl_bibid": "R1", "year": 1925,
        }
        with patch.object(work_identity, "_zotero_item", return_value=item), patch.object(
            work_identity, "search_ndl", return_value=[record]
        ), patch.object(work_identity, "save_work_link") as save:
            result = work_identity.detect_translation("ITEM", dry_run=False)
        self.assertEqual(result["status"], "needs_review")
        save.assert_not_called()

    def test_ndl_translation_accepts_matching_author(self):
        item = {
            "title": "贈与論", "extra": "", "language": "ja", "date": "2014",
            "creators": [{"firstName": "Marcel", "lastName": "Mauss"}],
        }
        record = {
            "title": "贈与論", "alternative_titles": ["Essai sur le don"],
            "authors": "Mauss, Marcel", "ndl_bibid": "R1", "year": 1925,
        }
        with patch.object(work_identity, "_zotero_item", return_value=item), patch.object(
            work_identity, "search_ndl", return_value=[record]
        ):
            result = work_identity.detect_translation("ITEM", dry_run=False)
        self.assertEqual(result["status"], "linked")


if __name__ == "__main__":
    unittest.main()
