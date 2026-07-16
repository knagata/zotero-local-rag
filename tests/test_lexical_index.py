from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from src.lexical_index import (
    delete_by_attachment_keys,
    delete_by_note_key,
    search_chunks,
    upsert_chunks,
)


class LexicalIndexTests(unittest.TestCase):
    def setUp(self):
        self.tempdir = tempfile.TemporaryDirectory()
        self.path = Path(self.tempdir.name) / "lexical.sqlite3"
        upsert_chunks(
            ["a", "b", "n"],
            ["贈与と互酬性の民族誌", "gift exchange and reciprocity", "贈与に関するメモ"],
            [
                {"itemKey": "I1", "lang": "ja", "source_type": "pdf", "attachmentKey": "A1"},
                {"itemKey": "I2", "lang": "en", "source_type": "pdf", "attachmentKey": "A2"},
                {"itemKey": "I1", "lang": "ja", "source_type": "note", "noteKey": "N1"},
            ],
            path=self.path,
        )

    def tearDown(self):
        self.tempdir.cleanup()

    def test_trigram_search_supports_japanese(self):
        rows = search_chunks("贈与", path=self.path)
        self.assertEqual([row["chunk_id"] for row in rows], ["a"])

    def test_multiword_query_matches_non_adjacent_terms(self):
        rows = search_chunks("gift reciprocity", path=self.path)
        self.assertEqual([row["chunk_id"] for row in rows], ["b"])

    def test_notes_and_item_filters(self):
        self.assertEqual(search_chunks("メモ", path=self.path), [])
        rows = search_chunks("メモ", include_notes=True, item_keys=["I1"], path=self.path)
        self.assertEqual(rows[0]["chunk_id"], "n")

    def test_upsert_replaces_and_delete_removes(self):
        upsert_chunks(
            ["a"], ["交換ではなく市場"],
            [{"itemKey": "I1", "source_type": "pdf", "attachmentKey": "A1"}],
            path=self.path,
        )
        self.assertEqual(search_chunks("贈与", path=self.path), [])
        delete_by_attachment_keys(["A2"], path=self.path)
        self.assertEqual(search_chunks("gift", path=self.path), [])
        delete_by_note_key("N1", path=self.path)
        self.assertEqual(search_chunks("メモ", include_notes=True, path=self.path), [])


if __name__ == "__main__":
    unittest.main()
