from __future__ import annotations

import sqlite3
import tempfile
import unittest
from pathlib import Path

from src.chunk_store import get_item_chunks, list_item_keys


class ChunkStoreTests(unittest.TestCase):
    def setUp(self):
        self.tempdir = tempfile.TemporaryDirectory()
        self.chroma_dir = Path(self.tempdir.name)
        connection = sqlite3.connect(self.chroma_dir / "chroma.sqlite3")
        connection.executescript('''
            CREATE TABLE collections (id TEXT PRIMARY KEY, name TEXT);
            CREATE TABLE segments (id TEXT PRIMARY KEY, scope TEXT, collection TEXT);
            CREATE TABLE embeddings (id INTEGER PRIMARY KEY, segment_id TEXT, embedding_id TEXT);
            CREATE TABLE embedding_metadata (
                id INTEGER, key TEXT, string_value TEXT, int_value INTEGER,
                float_value REAL, bool_value INTEGER
            );
            INSERT INTO collections VALUES ('c1', 'paragraphs');
            INSERT INTO segments VALUES ('s1', 'METADATA', 'c1');
            INSERT INTO embeddings VALUES (1, 's1', 'A:p10:para2:part0');
            INSERT INTO embeddings VALUES (2, 's1', 'A:p2:para1:part0');
            INSERT INTO embedding_metadata VALUES (1, 'itemKey', 'ITEM1', NULL, NULL, NULL);
            INSERT INTO embedding_metadata VALUES (1, 'chroma:document', 'later', NULL, NULL, NULL);
            INSERT INTO embedding_metadata VALUES (1, 'page', NULL, 10, NULL, NULL);
            INSERT INTO embedding_metadata VALUES (2, 'itemKey', 'ITEM1', NULL, NULL, NULL);
            INSERT INTO embedding_metadata VALUES (2, 'chroma:document', 'earlier', NULL, NULL, NULL);
            INSERT INTO embedding_metadata VALUES (2, 'page', NULL, 2, NULL, NULL);
        ''')
        connection.commit()
        connection.close()

    def tearDown(self):
        self.tempdir.cleanup()

    def test_reads_only_requested_collection_and_natural_sorts(self):
        chunks = get_item_chunks(
            "ITEM1", chroma_dir=self.chroma_dir, collection_name="paragraphs"
        )
        self.assertEqual([chunk["text"] for chunk in chunks], ["earlier", "later"])
        self.assertEqual(chunks[0]["metadata"]["page"], 2)
        self.assertEqual(
            list_item_keys(chroma_dir=self.chroma_dir, collection_name="paragraphs"),
            ["ITEM1"],
        )


if __name__ == "__main__":
    unittest.main()
