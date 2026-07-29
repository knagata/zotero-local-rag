from __future__ import annotations

import sqlite3
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from src.item_vectors import get_item_meta


class ItemVectorV3Tests(unittest.TestCase):
    def test_metadata_query_excludes_same_item_from_legacy_collection(self):
        with tempfile.TemporaryDirectory() as directory:
            db = Path(directory) / "chroma.sqlite3"
            connection = sqlite3.connect(db)
            connection.executescript("""
                CREATE TABLE collections (id TEXT PRIMARY KEY, name TEXT);
                CREATE TABLE segments (id TEXT PRIMARY KEY, scope TEXT, collection TEXT);
                CREATE TABLE embeddings (
                    id INTEGER PRIMARY KEY, segment_id TEXT, embedding_id TEXT
                );
                CREATE TABLE embedding_metadata (
                    id INTEGER, key TEXT, string_value TEXT, int_value INTEGER,
                    float_value REAL, bool_value INTEGER
                );
                INSERT INTO collections VALUES ('old', 'zotero_paragraphs');
                INSERT INTO collections VALUES ('v3', 'zotero_paragraphs_v3');
                INSERT INTO segments VALUES ('old-meta', 'METADATA', 'old');
                INSERT INTO segments VALUES ('v3-meta', 'METADATA', 'v3');
                INSERT INTO embeddings VALUES (1, 'old-meta', 'old-chunk');
                INSERT INTO embeddings VALUES (2, 'v3-meta', 'v3-chunk');
                INSERT INTO embedding_metadata VALUES (1, 'itemKey', 'ITEM', NULL, NULL, NULL);
                INSERT INTO embedding_metadata VALUES (1, 'title', 'Old title', NULL, NULL, NULL);
                INSERT INTO embedding_metadata VALUES (2, 'itemKey', 'ITEM', NULL, NULL, NULL);
                INSERT INTO embedding_metadata VALUES (2, 'title', 'V3 title', NULL, NULL, NULL);
            """)
            connection.commit()
            connection.close()
            with patch.dict(
                "os.environ", {"CHROMA_COLLECTION": "zotero_paragraphs_v3"}, clear=False,
            ):
                metadata = get_item_meta(["ITEM"], chroma_db=db)
        self.assertEqual(metadata["ITEM"]["title"], "V3 title")


if __name__ == "__main__":
    unittest.main()
