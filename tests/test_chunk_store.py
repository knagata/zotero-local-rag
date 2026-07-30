from __future__ import annotations

import importlib.util
import os
import sqlite3
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from src.chunk_store import get_item_chunks, list_chunk_ids_without_item, list_item_keys

ROOT = Path(__file__).resolve().parents[1]


def _fresh_chunk_store_module():
    """Re-import chunk_store so its module-level DEFAULT_CHROMA_DIR is
    recomputed from the currently patched CHROMA_DIR env var -- it's bound
    once at import time, so reusing the already-imported src.chunk_store
    module would see the value from whenever the test suite first imported
    it, not the value patched in this test."""
    src_dir = str(ROOT / "src")
    if src_dir not in sys.path:
        sys.path.insert(0, src_dir)  # chunk_store's ImportError fallback needs bare `v3_data_plane`
    spec = importlib.util.spec_from_file_location(
        "chunk_store_fresh", ROOT / "src" / "chunk_store.py",
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


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
            INSERT INTO embeddings VALUES (3, 's1', 'A:p3:para0:part0');
            INSERT INTO embedding_metadata VALUES (1, 'itemKey', 'ITEM1', NULL, NULL, NULL);
            INSERT INTO embedding_metadata VALUES (1, 'chroma:document', 'later', NULL, NULL, NULL);
            INSERT INTO embedding_metadata VALUES (1, 'page', NULL, 10, NULL, NULL);
            INSERT INTO embedding_metadata VALUES (2, 'itemKey', 'ITEM1', NULL, NULL, NULL);
            INSERT INTO embedding_metadata VALUES (2, 'chroma:document', 'earlier', NULL, NULL, NULL);
            INSERT INTO embedding_metadata VALUES (2, 'page', NULL, 2, NULL, NULL);
            -- Chunk 3 has no itemKey row at all (not even an empty string).
            INSERT INTO embedding_metadata VALUES (3, 'chroma:document', 'orphaned', NULL, NULL, NULL);
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

    def test_a_chunk_with_no_itemkey_row_at_all_is_found(self):
        # P6-3 (2026-07-29): a per-item audit iterating item keys never
        # examines a chunk that has no itemKey metadata row -- an inner join
        # against embedding_metadata would silently skip it the same way.
        orphaned = list_chunk_ids_without_item(
            chroma_dir=self.chroma_dir, collection_name="paragraphs",
        )
        self.assertEqual(orphaned, ["A:p3:para0:part0"])


class DefaultChromaDirExpansionTests(unittest.TestCase):
    def test_a_tilde_prefixed_chroma_dir_is_expanded(self):
        # 2026-07-30 regression: DEFAULT_CHROMA_DIR was computed directly from
        # the raw CHROMA_DIR env var with no .expanduser(), unlike every other
        # V3 entry point (which all go through
        # v3_data_plane.resolve_configured_path). A ~-prefixed CHROMA_DIR
        # would silently resolve to a literal "~" subdirectory here instead
        # of the user's home directory.
        with tempfile.TemporaryDirectory() as home_dir:
            with patch.dict(os.environ, {"HOME": home_dir, "CHROMA_DIR": "~/chroma-data"}):
                module = _fresh_chunk_store_module()
        self.assertEqual(module.DEFAULT_CHROMA_DIR, Path(home_dir) / "chroma-data")
        self.assertNotIn("~", str(module.DEFAULT_CHROMA_DIR))


if __name__ == "__main__":
    unittest.main()
