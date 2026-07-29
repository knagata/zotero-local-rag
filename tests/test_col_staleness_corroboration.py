"""Tests for the row-count corroboration on _col()'s mtime staleness check.

Constructing *any* chromadb.PersistentClient for CHROMA_DIR -- including a
read-only one from an unrelated audit/verification script -- bumps
chroma.sqlite3's mtime without writing a single new vector. Before this fix,
_col() treated any mtime increase as proof the indexer wrote new data and
called the expensive _reset_col(), discarding and re-mmapping the whole HNSW
segment (measured: p95 ~20s / worst-case 30-90s query latency, 2026-07-28).
"""
from __future__ import annotations

import os
import sys
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import rag_mcp_server  # noqa: E402


class ColStalenessCorroborationTests(unittest.TestCase):
    def setUp(self):
        self._orig_col = rag_mcp_server._COL
        self._orig_mtime = rag_mcp_server._COL_INIT_MTIME
        self._orig_db_mtime = rag_mcp_server._COL_INIT_DB_MTIME
        self._orig_manifest_mtime = rag_mcp_server._COL_INIT_MANIFEST_MTIME
        self._orig_row_count = rag_mcp_server._COL_INIT_ROW_COUNT
        self._orig_coll_name = rag_mcp_server._EMB_COLLECTION_NAME

    def tearDown(self):
        rag_mcp_server._COL = self._orig_col
        rag_mcp_server._COL_INIT_MTIME = self._orig_mtime
        rag_mcp_server._COL_INIT_DB_MTIME = self._orig_db_mtime
        rag_mcp_server._COL_INIT_MANIFEST_MTIME = self._orig_manifest_mtime
        rag_mcp_server._COL_INIT_ROW_COUNT = self._orig_row_count
        rag_mcp_server._EMB_COLLECTION_NAME = self._orig_coll_name

    def test_db_touch_with_unchanged_manifest_and_row_count_skips_reset(self):
        stale_col = Mock()
        rag_mcp_server._COL = stale_col
        rag_mcp_server._COL_INIT_MTIME = 100.0
        rag_mcp_server._COL_INIT_DB_MTIME = 60.0
        rag_mcp_server._COL_INIT_MANIFEST_MTIME = 40.0
        rag_mcp_server._COL_INIT_ROW_COUNT = 500
        rag_mcp_server._EMB_COLLECTION_NAME = "zotero_paragraphs_v3"

        with patch.object(rag_mcp_server, "_db_mtimes", return_value=(160.0, 40.0)), \
             patch.object(rag_mcp_server, "_collection_row_count", return_value=500) as row_count, \
             patch.object(rag_mcp_server, "_reset_col") as reset:
            result = rag_mcp_server._col()

        reset.assert_not_called()
        row_count.assert_called_once_with("zotero_paragraphs_v3")
        self.assertIs(result, stale_col)
        # The false-alarm mtime is still adopted so it stops re-triggering.
        self.assertEqual(rag_mcp_server._COL_INIT_MTIME, 200.0)
        self.assertEqual(rag_mcp_server._COL_INIT_DB_MTIME, 160.0)
        self.assertEqual(rag_mcp_server._COL_INIT_MANIFEST_MTIME, 40.0)

    def test_manifest_change_reloads_even_when_row_count_is_unchanged(self):
        stale_col = Mock()
        rag_mcp_server._COL = stale_col
        rag_mcp_server._COL_INIT_MTIME = 100.0
        rag_mcp_server._COL_INIT_DB_MTIME = 60.0
        rag_mcp_server._COL_INIT_MANIFEST_MTIME = 40.0
        rag_mcp_server._COL_INIT_ROW_COUNT = 500
        rag_mcp_server._EMB_COLLECTION_NAME = "zotero_paragraphs_v3"

        def _reset():
            rag_mcp_server._COL = None

        with patch.object(rag_mcp_server, "_db_mtimes", return_value=(160.0, 50.0)), \
             patch.object(rag_mcp_server, "_collection_row_count") as row_count, \
             patch.object(rag_mcp_server, "_reset_col", side_effect=_reset) as reset, \
             patch.object(rag_mcp_server, "_EMB_FN", Mock()), \
             patch.object(
                 rag_mcp_server, "open_chroma_collection",
                 return_value=Mock(count=lambda: 500),
             ):
            rag_mcp_server._col()

        reset.assert_called_once()
        row_count.assert_not_called()

    def test_mtime_rise_with_changed_row_count_triggers_reset(self):
        stale_col = Mock()
        rag_mcp_server._COL = stale_col
        rag_mcp_server._COL_INIT_MTIME = 100.0
        rag_mcp_server._COL_INIT_DB_MTIME = 60.0
        rag_mcp_server._COL_INIT_MANIFEST_MTIME = 40.0
        rag_mcp_server._COL_INIT_ROW_COUNT = 500
        rag_mcp_server._EMB_COLLECTION_NAME = "zotero_paragraphs_v3"

        def _reset():
            rag_mcp_server._COL = None

        with patch.object(rag_mcp_server, "_db_mtimes", return_value=(160.0, 40.0)), \
             patch.object(rag_mcp_server, "_collection_row_count", return_value=501), \
             patch.object(rag_mcp_server, "_reset_col", side_effect=_reset) as reset, \
             patch.object(rag_mcp_server, "_EMB_FN", Mock()), \
             patch.object(rag_mcp_server, "open_chroma_collection", return_value=Mock(count=lambda: 501)):
            rag_mcp_server._col()

        reset.assert_called_once()

    def test_mtime_rise_with_unknown_row_count_triggers_reset(self):
        """A None row count (DB unreadable) must fail safe, not skip the reset."""
        stale_col = Mock()
        rag_mcp_server._COL = stale_col
        rag_mcp_server._COL_INIT_MTIME = 100.0
        rag_mcp_server._COL_INIT_DB_MTIME = 60.0
        rag_mcp_server._COL_INIT_MANIFEST_MTIME = 40.0
        rag_mcp_server._COL_INIT_ROW_COUNT = 500
        rag_mcp_server._EMB_COLLECTION_NAME = "zotero_paragraphs_v3"

        def _reset():
            rag_mcp_server._COL = None

        with patch.object(rag_mcp_server, "_db_mtimes", return_value=(160.0, 40.0)), \
             patch.object(rag_mcp_server, "_collection_row_count", return_value=None), \
             patch.object(rag_mcp_server, "_reset_col", side_effect=_reset) as reset, \
             patch.object(rag_mcp_server, "_EMB_FN", Mock()), \
             patch.object(rag_mcp_server, "open_chroma_collection", return_value=Mock(count=lambda: 500)):
            rag_mcp_server._col()

        reset.assert_called_once()


class CollectionRowCountTests(unittest.TestCase):
    def test_missing_db_returns_none(self):
        with patch.object(rag_mcp_server, "CHROMA_DIR", "/nonexistent/path/for/test"):
            self.assertIsNone(rag_mcp_server._collection_row_count("any_collection"))

    def test_reads_actual_embedding_count_via_readonly_sqlite(self):
        import sqlite3
        import tempfile

        with tempfile.TemporaryDirectory() as directory:
            db_path = os.path.join(directory, "chroma.sqlite3")
            conn = sqlite3.connect(db_path)
            conn.executescript(
                """
                CREATE TABLE collections (id TEXT PRIMARY KEY, name TEXT);
                CREATE TABLE segments (id TEXT PRIMARY KEY, collection TEXT, scope TEXT);
                CREATE TABLE embeddings (id INTEGER PRIMARY KEY, segment_id TEXT, embedding_id TEXT);
                INSERT INTO collections VALUES ('c1', 'zotero_paragraphs_v3');
                INSERT INTO segments VALUES ('s1', 'c1', 'METADATA');
                INSERT INTO embeddings VALUES (1, 's1', 'chunk-1');
                INSERT INTO embeddings VALUES (2, 's1', 'chunk-2');
                """
            )
            conn.commit()
            conn.close()

            with patch.object(rag_mcp_server, "CHROMA_DIR", directory):
                count = rag_mcp_server._collection_row_count("zotero_paragraphs_v3")

        self.assertEqual(count, 2)


if __name__ == "__main__":
    unittest.main()
