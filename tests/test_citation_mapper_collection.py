"""Regression tests for citation mapper's active Chroma collection handling."""
from __future__ import annotations

import json
import os
import sqlite3
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import citation_mapper  # noqa: E402


def _create_db(directory: Path) -> sqlite3.Connection:
    connection = sqlite3.connect(directory / "chroma.sqlite3")
    connection.executescript("""
        CREATE TABLE collections (id TEXT PRIMARY KEY, name TEXT);
        CREATE TABLE segments (id TEXT PRIMARY KEY, collection TEXT, scope TEXT);
        CREATE TABLE embeddings (id INTEGER PRIMARY KEY, segment_id TEXT, embedding_id TEXT);
        CREATE TABLE embedding_metadata (id INTEGER, key TEXT, string_value TEXT);
    """)
    return connection


def _add_collection(
    connection: sqlite3.Connection, *, collection_id: str, name: str,
    segment_id: str, vector_id: str, chunks: list[tuple[str, str, str]],
) -> None:
    connection.execute("INSERT INTO collections VALUES (?, ?)", (collection_id, name))
    connection.execute("INSERT INTO segments VALUES (?, ?, 'METADATA')", (segment_id, collection_id))
    connection.execute("INSERT INTO segments VALUES (?, ?, 'VECTOR')", (vector_id, collection_id))
    for number, (embedding_id, item_key, document) in enumerate(chunks, start=1):
        row_id = int(f"{len(collection_id)}{number}{len(embedding_id)}")
        # Test fixture IDs need only be unique across a collection fixture.
        while connection.execute("SELECT 1 FROM embeddings WHERE id = ?", (row_id,)).fetchone():
            row_id += 100
        connection.execute("INSERT INTO embeddings VALUES (?, ?, ?)", (row_id, segment_id, embedding_id))
        connection.execute("INSERT INTO embedding_metadata VALUES (?, 'itemKey', ?)", (row_id, item_key))
        connection.execute("INSERT INTO embedding_metadata VALUES (?, 'chroma:document', ?)", (row_id, document))


class CitationMapperCollectionTests(unittest.TestCase):
    def setUp(self):
        self.tempdir = tempfile.TemporaryDirectory()
        self.chroma_dir = Path(self.tempdir.name)
        self.original_dir = citation_mapper.CHROMA_DIR
        self.original_segment = citation_mapper._SEGMENT_META
        self.original_item_cache = (
            citation_mapper._ITEM_CHUNKS_CACHE_KEY,
            citation_mapper._ITEM_CHUNK_IDS,
            citation_mapper._ITEM_CHUNK_MATRIX,
        )
        citation_mapper.CHROMA_DIR = self.chroma_dir
        citation_mapper._SEGMENT_META = None
        citation_mapper._ITEM_CHUNKS_CACHE_KEY = None
        citation_mapper._ITEM_CHUNK_IDS = []
        citation_mapper._ITEM_CHUNK_MATRIX = None

    def tearDown(self):
        citation_mapper.CHROMA_DIR = self.original_dir
        citation_mapper._SEGMENT_META = self.original_segment
        (citation_mapper._ITEM_CHUNKS_CACHE_KEY,
         citation_mapper._ITEM_CHUNK_IDS,
         citation_mapper._ITEM_CHUNK_MATRIX) = self.original_item_cache
        self.tempdir.cleanup()

    def test_explicit_collection_wins_over_larger_legacy_collection(self):
        connection = _create_db(self.chroma_dir)
        _add_collection(
            connection, collection_id="legacy-id", name="legacy", segment_id="legacy-meta",
            vector_id="legacy-vector", chunks=[(f"legacy-{i}", "ITEM", "old") for i in range(5)],
        )
        _add_collection(
            connection, collection_id="v3-id", name="zotero_paragraphs_v3", segment_id="v3-meta",
            vector_id="v3-vector", chunks=[("v3-1", "ITEM", "current")],
        )
        connection.commit()
        connection.close()

        with patch.dict(os.environ, {"CHROMA_COLLECTION": "zotero_paragraphs_v3"}, clear=False):
            segment = citation_mapper._get_segment_meta()

        self.assertEqual(segment["collection_name"], "zotero_paragraphs_v3")
        self.assertEqual(segment["metadata_segment_id"], "v3-meta")

    def test_legacy_active_config_is_ignored_when_collection_is_not_explicit(self):
        connection = _create_db(self.chroma_dir)
        _add_collection(
            connection, collection_id="legacy-id", name="legacy", segment_id="legacy-meta",
            vector_id="legacy-vector", chunks=[("legacy-1", "ITEM", "old")],
        )
        _add_collection(
            connection, collection_id="active-id", name="zotero_paragraphs_v3", segment_id="active-meta",
            vector_id="active-vector", chunks=[("active-1", "ITEM", "new")],
        )
        connection.commit()
        connection.close()
        (self.chroma_dir / "embedder_config_v3.json").write_text(
            json.dumps({"collection": "legacy"}), encoding="utf-8",
        )

        with patch.dict(os.environ, {"INGEST_STRUCTURED_V3_ENABLE": "1"}, clear=True):
            segment = citation_mapper._get_segment_meta()

        self.assertEqual(segment["collection_name"], "zotero_paragraphs_v3")

    def test_missing_or_ambiguous_configured_collection_fails_closed(self):
        connection = _create_db(self.chroma_dir)
        _add_collection(
            connection, collection_id="other-id", name="other", segment_id="other-meta",
            vector_id="other-vector", chunks=[("other-1", "ITEM", "other")],
        )
        connection.commit()
        connection.close()

        with patch.dict(os.environ, {"CHROMA_COLLECTION": "missing"}, clear=False):
            with self.assertRaisesRegex(RuntimeError, "Unsupported Chroma collection"):
                citation_mapper._get_segment_meta()

        connection = sqlite3.connect(self.chroma_dir / "chroma.sqlite3")
        connection.execute("INSERT INTO collections VALUES ('ambiguous-id', 'zotero_paragraphs_v3')")
        connection.execute("INSERT INTO segments VALUES ('ambiguous-meta-1', 'ambiguous-id', 'METADATA')")
        connection.execute("INSERT INTO segments VALUES ('ambiguous-meta-2', 'ambiguous-id', 'METADATA')")
        connection.execute("INSERT INTO segments VALUES ('ambiguous-vector', 'ambiguous-id', 'VECTOR')")
        connection.execute("INSERT INTO embeddings VALUES (999, 'ambiguous-meta-1', 'x')")
        connection.commit()
        connection.close()
        with patch.dict(os.environ, {"CHROMA_COLLECTION": "zotero_paragraphs_v3"}, clear=False):
            with self.assertRaisesRegex(RuntimeError, "ambiguous"):
                citation_mapper._get_segment_meta()

    def test_rebuild_generation_change_invalidates_item_chunk_cache(self):
        connection = _create_db(self.chroma_dir)
        _add_collection(
            connection, collection_id="before-id", name="zotero_paragraphs_v3", segment_id="before-meta",
            vector_id="before-vector", chunks=[("old-chunk", "ITEM", "old text")],
        )
        connection.commit()
        connection.close()

        def fake_embed(texts):
            return [[float(index + 1)] for index, _ in enumerate(texts)]

        with patch.dict(os.environ, {"CHROMA_COLLECTION": "zotero_paragraphs_v3"}, clear=False), \
             patch.object(citation_mapper, "_get_emb_fn", return_value=fake_embed):
            first_ids, _matrix = citation_mapper._load_chunks_for_item("ITEM")
            self.assertEqual(first_ids, ["old-chunk"])

            # Same collection name and row count, but the rebuild allocated a
            # new collection UUID.  Returning old-chunk here would map a new
            # document's citation against the deleted database generation.
            connection = sqlite3.connect(self.chroma_dir / "chroma.sqlite3")
            connection.executescript("""
                DELETE FROM embedding_metadata;
                DELETE FROM embeddings;
                DELETE FROM segments;
                DELETE FROM collections;
            """)
            _add_collection(
                connection, collection_id="after-id", name="zotero_paragraphs_v3", segment_id="after-meta",
                vector_id="after-vector", chunks=[("new-chunk", "ITEM", "new text")],
            )
            connection.commit()
            connection.close()

            second_ids, _matrix = citation_mapper._load_chunks_for_item("ITEM")

        self.assertEqual(second_ids, ["new-chunk"])


if __name__ == "__main__":
    unittest.main()
