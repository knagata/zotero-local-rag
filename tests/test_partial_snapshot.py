from __future__ import annotations

import json
import sqlite3
import sys
from types import SimpleNamespace

import pytest

from src import partial_snapshot


def _fake_chromadb(monkeypatch, collection):
    client = SimpleNamespace(get_collection=lambda *_args, **_kwargs: collection,
                             close=lambda: None)
    monkeypatch.setitem(sys.modules, 'chromadb', SimpleNamespace(
        PersistentClient=lambda **_kwargs: client))


def test_chroma_metadata_row_mismatch_fails_closed(tmp_path, monkeypatch):
    class Collection:
        def count(self):
            return 1

        def get(self, **_kwargs):
            return {'ids': ['one'], 'metadatas': []}

    _fake_chromadb(monkeypatch, Collection())
    with pytest.raises(ValueError, match='metadata rows'):
        partial_snapshot._chroma_rows(tmp_path, 'v3', {'GOOD'})


def test_chroma_count_change_fails_closed(tmp_path, monkeypatch):
    class Collection:
        def __init__(self):
            self.count_calls = 0

        def count(self):
            self.count_calls += 1
            return 1 if self.count_calls == 1 else 0

        def get(self, **_kwargs):
            return {'ids': ['one'], 'metadatas': [{'itemKey': 'GOOD'}]}

    _fake_chromadb(monkeypatch, Collection())
    with pytest.raises(ValueError, match='row count changed'):
        partial_snapshot._chroma_rows(tmp_path, 'v3', {'GOOD'})


def test_partial_copy_requires_unique_allowlist():
    with pytest.raises(ValueError, match='nonempty unique'):
        partial_snapshot.filter_query_copy(None, 'v3', None, None, None, ['GOOD', 'GOOD'])


def test_partial_copy_rejects_missing_manifest_inventory(tmp_path, monkeypatch):
    chroma = tmp_path / 'chroma'
    lexical = tmp_path / 'lexical.db'
    relations = tmp_path / 'relations.db'
    manifest = tmp_path / 'manifest.json'
    lexical_db = sqlite3.connect(lexical)
    lexical_db.execute('CREATE TABLE chunks_fts (chunk_id TEXT, item_key TEXT)')
    lexical_db.executemany('INSERT INTO chunks_fts VALUES (?,?)',
                           [('chunk1', 'GOOD'), ('chunk2', 'BAD')])
    lexical_db.commit()
    lexical_db.close()
    sqlite3.connect(relations).close()
    manifest.write_text(json.dumps({'files': []}))
    monkeypatch.setattr(partial_snapshot, '_chroma_rows',
                        lambda *_args: ({'chunk1'}, {'ATTGOOD'}))
    with pytest.raises(TypeError, match='file inventory'):
        partial_snapshot.filter_query_copy(chroma, 'v3', lexical, manifest, relations, ['GOOD'])
    with sqlite3.connect(lexical) as db:
        assert db.execute('SELECT chunk_id FROM chunks_fts').fetchall() == [('chunk1',)]


def test_partial_copy_rejects_chroma_lexical_id_difference(tmp_path, monkeypatch):
    lexical = tmp_path / 'lexical.db'
    db = sqlite3.connect(lexical)
    db.execute('CREATE TABLE chunks_fts (chunk_id TEXT, item_key TEXT)')
    db.execute('INSERT INTO chunks_fts VALUES (?,?)', ('lexical-id', 'GOOD'))
    db.commit()
    db.close()
    monkeypatch.setattr(partial_snapshot, '_chroma_rows',
                        lambda *_args: ({'chroma-id'}, {'ATTGOOD'}))
    with pytest.raises(ValueError, match='IDs differ'):
        partial_snapshot.filter_query_copy(None, 'v3', lexical, None, None, ['GOOD'])


def test_summary_collections_receive_the_same_allowlist(tmp_path, monkeypatch):
    calls = []
    client = SimpleNamespace(
        list_collections=lambda: [SimpleNamespace(name="body"), SimpleNamespace(name="summaries")],
        close=lambda: calls.append("closed"),
    )
    monkeypatch.setitem(sys.modules, "chromadb", SimpleNamespace(PersistentClient=lambda **kwargs: client))
    monkeypatch.setattr(partial_snapshot, "_chroma_rows", lambda *args: calls.append(args))
    partial_snapshot._filter_other_collections(tmp_path, "body", {"GOOD"})
    assert calls == ["closed", (tmp_path, "summaries", {"GOOD"})]
