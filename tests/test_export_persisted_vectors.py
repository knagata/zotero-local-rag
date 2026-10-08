from __future__ import annotations

import hashlib
import pickle
import sqlite3
import struct
from pathlib import Path

import pytest

from scripts import export_persisted_vectors as module


def snapshot(tmp_path):
    source = tmp_path / "snapshot"
    (tmp_path / "RAW_SNAPSHOT_COMPLETE").write_text("fixture snapshot complete\n")
    segment = source / "segment"
    segment.mkdir(parents=True)
    header = module.HEADER.pack(1, 0, 16, 2, 32, 24, 12, 0, 0, 1, 2, 1, 1.0, 100)
    (segment / "header.bin").write_bytes(header)
    raw = bytearray(64)
    for slot, label, vector in [(0, 9, [0, 1, 0]), (1, 7, [1, 0, 0])]:
        struct.pack_into("<3f", raw, slot * 32 + 12, *vector)
        struct.pack_into("<Q", raw, slot * 32 + 24, label)
    (segment / "data_level0.bin").write_bytes(raw)
    (segment / "index_metadata.pickle").write_bytes(pickle.dumps({
        "dimensionality": None, "id_to_label": {"A": 7, "B": 9},
        "label_to_id": {7: "A", 9: "B"},
    }))
    connection = sqlite3.connect(source / "chroma.sqlite3")
    connection.executescript("""
        CREATE TABLE collections (id TEXT, name TEXT, dimension INTEGER);
        CREATE TABLE segments (id TEXT, collection TEXT, scope TEXT);
        CREATE TABLE embeddings (embedding_id TEXT, segment_id TEXT);
        CREATE TABLE max_seq_id (segment_id TEXT, seq_id INTEGER);
        CREATE TABLE embeddings_queue (seq_id INTEGER, operation INTEGER,
            topic TEXT, id TEXT, vector BLOB, encoding TEXT);
        INSERT INTO collections VALUES ('c', 'test', 3);
        INSERT INTO segments VALUES ('segment', 'c', 'VECTOR'), ('meta', 'c', 'METADATA');
        INSERT INTO embeddings VALUES ('A', 'meta'), ('C', 'meta');
        INSERT INTO max_seq_id VALUES ('segment', 10);
    """)
    topic = "persistent://default/default/c"
    connection.executemany("INSERT INTO embeddings_queue VALUES (?, ?, ?, ?, ?, ?)", [
        (1, 2, topic, "A", struct.pack("<3f", 0, 1, 0), "FLOAT32"),
        (11, 3, topic, "B", None, None),
        (12, 2, topic, "C", struct.pack("<3f", 0, 0, 1), "FLOAT32"),
    ])
    connection.commit()
    connection.close()
    return source


def test_export_matches_canonical_ids_after_wal_without_native_client(tmp_path, monkeypatch):
    import chromadb
    monkeypatch.setattr(chromadb, "PersistentClient", lambda **kw: pytest.fail("native open"))
    source = snapshot(tmp_path)
    before = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in source.rglob("*") if p.is_file()}
    output = tmp_path / "vectors.sqlite3"
    report = module.export_snapshot(source, output)
    connection = sqlite3.connect(output)
    rows = connection.execute("SELECT id, vector FROM vectors ORDER BY id").fetchall()
    connection.close()
    assert rows == [("A", struct.pack("<3f", 1, 0, 0)), ("C", struct.pack("<3f", 0, 0, 1))]
    assert report["complete"] and report["collections"]["test"]["wal_operations"] == 2
    assert before == {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in source.rglob("*") if p.is_file()}


@pytest.mark.parametrize("kind", ["truncated", "duplicate", "missing", "bijection", "nan", "unsafe_pickle"])
def test_export_rejects_unprovable_physical_vectors(tmp_path, kind):
    source = snapshot(tmp_path)
    segment = source / "segment"
    binary = segment / "data_level0.bin"
    raw = bytearray(binary.read_bytes())
    if kind == "truncated":
        binary.write_bytes(raw[:-1])
    elif kind == "duplicate":
        struct.pack_into("<Q", raw, 56, 9)
        binary.write_bytes(raw)
    elif kind == "nan":
        struct.pack_into("<f", raw, 44, float("nan"))
        binary.write_bytes(raw)
    else:
        path = segment / "index_metadata.pickle"
        metadata = pickle.loads(path.read_bytes())
        if kind == "unsafe_pickle":
            metadata = Path("never-import-classes")
        elif kind == "missing":
            metadata["id_to_label"]["A"] = 42
            metadata["label_to_id"] = {42: "A", 9: "B"}
        else:
            metadata["label_to_id"][7] = "wrong"
        path.write_bytes(pickle.dumps(metadata))
    with pytest.raises(ValueError):
        module.export_snapshot(source, tmp_path / "export.sqlite3")
    assert not (tmp_path / "export.json").exists()


@pytest.mark.parametrize("sql", [
    "DELETE FROM embeddings_queue WHERE id='C'",
    "UPDATE embeddings_queue SET encoding='UNKNOWN' WHERE id='C'",
    "DELETE FROM max_seq_id",
    "UPDATE embeddings_queue SET topic='persistent://other/default/c'",
    "INSERT INTO embeddings VALUES ('A', 'meta')",
])
def test_export_requires_complete_proven_wal_replay(tmp_path, sql):
    source = snapshot(tmp_path)
    connection = sqlite3.connect(source / "chroma.sqlite3")
    connection.execute(sql)
    connection.commit()
    connection.close()
    with pytest.raises(ValueError):
        module.export_snapshot(source, tmp_path / "export.sqlite3")


def test_export_rejects_overwrite_and_output_inside_snapshot(tmp_path):
    source = snapshot(tmp_path)
    for output in [source / "new.sqlite3", source / "chroma.sqlite3"]:
        with pytest.raises(ValueError):
            module.export_snapshot(source, output)


def _copy_inputs(tmp_path):
    import json
    source = snapshot(tmp_path)
    connection = sqlite3.connect(source / 'chroma.sqlite3')
    connection.executescript('''
        ALTER TABLE embeddings ADD COLUMN id INTEGER;
        UPDATE embeddings SET id=rowid;
        ALTER TABLE collections ADD COLUMN schema_str TEXT;
        CREATE TABLE embedding_metadata (id INTEGER, key TEXT, string_value TEXT,
            int_value INTEGER, float_value REAL, bool_value INTEGER);
        CREATE TABLE collection_metadata (collection_id TEXT, key TEXT,
            str_value TEXT, int_value INTEGER, float_value REAL, bool_value INTEGER);
        INSERT INTO embedding_metadata VALUES
            (1, 'chroma:document', 'text A', NULL, NULL, NULL),
            (1, 'itemKey', 'ITEM-A', NULL, NULL, NULL),
            (2, 'chroma:document', 'text C', NULL, NULL, NULL),
            (2, 'itemKey', 'ITEM-C', NULL, NULL, NULL);
    ''')
    schema = {'keys': {'#embedding': {'float_list': {'vector_index': {'config': {
        'space': 'cosine', 'hnsw': {'max_neighbors': 16, 'ef_construction': 100,
                                 'ef_search': 100, 'num_threads': 8, 'sync_threshold': 100,
                                 'batch_size': 100, 'resize_factor': 1.2},
    }}}}}}
    connection.execute('UPDATE collections SET schema_str=?', (json.dumps(schema),))
    connection.commit()
    connection.close()
    vectors = tmp_path / 'vectors.sqlite3'
    module.export_snapshot(source, vectors)
    return source, vectors


def test_isolated_copy_uses_stored_vectors_and_reopens_without_embedding(tmp_path):
    import json
    import chromadb
    from scripts import rebuild_chroma_from_vectors as builder
    source, vectors = _copy_inputs(tmp_path)
    output = tmp_path / 'rebuilt'
    report = builder.build_copy(source, vectors, output, batch_size=1)
    assert report['complete']
    client = chromadb.PersistentClient(path=str(output))
    try:
        collection = client.get_collection('test', embedding_function=None)
        result = collection.get(ids=['A', 'C'], include=['documents', 'metadatas', 'embeddings'])
        assert result['documents'] == ['text A', 'text C']
        assert result['metadatas'] == [{'itemKey': 'ITEM-A'}, {'itemKey': 'ITEM-C'}]
        assert collection.query(query_embeddings=[[1, 0, 0]], n_results=1)['ids'] == [['A']]
        database = sqlite3.connect(output / 'chroma.sqlite3')
        schema = json.loads(database.execute('SELECT schema_str FROM collections').fetchone()[0])
        database.close()
        assert schema['keys']['#embedding']['float_list']['vector_index']['config']['hnsw']['num_threads'] == 1
    finally:
        client.close()


def test_isolated_copy_rejects_tampered_vector_proof(tmp_path):
    import json
    from scripts import rebuild_chroma_from_vectors as builder
    source, vectors = _copy_inputs(tmp_path)
    proof = json.loads(vectors.with_suffix('.json').read_text())
    proof['collections']['test']['id_vector_sha256'] = 'wrong'
    vectors.with_suffix('.json').write_text(json.dumps(proof))
    output = tmp_path / 'partial-copy'
    with pytest.raises(ValueError, match='hash/count'):
        builder.build_copy(source, vectors, output)
    assert not (output / 'stored-vector-build.json').exists()


def test_isolated_copy_requires_new_output_and_bounded_batches(tmp_path):
    from scripts import rebuild_chroma_from_vectors as builder
    source, vectors = _copy_inputs(tmp_path)
    for output, size in [(source / 'nested', 128), (source, 128), (tmp_path / 'bad', 0)]:
        with pytest.raises(ValueError):
            builder.build_copy(source, vectors, output, batch_size=size)


def test_isolated_recovery_verification_checks_all_data_and_native_reopen(tmp_path, monkeypatch):
    import chromadb
    from scripts import rebuild_chroma_from_vectors as builder, verify_vector_recovery as verifier
    source, vectors = _copy_inputs(tmp_path)
    candidate = tmp_path / 'verified-copy'
    builder.build_copy(source, vectors, candidate)
    native_client = chromadb.PersistentClient
    def candidate_only(*args, **kwargs):
        assert Path(kwargs['path']).resolve() != source.resolve()
        return native_client(*args, **kwargs)
    monkeypatch.setattr(chromadb, 'PersistentClient', candidate_only)
    report = verifier.verify_copy(source, vectors, candidate)
    assert report['verified'] and report['reopened']
    assert report['collections']['test']['count'] == 2
    assert report['collections']['test']['logical_signature']['metadata_rows'] == 4
    assert report['collections']['test']['max_abs_float32_difference'] == 0


def test_recovery_verification_rejects_changed_metadata_and_invalidates_old_proof(tmp_path):
    import json
    from scripts import rebuild_chroma_from_vectors as builder, verify_vector_recovery as verifier
    source, vectors = _copy_inputs(tmp_path)
    candidate = tmp_path / 'changed-copy'
    builder.build_copy(source, vectors, candidate)
    certificate = candidate.parent / f'{candidate.name}-verification.json'
    certificate.write_text('{"verified":true}')
    connection = sqlite3.connect(candidate / 'chroma.sqlite3')
    connection.execute("UPDATE embedding_metadata SET string_value='wrong' WHERE key='itemKey'")
    connection.commit()
    connection.close()
    with pytest.raises(ValueError, match='content differs'):
        verifier.verify_copy(source, vectors, candidate)
    assert json.loads(certificate.read_text())['verified'] is False


def test_recovery_verification_rejects_vector_changes_beyond_machine_rounding():
    from unittest.mock import MagicMock
    from scripts import verify_vector_recovery as verifier
    vector_db = sqlite3.connect(':memory:')
    vector_db.execute('CREATE TABLE vectors (collection TEXT, id TEXT, vector BLOB)')
    vector_db.execute('INSERT INTO vectors VALUES (?, ?, ?)', ('test', 'A', struct.pack('<3f', 1, 0, 0)))
    collection = MagicMock()
    collection.count.return_value = 1
    collection.get.return_value = {'ids': ['A'], 'embeddings': [[0.99, 0, 0]]}
    with pytest.raises(ValueError, match='rounding'):
        verifier.verify_collection(collection, vector_db, 'test', {'count': 1, 'dimension': 3})
    vector_db.close()
