from __future__ import annotations

import json
import sqlite3
from contextlib import nullcontext
from pathlib import Path

import pytest

from scripts import backup_v3_generation


def _database(path: Path, schema: str, rows: list[tuple]) -> None:
    connection = sqlite3.connect(path)
    try:
        connection.execute(schema)
        if rows:
            table = "chunks_fts" if "chunks_fts" in schema else "items"
            connection.executemany(f"INSERT INTO {table} VALUES (?)", rows)
        connection.commit()
    finally:
        connection.close()


def _sources(root: Path) -> backup_v3_generation.GenerationSources:
    chroma = root / "active-chroma"
    chroma.mkdir()
    (chroma / "segment.bin").write_bytes(b"vector-data")
    connection = sqlite3.connect(chroma / "chroma.sqlite3")
    connection.executescript("""
        CREATE TABLE collections(id TEXT, name TEXT);
        CREATE TABLE segments(id TEXT, scope TEXT, collection TEXT);
        CREATE TABLE embeddings(id TEXT, segment_id TEXT);
        INSERT INTO collections VALUES ('C', 'zotero_paragraphs_v3'), ('OTHER', 'other');
        INSERT INTO segments VALUES ('S', 'METADATA', 'C'), ('O', 'METADATA', 'OTHER');
        INSERT INTO embeddings VALUES ('one', 'S'), ('two', 'S'), ('unrelated', 'O');
    """)
    connection.close()

    (chroma / "embedder_config_v3.json").write_text(
        json.dumps({
            "embedding_dim": 1024,
            "normalize_embeddings": True,
            "model_state_fingerprint": "sha256:test",
            "embedder": {"emb_model": "/local/models/bge-m3", "embedding_dim": 1024},
        }),
        encoding="utf-8",
    )
    manifest = root / "manifest_v3.json"
    manifest.write_text(json.dumps({"files": {"A": {}, "B": {}}}), encoding="utf-8")
    lexical = root / "lexical_v3.sqlite3"
    relations = root / "relations.db"
    _database(lexical, "CREATE TABLE chunks_fts(chunk_id TEXT)", [("one",), ("two",)])
    _database(relations, "CREATE TABLE items(item_key TEXT)", [("ITEM",)])
    return backup_v3_generation.GenerationSources(
        chroma=chroma,
        manifest=manifest,
        lexical=lexical,
        relations=relations,
        pipeline_config=chroma / "embedder_config_v3.json",
        lock=root / "indexing.lock",
    )


def test_backup_copies_and_verifies_the_complete_generation(tmp_path, monkeypatch):
    sources = _sources(tmp_path)
    destination = tmp_path / "snapshot"
    monkeypatch.setattr(backup_v3_generation, "_chroma_count", lambda _path: 2)
    monkeypatch.setattr(backup_v3_generation.indexing_lock, "held", lambda _path: nullcontext())
    monkeypatch.setattr(
        backup_v3_generation.subprocess,
        "run",
        lambda *args, **kwargs: type("Result", (), {"stdout": "commit\n"})(),
    )

    report = backup_v3_generation.backup_generation(
        destination, root=tmp_path, sources=sources,
    )

    assert report["verified"] is True
    assert report["chroma_count"] == 2
    assert report["lexical_count"] == 2
    assert report["manifest_attachments"] == 2
    assert report["source_commit"] == "commit"
    assert report["pipeline"]["dimension"] == 1024
    assert not (destination / "INCOMPLETE").exists()
    assert (destination / "backup-report.json").exists()
    assert (destination / "chroma" / "segment.bin").read_bytes() == b"vector-data"


def test_backup_refuses_to_overwrite_an_existing_destination(tmp_path):
    destination = tmp_path / "snapshot"
    destination.mkdir()

    with pytest.raises(FileExistsError, match="already exists"):
        backup_v3_generation.backup_generation(
            destination, root=tmp_path, sources=_sources(tmp_path),
        )


def test_backup_refuses_a_destination_inside_source_chroma(tmp_path):
    sources = _sources(tmp_path)
    destination = sources.chroma / "backups" / "snapshot"

    with pytest.raises(ValueError, match="outside the source Chroma directory"):
        backup_v3_generation.backup_generation(
            destination, root=tmp_path, sources=sources,
        )

    assert not destination.exists()
    assert (sources.chroma / "segment.bin").read_bytes() == b"vector-data"


def test_verification_refuses_an_incomplete_snapshot(tmp_path):
    snapshot = tmp_path / "snapshot"
    snapshot.mkdir()
    (snapshot / "INCOMPLETE").write_text("not done\n", encoding="utf-8")

    with pytest.raises(RuntimeError, match="marked incomplete"):
        backup_v3_generation.verify_backup(snapshot)


def test_backup_preserves_raw_files_before_native_verification_fails(tmp_path, monkeypatch):
    sources = _sources(tmp_path)
    destination = tmp_path / "snapshot"
    opened = []
    def native_count(path):
        opened.append(path)
        assert path != sources.chroma
        assert (destination / "RAW_SNAPSHOT_COMPLETE").exists()
        assert (destination / "lexical_v3.sqlite3").exists()
        assert (destination / "relations.db").exists()
        raise RuntimeError("native verification failed")
    monkeypatch.setattr(backup_v3_generation, "_chroma_count", native_count)
    monkeypatch.setattr(backup_v3_generation.indexing_lock, "held", lambda path: nullcontext())
    with pytest.raises(RuntimeError, match="native verification failed"):
        backup_v3_generation.backup_generation(destination, root=tmp_path, sources=sources)
    assert opened == [destination / "chroma"]
    assert (destination / "INCOMPLETE").exists()
    assert (destination / "chroma" / "segment.bin").read_bytes() == b"vector-data"
