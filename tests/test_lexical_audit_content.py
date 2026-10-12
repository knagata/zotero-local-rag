import sqlite3
from unittest.mock import MagicMock, patch

from src.lexical_index import audit_content, upsert_chunks
from src.database_gate import database_state_fingerprint
from src.database_gate import live_database_state_fingerprint


def test_same_ids_with_different_body_or_metadata_fail_content_check(tmp_path):
    path = tmp_path / "fts.sqlite3"
    metadata = {"itemKey": "ITEM", "title": "Title"}
    rows = [{"id": "CHUNK", "text": "original", "metadata": metadata}]
    upsert_chunks(["CHUNK"], ["original"], [metadata], path=path)
    fingerprint, mismatches = audit_content(rows, path=path)
    assert not mismatches
    with sqlite3.connect(path) as connection:
        connection.execute("UPDATE chunks_fts SET body='wrong', title='changed'")
    changed, mismatches = audit_content(rows, path=path)
    assert mismatches == ["CHUNK"] and changed != fingerprint


def test_duplicate_fts_rows_fail_even_when_body_is_identical(tmp_path):
    path = tmp_path / "fts.sqlite3"
    row = {"id": "CHUNK", "text": "body", "metadata": {}}
    upsert_chunks(["CHUNK"], ["body"], [{}], path=path)
    upsert_chunks(["CHUNK"], ["body"], [{}], path=path, replace=False)
    assert audit_content([row], path=path)[1] == ["CHUNK"]


def test_missing_fts_is_not_created_by_audit(tmp_path):
    path = tmp_path / "missing.sqlite3"
    assert audit_content([], path=path) == ("missing", [])
    assert not path.exists()


def test_generation_attestation_includes_fts_contents(tmp_path):
    manifest = tmp_path / "manifest.json"
    pipeline = tmp_path / "pipeline.json"
    manifest.write_text("{}")
    pipeline.write_text("{}")
    kwargs = dict(manifest_path=manifest, pipeline_config_path=pipeline,
                  chroma_rows=[], lexical_ids=["SAME"], structure_rows=[])
    assert database_state_fingerprint(**kwargs, lexical_content_fingerprint="before") != database_state_fingerprint(
        **kwargs, lexical_content_fingerprint="after")


def test_live_attestation_reads_fts_content_and_closes_connection(tmp_path):
    connection = MagicMock()
    connection.execute.return_value.fetchall.return_value = []
    manifest, pipeline, lexical = (tmp_path / name for name in ("manifest", "pipeline", "lexical"))
    with patch("src.db_relations.get_db_connection", return_value=connection), \
         patch("src.chunk_store.get_collection_chunks", return_value=[]), \
         patch("src.lexical_index.list_chunk_ids", return_value=[]), \
         patch("src.lexical_index.audit_content", return_value=("fts-proof", [])) as content:
        actual = live_database_state_fingerprint(collection_name="zotero_paragraphs_v3", manifest_path=manifest,
                                                lexical_db_path=lexical, pipeline_config_path=pipeline)
    assert actual == database_state_fingerprint(manifest_path=manifest, pipeline_config_path=pipeline,
                                               chroma_rows=[], lexical_ids=[], structure_rows=[],
                                               lexical_content_fingerprint="fts-proof")
    connection.close.assert_called_once()
    content.assert_called_once_with([], path=lexical)
