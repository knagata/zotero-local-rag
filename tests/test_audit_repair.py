import json
import sqlite3
from unittest.mock import patch

import pytest

from src import audit_repair


def row(key="A", item="ITEM"):
    return {"id": key, "text": "original body", "metadata": {"itemKey": item}}


def test_only_known_derived_failures_are_repaired():
    report = {"items": [
        {"item_key": "GOOD", "failures": ["stale_structure_fingerprint"]},
        {"item_key": "BAD", "failures": ["stale_structure_fingerprint", "no_retrievable_chunks"]},
        {"item_key": "UNKNOWN", "failures": ["future_reason"]},
    ]}
    assert audit_repair.repair_plan(report, [], []) == {"items": ["GOOD"], "lexical_rows": []}


def test_lexical_repair_only_adds_missing_and_never_deletes_extra_or_duplicates():
    report = {"gate": {"global_failures": ["chroma_lexical_id_mismatch"]}}
    rows = [row("A"), row("B")]
    assert audit_repair.repair_plan(report, rows, ["A"])["lexical_rows"] == [rows[1]]
    assert audit_repair.repair_plan(report, rows, ["A", "X"])["lexical_rows"] == []
    assert audit_repair.repair_plan(report, rows, ["A", "A"])["lexical_rows"] == []
    assert audit_repair.repair_plan({}, rows, [])["lexical_rows"] == []


def test_dangling_references_select_only_affected_items():
    plan = audit_repair.repair_plan({"dangling_node_chunks": ["B"]}, [row("A", "ONE"), row("B", "TWO")], [])
    assert plan["items"] == ["TWO"]


@pytest.mark.parametrize("kind", ["items", "lexical"])
def test_repair_bounds_fail_before_any_write(kind, monkeypatch):
    monkeypatch.setattr(audit_repair, "MAX_ITEMS", 1)
    monkeypatch.setattr(audit_repair, "MAX_LEXICAL_ROWS", 1)
    report = ({"items": [{"item_key": key, "failures": ["missing_v3_structure"]} for key in ["A", "B"]]}
              if kind == "items" else {"gate": {"global_failures": ["chroma_lexical_id_mismatch"]}})
    with pytest.raises(RuntimeError, match="上限"):
        audit_repair.repair_plan(report, [row("A"), row("B")], [])


def test_backup_includes_uncheckpointed_sqlite_and_exact_original_rows(tmp_path):
    source = tmp_path / "original.sqlite3"
    connection = sqlite3.connect(source)
    connection.execute("PRAGMA journal_mode=WAL")
    connection.execute("CREATE TABLE example(value)")
    connection.execute("INSERT INTO example VALUES ('original')")
    connection.commit()
    saved = audit_repair._backup(tmp_path, [row()], source, source)
    connection.close()
    assert not (saved / "INCOMPLETE").exists()
    assert json.loads((saved / "original-chunks.json").read_text()) == [row()]
    with sqlite3.connect(saved / "relations.sqlite3") as copy:
        assert copy.execute("SELECT value FROM example").fetchone() == ("original",)


def test_failed_preflight_never_backs_up_or_writes(tmp_path):
    with patch("src.chunk_store.get_collection_chunks", return_value=[row()]), \
         patch("src.lexical_index.list_chunk_ids", return_value=["A"]), \
         patch("scripts.rebuild_document_structure.rebuild_item", return_value={"source_refresh_error": "mapping changed"}) as rebuild, \
         patch.object(audit_repair, "_backup") as backup:
        with pytest.raises(RuntimeError, match="修復前"):
            audit_repair.repair_derived_data({"items": [{"item_key": "ITEM", "failures": ["stale_structure_fingerprint"]}]}, tmp_path)
    assert rebuild.call_args.kwargs["dry_run"] is True
    backup.assert_not_called()


def test_repair_uses_backup_before_write_and_does_not_embed(tmp_path):
    events = []
    def rebuild(key, **kwargs):
        events.append("preview" if kwargs["dry_run"] else "write")
        return {"diagnostics": {"valid": True}, "status": "exact"}
    def backup(*args):
        events.append("backup")
        return tmp_path / "backup"
    with patch("src.chunk_store.get_collection_chunks", return_value=[row()]), \
         patch("src.lexical_index.list_chunk_ids", return_value=["A"]), \
         patch("scripts.rebuild_document_structure.rebuild_item", side_effect=rebuild), \
         patch.object(audit_repair, "_backup", side_effect=backup):
        result = audit_repair.repair_derived_data({"items": [{"item_key": "ITEM", "failures": ["stale_structure_version"]}]}, tmp_path)
    assert events == ["preview", "backup", "write"]
    assert result["repaired"] is True


def test_lexical_repair_copies_original_text_and_metadata(tmp_path):
    with patch("src.chunk_store.get_collection_chunks", return_value=[row()]), \
         patch("src.lexical_index.list_chunk_ids", return_value=[]), \
         patch("src.lexical_index.upsert_chunks") as upsert, \
         patch.object(audit_repair, "_backup", return_value=tmp_path / "backup"):
        result = audit_repair.repair_derived_data({"gate": {"global_failures": ["chroma_lexical_id_mismatch"]}}, tmp_path)
    assert result["lexical_rows"] == 1
    assert upsert.call_args.args == (["A"], ["original body"], [{"itemKey": "ITEM"}])


@pytest.mark.parametrize("report", [{}, {"gate": {"global_failures": ["chroma_lexical_id_mismatch"]}}])
def test_no_safe_repair_changes_nothing(tmp_path, report):
    with patch("src.chunk_store.get_collection_chunks", return_value=[row()]), \
         patch("src.lexical_index.list_chunk_ids", return_value=["EXTRA"]), \
         patch.object(audit_repair, "_backup") as backup:
        assert audit_repair.repair_derived_data(report, tmp_path)["repaired"] is False
    backup.assert_not_called()


def test_lexical_content_drift_is_repaired_from_complete_live_comparison(tmp_path):
    report = {"gate": {"global_failures": ["lexical_content_mismatch"]},
              "lexical_content_mismatches": ["OUTDATED_SAMPLE"]}
    with patch("src.chunk_store.get_collection_chunks", return_value=[row()]), \
         patch("src.lexical_index.list_chunk_ids", return_value=["A"]), \
         patch("src.lexical_index.audit_content", return_value=("fp", ["A"])), \
         patch("src.lexical_index.upsert_chunks") as upsert, \
         patch.object(audit_repair, "_backup", return_value=tmp_path / "backup"):
        assert audit_repair.repair_derived_data(report, tmp_path)["repaired"] is True
    assert upsert.call_args.args[1] == ["original body"]


def test_changed_source_during_repair_remains_failure_with_backup(tmp_path):
    report = {"items": [{"item_key": "ITEM", "failures": ["stale_structure_fingerprint"]}]}
    with patch("src.chunk_store.get_collection_chunks", return_value=[row()]), \
         patch("src.lexical_index.list_chunk_ids", return_value=["A"]), \
         patch("scripts.rebuild_document_structure.rebuild_item", side_effect=[
             {"diagnostics": {"valid": True}}, {"source_refresh_error": "changed"},
         ]), patch.object(audit_repair, "_backup", return_value=tmp_path / "saved"):
        with pytest.raises(RuntimeError, match="バックアップ"):
            audit_repair.repair_derived_data(report, tmp_path)


def test_backup_refuses_overlap(tmp_path):
    with patch("src.audit_repair.uuid.uuid4", return_value=type("ID", (), {"hex": "fixed"})()):
        source = tmp_path / "data/backups/audit-repair-fixed/active.sqlite3"
        with pytest.raises(RuntimeError, match="重複"):
            audit_repair._backup(tmp_path, [], source, source)


def test_backup_validation_failure_keeps_incomplete_marker(tmp_path):
    from unittest.mock import MagicMock
    original = MagicMock()
    copy = MagicMock()
    copy.execute.return_value.fetchone.return_value = ("failed",)
    with patch("src.audit_repair.sqlite3.connect", side_effect=[original, copy]):
        with pytest.raises(RuntimeError, match="SQLite"):
            audit_repair._backup(tmp_path, [], tmp_path / "lexical", tmp_path / "relations")
    assert list((tmp_path / "data/backups").glob("*/INCOMPLETE"))
    assert original.close.called and copy.close.called


def test_chunk_backup_validation_failure_keeps_incomplete_marker(tmp_path):
    source = tmp_path / "db.sqlite3"
    with sqlite3.connect(source) as connection:
        connection.execute("CREATE TABLE example(value)")
    with patch("pathlib.Path.read_text", return_value="wrong"):
        with pytest.raises(RuntimeError, match="本文"):
            audit_repair._backup(tmp_path, [row()], source, source)
    assert list((tmp_path / "data/backups").glob("*/INCOMPLETE"))
