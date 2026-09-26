from __future__ import annotations

import sqlite3
from pathlib import Path
from types import SimpleNamespace

from scripts.check_admin_update_status import (
    _citation_freshness,
    _index_freshness,
    _note_freshness,
    _summary_freshness,
)
from src import db_relations


def attachment(key: str, path: Path, parent: str = "ITEM") -> SimpleNamespace:
    return SimpleNamespace(attachmentKey=key, parentItemKey=parent, pdf_path=str(path))


def test_index_freshness_reports_new_changed_retired_and_excluded(tmp_path):
    current = tmp_path / "current.pdf"
    changed = tmp_path / "changed.pdf"
    new = tmp_path / "new.pdf"
    for path in (current, changed, new):
        path.write_bytes(b"pdf")
    current_stat = current.stat()
    changed_stat = changed.stat()
    manifest = {
        "pipeline_fingerprint": "pipeline",
        "files": {
            "CURRENT": {"mtime": current_stat.st_mtime, "size": current_stat.st_size, "pipeline_fingerprint": "pipeline"},
            "CHANGED": {"mtime": changed_stat.st_mtime - 1, "size": changed_stat.st_size, "pipeline_fingerprint": "pipeline"},
            "RETIRED": {},
            "EXCLUDED": {},
        },
    }
    result = _index_freshness(
        [attachment("CURRENT", current), attachment("CHANGED", changed), attachment("NEW", new)],
        manifest,
        excluded_keys={"EXCLUDED"},
    )
    assert result["pending"] == 4
    assert result["new_or_changed"] == 2
    assert result["retired"] == 1
    assert result["excluded_tracked"] == 1


def test_index_freshness_reports_an_unresolved_attachment_as_missing_source():
    row = SimpleNamespace(attachmentKey="MISSING", parentItemKey="ITEM", pdf_path="")
    result = _index_freshness(
        [row], {"pipeline_fingerprint": "pipeline", "files": {}}, excluded_keys=set(),
    )
    assert result["missing_source"] == 1
    assert result["sample_keys"] == ["MISSING"]


def test_citation_freshness_treats_only_terminal_statuses_as_current(tmp_path):
    database = tmp_path / "relations.db"
    with sqlite3.connect(database) as connection:
        connection.execute(
            "CREATE TABLE item_citation_status (item_key TEXT, s2_status TEXT, doi TEXT, isbn TEXT)"
        )
        connection.executemany("INSERT INTO item_citation_status VALUES (?, ?, ?, ?)", [
            ("DONE", "mapped", "10/done", ""),
            ("LIMIT", "limited", "", "978-old"),
            ("REMOVED", "mapped", "10/obsolete", ""),
            ("ERROR", "error", "", ""),
        ])
    items = [
        {"key": "DONE", "data": {"DOI": "10/done"}},
        {"key": "LIMIT", "data": {"ISBN": "978-new"}},
        {"key": "REMOVED", "data": {}},
        {"key": "ERROR", "data": {}},
        {"key": "NEW", "data": {}},
        {"key": "NO_ATTACHMENT", "data": {}},
    ]
    result = _citation_freshness(items, database)
    assert result["pending"] == 5
    assert result["unprocessed"] == 3
    assert result["metadata_changed"] == 2
    assert result["errors"] == 1
    assert result["sample_keys"] == ["ERROR", "LIMIT", "NEW", "NO_ATTACHMENT", "REMOVED"]


def test_note_freshness_reports_new_changed_and_retired_notes():
    manifest = {"notes": {
        "CURRENT": {"version": 1}, "CHANGED": {"version": 1}, "RETIRED": {"version": 1},
    }}
    result = _note_freshness([
        {"noteKey": "CURRENT", "version": 1},
        {"noteKey": "CHANGED", "version": 2},
        {"noteKey": "NEW", "version": 1},
    ], manifest)
    assert result == {
        "pending": 3, "new_or_changed": 2, "retired": 1,
        "sample_keys": ["CHANGED", "NEW", "RETIRED"],
    }


def test_summary_freshness_reports_pending_items_and_index_integrity(monkeypatch):
    monkeypatch.setattr(
        "scripts.check_admin_update_status.build_summary_audit",
        lambda **_kwargs: {
            "passed": False,
            "failures": ["nonterminal_or_degraded_summary_status", "summary_index_id_mismatch"],
            "counts": {"expected_index_rows": 12, "actual_index_rows": 11},
            "details": {
                "missing_status_items": ["MISSING"],
                "failed_status_items": ["FAILED"],
                "stale_source_items": ["STALE", "FAILED"],
                "missing_index_ids": ["sum:node:1"],
                "unexpected_index_ids": [],
                "content_mismatch_index_ids": [],
            },
        },
    )

    result = _summary_freshness()

    assert result["pending"] == 3
    assert result["failed"] == 1
    assert result["index_issues"] == 1
    assert result["expected_index_rows"] == 12
    assert result["actual_index_rows"] == 11
    assert result["attention"] == 2
    assert result["sample_keys"] == ["FAILED", "MISSING", "STALE"]


def test_citation_identifier_sync_can_clear_removed_zotero_values(tmp_path, monkeypatch):
    database = tmp_path / "relations.db"
    monkeypatch.setattr(db_relations, "DB_PATH", str(database))
    monkeypatch.setattr(db_relations, "_db_initialized", False)
    monkeypatch.setattr(db_relations, "_initialized_db_path", None)
    db_relations.update_item_citation_status(
        "ITEM", "mapped", doi="10/obsolete", isbn="978-obsolete",
    )

    assert db_relations.sync_item_citation_identifiers("ITEM", doi=None, isbn=None) is True
    assert db_relations.sync_item_citation_identifiers("ITEM", doi=None, isbn=None) is False

    with sqlite3.connect(database) as connection:
        assert connection.execute(
            "SELECT doi, isbn FROM item_citation_status WHERE item_key = 'ITEM'"
        ).fetchone() == (None, None)
