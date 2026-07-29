from __future__ import annotations

from unittest.mock import MagicMock, patch

from scripts import audit_structure_summaries


def _connection():
    connection = MagicMock()
    connection.execute.side_effect = [
        MagicMock(fetchall=MagicMock(return_value=[("ITEM", "sha256:source")])),
        MagicMock(fetchall=MagicMock(return_value=[("ITEM", "success")])),
    ]
    return connection


def _summary_row():
    return {
        "node_id": "node-1",
        "item_key": "ITEM",
        "summary": "根拠に基づく有効な階層要約です。",
        "summary_kind": "llm",
        "model": "deepseek:test",
        "prompt_version": audit_structure_summaries.PROMPT_VERSION,
        "source_fingerprint": "sha256:source",
        "searchable": 1,
        "quality_status": "accepted",
        "parent_node_id": "",
        "attachment_key": "ATT",
        "node_type": "semantic_segment",
        "depth": 2,
        "title": "Chapter",
    }


def test_summary_audit_passes_when_db_and_index_match():
    collection = MagicMock()
    collection.get.return_value = {
        "ids": ["sum:node:node-1"],
        "documents": ["Chapter\n根拠に基づく有効な階層要約です。"],
        "metadatas": [{
            "itemKey": "ITEM", "attachmentKey": "ATT", "node_id": "node-1",
            "parent_node_id": "", "node_type": "semantic_segment", "depth": 2,
            "title": "Chapter", "summary_kind": "llm", "structure_version": "3",
            "source_fingerprint": "sha256:source",
        }],
    }
    client = MagicMock()
    client.get_collection.return_value = collection
    with (
        patch.object(audit_structure_summaries, "get_db_connection", return_value=_connection()),
        patch.object(
            audit_structure_summaries,
            "get_all_document_node_summaries",
            return_value=[_summary_row()],
        ),
        patch.object(audit_structure_summaries.chromadb, "PersistentClient", return_value=client),
    ):
        report = audit_structure_summaries.build_report(collection_name="v3")
    assert report["passed"] is True
    assert report["failures"] == []


def test_summary_audit_rejects_missing_summary_index_row():
    collection = MagicMock()
    collection.get.return_value = {"ids": [], "documents": [], "metadatas": []}
    client = MagicMock()
    client.get_collection.return_value = collection
    with (
        patch.object(audit_structure_summaries, "get_db_connection", return_value=_connection()),
        patch.object(
            audit_structure_summaries,
            "get_all_document_node_summaries",
            return_value=[_summary_row()],
        ),
        patch.object(audit_structure_summaries.chromadb, "PersistentClient", return_value=client),
    ):
        report = audit_structure_summaries.build_report(collection_name="v3")
    assert report["passed"] is False
    assert "summary_index_id_mismatch" in report["failures"]


def test_summary_audit_rejects_stale_index_content_with_same_id():
    collection = MagicMock()
    collection.get.return_value = {
        "ids": ["sum:node:node-1"],
        "documents": ["stale summary text"],
        "metadatas": [{
            "itemKey": "ITEM", "attachmentKey": "ATT", "node_id": "node-1",
            "parent_node_id": "", "node_type": "semantic_segment", "depth": 2,
            "title": "Chapter", "summary_kind": "llm", "structure_version": "3",
            "source_fingerprint": "sha256:source",
        }],
    }
    client = MagicMock()
    client.get_collection.return_value = collection
    with (
        patch.object(audit_structure_summaries, "get_db_connection", return_value=_connection()),
        patch.object(
            audit_structure_summaries,
            "get_all_document_node_summaries",
            return_value=[_summary_row()],
        ),
        patch.object(audit_structure_summaries.chromadb, "PersistentClient", return_value=client),
    ):
        report = audit_structure_summaries.build_report(collection_name="v3")
    assert report["passed"] is False
    assert "summary_index_content_mismatch" in report["failures"]
