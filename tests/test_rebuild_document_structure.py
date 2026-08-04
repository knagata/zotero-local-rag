from __future__ import annotations

from unittest.mock import patch

from scripts import rebuild_document_structure
from src.document_structure import build_document_structure


def _chunks():
    return [{
        "id": "ABCD1234:p1",
        "text": "argument " * 200,
        "metadata": {
            "attachmentKey": "ABCD1234",
            "source_type": "pdf",
            "structure_path": ["Chapter 1"],
            "structure_roles": ["chapter"],
            "zone": "body",
        },
    }]


def test_structure_rebuild_keeps_paragraph_embeddings_current():
    statuses: list[tuple[str, str]] = []

    def record_status(_item_key, artifact_type, status, **_kwargs):
        statuses.append((artifact_type, status))

    with patch.object(rebuild_document_structure, "get_item_chunks", return_value=_chunks()), \
            patch.object(rebuild_document_structure, "get_document_structure", return_value=None), \
            patch.object(rebuild_document_structure, "replace_document_structure") as replace, \
            patch.object(rebuild_document_structure, "_resync_chunk_metadata", return_value=1), \
            patch.object(rebuild_document_structure, "mark_artifact_status", side_effect=record_status):
        result = rebuild_document_structure.rebuild_item(
            "ITEM", dry_run=False, force=False, run_id="test", collection_name="test",
        )

    replace.assert_called_once()
    assert result["action"] == "rebuilt"
    assert result["embeddings_unchanged"] is True
    assert ("summary", "stale") in statuses
    assert ("summary_index", "stale") in statuses
    assert not any(artifact == "embeddings" for artifact, _status in statuses)


def test_structure_dry_run_writes_nothing():
    with patch.object(rebuild_document_structure, "get_item_chunks", return_value=_chunks()), \
            patch.object(rebuild_document_structure, "get_document_structure", return_value=None), \
            patch.object(rebuild_document_structure, "replace_document_structure") as replace, \
            patch.object(rebuild_document_structure, "_resync_chunk_metadata") as resync, \
            patch.object(rebuild_document_structure, "mark_artifact_status") as mark:
        result = rebuild_document_structure.rebuild_item(
            "ITEM", dry_run=True, force=False, run_id="test", collection_name="test",
        )

    assert result["action"] == "dry_run"
    replace.assert_not_called()
    resync.assert_not_called()
    mark.assert_not_called()


def test_role_only_change_is_detected_for_an_existing_structure():
    old_chunks = _chunks()
    old_chunks[0]["metadata"]["structure_roles"] = ["section"]
    previous = build_document_structure("ITEM", old_chunks)

    with patch.object(rebuild_document_structure, "get_item_chunks", return_value=_chunks()), \
            patch.object(rebuild_document_structure, "get_document_structure", return_value=previous):
        result = rebuild_document_structure.rebuild_item(
            "ITEM", dry_run=True, force=False, run_id="test", collection_name="test",
        )

    assert result["changed"] is True
    assert result["action"] == "dry_run"
