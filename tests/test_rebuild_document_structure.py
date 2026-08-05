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
            refresh_source=False,
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
            refresh_source=False,
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
            refresh_source=False,
        )

    assert result["changed"] is True
    assert result["action"] == "dry_run"


def test_stored_structure_json_is_decoded_before_reserializing():
    decode = rebuild_document_structure._stored_sequence
    assert decode('["Chapter", "Section"]', plain_string_is_value=True) == [
        "Chapter", "Section",
    ]
    assert decode('["chapter", "section"]', plain_string_is_value=False) == [
        "chapter", "section",
    ]
    assert decode(["Chapter"], plain_string_is_value=True) == ["Chapter"]
    assert decode("Legacy heading", plain_string_is_value=True) == ["Legacy heading"]
    assert decode("invalid-role", plain_string_is_value=False) == []


def test_resync_does_not_double_encode_stored_structure_json():
    source_chunks = [{
        "id": "CHUNK",
        "metadata": {
            "structure_path": '["Chapter"]',
            "structure_roles": '["chapter"]',
            "chapter": "Chapter",
        },
    }]
    connection = type("Connection", (), {"execute": lambda self, _sql, _args: []})()
    collection = type("Collection", (), {
        "get": lambda self, **_kwargs: {
            "ids": ["CHUNK"], "metadatas": [{"node_id": "old"}],
        },
        "update": lambda self, **kwargs: setattr(self, "updated", kwargs),
    })()
    client = type("Client", (), {"get_collection": lambda self, _name: collection})()
    with patch("src.db_relations.get_db_connection", return_value=connection), \
            patch("src.structure_metadata_sync.desired_chunk_metadata", return_value={
                "CHUNK": {
                    "node_id": "new", "zone": "body", "summary_policy": "include",
                    "retrieval_policy": "normal", "citation_policy": "none",
                },
            }), patch("chromadb.PersistentClient", return_value=client):
        updated = rebuild_document_structure._resync_chunk_metadata(
            "ITEM", "test", source_chunks,
        )

    assert updated == 1
    metadata = collection.updated["metadatas"][0]
    assert metadata["structure_path"] == '["Chapter"]'
    assert metadata["structure_roles"] == '["chapter"]'


def test_metadata_refresh_is_persisted_when_tree_is_unchanged():
    refreshed = _chunks()
    refreshed[0]["metadata"]["chapter"] = "Corrected display chapter"
    previous = build_document_structure("ITEM", refreshed)
    with patch.object(rebuild_document_structure, "get_item_chunks", return_value=_chunks()), \
            patch.object(
                rebuild_document_structure, "refresh_source_structure_metadata",
                return_value=(refreshed, [{"metadata_changed": 1}]),
            ), patch.object(
                rebuild_document_structure, "get_document_structure", return_value=previous,
            ), patch.object(
                rebuild_document_structure, "replace_document_structure",
            ) as replace, patch.object(
                rebuild_document_structure, "_resync_chunk_metadata", return_value=1,
            ) as resync:
        result = rebuild_document_structure.rebuild_item(
            "ITEM", dry_run=False, force=False, run_id="test", collection_name="test",
        )

    replace.assert_not_called()
    resync.assert_called_once_with("ITEM", "test", refreshed)
    assert result["action"] == "metadata_resynced"
    assert result["chunk_metadata_resynced"] == 1
    assert result["changed"] is False
    assert result["embeddings_unchanged"] is True
