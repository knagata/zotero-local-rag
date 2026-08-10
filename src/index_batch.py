"""Compensating unit-of-work for replacing indexed attachment batches."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Iterable

try:
    from .lexical_index import delete_by_attachment_keys, upsert_chunks
except ImportError:  # pragma: no cover - direct src/ entrypoints
    from lexical_index import delete_by_attachment_keys, upsert_chunks


@dataclass(frozen=True)
class AttachmentBatch:
    attachment_keys: tuple[str, ...]
    ids: list[str]
    documents: list[str]
    metadatas: list[dict[str, Any]]
    expected_ids: dict[str, set[str]]
    attachment_item_keys: dict[str, str]
    subbatch_size: int
    show_progress: bool
    label: str
    context_label: str
    strict_lexical: bool

    @classmethod
    def create(
        cls, *, attachment_keys: Iterable[str], **values: Any,
    ) -> "AttachmentBatch":
        keys = tuple(sorted({str(key) for key in attachment_keys if key}))
        return cls(attachment_keys=keys, **values)


@dataclass
class PendingIndexBatch:
    ids: list[str]
    documents: list[str]
    metadatas: list[dict[str, Any]]
    manifest_updates: dict[str, dict[str, Any]]
    extraction_statuses: dict[str, tuple[str, str, dict[str, Any]]]
    delete_attachment_keys: set[str]
    source_types: dict[str, str]
    item_keys: dict[str, str]

    @classmethod
    def empty(cls) -> "PendingIndexBatch":
        return cls([], [], [], {}, {}, set(), {}, {})

    def expected_ids(self) -> dict[str, set[str]]:
        expected = {key: set() for key in self.delete_attachment_keys}
        for chunk_id, metadata in zip(self.ids, self.metadatas, strict=True):
            attachment_key = str(metadata.get("attachmentKey") or "")
            if attachment_key in expected:
                expected[attachment_key].add(str(chunk_id))
        return expected

    def add_attachment(self, candidate: "PendingAttachmentCandidate") -> None:
        """Stage one already-validated attachment as one in-memory unit."""
        self.ids.extend(candidate.ids)
        self.documents.extend(candidate.documents)
        self.metadatas.extend(candidate.metadatas)
        self.manifest_updates[candidate.attachment_key] = candidate.manifest_entry
        self.extraction_statuses[candidate.attachment_key] = candidate.extraction_status
        self.delete_attachment_keys.add(candidate.attachment_key)
        self.source_types[candidate.attachment_key] = candidate.source_type
        self.item_keys[candidate.attachment_key] = candidate.item_key

    def clear(self) -> None:
        self.ids.clear()
        self.documents.clear()
        self.metadatas.clear()
        self.manifest_updates.clear()
        self.extraction_statuses.clear()
        self.delete_attachment_keys.clear()
        self.source_types.clear()
        self.item_keys.clear()


@dataclass(frozen=True)
class PendingAttachmentCandidate:
    """All state an accepted attachment contributes to the next flush."""

    attachment_key: str
    item_key: str
    source_type: str
    ids: list[str]
    documents: list[str]
    metadatas: list[dict[str, Any]]
    manifest_entry: dict[str, Any]
    extraction_status: tuple[str, str, dict[str, Any]]


@dataclass(frozen=True)
class FlushOutcome:
    updated_pdf: int = 0
    updated_html: int = 0
    updated_epub: int = 0
    last_written_id: str | None = None
    committed_item_keys: frozenset[str] = frozenset()


def snapshot_attachment_chunks(col: Any, attachment_keys: Iterable[str]) -> dict[str, Any]:
    """Capture enough of the active generation to compensate a failed flush."""
    keys = sorted({str(key) for key in attachment_keys if key})
    if not keys:
        return {"ids": [], "documents": [], "metadatas": [], "embeddings": []}
    snapshot = col.get(
        where={"attachmentKey": {"$in": keys}},
        include=["documents", "metadatas", "embeddings"],
    )
    ids = list(snapshot.get("ids") or [])
    documents = list(snapshot.get("documents") or [])
    metadatas = list(snapshot.get("metadatas") or [])
    embeddings_raw = snapshot.get("embeddings")
    embeddings = list(embeddings_raw) if embeddings_raw is not None else []
    if not (len(ids) == len(documents) == len(metadatas)):
        raise RuntimeError(
            "Cannot safely replace attachments: active Chroma snapshot is inconsistent "
            f"(ids={len(ids)}, documents={len(documents)}, metadatas={len(metadatas)})"
        )
    if embeddings and len(embeddings) != len(ids):
        raise RuntimeError(
            "Cannot safely replace attachments: active embedding snapshot is inconsistent "
            f"(ids={len(ids)}, embeddings={len(embeddings)})"
        )
    return {
        "ids": ids,
        "documents": documents,
        "metadatas": metadatas,
        "embeddings": embeddings,
    }


def restore_attachment_snapshot(
    col: Any, attachment_keys: Iterable[str], snapshot: dict[str, Any],
) -> None:
    """Remove a partial generation and restore both vector and lexical rows."""
    keys = sorted({str(key) for key in attachment_keys if key})
    col.delete(where={"attachmentKey": {"$in": keys}})
    delete_by_attachment_keys(keys)
    old_ids = list(snapshot.get("ids") or [])
    if not old_ids:
        return
    old_documents = list(snapshot.get("documents") or [])
    old_metadatas = list(snapshot.get("metadatas") or [])
    upsert_args: dict[str, Any] = {
        "ids": old_ids,
        "documents": old_documents,
        "metadatas": old_metadatas,
    }
    embeddings = snapshot.get("embeddings")
    if embeddings is not None and len(embeddings) == len(old_ids):
        upsert_args["embeddings"] = embeddings
    col.upsert(**upsert_args)
    upsert_chunks(old_ids, old_documents, old_metadatas)


def replace_attachment_batch(
    col: Any,
    batch: AttachmentBatch,
    *,
    delete_batch: Callable[..., None],
    upsert_batch: Callable[..., None],
    health_check: Callable[..., None],
    verify_written: Callable[[Any, dict[str, set[str]]], None] | None,
) -> None:
    """Replace a Chroma/lexical flush and compensate every failed phase."""
    snapshot = snapshot_attachment_chunks(col, batch.attachment_keys)
    try:
        delete_batch(col, batch.attachment_keys, strict=batch.strict_lexical)
        upsert_batch(
            col, batch.ids, batch.documents, batch.metadatas,
            subbatch_size=batch.subbatch_size,
            show_progress=batch.show_progress,
            label=batch.label,
            strict_lexical=batch.strict_lexical,
        )
        health_check(
            col, batch.attachment_item_keys, context_label=batch.context_label,
        )
        if verify_written is not None:
            verify_written(col, batch.expected_ids)
    except BaseException as original:
        try:
            restore_attachment_snapshot(col, batch.attachment_keys, snapshot)
        except BaseException as rollback_error:
            raise RuntimeError(
                f"Attachment batch failed and rollback also failed: {rollback_error}"
            ) from original
        raise
