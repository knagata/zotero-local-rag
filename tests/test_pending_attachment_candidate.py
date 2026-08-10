"""Contracts for staging one accepted attachment before a flush."""
from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import index_from_zotero as module  # noqa: E402
from index_batch import PendingIndexBatch  # noqa: E402


def test_candidate_builds_every_pending_field_without_deduplicating_chunks():
    metadata = {"attachmentKey": "ATT", "itemKey": "ITEM"}
    status = ("ITEM", "success", {"attachment_key": "ATT"})
    candidate = module._build_pending_attachment_candidate(
        attachment_key="ATT",
        item_key="ITEM",
        source_type="pdf",
        chunks=[("same", "first", metadata), ("same", "second", metadata)],
        extraction_status=status,
        mtime=100.0,
        size=2048,
        source_path=Path("synthetic.pdf"),
        title="Synthetic",
        quality={"parser": "pymupdf"},
        content_signature_value="sha256:content",
        pipeline_fingerprint="sha256:pipeline",
    )

    assert candidate.ids == ["same", "same"]
    assert candidate.documents == ["first", "second"]
    assert candidate.metadatas == [metadata, metadata]
    assert candidate.extraction_status is status
    assert candidate.manifest_entry["mtime"] == 100.0
    assert candidate.manifest_entry["size"] == 2048
    assert candidate.manifest_entry["content_signature"] == "sha256:content"
    assert candidate.manifest_entry["pipeline_fingerprint"] == "sha256:pipeline"


def test_add_attachment_stages_the_candidate_as_one_flush_unit():
    candidate = module._build_pending_attachment_candidate(
        attachment_key="ATT",
        item_key="ITEM",
        source_type="html",
        chunks=[("chunk-1", "text", {"attachmentKey": "ATT", "itemKey": "ITEM"})],
        extraction_status=("ITEM", "degraded", {"reason_code": "quality"}),
        mtime=1.0,
        size=2,
        source_path=Path("snapshot.html"),
        title=None,
        quality={},
        content_signature_value=None,
        pipeline_fingerprint="sha256:pipeline",
    )
    pending = PendingIndexBatch.empty()

    pending.add_attachment(candidate)

    assert pending.ids == ["chunk-1"]
    assert pending.documents == ["text"]
    assert pending.delete_attachment_keys == {"ATT"}
    assert pending.source_types == {"ATT": "html"}
    assert pending.item_keys == {"ATT": "ITEM"}
    assert pending.extraction_statuses["ATT"][1] == "degraded"
    assert pending.manifest_updates["ATT"] == candidate.manifest_entry
    assert pending.expected_ids() == {"ATT": {"chunk-1"}}
