"""Pure helpers for safe V3 migration of legacy OCR-derived chunks."""
from __future__ import annotations

import json
from typing import Any, Mapping, Sequence

try:
    from .text_utils import (
        MAX_CHARS, MAX_CHARS_CJK, MIN_CHUNK_CHARS, MIN_CHUNK_CHARS_NO_SPACE,
        is_no_space_language_document, merge_short_chunk_records,
    )
except ImportError:  # pragma: no cover - direct src entrypoint
    from text_utils import (
        MAX_CHARS, MAX_CHARS_CJK, MIN_CHUNK_CHARS, MIN_CHUNK_CHARS_NO_SPACE,
        is_no_space_language_document, merge_short_chunk_records,
    )


def is_ocr_derived(quality: Mapping[str, Any] | None) -> bool:
    quality = quality or {}
    parser = str(quality.get("parser") or "").casefold()
    return bool(
        "ocr" in parser
        or quality.get("is_scanned")
        or quality.get("scanned_pages")
        or float(quality.get("scanned_ratio") or 0) > 0
        or quality.get("ocr_pages")
    )


def reuse_ocr_chunks_for_v3(
    chunks: Sequence[Mapping[str, Any]], attachment_key: str,
    meta_base: Mapping[str, Any], *, original_quality: Mapping[str, Any] | None = None,
) -> tuple[list[tuple[str, str, dict[str, Any]]], dict[str, Any]]:
    """Treat legacy OCR chunks as retained source blocks and apply V3 merging."""
    source: list[tuple[str, str, dict[str, Any]]] = []
    texts: list[str] = []
    for ordinal, row in enumerate(chunks):
        metadata = dict(meta_base)
        metadata.update(dict(row.get("metadata") or {}))
        if str(metadata.get("attachmentKey") or "") != attachment_key:
            continue
        text = str(row.get("text") or "").strip()
        if not text:
            continue
        metadata.update({
            "itemKey": meta_base.get("itemKey"), "attachmentKey": attachment_key,
            "source_type": "pdf", "ocr_text_reused": True,
            "original_extraction_engine": metadata.get("extraction_engine") or (original_quality or {}).get("parser") or "unknown",
            "original_extraction_version": metadata.get("extraction_version") or (original_quality or {}).get("parser_version") or "unknown",
            "extraction_engine": "legacy_ocr_reuse", "extraction_version": "v3-1",
        })
        source.append((f"{attachment_key}:v3reuse:block:{ordinal:06d}", text, metadata))
        texts.append(text)
    if not source:
        return [], {"parser": "legacy-ocr-reuse", "total_pages": 0, "ocr_text_reused": True}

    no_space = is_no_space_language_document("\n".join(texts))
    merged = merge_short_chunk_records(
        source,
        min_chars=MIN_CHUNK_CHARS_NO_SPACE if no_space else MIN_CHUNK_CHARS,
        max_chars=MAX_CHARS_CJK if no_space else MAX_CHARS,
        boundary_key=lambda _cid, _text, md: (
            json.dumps(md.get("structure_path") or md.get("heading_path") or [], ensure_ascii=False, sort_keys=True),
            str(md.get("zone") or "body"),
        ),
    )
    canonical = [
        (f"{attachment_key}:v3reuse:{ordinal:06d}", text, metadata)
        for ordinal, (_old_id, text, metadata) in enumerate(merged)
    ]
    quality = dict(original_quality or {})
    quality.update({
        "parser": "legacy-ocr-reuse", "parser_version": "v3-1",
        "ocr_text_reused": True, "source_blocks": len(source), "chunks": len(canonical),
    })
    return canonical, quality


__all__ = ["is_ocr_derived", "reuse_ocr_chunks_for_v3"]
