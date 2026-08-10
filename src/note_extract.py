from __future__ import annotations

from typing import Any, Callable, Optional

try:
    from .html_extract import extract_main_text_from_html
    from .text_utils import (
        HARD_MIN_CHARS, MAX_CHARS, TARGET_CHARS, MAX_CHARS_CJK, TARGET_CHARS_CJK,
        MIN_CHUNK_CHARS, MIN_CHUNK_CHARS_NO_SPACE, clean_extracted_text, detect_lang,
        is_no_space_language_document, joiner_for_text, looks_like_gibberish,
        merge_short_chunk_records, normalize_paragraphs, split_long_paragraph,
    )
except ImportError:  # pragma: no cover - direct src entrypoint
    from html_extract import extract_main_text_from_html
    from text_utils import (
        HARD_MIN_CHARS, MAX_CHARS, TARGET_CHARS, MAX_CHARS_CJK, TARGET_CHARS_CJK,
        MIN_CHUNK_CHARS, MIN_CHUNK_CHARS_NO_SPACE, clean_extracted_text, detect_lang,
        is_no_space_language_document, joiner_for_text, looks_like_gibberish,
        merge_short_chunk_records, normalize_paragraphs, split_long_paragraph,
    )


DedupeFn = Callable[
    [list[str], list[str], list[dict[str, Any]]],
    tuple[list[str], list[str], list[dict[str, Any]]],
]
UpsertFn = Callable[..., None]


def _delete_note_rows(
    col: Any,
    note_key: str,
    lexical_delete_fn: Callable[[str], None] | None,
    strict_lexical: bool,
) -> bool:
    """Delete one note from Chroma and the optional lexical index."""
    chroma_deleted = False
    try:
        col.delete(where={"noteKey": note_key})
        chroma_deleted = True
    except Exception:
        if strict_lexical:
            raise
    if lexical_delete_fn:
        try:
            lexical_delete_fn(note_key)
        except Exception:
            if strict_lexical:
                raise
    return chroma_deleted


def _flush_note_batch(
    col: Any,
    pending_ids: list[str],
    pending_docs: list[str],
    pending_metas: list[dict[str, Any]],
    *,
    batch_size: int,
    show_progress: bool,
    dedupe_fn: DedupeFn,
    upsert_fn: UpsertFn,
    label: str,
    strict_lexical: bool,
) -> tuple[list[str], list[str], list[dict[str, Any]]]:
    """Dedupe and upsert pending note chunks, then reset the buffers."""
    if not pending_ids:
        return pending_ids, pending_docs, pending_metas
    ids, docs, metas = dedupe_fn(pending_ids, pending_docs, pending_metas)
    kwargs: dict[str, Any] = {
        "subbatch_size": batch_size,
        "show_progress": show_progress,
        "label": label,
    }
    if strict_lexical:
        kwargs["strict_lexical"] = True
    upsert_fn(col, ids, docs, metas, **kwargs)
    return [], [], []


def _note_metadata(note: dict[str, Any], note_key: str) -> dict[str, Any]:
    creators = note.get("creators")
    creators_str: Optional[str] = None
    if isinstance(creators, list):
        creators_str = "; ".join(
            c for c in creators if isinstance(c, str) and c.strip()
        ) or None
    return {
        "itemKey": note.get("parentItemKey"),
        "attachmentKey": None,
        "noteKey": note_key,
        "title": note.get("title"),
        "year": note.get("year"),
        "creators": creators_str,
        "source_type": "note",
        "path": None,
        "pdf_path": None,
        "locator": None,
        "lang": detect_lang("", note.get("language")),
    }


def _extract_note_chunks(
    note: dict[str, Any],
    note_key: str,
    meta_base: dict[str, Any],
) -> list[tuple[str, str, dict[str, Any]]]:
    """Extract valid note chunks, returning an empty list for unusable text."""
    note_html = note.get("note_html") or ""
    raw_html = note_html if isinstance(note_html, str) else ""
    note_text = clean_extracted_text(extract_main_text_from_html(raw_html))
    joiner = joiner_for_text(note_text[:20000])
    paras = normalize_paragraphs(note_text, joiner=joiner)
    if not paras:
        return []
    joined = "\n\n".join(paras)
    if looks_like_gibberish(joined):
        return []

    is_cjk = is_no_space_language_document(joined)
    local_min = MIN_CHUNK_CHARS_NO_SPACE if is_cjk else MIN_CHUNK_CHARS
    local_max = MAX_CHARS_CJK if is_cjk else MAX_CHARS
    local_target = TARGET_CHARS_CJK if is_cjk else TARGET_CHARS
    chunks: list[tuple[str, str, dict[str, Any]]] = []
    for para_index, para_text in enumerate(paras):
        para_text = para_text.strip()
        if not para_text:
            continue
        parts = split_long_paragraph(
            para_text, max_chars=local_max, target_chars=local_target,
        )
        for part_index, part in enumerate(parts):
            part = part.strip()
            if len(part) < HARD_MIN_CHARS:
                continue
            chunk_id = f"{note_key}:note:para{para_index}:part{part_index}"
            metadata = dict(meta_base)
            metadata.update({
                "locator": f"note:para{para_index}",
                "para_index": int(para_index),
                "part_index": int(part_index),
                "lang": detect_lang(part, note.get("language")),
            })
            chunks.append((chunk_id, part, metadata))
    return merge_short_chunk_records(
        chunks, min_chars=local_min, max_chars=local_max,
    )


def index_notes(
    notes: list[dict[str, Any]],
    *,
    col: Any,
    notes_manifest: dict[str, dict[str, Any]],
    batch_size: int,
    show_progress: bool,
    dedupe_fn: DedupeFn,
    upsert_fn: UpsertFn,
    lexical_delete_fn: Callable[[str], None] | None = None,
    delete_stale: bool = True,
    strict_lexical: bool = False,
) -> tuple[dict[str, dict[str, Any]], dict[str, int]]:
    """
    Index Zotero notes into Chroma.

    - Notes are chunked into paragraph-level (with long paragraph splitting + short-merge).
    - Records are written with metadatas including `noteKey`, `source_type="note"`.
    - `notes_manifest` stores note version to skip unchanged notes.
    - Returns (updated_notes_manifest, stats).
    """

    current_note_keys = {n.get("noteKey") for n in notes if isinstance(n, dict) and n.get("noteKey")}
    stale_note_keys = (
        set(notes_manifest.keys()) - set(current_note_keys) if delete_stale else set()
    )

    deleted_stale_notes = 0
    for nk in stale_note_keys:
        deleted_stale_notes += int(
            _delete_note_rows(col, nk, lexical_delete_fn, strict_lexical)
        )
        notes_manifest.pop(nk, None)

    updated_notes = 0
    skipped_notes = 0

    pending_ids: list[str] = []
    pending_docs: list[str] = []
    pending_metas: list[dict[str, Any]] = []

    for n in notes:
        if not isinstance(n, dict):
            continue

        note_key = n.get("noteKey")
        if not isinstance(note_key, str) or not note_key:
            continue

        ver = n.get("version")
        prev = notes_manifest.get(note_key)
        prev_ver = prev.get("version") if isinstance(prev, dict) else None
        if prev is not None and prev_ver == ver:
            skipped_notes += 1
            continue

        _delete_note_rows(col, note_key, lexical_delete_fn, strict_lexical)
        meta_base = _note_metadata(n, note_key)
        note_chunks = _extract_note_chunks(n, note_key, meta_base)

        for cid, part, md in note_chunks:
            pending_ids.append(cid)
            pending_docs.append(part)
            pending_metas.append(md)

            if len(pending_ids) >= batch_size:
                pending_ids, pending_docs, pending_metas = _flush_note_batch(
                    col, pending_ids, pending_docs, pending_metas,
                    batch_size=batch_size, show_progress=show_progress,
                    dedupe_fn=dedupe_fn, upsert_fn=upsert_fn,
                    label="notes upsert", strict_lexical=strict_lexical,
                )

        pending_ids, pending_docs, pending_metas = _flush_note_batch(
            col, pending_ids, pending_docs, pending_metas,
            batch_size=batch_size, show_progress=show_progress,
            dedupe_fn=dedupe_fn, upsert_fn=upsert_fn,
            label="notes upsert", strict_lexical=strict_lexical,
        )

        notes_manifest[note_key] = {"version": ver}
        updated_notes += 1

    stats = {
        "updated_notes": int(updated_notes),
        "skipped_notes": int(skipped_notes),
        "deleted_stale_notes": int(deleted_stale_notes),
    }
    return notes_manifest, stats
