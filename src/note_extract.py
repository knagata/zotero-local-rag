from __future__ import annotations

from typing import Any, Callable, Optional

from html_extract import extract_main_text_from_html
from text_utils import (
    HARD_MIN_CHARS,
    MAX_CHARS,
    MIN_CHUNK_CHARS,
    MIN_CHUNK_CHARS_NO_SPACE,
    clean_extracted_text,
    is_no_space_language_document,
    joiner_for_text,
    looks_like_gibberish,
    merge_short_chunk_records,
    normalize_paragraphs,
    split_long_paragraph,
)


DedupeFn = Callable[
    [list[str], list[str], list[dict[str, Any]]],
    tuple[list[str], list[str], list[dict[str, Any]]],
]
UpsertFn = Callable[..., None]


def index_notes(
    notes: list[dict[str, Any]],
    *,
    col: Any,
    notes_manifest: dict[str, dict[str, Any]],
    batch_size: int,
    show_progress: bool,
    dedupe_fn: DedupeFn,
    upsert_fn: UpsertFn,
) -> tuple[dict[str, dict[str, Any]], dict[str, int]]:
    """
    Index Zotero notes into Chroma.

    - Notes are chunked into paragraph-level (with long paragraph splitting + short-merge).
    - Records are written with metadatas including `noteKey`, `source_type="note"`.
    - `notes_manifest` stores note version to skip unchanged notes.
    - Returns (updated_notes_manifest, stats).
    """

    # --- stale note delete ---
    current_note_keys = {n.get("noteKey") for n in notes if isinstance(n, dict) and n.get("noteKey")}
    stale_note_keys = set(notes_manifest.keys()) - set(current_note_keys)

    deleted_stale_notes = 0
    for nk in stale_note_keys:
        try:
            col.delete(where={"noteKey": nk})
            deleted_stale_notes += 1
        except Exception:
            pass
        notes_manifest.pop(nk, None)

    updated_notes = 0
    skipped_notes = 0

    pending_ids: list[str] = []
    pending_docs: list[str] = []
    pending_metas: list[dict[str, Any]] = []

    def _flush(label: str) -> None:
        nonlocal pending_ids, pending_docs, pending_metas
        if not pending_ids:
            return
        ids, docs, metas = dedupe_fn(pending_ids, pending_docs, pending_metas)
        upsert_fn(
            col,
            ids,
            docs,
            metas,
            subbatch_size=batch_size,
            show_progress=show_progress,
            label=label,
        )
        pending_ids = []
        pending_docs = []
        pending_metas = []

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

        # delete existing chunks for this noteKey (best-effort)
        try:
            col.delete(where={"noteKey": note_key})
        except Exception:
            pass

        creators_str: Optional[str] = None
        creators = n.get("creators")
        if isinstance(creators, list):
            creators_str = "; ".join([c for c in creators if isinstance(c, str) and c.strip()]) or None

        meta_base: dict[str, Any] = {
            "itemKey": n.get("parentItemKey"),
            "attachmentKey": None,
            "noteKey": note_key,
            "title": n.get("title"),
            "year": n.get("year"),
            "creators": creators_str,
            "source_type": "note",
            "path": None,
            "pdf_path": None,
            "locator": None,
        }

        note_html = n.get("note_html") or ""
        note_text = clean_extracted_text(extract_main_text_from_html(note_html if isinstance(note_html, str) else ""))

        # normalize paragraphing
        joiner = joiner_for_text(note_text[:20000])
        paras = normalize_paragraphs(note_text, joiner=joiner)

        # still update manifest even if empty/gibberish, so we don't retry forever
        if not paras:
            notes_manifest[note_key] = {"version": ver}
            updated_notes += 1
            continue

        joined = "\n\n".join(paras)
        if looks_like_gibberish(joined):
            notes_manifest[note_key] = {"version": ver}
            updated_notes += 1
            continue

        local_min_chunk = MIN_CHUNK_CHARS_NO_SPACE if is_no_space_language_document(joined) else MIN_CHUNK_CHARS

        note_chunks: list[tuple[str, str, dict[str, Any]]] = []

        for para_index, para_text in enumerate(paras):
            para_text = para_text.strip()
            if not para_text:
                continue

            parts = split_long_paragraph(para_text, max_chars=MAX_CHARS)
            for part_index, part in enumerate(parts):
                part = part.strip()
                if len(part) < HARD_MIN_CHARS:
                    continue

                cid = f"{note_key}:note:para{para_index}:part{part_index}"
                md = dict(meta_base)
                md.update(
                    {
                        "locator": f"note:para{para_index}",
                        "para_index": int(para_index),
                        "part_index": int(part_index),
                    }
                )
                note_chunks.append((cid, part, md))

        note_chunks = merge_short_chunk_records(note_chunks, min_chars=local_min_chunk, max_chars=MAX_CHARS)

        for cid, part, md in note_chunks:
            pending_ids.append(cid)
            pending_docs.append(part)
            pending_metas.append(md)

            if len(pending_ids) >= batch_size:
                _flush(label="notes upsert")

        _flush(label="notes upsert")

        notes_manifest[note_key] = {"version": ver}
        updated_notes += 1

    stats = {
        "updated_notes": int(updated_notes),
        "skipped_notes": int(skipped_notes),
        "deleted_stale_notes": int(deleted_stale_notes),
    }
    return notes_manifest, stats