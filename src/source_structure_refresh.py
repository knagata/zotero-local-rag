"""Refresh source-derived heading metadata without replacing chunk text or vectors."""
from __future__ import annotations

import os
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Sequence

try:
    from .chapter_detect import build_pdf_page_structure_path_lookup, get_pdf_toc, infer_structure_roles
    from .document_structure import _attachment_key
    from .heading_zone import classify_heading_path
    from .html_extract import extract_chunks_from_epub_snapshot
    from .pdf_extract import _outline_events_by_page, _resolve_record_structure_paths
except ImportError:  # pragma: no cover
    from chapter_detect import build_pdf_page_structure_path_lookup, get_pdf_toc, infer_structure_roles
    from document_structure import _attachment_key
    from heading_zone import classify_heading_path
    from html_extract import extract_chunks_from_epub_snapshot
    from pdf_extract import _outline_events_by_page, _resolve_record_structure_paths


_EPUB_LOCATOR_RE = re.compile(r"^epub:spine(?P<spine>\d+):block(?P<block>\d+)$")
STRUCTURE_METADATA_KEYS = ("structure_path", "structure_roles", "chapter", "section", "zone")
INTRINSIC_ZONES = {"corrupted", "footnote"}


def _clear_source_structure_metadata(metadata: Dict[str, Any]) -> None:
    """Drop outline-derived fields while retaining intrinsic extraction zones."""
    for key in ("structure_path", "structure_roles", "chapter", "section"):
        metadata.pop(key, None)
    if str(metadata.get("zone") or "body") not in INTRINSIC_ZONES:
        metadata.pop("zone", None)


def _epub_source_path(rows: Sequence[Dict[str, Any]], attachment_key: str) -> Path:
    first = rows[0].get("metadata") or {}
    for raw in (first.get("path"), first.get("pdf_path")):
        if raw:
            candidate = Path(str(raw)).expanduser()
            if candidate.is_file() and candidate.suffix.casefold() == ".epub":
                return candidate
    data_dir = Path(os.environ.get("ZOTERO_DATA_DIR") or Path.home() / "Zotero").expanduser()
    storage = data_dir / "storage" / attachment_key
    filename = str(first.get("filename") or "").strip()
    if filename:
        candidate = storage / filename
        if candidate.is_file():
            return candidate
    candidates = sorted(storage.glob("*.epub")) if storage.is_dir() else []
    if len(candidates) == 1:
        return candidates[0]
    raise FileNotFoundError(
        f"unable to resolve one EPUB source for attachment {attachment_key}: {storage}"
    )


def _pdf_source_path(rows: Sequence[Dict[str, Any]], attachment_key: str) -> Path:
    first = rows[0].get("metadata") or {}
    for raw in (first.get("pdf_path"), first.get("path")):
        if raw:
            candidate = Path(str(raw)).expanduser()
            if candidate.is_file() and candidate.suffix.casefold() == ".pdf":
                return candidate
    data_dir = Path(os.environ.get("ZOTERO_DATA_DIR") or Path.home() / "Zotero").expanduser()
    storage = data_dir / "storage" / attachment_key
    filename = str(first.get("filename") or "").strip()
    if filename:
        candidate = storage / filename
        if candidate.is_file():
            return candidate
    candidates = sorted(storage.glob("*.pdf")) if storage.is_dir() else []
    if len(candidates) == 1:
        return candidates[0]
    raise FileNotFoundError(
        f"unable to resolve one PDF source for attachment {attachment_key}: {storage}"
    )


def _locator_blocks(metadata: Dict[str, Any]) -> list[tuple[int, int]]:
    start = _EPUB_LOCATOR_RE.fullmatch(str(metadata.get("locator") or ""))
    if not start:
        return []
    spine = int(start.group("spine"))
    first = int(start.group("block"))
    end = _EPUB_LOCATOR_RE.fullmatch(str(metadata.get("locator_end") or ""))
    last = int(end.group("block")) if end and int(end.group("spine")) == spine else first
    return [(spine, block) for block in range(first, max(first, last) + 1)]


def _refresh_epub_rows(
    rows: Sequence[Dict[str, Any]], attachment_key: str,
) -> tuple[list[Dict[str, Any]], Dict[str, Any]]:
    source_path = _epub_source_path(rows, attachment_key)
    first_metadata = dict(rows[0].get("metadata") or {})
    meta_base = {
        key: first_metadata[key]
        for key in ("itemKey", "attachmentKey", "source_type", "title", "filename")
        if key in first_metadata
    }
    fresh, _quality = extract_chunks_from_epub_snapshot(source_path, attachment_key, meta_base)
    by_block: dict[tuple[int, int], list[Dict[str, Any]]] = defaultdict(list)
    for _chunk_id, _text, metadata in fresh:
        for block in _locator_blocks(metadata):
            by_block[block].append(metadata)

    output: list[Dict[str, Any]] = []
    changed = 0
    for row in rows:
        blocks = _locator_blocks(dict(row.get("metadata") or {}))
        # An old merged chunk can span a newly recognized heading boundary.
        # Its immutable text cannot be split during a structure-only refresh;
        # assign it to the heading active at its first source block. Subsequent
        # old chunks move to the later heading through their own start locator.
        candidates = list(by_block.get(blocks[0], [])) if blocks else []
        if not candidates:
            candidates = [candidate for block in blocks for candidate in by_block.get(block, [])]
        signatures = {
            tuple(
                tuple(value) if isinstance(value, list) else str(value or "")
                for value in (candidate.get(key) for key in STRUCTURE_METADATA_KEYS)
            )
            for candidate in candidates
        }
        if not candidates:
            raise RuntimeError(f"no fresh EPUB structure match for chunk {row.get('id')}")
        if len(signatures) != 1:
            raise RuntimeError(f"ambiguous fresh EPUB structure for chunk {row.get('id')}")
        source_metadata = candidates[0]
        metadata = dict(row.get("metadata") or {})
        before = {key: metadata.get(key) for key in STRUCTURE_METADATA_KEYS}
        for key in STRUCTURE_METADATA_KEYS:
            if key in source_metadata:
                metadata[key] = source_metadata[key]
            else:
                metadata.pop(key, None)
        after = {key: metadata.get(key) for key in STRUCTURE_METADATA_KEYS}
        changed += before != after
        output.append({**row, "metadata": metadata})
    return output, {
        "attachment_key": attachment_key,
        "source_type": "epub",
        "source_path": str(source_path),
        "chunks": len(output),
        "metadata_changed": changed,
    }


def _refresh_pdf_rows(
    rows: Sequence[Dict[str, Any]], attachment_key: str,
) -> tuple[list[Dict[str, Any]], Dict[str, Any]]:
    source_path = _pdf_source_path(rows, attachment_key)
    toc = get_pdf_toc(str(source_path))
    if not toc:
        output = []
        changed = 0
        for row in rows:
            metadata = dict(row.get("metadata") or {})
            before = {key: metadata.get(key) for key in STRUCTURE_METADATA_KEYS}
            _clear_source_structure_metadata(metadata)
            after = {key: metadata.get(key) for key in STRUCTURE_METADATA_KEYS}
            changed += before != after
            output.append({**row, "metadata": metadata})
        return output, {
            "attachment_key": attachment_key, "source_type": "pdf",
            "source_path": str(source_path), "chunks": len(rows),
            "metadata_changed": changed, "outline_entries": 0,
        }
    lookup = build_pdf_page_structure_path_lookup(toc)
    events = _outline_events_by_page(toc)
    page_rows: dict[int, list[tuple[int, Dict[str, Any]]]] = defaultdict(list)
    for index, row in enumerate(rows):
        try:
            page = int((row.get("metadata") or {}).get("page") or 0)
        except (TypeError, ValueError):
            page = 0
        if page < 1:
            raise RuntimeError(f"PDF chunk has no page locator: {row.get('id')}")
        page_rows[page].append((index, row))

    output = [{**row, "metadata": dict(row.get("metadata") or {})} for row in rows]
    changed = 0
    for page, indexed_rows in sorted(page_rows.items()):
        indexed_rows.sort(key=lambda value: (
            int((value[1].get("metadata") or {}).get("reading_order") or 0),
            int((value[1].get("metadata") or {}).get("para_index") or 0),
            int((value[1].get("metadata") or {}).get("part_index") or 0),
            str(value[1].get("id") or ""),
        ))
        records = [{"text": str(row.get("text") or "")} for _index, row in indexed_rows]
        page_events = events.get(page) or []
        paths = _resolve_record_structure_paths(
            records, previous_path=lookup(page - 1) if page > 1 else [],
            page_events=page_events,
        ) if page_events else []
        fallback_path = lookup(page)
        for position, (index, row) in enumerate(indexed_rows):
            path = list(paths[position]) if position < len(paths) else list(fallback_path)
            metadata = dict(row.get("metadata") or {})
            before = {key: metadata.get(key) for key in STRUCTURE_METADATA_KEYS}
            if path:
                metadata["structure_path"] = path
                metadata["structure_roles"] = infer_structure_roles(path)
                metadata["chapter"] = path[0]
                if len(path) > 1:
                    metadata["section"] = path[1]
                else:
                    metadata.pop("section", None)
                inferred_zone = classify_heading_path(path)
                if str(metadata.get("zone") or "body") not in {"corrupted", "footnote"}:
                    if inferred_zone == "body":
                        metadata.pop("zone", None)
                    else:
                        metadata["zone"] = inferred_zone
            else:
                _clear_source_structure_metadata(metadata)
            after = {key: metadata.get(key) for key in STRUCTURE_METADATA_KEYS}
            changed += before != after
            output[index] = {**row, "metadata": metadata}
    return output, {
        "attachment_key": attachment_key, "source_type": "pdf",
        "source_path": str(source_path), "chunks": len(output),
        "metadata_changed": changed, "outline_entries": len(toc),
    }


def refresh_source_structure_metadata(
    chunks: Sequence[Dict[str, Any]],
) -> tuple[list[Dict[str, Any]], list[Dict[str, Any]]]:
    """Refresh supported source structure while preserving rows, IDs, and text."""
    groups: dict[tuple[str, str], list[tuple[int, Dict[str, Any]]]] = defaultdict(list)
    output = [{**row, "metadata": dict(row.get("metadata") or {})} for row in chunks]
    for index, row in enumerate(output):
        metadata = row.get("metadata") or {}
        source_type = str(metadata.get("source_type") or "").casefold()
        if source_type not in {"epub", "pdf"}:
            continue
        key = _attachment_key(row)
        if key:
            groups[(source_type, key)].append((index, row))
    reports: list[Dict[str, Any]] = []
    for (source_type, attachment_key), indexed_rows in groups.items():
        source_rows = [row for _index, row in indexed_rows]
        refreshed, report = (
            _refresh_epub_rows(source_rows, attachment_key)
            if source_type == "epub"
            else _refresh_pdf_rows(source_rows, attachment_key)
        )
        for (index, _old), new in zip(indexed_rows, refreshed, strict=True):
            output[index] = new
        reports.append(report)
    return output, reports


__all__ = ["STRUCTURE_METADATA_KEYS", "refresh_source_structure_metadata"]
