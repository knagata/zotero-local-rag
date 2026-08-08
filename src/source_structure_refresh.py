"""Refresh source-derived heading metadata without replacing chunk text or vectors."""
from __future__ import annotations

import os
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Sequence

try:
    from .chapter_detect import (
        build_pdf_page_structure_path_lookup, get_epub_chapter_index_to_toc_entries,
        get_pdf_toc, infer_structure_roles,
    )
    from .document_structure import _attachment_key
    from .heading_structure import (
        CHAPTER_RE, INDEX_GROUP_RE, PART_RE, ROMAN_SUBHEADING_RE, SECTION_RE,
        normalize as normalize_heading, ordinal as heading_ordinal,
    )
    from .heading_zone import TOC_RE, classify_heading_path
    from .html_extract import extract_chunks_from_epub_snapshot
    from .manifest import load_manifest
    from .pdf_extract import _outline_events_by_page, _resolve_record_structure_paths
    from .pdf_toc_recovery import HeadingAnchor, apply_anchors
    from .v3_data_plane import manifest_path
except ImportError:  # pragma: no cover
    from chapter_detect import (
        build_pdf_page_structure_path_lookup, get_epub_chapter_index_to_toc_entries,
        get_pdf_toc, infer_structure_roles,
    )
    from document_structure import _attachment_key
    from heading_structure import (
        CHAPTER_RE, INDEX_GROUP_RE, PART_RE, ROMAN_SUBHEADING_RE, SECTION_RE,
        normalize as normalize_heading, ordinal as heading_ordinal,
    )
    from heading_zone import TOC_RE, classify_heading_path
    from html_extract import extract_chunks_from_epub_snapshot
    from manifest import load_manifest
    from pdf_extract import _outline_events_by_page, _resolve_record_structure_paths
    from pdf_toc_recovery import HeadingAnchor, apply_anchors
    from v3_data_plane import manifest_path


_EPUB_LOCATOR_RE = re.compile(r"^epub:spine(?P<spine>\d+):block(?P<block>\d+)$")
_EPUB_SPINE_RE = re.compile(r"^epub:spine(?P<spine>\d+)(?::|$)")
STRUCTURE_METADATA_KEYS = ("structure_path", "structure_roles", "chapter", "section", "zone")
INTRINSIC_ZONES = {"corrupted", "footnote"}
_TOP_LEVEL_HEADING_RE = re.compile(
    r"^(?:(?:序言|はじめに|訳者あとがき|終章|補論)(?:\s|$)|"
    r"(?:初出一覧|参考文献|謝辞|索引|人名索引)$)"
)
_FUNCTIONAL_BODY_HEADINGS = {"参考文献", "初出一覧", "謝辞", "索引", "人名索引"}
_PROJECT_ROOT = Path(__file__).resolve().parents[1]

#: A folio -- the printed page number a running header carries alongside the
#: chapter name ("序章] 13", "序章〕 15"). Every occurrence of such a header is
#: textually unique, so counting the raw text cannot tell a header apart from a
#: chapter opener until the number is folded away.
_TRAILING_FOLIO_RE = re.compile(r"[^\w぀-ヿ㐀-鿿]*[0-9]{1,4}$")
#: How often a heading's text may occur before it is read as a running header
#: rather than a boundary. Measured over the 84 flat PDFs here, genuine chapter
#: openers appear once or twice and running headers 3 to 28 times.
_MAX_BOUNDARY_REPEATS = 2


def _clear_source_structure_metadata(metadata: Dict[str, Any]) -> None:
    """Drop outline-derived fields while retaining intrinsic extraction zones."""
    for key in ("structure_path", "structure_roles", "chapter", "section"):
        metadata.pop(key, None)
    if str(metadata.get("zone") or "body") not in INTRINSIC_ZONES:
        metadata.pop("zone", None)


def _set_structure_metadata(
    metadata: Dict[str, Any], path: Sequence[str], roles: Sequence[str] | None = None,
) -> None:
    """Set path-derived fields without confusing a section with its chapter."""
    clean_path = [str(value).strip() for value in path if str(value).strip()]
    clean_roles = [str(value).strip() for value in (roles or []) if str(value).strip()]
    if len(clean_roles) != len(clean_path):
        clean_roles = infer_structure_roles(clean_path)
    metadata["structure_path"] = clean_path
    metadata["structure_roles"] = clean_roles
    for field, role in (("chapter", "chapter"), ("section", "section")):
        value = next((title for title, kind in zip(clean_path, clean_roles) if kind == role), None)
        if value is None:
            metadata.pop(field, None)
        else:
            metadata[field] = value


def _body_heading_title(row: Dict[str, Any]) -> str:
    """The heading text a boundary would be named after, or "".

    Normalised through the shared vocabulary so a heading that layout or OCR
    prefixed with decoration ("■ ■ ■ ■ ■ CHAPTER ONE") is still
    recognisable as the chapter opener it is.
    """
    text = normalize_heading(row.get("text") or "")
    return text if 0 < len(text) <= 180 else ""


def _row_page(row: Dict[str, Any]) -> int:
    try:
        return int((row.get("metadata") or {}).get("page") or 0)
    except (TypeError, ValueError):
        return 0


def _persisted_ai_toc_anchors(attachment_key: str) -> list[HeadingAnchor]:
    """Load previously accepted AI-TOC anchors without invoking an LLM."""
    entry = (load_manifest(manifest_path(_PROJECT_ROOT)).get("files") or {}).get(
        attachment_key, {}
    )
    quality = entry.get("quality") if isinstance(entry, dict) else {}
    diagnostics = quality.get("ai_toc_diagnostics") if isinstance(quality, dict) else {}
    if not isinstance(diagnostics, dict) or diagnostics.get("accepted") is not True:
        return []
    payload = diagnostics.get("anchor_payload")
    try:
        values = json.loads(payload) if isinstance(payload, str) else payload
        anchors = [HeadingAnchor(**value) for value in values if isinstance(value, dict)]
    except (TypeError, ValueError):
        return []
    return anchors if len(anchors) >= 2 else []


def _refresh_pdf_rows_from_persisted_anchors(
    rows: Sequence[Dict[str, Any]], attachment_key: str, source_path: Path,
) -> tuple[list[Dict[str, Any]], Dict[str, Any]] | None:
    """Reapply accepted, persisted AI-TOC boundaries to unchanged chunks."""
    anchors = _persisted_ai_toc_anchors(attachment_key)
    if not anchors:
        return None
    cleared: list[tuple[str, str, Dict[str, Any]]] = []
    for row in rows:
        metadata = dict(row.get("metadata") or {})
        _clear_source_structure_metadata(metadata)
        cleared.append((str(row.get("id") or ""), str(row.get("text") or ""), metadata))
    structured = apply_anchors(cleared, anchors)
    output: list[Dict[str, Any]] = []
    changed = mapped = 0
    for row, (_chunk_id, _text, metadata) in zip(rows, structured, strict=True):
        before = {key: (row.get("metadata") or {}).get(key) for key in STRUCTURE_METADATA_KEYS}
        path = metadata.get("structure_path") or []
        if path:
            _set_structure_metadata(metadata, path)
            mapped += 1
        after = {key: metadata.get(key) for key in STRUCTURE_METADATA_KEYS}
        changed += before != after
        output.append({**row, "metadata": metadata})
    if mapped / max(1, len(output)) < 0.8:
        return None
    return output, {
        "attachment_key": attachment_key, "source_type": "pdf",
        "source_path": str(source_path), "chunks": len(output),
        "metadata_changed": changed, "outline_entries": len(anchors),
        "mapping_mode": "persisted_ai_toc_anchors", "mapped_chunks": mapped,
    }


def _is_structural(title: str) -> bool:
    return bool(PART_RE.match(title) or CHAPTER_RE.match(title) or SECTION_RE.match(title))


def _ordinal_key(title: str) -> tuple[str, int] | None:
    """The (level, number) a numbered structural heading carries, or None.

    A printed contents entry and the opener it points at seldom share their
    exact text: the contents page says "PART II" where the body page says
    "PART TWO", and adds the folio. The level and the number survive both, so
    matching on those is what lets the contents region be recognised at all in
    a book that renders its ordinals two ways.
    """
    for level, pattern in (("part", PART_RE), ("chapter", CHAPTER_RE), ("section", SECTION_RE)):
        if pattern.match(title):
            number = heading_ordinal(title)
            return (level, number) if number else None
    return None


def _repetition_key(title: str) -> str:
    """The form of a title used to ask how often the document repeats it.

    A running header that prints the folio beside the chapter name gives every
    page a unique string, so the raw text always looks unrepeated. Dropping a
    trailing number recovers the shared form -- but only when what is left is
    still a structural heading, since stripping the ordinal off "CHAPTER 1"
    would fold every chapter in the book onto one key.
    """
    folded = _TRAILING_FOLIO_RE.sub("", title).strip()
    return folded if folded != title and _is_structural(folded) else title


#: A top-level division holding this share of a document, while another holds
#: almost nothing, is what divisions read off a contents page look like.
_DOMINANT_SHARE = 0.90
_NEGLIGIBLE_CHUNKS = 5


def _divides_the_document(
    events: Dict[int, tuple[list[str], list[str]]], total_rows: int,
) -> bool:
    """Whether the boundaries actually divide the document between them.

    A book's divisions divide it. When one holds nearly everything and its
    siblings hold a handful of chunks each, those siblings are not divisions of
    the body -- they are the lines of a printed contents page, and the one that
    swallowed the rest is simply the last of them before the body began.

    Japan-ness in Architecture is in that state: its contents page lost its own
    "Contents" heading during extraction, so nothing marks the region, and the
    four part titles it lists were read as the parts themselves. 1,711 of 1,718
    chunks land under the fourth while the other three hold two or three each.
    The list of parts looks right; only the weights show it is one part wearing
    the name of the fourth, which is worse than leaving the document flat.

    Chunks are counted against the top-level division they sit under, since a
    chunk inside a subsection is just as much inside the chapter above it.
    """
    if len(events) < 2:
        return True
    weights: Dict[str, int] = {}
    boundaries = sorted(events)
    for position, index in enumerate(boundaries):
        end = boundaries[position + 1] if position + 1 < len(boundaries) else total_rows
        division = events[index][0][0]
        weights[division] = weights.get(division, 0) + max(0, end - index)
    total = sum(weights.values())
    if total <= 0 or len(weights) < 2:
        return True
    if max(weights.values()) / total < _DOMINANT_SHARE:
        return True
    return not any(chunks <= _NEGLIGIBLE_CHUNKS for chunks in weights.values())


def _numbering_is_contiguous(ordinals: Sequence[int]) -> bool:
    """Whether a run of part/chapter numbers is a complete 1..n.

    Unnumbered openers (序章, "Introduction") report 0 and are ignored: they
    are real boundaries that simply carry no ordinal.
    """
    numbered = sorted({value for value in ordinals if value})
    if len(numbered) < 2:
        return True
    # Must start at 1, not merely be gapless. An extractor that loses the
    # opening chapters leaves 3, 4, 5 -- contiguous, and a tree built from it
    # silently omits the beginning of the book.
    #
    # The known cost: a volume whose chapters continue the previous volume's
    # numbering (12 to 20) is a complete run for that volume and is rejected
    # anyway. Nothing in the chunks distinguishes it from an extractor that lost
    # chapters 1 to 11, and staying flat is the safer of the two mistakes.
    return numbered == list(range(1, len(numbered) + 1))


def _refresh_pdf_rows_from_numbered_body_headings(
    rows: Sequence[Dict[str, Any]], attachment_key: str, source_path: Path,
) -> tuple[list[Dict[str, Any]], Dict[str, Any]] | None:
    """Recover a conservative TOC-level hierarchy from body headings.

    Language-neutral: the part/chapter/section vocabulary comes from
    heading_structure, so an English book is read the same way a Japanese one
    is. Notes attached to a chapter become a child of that chapter rather than
    a boundary of their own -- chapter-end notes are the ordinary form in an
    edited volume, and treating each "NOTES" as a top-level section would cut
    every chapter in half. As a child they also pick up zone="endnote" from
    classify_heading_path, which keeps them out of the chapter's summary and
    puts their references in front of the citation extractor.
    """
    events: dict[int, tuple[list[str], list[str]]] = {}
    chapter_ordinals: list[int] = []
    part_ordinals: list[int] = []
    part_path: list[str] = []
    chapter_path: list[str] = []
    section_path: list[str] = []
    functional_scope = False
    inside_printed_toc = False
    printed_toc_page = 0
    # How often the document carries each heading's text, over every block that
    # could hold one. Two questions are asked of it: how often in total, which
    # separates a chapter opener from the running header repeating its words on
    # every page; and how often is left ahead, which separates a printed
    # contents entry from the opener it points at.
    repeated_titles = Counter(
        _repetition_key(title)
        for row in rows
        if (title := _body_heading_title(row))
        and str((row.get("metadata") or {}).get("block_type") or "").casefold()
        in {"heading", "page_furniture"}
    )
    remaining_titles = Counter(repeated_titles)
    remaining_ordinals: Counter[tuple[str, int]] = Counter(
        key
        for row in rows
        if (title := _body_heading_title(row)) and (key := _ordinal_key(title))
        and str((row.get("metadata") or {}).get("block_type") or "").casefold()
        in {"heading", "page_furniture"}
    )
    # Which numbers have already opened something. Parts are numbered once
    # across the book; chapter numbers restart inside each part in many edited
    # collections and multi-volume works, so the chapter set is cleared at every
    # part boundary. Without that, a book running PART ONE/CHAPTER 1..3 then
    # PART TWO/CHAPTER 1..3 lost all three of part two's chapters and put its
    # whole body under the bare part node.
    seen_part_ordinals: set[int] = set()
    seen_chapter_ordinals: set[int] = set()
    part_count = chapter_count = section_count = roman_count = 0
    for index, row in enumerate(rows):
        title = _body_heading_title(row)
        if not title:
            continue
        block_type = str((row.get("metadata") or {}).get("block_type") or "").casefold()
        if block_type in {"heading", "page_furniture"}:
            remaining_titles[_repetition_key(title)] -= 1
            if ordinal_key := _ordinal_key(title):
                remaining_ordinals[ordinal_key] -= 1
        # A printed table of contents lists every chapter opener verbatim, so
        # reading it as body gives the book a second, spurious copy of its own
        # hierarchy in front of the real one. The guard keyed on the literal
        # 目次 and so never fired for an English book, which says "Contents".
        if TOC_RE.match(title):
            inside_printed_toc = True
            printed_toc_page = _row_page(row)
            continue
        if inside_printed_toc and (page := _row_page(row)) and printed_toc_page \
                and page > printed_toc_page + 1:
            # A printed contents occupies its own page and at most the next; in
            # all 25 candidates here that carry one, every entry sits on the
            # contents page or the one after, and the first body heading is 2 to
            # 172 pages further on. The bound is pinned to the contents heading
            # rather than moving with the region: letting it advance with each
            # row skipped meant a book with a heading on every page never
            # reached it, and the contents then ran until something else ended
            # it. In one book here nothing did until page 171, so its four real
            # part openers were read as contents and the recovered tree was the
            # back-of-book notes index instead.
            inside_printed_toc = False
        if inside_printed_toc:
            # What distinguishes a contents entry from the opener it lists is
            # that the entry comes first: the same heading occurs again further
            # on. Occurrences are counted over furniture as well as headings,
            # because an opener the extractor labelled page_furniture would
            # otherwise look like it never recurs, and the contents region would
            # swallow the whole book.
            numbered_opener = bool(PART_RE.match(title) or CHAPTER_RE.match(title))
            key = _ordinal_key(title)
            repeated_later = (
                remaining_ordinals[key] > 0 if key
                else remaining_titles[_repetition_key(title)] > 0
            )
            explicit_opener_later = any(
                remaining_titles[value] > 0 for value in ("はじめに", "序章")
            )
            body_opener = (
                title in {"はじめに", "序章"}
                or (
                    numbered_opener and _row_page(row) > printed_toc_page
                    and not repeated_later and not explicit_opener_later
                )
            )
            if body_opener and block_type in {"heading", "page_furniture"}:
                inside_printed_toc = False
            else:
                continue
        # OCR/layout extractors often label a genuine Part opener as text, and
        # they label chapter openers "page_furniture" as readily as "heading" --
        # in one book here five of fifteen chapter openers landed in each. A
        # furniture block is admitted when it speaks the structural vocabulary
        # and the document does not repeat it, which is exactly what tells an
        # opener from the running header that carries the same words on every
        # page of the chapter. Everything else stays out: incidental prose
        # references to a chapter are not boundaries.
        if block_type != "heading" and not PART_RE.match(title):
            if not (
                block_type == "page_furniture" and _is_structural(title)
                and repeated_titles[_repetition_key(title)] <= _MAX_BOUNDARY_REPEATS
            ):
                continue
        if INDEX_GROUP_RE.match(title) and not functional_scope:
            # Multi-column indexes sometimes expose their first kana-group
            # heading but lose the section title. Start at that PDF page's
            # first chunk so entries preceding the group are not assigned to
            # the previous chapter. A later running ``人名索引`` header is
            # treated as the same functional region below.
            page = _row_page(row)
            page_start = index if not page else next((
                candidate for candidate, value in enumerate(rows)
                if _row_page(value) == page
            ), index)
            part_path = []
            chapter_path = ["索引"]
            section_path = []
            functional_scope = True
            events[page_start] = (["索引"], ["chapter"])
            continue
        # Notes belonging to the chapter just opened. Only while a chapter is
        # open: a "NOTES" before any chapter is the book's own back matter.
        if chapter_path and classify_heading_path([title]) in {"endnote", "footnote"}:
            section_path = [*chapter_path, title]
            functional_scope = True
            events[index] = (list(section_path), infer_structure_roles(section_path))
            continue
        if _TOP_LEVEL_HEADING_RE.match(title):
            if title in {"索引", "人名索引"} and chapter_path == ["索引"]:
                continue
            part_path = []
            chapter_path = [title]
            section_path = []
            functional_scope = title in _FUNCTIONAL_BODY_HEADINGS
            events[index] = ([title], ["chapter"])
        elif PART_RE.match(title) and heading_ordinal(title) in seen_part_ordinals:
            # Already opened -- see the chapter case below for why a repeat is
            # a notes label rather than a boundary.
            if functional_scope and section_path:
                path = [*section_path, title]
                events[index] = (path, infer_structure_roles(path))
        elif PART_RE.match(title):
            if number := heading_ordinal(title):
                seen_part_ordinals.add(number)
            seen_chapter_ordinals.clear()
            part_path = [title]
            chapter_path = []
            section_path = []
            functional_scope = False
            part_count += 1
            part_ordinals.append(heading_ordinal(title) or 0)
            events[index] = (list(part_path), ["part"])
        elif CHAPTER_RE.match(title) and heading_ordinal(title) in seen_chapter_ordinals:
            # The book has already opened this chapter, so this is not a
            # boundary. Notes collected at the back are grouped under the
            # chapter they serve and so repeat every chapter heading -- read as
            # openers, they gave a seven-chapter book fourteen chapters. Inside
            # such a region the heading still names which chapter the notes
            # belong to, so it is kept as a child; elsewhere it is a running
            # header that the repetition test happened to let through.
            if functional_scope and section_path:
                path = [*section_path, title]
                events[index] = (path, infer_structure_roles(path))
        elif CHAPTER_RE.match(title):
            if number := heading_ordinal(title):
                seen_chapter_ordinals.add(number)
            chapter_path = [*part_path, title]
            section_path = []
            functional_scope = False
            chapter_count += 1
            chapter_ordinals.append(heading_ordinal(title) or 0)
            roles = ["part", "chapter"] if part_path else ["chapter"]
            events[index] = (list(chapter_path), roles)
        elif SECTION_RE.match(title) and chapter_path:
            section_path = [*chapter_path, title]
            functional_scope = False
            section_count += 1
            roles = infer_structure_roles(section_path)
            events[index] = (list(section_path), roles)
        elif ROMAN_SUBHEADING_RE.match(title) and not functional_scope:
            parent = section_path or chapter_path
            if parent:
                roman_count += 1
                path = [*parent, title]
                events[index] = (path, infer_structure_roles(path))

    # A numbered run with holes in it means the extractor lost some openers,
    # not that the book skips chapters. Storing 1, 3, 4, 6 asserts a
    # six-chapter book has four; four of the ten candidates measured here were
    # in that state, so the run has to be contiguous or the document stays flat.
    if not _numbering_is_contiguous(chapter_ordinals):
        return None
    if not _numbering_is_contiguous(part_ordinals):
        return None

    # Avoid promoting incidental numbered phrases. A usable book/thesis tree
    # needs either explicit Parts or several Chapters, plus multiple boundaries.
    if not ((part_count >= 1 or chapter_count >= 3) and len(events) >= 5):
        return None

    if not _divides_the_document(events, len(rows)):
        return None

    output: list[Dict[str, Any]] = []
    active_path: list[str] = []
    active_roles: list[str] = []
    changed = 0
    mapped = 0
    for index, row in enumerate(rows):
        if index in events:
            active_path, active_roles = events[index]
        metadata = dict(row.get("metadata") or {})
        before = {key: metadata.get(key) for key in STRUCTURE_METADATA_KEYS}
        if active_path:
            _set_structure_metadata(metadata, active_path, active_roles)
            zone = classify_heading_path(active_path)
            if str(metadata.get("zone") or "body") not in INTRINSIC_ZONES:
                if zone == "body":
                    metadata.pop("zone", None)
                else:
                    metadata["zone"] = zone
            mapped += 1
        else:
            _clear_source_structure_metadata(metadata)
        after = {key: metadata.get(key) for key in STRUCTURE_METADATA_KEYS}
        changed += before != after
        output.append({**row, "metadata": metadata})
    return output, {
        "attachment_key": attachment_key, "source_type": "pdf",
        "source_path": str(source_path), "chunks": len(output),
        "metadata_changed": changed, "outline_entries": 0,
        "mapping_mode": "numbered_body_headings", "mapped_chunks": mapped,
        "heading_counts": {
            "parts": part_count, "chapters": chapter_count,
            "sections": section_count, "roman_subheadings": roman_count,
            "total": len(events),
        },
    }


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


def _refresh_epub_rows_from_spine_toc(
    rows: Sequence[Dict[str, Any]], attachment_key: str, source_path: Path,
) -> tuple[list[Dict[str, Any]], Dict[str, Any]]:
    """Project TOC paths onto fixed-layout/OCR chunks using stable spine IDs."""
    entries = get_epub_chapter_index_to_toc_entries(str(source_path))
    if sum(len(values) for values in entries.values()) < 2:
        raise RuntimeError(
            f"EPUB spine TOC has fewer than two usable entries for {attachment_key}"
        )
    starts = sorted(entries)
    output: list[Dict[str, Any]] = []
    changed = 0
    mapped = 0
    for row in rows:
        metadata = dict(row.get("metadata") or {})
        match = _EPUB_SPINE_RE.match(str(metadata.get("locator") or ""))
        spine = int(match.group("spine")) if match else -1
        active = max((value for value in starts if value <= spine), default=None)
        before = {key: metadata.get(key) for key in STRUCTURE_METADATA_KEYS}
        if active is None:
            _clear_source_structure_metadata(metadata)
        else:
            candidates = entries[active]
            candidate_paths = [
                [str(value).strip() for value in candidate.get("path") or [] if str(value).strip()]
                for candidate in candidates
            ]
            path = list(candidate_paths[0]) if candidate_paths else []
            for candidate_path in candidate_paths[1:]:
                common = 0
                while (
                    common < len(path) and common < len(candidate_path)
                    and path[common] == candidate_path[common]
                ):
                    common += 1
                path = path[:common]
            if len(candidate_paths) > 1 and not path:
                raise RuntimeError(
                    f"ambiguous spine TOC has no common parent for {attachment_key} spine {active}"
                )
            if path:
                _set_structure_metadata(metadata, path)
                zone = classify_heading_path(path)
                if str(metadata.get("zone") or "body") not in INTRINSIC_ZONES:
                    if zone == "body":
                        metadata.pop("zone", None)
                    else:
                        metadata["zone"] = zone
                mapped += 1
            else:
                _clear_source_structure_metadata(metadata)
        after = {key: metadata.get(key) for key in STRUCTURE_METADATA_KEYS}
        changed += before != after
        output.append({**row, "metadata": metadata})
    if not mapped:
        raise RuntimeError(f"no existing EPUB chunks map to spine TOC for {attachment_key}")
    return output, {
        "attachment_key": attachment_key, "source_type": "epub",
        "source_path": str(source_path), "chunks": len(output),
        "metadata_changed": changed, "mapping_mode": "spine_toc",
        "toc_entries": len(entries), "mapped_chunks": mapped,
    }


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
    if not fresh:
        return _refresh_epub_rows_from_spine_toc(rows, attachment_key, source_path)
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
        path = metadata.get("structure_path") or []
        if path:
            _set_structure_metadata(metadata, path, metadata.get("structure_roles") or [])
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
        persisted = _refresh_pdf_rows_from_persisted_anchors(
            rows, attachment_key, source_path,
        )
        if persisted is not None:
            return persisted
        recovered = _refresh_pdf_rows_from_numbered_body_headings(
            rows, attachment_key, source_path,
        )
        if recovered is not None:
            return recovered
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
                _set_structure_metadata(metadata, path)
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
