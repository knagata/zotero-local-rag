"""Refresh source-derived heading metadata without replacing chunk text or vectors."""
from __future__ import annotations

import os
import json
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Sequence

try:
    from .chapter_detect import (
        build_pdf_page_structure_path_lookup, get_epub_chapter_index_to_toc_entries,
        get_pdf_toc, infer_structure_roles,
    )
    from .document_structure import (
        HEADING_BLOCK_TYPES, HEADING_CANDIDATE_BLOCK_TYPES, _attachment_key,
    )
    from .heading_structure import (
        CHAPTER_RE, INDEX_GROUP_RE, PART_RE, ROMAN_SUBHEADING_RE, SECTION_RE,
        normalize as normalize_heading, ordinal as heading_ordinal,
    )
    from .heading_zone import TOC_RE, classify_heading_path
    from .html_extract import extract_chunks_from_epub_snapshot
    from .manifest import load_manifest
    from .mistral_ocr_extract import extract_chunks_from_mistral_ocr_result
    from .ocr_cache import MISTRAL_REQUEST_CONTRACT, load_result_any_model, source_digest
    from .pdf_extract import _outline_events_by_page, _resolve_record_structure_paths
    from .pdf_toc_recovery import HeadingAnchor, apply_anchors
    from .v3_data_plane import manifest_path
except ImportError:  # pragma: no cover
    from chapter_detect import (
        build_pdf_page_structure_path_lookup, get_epub_chapter_index_to_toc_entries,
        get_pdf_toc, infer_structure_roles,
    )
    from document_structure import (
        HEADING_BLOCK_TYPES, HEADING_CANDIDATE_BLOCK_TYPES, _attachment_key,
    )
    from heading_structure import (
        CHAPTER_RE, INDEX_GROUP_RE, PART_RE, ROMAN_SUBHEADING_RE, SECTION_RE,
        normalize as normalize_heading, ordinal as heading_ordinal,
    )
    from heading_zone import TOC_RE, classify_heading_path
    from html_extract import extract_chunks_from_epub_snapshot
    from manifest import load_manifest
    from mistral_ocr_extract import extract_chunks_from_mistral_ocr_result
    from ocr_cache import MISTRAL_REQUEST_CONTRACT, load_result_any_model, source_digest
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
#: Named by document_structure so the module that decides which documents are
#: worth recovering and the module that recovers them cannot disagree about
#: what a heading block is. They did: the diagnostics counted section_header,
#: title, chapter and chapter_title, none of which this module looked at, so an
#: attachment could be listed as a recovery candidate on the strength of blocks
#: the recovery would then ignore.
_HEADING_BLOCKS = HEADING_CANDIDATE_BLOCK_TYPES
#: Openers that carry no number and so cannot be matched by level and ordinal.
_UNNUMBERED_OPENERS = ("はじめに", "序章")


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


def _block_type(row: Dict[str, Any]) -> str:
    return str((row.get("metadata") or {}).get("block_type") or "").casefold()


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


def _refresh_pdf_rows_from_mistral_cache(
    rows: Sequence[Dict[str, Any]], attachment_key: str, source_path: Path,
) -> tuple[list[Dict[str, Any]], Dict[str, Any]] | None:
    """Replay paid OCR structure only when it maps exactly onto stored chunks.

    A structure-only maintenance run must not call a hosted service.  The raw
    Mistral response is content-addressed, so it can be parsed again locally;
    requiring identical chunk ids keeps this a metadata repair rather than a
    hidden rechunk or re-embedding operation.
    """
    engines = {
        str((row.get("metadata") or {}).get("extraction_engine") or "").casefold()
        for row in rows
    }
    if not any("mistral" in engine for engine in engines):
        return None

    try:
        digest = source_digest(source_path)
    except OSError:
        return None
    found = load_result_any_model(
        _PROJECT_ROOT / "data", engine="mistral_ocr",
        contract_version=MISTRAL_REQUEST_CONTRACT, digest=digest,
    )
    if found is None:
        return None
    model, result = found
    first = dict(rows[0].get("metadata") or {})
    meta_base = {
        key: first[key]
        for key in ("itemKey", "attachmentKey", "source_type", "title", "filename")
        if key in first
    }
    fresh, _quality = extract_chunks_from_mistral_ocr_result(
        source_path, attachment_key, meta_base, result, model=model,
    )
    fresh_by_id = {chunk_id: metadata for chunk_id, _text, metadata in fresh}
    stored_ids = {str(row.get("id") or "") for row in rows}
    if not stored_ids or set(fresh_by_id) != stored_ids:
        return None

    output: list[Dict[str, Any]] = []
    changed = mapped = 0
    paths: set[tuple[str, ...]] = set()
    for row in rows:
        metadata = dict(row.get("metadata") or {})
        before = {key: metadata.get(key) for key in STRUCTURE_METADATA_KEYS}
        source_metadata = fresh_by_id[str(row.get("id") or "")]
        for key in STRUCTURE_METADATA_KEYS:
            if key in source_metadata:
                metadata[key] = source_metadata[key]
            else:
                metadata.pop(key, None)
        if metadata.get("structure_path"):
            mapped += 1
            paths.add(tuple(str(value) for value in metadata["structure_path"]))
        after = {key: metadata.get(key) for key in STRUCTURE_METADATA_KEYS}
        changed += before != after
        output.append({**row, "metadata": metadata})
    if mapped != len(output):
        return None
    return output, {
        "attachment_key": attachment_key, "source_type": "pdf",
        "source_path": str(source_path), "chunks": len(output),
        "metadata_changed": changed, "outline_entries": len(paths),
        "mapping_mode": "mistral_ocr_cache_exact_ids", "mapped_chunks": mapped,
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


@dataclass
class _HeadingCensus:
    """How often the document carries each heading, and where its pages start.

    Two different questions are asked of the same tally, and keeping them apart
    is the point. *How often in total* separates a chapter opener from the
    running header that repeats its words on every page of the chapter. *How
    many are still ahead* separates a printed contents entry from the opener it
    points at, since the entry always comes first.
    """

    total: Counter
    remaining: Counter
    remaining_ordinals: Counter
    first_row_on_page: Dict[int, int]

    @classmethod
    def of(cls, rows: Sequence[Dict[str, Any]]) -> "_HeadingCensus":
        total: Counter = Counter()
        ordinals: Counter = Counter()
        first_row_on_page: Dict[int, int] = {}
        for index, row in enumerate(rows):
            first_row_on_page.setdefault(_row_page(row), index)
            title = _body_heading_title(row)
            if not title or _block_type(row) not in _HEADING_BLOCKS:
                continue
            total[_repetition_key(title)] += 1
            if key := _ordinal_key(title):
                ordinals[key] += 1
        return cls(total, Counter(total), ordinals, first_row_on_page)

    def passing(self, title: str) -> None:
        """Record that the walk has reached this heading."""
        self.remaining[_repetition_key(title)] -= 1
        if key := _ordinal_key(title):
            self.remaining_ordinals[key] -= 1

    def repeats(self, title: str) -> bool:
        """Whether the document carries this heading too often to be a boundary."""
        return self.total[_repetition_key(title)] > _MAX_BOUNDARY_REPEATS

    def recurs_later(self, title: str) -> bool:
        """Whether this heading appears again further on.

        Matched on the level and number where the heading carries one: a
        contents page says "PART II" where the body page says "PART TWO", and
        the two share no text at all.
        """
        if key := _ordinal_key(title):
            return self.remaining_ordinals[key] > 0
        return self.remaining[_repetition_key(title)] > 0

    def unnumbered_opener_ahead(self) -> bool:
        return any(self.remaining[value] > 0 for value in _UNNUMBERED_OPENERS)


class _PrintedContents:
    """The region a printed table of contents occupies.

    It lists every opener in the book, so read as body it gives the document a
    second, spurious copy of its own hierarchy in front of the real one.
    """

    def __init__(self) -> None:
        self.inside = False
        self.heading_page = 0

    def opens_at(self, title: str, page: int) -> bool:
        # The guard keyed on the literal 目次 and so never fired for an English
        # book, which says "Contents".
        if not TOC_RE.match(title):
            return False
        self.inside, self.heading_page = True, page
        return True

    def holds(
        self, title: str, page: int, block_type: str, census: _HeadingCensus,
    ) -> bool:
        """Whether this row belongs to the contents rather than to the body."""
        if not self.inside:
            return False
        # A printed contents occupies its own page and at most the next; in all
        # 25 candidates here that carry one, every entry sits on the contents
        # page or the one after, and the first body heading is 2 to 172 pages
        # further on. The bound is pinned to the contents heading rather than
        # moving with the region: letting it advance with each row skipped meant
        # a book with a heading on every page never reached it, and the contents
        # then ran until something else ended it. In one book here nothing did
        # until page 171, so its four real part openers were read as contents
        # and the recovered tree was the back-of-book notes index instead.
        if page and self.heading_page and page > self.heading_page + 1:
            self.inside = False
            return False
        opener = title in _UNNUMBERED_OPENERS or (
            bool(PART_RE.match(title) or CHAPTER_RE.match(title))
            and page > self.heading_page
            and not census.recurs_later(title)
            and not census.unnumbered_opener_ahead()
        )
        if opener and block_type in _HEADING_BLOCKS:
            self.inside = False
            return False
        return True


def _admits_as_boundary(title: str, block_type: str, census: _HeadingCensus) -> bool:
    """Whether a row may be considered for a boundary at all.

    OCR and layout extractors often file a genuine Part opener as plain text,
    and they label chapter openers "page_furniture" as readily as "heading" --
    in one book here five of fifteen chapter openers landed in each. Furniture
    is admitted when it speaks the structural vocabulary and the document does
    not repeat it, which is exactly what tells an opener from the running header
    carrying the same words on every page of its chapter. Everything else stays
    out: an incidental prose reference to a chapter is not a boundary.
    """
    if block_type in HEADING_BLOCK_TYPES or PART_RE.match(title):
        return True
    return (
        block_type == "page_furniture"
        and _is_structural(title)
        and not census.repeats(title)
    )


class _RecoveredTree:
    """Where each admitted heading goes, and whether the result is usable.

    Separated from the walk that feeds it because the placement rules interact
    only through this state. They used to share fourteen locals with the reading
    of the rows, the counting of repetitions and the contents region, and a rule
    added to one of those reached all of them.
    """

    def __init__(self) -> None:
        self.events: Dict[int, tuple[list[str], list[str]]] = {}
        self._part_path: list[str] = []
        self._chapter_path: list[str] = []
        self._section_path: list[str] = []
        self._functional_scope = False
        # Which numbers have already opened something. Parts are numbered once
        # across the book; chapter numbers restart inside each part in many
        # edited collections and multi-volume works, so the chapter set is
        # cleared at every part boundary. Without that, a book running PART
        # ONE/CHAPTER 1..3 then PART TWO/CHAPTER 1..3 lost all three of part
        # two's chapters and put its whole body on the bare part node.
        self._seen_parts: set[int] = set()
        self._seen_chapters: set[int] = set()
        self._part_ordinals: list[int] = []
        self._chapter_ordinals: list[int] = []
        self.counts = {"parts": 0, "chapters": 0, "sections": 0, "roman": 0}

    def _record(self, index: int, path: Sequence[str], roles: Sequence[str] | None = None) -> None:
        path = list(path)
        self.events[index] = (path, list(roles) if roles else infer_structure_roles(path))

    def _as_notes_label(self, index: int, title: str) -> None:
        """Keep a heading that names, rather than opens, the region it sits in.

        Notes collected at the back of a book are grouped under the chapter they
        serve and so repeat every chapter heading. Read as openers they gave a
        seven-chapter book fourteen chapters; the heading still says which
        chapter the notes belong to, so it is kept as a child. Outside such a
        region a repeat is a running header the repetition test let through, and
        is dropped.
        """
        if self._functional_scope and self._section_path:
            self._record(index, [*self._section_path, title])

    def open_index(self, index: int) -> None:
        self._part_path, self._chapter_path, self._section_path = [], ["索引"], []
        self._functional_scope = True
        self._record(index, ["索引"], ["chapter"])

    def place(self, index: int, title: str) -> None:
        if self._chapter_path and classify_heading_path([title]) in {"endnote", "footnote"}:
            # Notes belonging to the chapter just opened. Only while a chapter
            # is open: a "NOTES" before any chapter is the book's own back
            # matter.
            self._section_path = [*self._chapter_path, title]
            self._functional_scope = True
            self._record(index, self._section_path)
        elif _TOP_LEVEL_HEADING_RE.match(title):
            if title in {"索引", "人名索引"} and self._chapter_path == ["索引"]:
                return
            self._part_path, self._chapter_path, self._section_path = [], [title], []
            self._functional_scope = title in _FUNCTIONAL_BODY_HEADINGS
            self._record(index, [title], ["chapter"])
        elif PART_RE.match(title):
            if heading_ordinal(title) in self._seen_parts:
                self._as_notes_label(index, title)
                return
            if number := heading_ordinal(title):
                self._seen_parts.add(number)
            self._seen_chapters.clear()
            self._part_path, self._chapter_path, self._section_path = [title], [], []
            self._functional_scope = False
            self.counts["parts"] += 1
            self._part_ordinals.append(heading_ordinal(title) or 0)
            self._record(index, self._part_path, ["part"])
        elif CHAPTER_RE.match(title):
            if heading_ordinal(title) in self._seen_chapters:
                self._as_notes_label(index, title)
                return
            if number := heading_ordinal(title):
                self._seen_chapters.add(number)
            self._chapter_path = [*self._part_path, title]
            self._section_path = []
            self._functional_scope = False
            self.counts["chapters"] += 1
            self._chapter_ordinals.append(heading_ordinal(title) or 0)
            self._record(index, self._chapter_path,
                         ["part", "chapter"] if self._part_path else ["chapter"])
        elif SECTION_RE.match(title) and self._chapter_path:
            self._section_path = [*self._chapter_path, title]
            self._functional_scope = False
            self.counts["sections"] += 1
            self._record(index, self._section_path)
        elif ROMAN_SUBHEADING_RE.match(title) and not self._functional_scope:
            parent = self._section_path or self._chapter_path
            if parent:
                self.counts["roman"] += 1
                self._record(index, [*parent, title])

    @property
    def in_index_region(self) -> bool:
        return self._functional_scope and self._chapter_path == ["索引"]

    def is_usable(self, total_rows: int) -> bool:
        """Whether this tree is worth storing in place of a flat document."""
        # A numbered run with holes in it means the extractor lost some openers,
        # not that the book skips chapters. Storing 1, 3, 4, 6 asserts a
        # six-chapter book has four; four of the ten candidates measured here
        # were in that state.
        if not _numbering_is_contiguous(self._chapter_ordinals):
            return False
        if not _numbering_is_contiguous(self._part_ordinals):
            return False
        # Avoid promoting incidental numbered phrases. A usable book or thesis
        # tree needs either explicit Parts or several Chapters, plus multiple
        # boundaries.
        if not ((self.counts["parts"] >= 1 or self.counts["chapters"] >= 3)
                and len(self.events) >= 5):
            return False
        return _divides_the_document(self.events, total_rows)


def _apply_events(
    rows: Sequence[Dict[str, Any]], events: Dict[int, tuple[list[str], list[str]]],
) -> tuple[list[Dict[str, Any]], int, int]:
    """Write each boundary's path onto the chunks that follow it."""
    output: list[Dict[str, Any]] = []
    active_path: list[str] = []
    active_roles: list[str] = []
    changed = mapped = 0
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
    return output, changed, mapped


def _refresh_pdf_rows_from_numbered_body_headings(
    rows: Sequence[Dict[str, Any]], attachment_key: str, source_path: Path,
) -> tuple[list[Dict[str, Any]], Dict[str, Any]] | None:
    """Recover a conservative TOC-level hierarchy from body headings.

    Language-neutral: the part/chapter/section vocabulary comes from
    heading_structure, so an English book is read the same way a Japanese one
    is.

    Four questions are asked of each row, and they are kept apart because they
    used to be one 250-line loop over fourteen shared locals, where a rule added
    for one of them reached all four. Does the document repeat this heading
    (`_HeadingCensus`); is the row still inside a printed contents
    (`_PrintedContents`); may it be a boundary at all (`_admits_as_boundary`);
    and where does it belong (`_RecoveredTree`).
    """
    census = _HeadingCensus.of(rows)
    contents = _PrintedContents()
    tree = _RecoveredTree()
    for index, row in enumerate(rows):
        title = _body_heading_title(row)
        if not title:
            continue
        block_type = _block_type(row)
        page = _row_page(row)
        if block_type in _HEADING_BLOCKS:
            census.passing(title)
        if contents.opens_at(title, page):
            continue
        if contents.holds(title, page, block_type, census):
            continue
        if not _admits_as_boundary(title, block_type, census):
            continue
        if INDEX_GROUP_RE.match(title) and not tree.in_index_region:
            # Multi-column indexes sometimes expose their first kana-group
            # heading but lose the section title. Start at that PDF page's first
            # chunk so entries preceding the group are not assigned to the
            # previous chapter. A later running 人名索引 header is treated as
            # the same functional region.
            tree.open_index(census.first_row_on_page.get(page, index) if page else index)
            continue
        tree.place(index, title)

    if not tree.is_usable(len(rows)):
        return None

    output, changed, mapped = _apply_events(rows, tree.events)
    return output, {
        "attachment_key": attachment_key, "source_type": "pdf",
        "source_path": str(source_path), "chunks": len(output),
        "metadata_changed": changed, "outline_entries": 0,
        "mapping_mode": "numbered_body_headings", "mapped_chunks": mapped,
        "heading_counts": {
            "parts": tree.counts["parts"], "chapters": tree.counts["chapters"],
            "sections": tree.counts["sections"],
            "roman_subheadings": tree.counts["roman"],
            "total": len(tree.events),
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
        cached = _refresh_pdf_rows_from_mistral_cache(rows, attachment_key, source_path)
        if cached is not None:
            return cached
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
