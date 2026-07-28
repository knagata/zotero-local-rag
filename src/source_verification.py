# src/source_verification.py
"""Verify the index against the original files, not against another index.

Every existing check compares one index with another -- V3 with legacy, the
manifest with Chroma, an item with the same item. Two stores that are missing
the same text agree with each other, so on 2026-07-28 the cutover audit
returned ``passed: true`` while five documents had lost body text, 1,774 chunks
belonged to no item, and one document had vanished from search entirely.

The invariants here are deliberately about *presence*, not proportion. A
threshold turns a missing section into a number to be compared against a
tolerance, and three of the four documents that lost their endnotes sat inside
the tolerance (ratio 0.81-0.95). Presence has no tolerance to sit inside.

The core invariant needs the source file: a page whose text PyMuPDF can read
must have text in the index. The companion invariants need only the index, but
they close the holes that let a per-item comparison miss whole populations.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence

#: A page holding less than this is a running head, a folio, or a caption
#: fragment -- not a page of text whose absence would mean anything. Kept small
#: on purpose: this decides what counts as *evidence of text*, not what counts
#: as an acceptable loss.
MIN_SOURCE_PAGE_CHARS = 20

#: Pages whose entire text is one of these carry no content, and an extractor
#: is right to drop them. They were 193 of the first run's 539 "lost" pages --
#: 36% of the finding was the check misreading a deliberately blank page as a
#: loss (2026-07-28). A verification that cries wolf on a third of its output
#: teaches its reader to discount it, so the boilerplate is recognised rather
#: than the threshold raised: lifting the floor to cover a 34-character notice
#: would also stop the check seeing a genuinely short page of text.
BLANK_PAGE_MARKERS = (
    "this page intentionally left blank",
    "page intentionally left blank",
    "intentionally left blank",
    "this page has been intentionally left blank",
    "本ページは意図的に空白としています",
    "白紙",
)


def is_blank_page_notice(text: str) -> bool:
    """Whether a page's whole text is a notice that the page is empty."""
    normalised = " ".join(str(text or "").split()).strip().strip(".").casefold()
    if not normalised:
        return True
    return any(marker in normalised for marker in BLANK_PAGE_MARKERS) and len(normalised) <= 80


def _meaningful_char_count(text: str) -> int:
    """Count only characters that could plausibly be prose: letters or digits.

    ``len(text)`` treats a run of decorative glyphs -- bullets, dashes, box-
    drawing characters used as a page rule -- as evidence of a page of text.
    One page of roughly 1,000 "• " bullets cleared MIN_SOURCE_PAGE_CHARS
    and was reported lost when the index correctly held nothing for it
    (VU5FWYMC p1, 2026-07-28). Filtering by character class rather than
    raising the length threshold keeps a genuinely short page of real prose
    visible to the same check.
    """
    return sum(1 for ch in str(text or "") if ch.isalnum())


@dataclass
class DocumentVerdict:
    """What the source says about one attachment, next to what the index holds."""

    attachment_key: str
    source_pages: int = 0
    #: Pages whose source text is readable but which have nothing in the index.
    lost_pages: List[int] = field(default_factory=list)
    #: Pages with no readable source text and nothing indexed. Not a loss on its
    #: own -- a scanned page legitimately has no text layer -- but it is the
    #: population that OCR was supposed to cover, so it is reported separately
    #: rather than folded into either bucket.
    unreadable_pages: List[int] = field(default_factory=list)
    source_chars: int = 0
    indexed_chars: int = 0

    @property
    def failed(self) -> bool:
        return bool(self.lost_pages)


def source_page_chars(pdf_path: Path) -> Dict[int, int]:
    """Characters PyMuPDF can read from each 1-based page of the original."""
    import fitz  # imported here so the module stays importable without PyMuPDF

    result: Dict[int, int] = {}
    with fitz.open(str(pdf_path)) as document:
        for index in range(document.page_count):
            text = document[index].get_text().strip()
            # A page whose only text says the page is blank holds no content,
            # so an extractor dropping it is correct and counting it as a loss
            # is not. Reported as zero rather than filtered out, so the page
            # still appears in the page census. Length is measured in
            # meaningful (alnum) characters, not raw length, so a run of
            # decorative glyphs cannot pass as evidence of a page of text.
            result[index + 1] = 0 if is_blank_page_notice(text) else _meaningful_char_count(text)
    return result


def indexed_page_chars(rows: Iterable[Mapping[str, Any]]) -> Dict[int, int]:
    """Characters the index holds for each page of one attachment."""
    totals: Dict[int, int] = defaultdict(int)
    for row in rows:
        metadata = row.get("metadata") or {}
        try:
            page = int(metadata.get("page") or 0)
        except (TypeError, ValueError):
            continue
        if page > 0:
            totals[page] += len(str(row.get("text") or "").strip())
    return dict(totals)


def compare_document(
    attachment_key: str,
    source: Mapping[int, int],
    indexed: Mapping[int, int],
    *,
    min_source_chars: int = MIN_SOURCE_PAGE_CHARS,
) -> DocumentVerdict:
    """Pages the source can read but the index does not hold.

    The asymmetry is the point. Text in the index that is absent from the
    source layer is normal -- that is what OCR produces -- so it is never a
    failure here. Text in the source that is absent from the index is the thing
    no other check in this project can see.
    """
    verdict = DocumentVerdict(attachment_key=attachment_key, source_pages=len(source))
    for page, chars in sorted(source.items()):
        held = int(indexed.get(page) or 0)
        if chars >= min_source_chars:
            verdict.source_chars += chars
            if held == 0:
                verdict.lost_pages.append(page)
        elif held == 0:
            verdict.unreadable_pages.append(page)
    verdict.indexed_chars = sum(int(value) for value in indexed.values())
    return verdict


# --- Companion invariants (index-only) -------------------------------------
#
# These need no source file. Each one closes a hole through which a per-item
# comparison cannot see, and each is a statement about presence.


def chunks_without_item(rows: Sequence[Mapping[str, Any]]) -> List[str]:
    """Chunk ids belonging to no Zotero item.

    A per-item audit iterates over item keys, so a chunk with no ``itemKey``
    is never examined by it -- 1,774 such chunks were being served while every
    gate passed (2026-07-28). They answer vector search but cannot be reached
    by citation, by item filters, or by hierarchical routing.
    """
    return [
        str(row.get("id") or "")
        for row in rows
        if not str((row.get("metadata") or {}).get("itemKey") or "").strip()
    ]


def dangling_node_ids(
    rows: Sequence[Mapping[str, Any]], live_node_ids: Iterable[str],
) -> List[str]:
    """Chunk ids whose ``node_id`` names a node that does not exist.

    The cutover audit counts a chunk as structurally covered when the metadata
    holds a non-empty ``node_id`` string, without asking whether anything
    answers to it. 45% of chunks pointed at nodes that no longer existed, and
    53 items where *every* chunk did still scored a node coverage of 1.0.
    """
    live = set(live_node_ids)
    return [
        str(row.get("id") or "")
        for row in rows
        if str((row.get("metadata") or {}).get("node_id") or "").strip()
        and str((row.get("metadata") or {}).get("node_id")).strip() not in live
    ]


def unretrievable_documents(
    rows_by_item: Mapping[str, Sequence[Mapping[str, Any]]],
) -> List[str]:
    """Items that hold text but expose none of it to ordinary search.

    A zone heuristic applied to a whole document stops describing structure and
    removes the document: one 65,000-character essay was marked ``zone=index``
    throughout and disappeared from search while every aggregate looked healthy.
    Zone assignment is never asserted by the existing gates, and
    ``retrieval_policy`` does not appear in them at all.
    """
    failures = []
    for item_key, rows in rows_by_item.items():
        if not rows:
            continue
        if not any(str((row.get("text") or "")).strip() for row in rows):
            continue
        reachable = any(
            str((row.get("metadata") or {}).get("retrieval_policy") or "normal") == "normal"
            for row in rows
        )
        if not reachable:
            failures.append(item_key)
    return failures


__all__ = [
    "DocumentVerdict", "MIN_SOURCE_PAGE_CHARS", "chunks_without_item",
    "compare_document", "dangling_node_ids", "indexed_page_chars",
    "source_page_chars", "unretrievable_documents",
]
