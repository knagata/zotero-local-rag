# src/heading_zone.py
"""Classify a heading path into a canonical zone. The single definition.

Three extractors each carried their own version of this vocabulary --
``html_extract.py``, ``docling_extract.py``, and the AI-TOC path in
``pdf_toc_recovery.py`` -- and PyMuPDF's own path (``pdf_extract.py``, the
extractor most PDFs in this library actually go through) carried none at all.
The PyMuPDF gap alone left an estimated 10.6 million characters across 217
items classified as body text while sitting under headings like "Sources",
"Bibliography", or "Notes" (2026-07-28). The three copies disagreed on
coverage too: only this module's vocabulary recognises "Sources" or "Further
Reading" as bibliography, or "Notes to Chapter 3" as endnote rather than
requiring an exact "Notes".

``classify_heading_path`` takes the whole ancestor chain, outermost first, and
returns the first non-body zone found. A single-heading classifier that only
looks at the immediate title misses the ordinary case of a subsection nested
under a paratext section -- "Notes" containing "Chapter 3" classified the
chapter's notes as body, because only "Chapter 3" was checked. 2,715 chunks
were affected by exactly this in the EPUB/HTML path alone.
"""
from __future__ import annotations

import re
from typing import Sequence

#: Trailing "to X" / "for X" is as important as the bare word: a document that
#: numbers its note sections ("Notes to Chapter 3") or its source lists
#: ("Sources for Part Two") repeats the heading once per chapter, and a bare
#: fullmatch against "Notes" alone misses every one of them.
_TRAILING_QUALIFIER = r"(?:\s*(?:to|for)\s+.+)?"

BIBLIOGRAPHY_RE = re.compile(
    r"^(?:references?|bibliography|select\s+bibliography|further\s+reading|sources|"
    r"works\s+cited|参考文献|引用文献|文献一覧|文献目録|参考資料)" + _TRAILING_QUALIFIER + r"$",
    re.IGNORECASE,
)
ENDNOTE_RE = re.compile(
    r"^(?:notes?|endnotes?|巻末注|後注|原注|訳注|注記|注)" + _TRAILING_QUALIFIER + r"$",
    re.IGNORECASE,
)
FOOTNOTE_RE = re.compile(r"^(?:footnotes?|脚注|註)$", re.IGNORECASE)
TOC_RE = re.compile(r"^(?:目次|contents?|table\s*of\s*contents)$", re.IGNORECASE)
INDEX_RE = re.compile(
    r"^(?:索引|人名索引|事項索引|地名索引|subject\s*index|name\s*index|general\s*index|index)$",
    re.IGNORECASE,
)
COLOPHON_RE = re.compile(r"^(?:奥付|colophon|copyright)$", re.IGNORECASE)
FRONT_MATTER_RE = re.compile(r"^(?:凡例|謝辞|序文|まえがき|acknowledg(?:e)?ments?)$", re.IGNORECASE)
BACK_MATTER_RE = re.compile(
    r"^(?:あとがき|後書き|付録|付表|appendix|appendices|afterword|glossary|用語集)$",
    re.IGNORECASE,
)

#: Order matters only in that each pattern is mutually exclusive by design;
#: first match wins, but no heading should match two of these.
_ORDERED_PATTERNS = (
    (BIBLIOGRAPHY_RE, "bibliography"),
    (FOOTNOTE_RE, "footnote"),
    (ENDNOTE_RE, "endnote"),
    (TOC_RE, "toc"),
    (INDEX_RE, "index"),
    (COLOPHON_RE, "colophon"),
    (FRONT_MATTER_RE, "front_matter"),
    (BACK_MATTER_RE, "back_matter"),
)


def classify_heading(title: str) -> str:
    """Zone implied by a single heading's text, or "body" if none matches."""
    normalized = " ".join(str(title or "").split())
    if not normalized:
        return "body"
    for pattern, zone in _ORDERED_PATTERNS:
        if pattern.match(normalized):
            return zone
    return "body"


def classify_heading_path(path: Sequence[str]) -> str:
    """Zone implied by a heading path, checking ancestors before the leaf.

    A leaf-only check misses "Notes" > "Chapter 3": the chapter title carries
    no paratext vocabulary of its own, but everything under "Notes" is a note.
    Checking outermost-first also means a genuine chapter titled "Sources of
    Power" is not misread as a bibliography merely because a later, unrelated
    ancestor happens to say "Sources" -- each level is judged independently
    and the first paratext match, from the top, wins.
    """
    for title in path:
        zone = classify_heading(title)
        if zone != "body":
            return zone
    return "body"


__all__ = [
    "BACK_MATTER_RE", "BIBLIOGRAPHY_RE", "COLOPHON_RE", "ENDNOTE_RE", "FOOTNOTE_RE",
    "FRONT_MATTER_RE", "INDEX_RE", "TOC_RE", "classify_heading", "classify_heading_path",
]
