"""Build reference candidates from canonical V3 ``zone=bibliography`` chunks.

This replaces the legacy ``epub_reference_extractor`` double-parse (R6): the
body was already parsed into structured chunks at ingestion, and bibliography
blocks carry ``zone == 'bibliography'``.  Reusing those chunks keeps the body
and reference views consistent and drops a second EPUB DOM parse.

The output contract matches ``extract_epub_references`` so
``citation_mapper.map_item_local_references`` can consume it unchanged:
``raw_reference_text``, ``context_snippet``, ``citing_chunk_id``,
``similarity_distance``, ``source_zone``, ``cite_count``.
"""
from __future__ import annotations

import re
from typing import Any, Dict, List

try:
    from .chunk_store import get_item_chunks
except ImportError:  # pragma: no cover
    from chunk_store import get_item_chunks


MIN_REFERENCE_CHARS = 20
# Zones whose chunks carry outgoing reference/citation candidates.
_REFERENCE_ZONES = {"bibliography", "endnote", "footnote"}
# A new reference entry typically starts at the left margin; wrapped continuation
# lines are indented or begin lowercase / with punctuation.  We merge lines that
# look like continuations back into the preceding entry.
_CONTINUATION_RE = re.compile(r"^[\s]+|^[a-z぀-ヿ）」』】、,.;:]")
# A run-on numbered list ("1. ... 2. ... 3. ...") — common for PDF endnotes/
# numbered references where line breaks were flattened during chunking.  Only a
# sequential ascending run triggers a split, so incidental numbers (page refs,
# years) inside a single reference never over-split it (E2a, dev-notes 77).
_NUMBERED_MARKER_RE = re.compile(r"(?:^|\s)(\d{1,3})[.)]\s+(?=\S)")


def _split_numbered_entries(text: str) -> List[str] | None:
    """Split a flattened numbered list on sequential markers, else return None."""
    matches = list(_NUMBERED_MARKER_RE.finditer(text))
    if len(matches) < 2:
        return None
    numbers = [int(match.group(1)) for match in matches]
    ascending = sum(1 for a, b in zip(numbers, numbers[1:]) if b == a + 1)
    if ascending < max(1, len(matches) - 2):
        return None
    entries: List[str] = []
    for index, match in enumerate(matches):
        start = match.start(1)
        end = matches[index + 1].start(1) if index + 1 < len(matches) else len(text)
        entries.append(" ".join(text[start:end].split()))
    kept = [entry for entry in entries if len(entry) >= MIN_REFERENCE_CHARS]
    return kept or None


def _split_reference_lines(text: str) -> List[str]:
    """Split a bibliography/notes block into one string per reference entry."""
    entries: List[str] = []
    for raw_line in str(text or "").splitlines():
        line = raw_line.rstrip()
        if not line.strip():
            continue
        if entries and _CONTINUATION_RE.match(raw_line):
            entries[-1] = f"{entries[-1]} {line.strip()}"
        else:
            entries.append(line.strip())
    result: List[str] = []
    for entry in entries:
        if len(entry) < MIN_REFERENCE_CHARS:
            continue
        # A single line may still pack a whole flattened numbered list.
        numbered = _split_numbered_entries(entry)
        result.extend(numbered if numbered else [entry])
    return result


def extract_references_from_chunks(
    item_key: str, *, collection_name: str | None = None,
) -> List[Dict[str, Any]]:
    """Return reference candidates from an item's reference/note chunks.

    Bibliography, endnote and footnote zone chunks are split into one entry per
    reference/note.  Each candidate is anchored to a body chunk id
    (``citing_chunk_id``) with no similarity search: bibliography entries point at
    their own block (a standalone list), while notes reuse the deterministic
    ``citing_chunk_id`` recorded from the body ``noteref`` at ingestion (Part C,
    dev-notes/current/77).  Falls back to the note's own chunk id when unlinked.
    """
    chunks = get_item_chunks(item_key, collection_name=collection_name)
    references: List[Dict[str, Any]] = []
    seen: set[str] = set()
    for chunk in chunks:
        metadata = chunk.get("metadata") or {}
        zone = metadata.get("zone")
        if zone not in _REFERENCE_ZONES:
            continue
        chunk_id = str(chunk.get("id") or "")
        # Notes cite from the body chunk recorded at ingestion; a bibliography
        # list has no single citer, so it anchors to its own block.
        cited_from = str(metadata.get("citing_chunk_id") or "") if zone != "bibliography" else ""
        citing_chunk_id = cited_from or chunk_id
        for entry in _split_reference_lines(chunk.get("text") or ""):
            key = " ".join(entry.casefold().split())
            if key in seen:
                continue
            seen.add(key)
            references.append({
                "context_snippet": entry,
                "raw_reference_text": entry,
                "citing_chunk_id": citing_chunk_id,
                "similarity_distance": 0.0,
                "cite_count": 1,
                "source_zone": zone,
            })
    return references


__all__ = ["extract_references_from_chunks"]
