# src/text_utils.py
from __future__ import annotations

import os
import re
import unicodedata
from typing import Any, Dict, List, Optional, Tuple

# Chunking controls (same env defaults as index_from_zotero.py)
MAX_CHARS = int(os.environ.get("MAX_CHARS", "1200"))
MIN_CHUNK_CHARS = int(os.environ.get("MIN_CHUNK_CHARS", "200"))

# Light overlap (characters) to improve retrieval around boundaries.
# Backward compatible:
# - If OVERLAP_CHARS is set, it acts as a global default.
# - Otherwise, we use language-sensitive defaults below.
OVERLAP_CHARS_DEFAULT = int(os.environ.get("OVERLAP_CHARS", "0"))
OVERLAP_CHARS_LATIN = int(os.environ.get("OVERLAP_CHARS_LATIN", "80"))
OVERLAP_CHARS_CJK = int(os.environ.get("OVERLAP_CHARS_CJK", "60"))

# For languages that typically do not use whitespace word segmentation (e.g., Japanese/Chinese).
MIN_CHUNK_CHARS_NO_SPACE = int(os.environ.get("MIN_CHUNK_CHARS_NO_SPACE", "120"))

# Hard minimum to avoid indexing obvious noise (page numbers, single tokens, etc.).
HARD_MIN_CHARS = int(os.environ.get("HARD_MIN_CHARS", "40"))

# Text cleaning / filtering
CONTROL_CHARS = re.compile(r"[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]")


def clean_extracted_text(s: str) -> str:
    s = CONTROL_CHARS.sub("", s)
    s = unicodedata.normalize("NFKC", s)
    s = s.replace("\uFFFD", "")
    return s


def normalize_paragraphs(raw: str, joiner: str = " ") -> List[str]:
    lines = raw.splitlines()
    paras: List[str] = []
    buf: List[str] = []

    def flush() -> None:
        nonlocal buf
        if not buf:
            return

        parts: List[str] = []
        for ln in buf:
            ln = ln.strip()
            if not ln:
                continue

            if parts and parts[-1].endswith("-"):
                parts[-1] = parts[-1][:-1] + ln
            else:
                parts.append(ln)

        merged = joiner.join(parts)
        merged = re.sub(r"\s+", " ", merged).strip()
        if merged:
            paras.append(merged)
        buf = []

    for ln in lines:
        if not ln.strip():
            flush()
        else:
            buf.append(ln)
    flush()
    return paras


def looks_like_gibberish(text: str) -> bool:
    s = re.sub(r"\s+", "", text)

    # Short excerpts (e.g., titles, headers, short notes) should not be auto-rejected.
    if len(s) < 50:
        return False

    printable_ratio = sum(ch.isprintable() for ch in s) / max(len(s), 1)
    letters_ratio = sum(ch.isalpha() for ch in s) / max(len(s), 1)
    digits_ratio = sum(ch.isdigit() for ch in s) / max(len(s), 1)

    return (printable_ratio < 0.90) or ((letters_ratio + digits_ratio) < 0.20)


# --- Heuristics for no-space (CJK) language detection ---

def _cjk_ratio(text: str) -> float:
    """Return ratio of CJK (Han/Hiragana/Katakana) characters in text."""
    if not text:
        return 0.0
    total = 0
    cjk = 0
    for ch in text:
        if ch.isspace():
            continue
        total += 1
        o = ord(ch)
        # Hiragana, Katakana, CJK Unified Ideographs (+ extensions), Halfwidth Katakana
        if (
            0x3040 <= o <= 0x30FF
            or 0x3400 <= o <= 0x4DBF
            or 0x4E00 <= o <= 0x9FFF
            or 0xF900 <= o <= 0xFAFF
            or 0xFF66 <= o <= 0xFF9D
        ):
            cjk += 1
    return cjk / max(total, 1)


def _latin_ratio(text: str) -> float:
    """Return ratio of Latin alphabet characters in text (A-Z/a-z)."""
    if not text:
        return 0.0
    total = 0
    latin = 0
    for ch in text:
        if ch.isspace():
            continue
        total += 1
        o = ord(ch)
        if (0x0041 <= o <= 0x005A) or (0x0061 <= o <= 0x007A):
            latin += 1
    return latin / max(total, 1)


def is_no_space_language_document(sample: str) -> bool:
    """Heuristic: treat as a no-whitespace language doc (e.g., Japanese/Chinese).

    NOTE:
    - We avoid using a strict `space_ratio` gate because upstream normalization may insert
      spaces when joining hard line breaks (common in PDFs/OCR), which would incorrectly
      downgrade CJK documents.
    """
    if not sample:
        return False
    s = sample.strip()
    if not s:
        return False
    s = s[:20000]

    cjk = _cjk_ratio(s)
    latin = _latin_ratio(s)

    # Conservative rule: mostly CJK and not dominated by Latin letters.
    return (cjk >= 0.20) and (latin <= 0.40)


def joiner_for_text(text: str) -> str:
    """Preferred joiner between segments for this text."""
    if not text:
        return " "
    cjk = _cjk_ratio(text)
    latin = _latin_ratio(text)
    return "" if (cjk >= 0.20 and latin <= 0.40) else " "


def overlap_for_text(text: str) -> int:
    """Overlap chars for this text (CJK vs Latin tuned)."""
    if OVERLAP_CHARS_DEFAULT and OVERLAP_CHARS_DEFAULT > 0:
        return int(OVERLAP_CHARS_DEFAULT)
    if not text:
        return int(OVERLAP_CHARS_LATIN)
    cjk = _cjk_ratio(text)
    latin = _latin_ratio(text)
    return int(OVERLAP_CHARS_CJK) if (cjk >= 0.20 and latin <= 0.40) else int(OVERLAP_CHARS_LATIN)


def split_long_paragraph(p: str, max_chars: int = MAX_CHARS) -> List[str]:
    p = p.strip()
    if len(p) <= max_chars:
        return [p]

    joiner = joiner_for_text(p)

    # Prefer splitting on punctuation followed by whitespace (works well for English).
    # If that doesn't split (common in Japanese where punctuation may not be followed by whitespace),
    # fall back to splitting on punctuation regardless of following whitespace.
    sentences = re.split(r"(?<=[\.\?\!。！？])\s+", p)
    if len(sentences) <= 1:
        sentences = re.split(r"(?<=[\.\?\!。！？])\s*", p)

    parts: List[str] = []
    cur = ""
    for s in sentences:
        s = s.strip()
        if not s:
            continue

        add_len = len(joiner) if cur else 0
        if len(cur) + add_len + len(s) <= max_chars:
            cur = (cur + joiner + s).strip() if cur else s
        else:
            if cur:
                parts.append(cur)
            cur = s
    if cur:
        parts.append(cur)

    # Light overlap between adjacent parts to improve retrieval around boundaries.
    overlap = max(0, int(overlap_for_text(p)))
    if overlap > 0 and len(parts) > 1:
        for i in range(1, len(parts)):
            prev = parts[i - 1]
            cur_part = parts[i]
            if not prev or not cur_part:
                continue
            tail = prev[-overlap:]
            keep = max(0, max_chars - len(cur_part))
            if keep <= 0:
                continue
            if len(tail) > keep:
                tail = tail[-keep:]
            parts[i] = tail + cur_part

    final_parts: List[str] = []
    for part in parts:
        if len(part) <= max_chars:
            final_parts.append(part)
        else:
            # Overlapping fixed windows for extremely long segments.
            overlap = max(0, int(overlap_for_text(p)))
            if overlap >= max_chars:
                overlap = max(0, max_chars // 4)
            step = max(1, max_chars - overlap)
            for i in range(0, len(part), step):
                window = part[i : i + max_chars].strip()
                if window:
                    final_parts.append(window)
                if i + max_chars >= len(part):
                    break

    return [x for x in final_parts if x]


def merge_short_chunk_records(
    chunks: List[Tuple[str, str, Dict[str, Any]]],
    *,
    min_chars: int,
    max_chars: int,
) -> List[Tuple[str, str, Dict[str, Any]]]:
    """Merge consecutive short chunks until they reach `min_chars`.

    - Keeps ordering.
    - Uses a paragraph-style separator (`\\n\\n`) when concatenating.
    - Preserves the first chunk's id/metadata; updates metadata when merges occur.
    - Never grows a merged chunk beyond `max_chars`.
    - Drops chunks below HARD_MIN_CHARS.
    """
    if not chunks:
        return []

    out: List[Tuple[str, str, Dict[str, Any]]] = []
    buf_id: Optional[str] = None
    buf_text: str = ""
    buf_md: Optional[Dict[str, Any]] = None
    merge_count: int = 0
    locator_end: Optional[str] = None

    def _finalize_buf() -> None:
        nonlocal buf_id, buf_text, buf_md, merge_count, locator_end
        if buf_id is None or buf_md is None:
            return
        if merge_count > 1:
            buf_md["merged"] = True
            buf_md["merge_count"] = int(merge_count)
            if locator_end:
                buf_md["locator_end"] = locator_end
        out.append((buf_id, buf_text.strip(), buf_md))
        buf_id = None
        buf_text = ""
        buf_md = None
        merge_count = 0
        locator_end = None

    sep = "\n\n"

    for cid, text, md in chunks:
        t = (text or "").strip()
        if not t:
            continue
        if len(t) < HARD_MIN_CHARS:
            continue

        if buf_id is None:
            buf_id = cid
            buf_text = t
            buf_md = md
            merge_count = 1
            locator_end = md.get("locator") if isinstance(md, dict) else None
            continue

        # If the current buffer is short, try to grow it by appending the next chunk.
        if len(buf_text) < min_chars:
            if len(buf_text) + len(sep) + len(t) <= max_chars:
                buf_text = buf_text + sep + t
                merge_count += 1
                loc = md.get("locator") if isinstance(md, dict) else None
                if isinstance(loc, str) and loc:
                    locator_end = loc
                continue

        # Otherwise, flush the buffer and start a new one.
        _finalize_buf()
        buf_id = cid
        buf_text = t
        buf_md = md
        merge_count = 1
        locator_end = md.get("locator") if isinstance(md, dict) else None

    if buf_id is not None and buf_md is not None:
        # If the last buffer is still short, try to append it to the previous chunk if it fits.
        if len(buf_text) < min_chars and out:
            prev_id, prev_text, prev_md = out[-1]
            if len(prev_text) + len(sep) + len(buf_text) <= max_chars:
                out[-1] = (prev_id, (prev_text + sep + buf_text).strip(), prev_md)
                if isinstance(prev_md, dict):
                    prev_md["merged"] = True
                    prev_md["merge_count"] = int(prev_md.get("merge_count", 1)) + 1
                    loc = locator_end
                    if isinstance(loc, str) and loc:
                        prev_md["locator_end"] = loc
            else:
                _finalize_buf()
        else:
            _finalize_buf()

    return out