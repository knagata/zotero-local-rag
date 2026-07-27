# src/text_utils.py
from __future__ import annotations

import os
import re
import unicodedata
from typing import Any, Callable, Dict, Hashable, Iterable, List, Optional, Tuple

# -- Chunking limits: model-aware with env-var overrides --

def _resolve_chunk_limits():
    """Return (max_latin, target_latin, max_cjk, target_cjk) based on EMB_PROFILE.

    Limits are derived from the natural semantic unit of each writing system
    (1–2 paragraphs), NOT from character counts alone.  Japanese packs roughly
    twice the information per character compared to English, so CJK limits are
    proportionally lower.

    The model's token budget acts only as a **hard ceiling** — it constrains
    CJK on token-tight models (MiniLM) but does not inflate limits on models
    with abundant headroom.  Explicit env vars always take precedence.

    Semantic-unit rationale (per 1–2 paragraphs):
      English:  ~500–1000 chars  →  target 800, max 1200
      Japanese: ~200– 500 chars  →  target 400, max  600
      (ratio ≈ 2:1, matching information-density difference)
    """
    profile = os.environ.get("EMB_PROFILE", "fast").strip().lower()

    if profile == "bge":
        token_budget = 8000   # BGE-M3: 8192 tokens
    else:
        token_budget = 400    # MiniLM: 512 tokens, ~400 effective

    # --- Latin (English) ---
    # 800 chars ≈ 1–2 paragraphs.  Model token budget is never the bottleneck.
    if "MAX_CHARS" in os.environ:
        max_latin = int(os.environ["MAX_CHARS"])
    else:
        max_latin = 1200

    if "TARGET_CHARS" in os.environ:
        target_latin = int(os.environ["TARGET_CHARS"])
    else:
        target_latin = 800

    # --- CJK (Japanese) ---
    # 400 chars ≈ 1–2 dense paragraphs.  Constrained by token budget on MiniLM.
    _best_max_cjk = 600   # semantic ceiling (≈2–3 Japanese paragraphs)
    _best_target_cjk = 400  # semantic sweet spot (≈1–2 Japanese paragraphs)

    if "MAX_CHARS_CJK" in os.environ:
        max_cjk = int(os.environ["MAX_CHARS_CJK"])
    else:
        # Model ceiling may force us below the semantic ceiling
        max_cjk = min(_best_max_cjk, int(token_budget * 1.0))

    if "TARGET_CHARS_CJK" in os.environ:
        target_cjk = int(os.environ["TARGET_CHARS_CJK"])
    else:
        target_cjk = min(_best_target_cjk, int(max_cjk * 0.7))

    return max_latin, target_latin, max_cjk, target_cjk


MAX_CHARS, TARGET_CHARS, MAX_CHARS_CJK, TARGET_CHARS_CJK = _resolve_chunk_limits()
MIN_CHUNK_CHARS = int(os.environ.get("MIN_CHUNK_CHARS", "200"))

# Overlap: ~12% of target size, approximating one sentence of context
# at chunk boundaries.  OVERLAP_CHARS overrides everything.
_OVERLAP_RATIO = 0.12
_OVERLAP_LATIN = max(20, int(TARGET_CHARS * _OVERLAP_RATIO))
_OVERLAP_CJK = max(15, int(TARGET_CHARS_CJK * _OVERLAP_RATIO))
OVERLAP_CHARS_DEFAULT = int(os.environ.get("OVERLAP_CHARS", "0"))
OVERLAP_CHARS_LATIN = int(os.environ.get("OVERLAP_CHARS_LATIN", str(_OVERLAP_LATIN)))
OVERLAP_CHARS_CJK = int(os.environ.get("OVERLAP_CHARS_CJK", str(_OVERLAP_CJK)))

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


def detect_lang(text: str, hint: Optional[str] = None) -> str:
    """Return ``ja``, ``zh``, ``en``, or ``other`` using metadata then script ratios."""
    if hint:
        normalized = unicodedata.normalize("NFKC", hint).strip().casefold()
        primary = re.split(r"[-_\s]", normalized, maxsplit=1)[0]
        aliases = {
            "ja": "ja", "jpn": "ja", "japanese": "ja", "日本語": "ja",
            "zh": "zh", "zho": "zh", "chi": "zh", "chinese": "zh", "中国語": "zh",
            "en": "en", "eng": "en", "english": "en", "英語": "en",
        }
        if primary in aliases:
            return aliases[primary]
        for alias, code in aliases.items():
            if alias in normalized:
                return code

    sample = (text or "")[:20000]
    if not sample.strip():
        return "other"
    kana = sum(
        1
        for char in sample
        if 0x3040 <= ord(char) <= 0x30FF or 0xFF66 <= ord(char) <= 0xFF9D
    )
    han = sum(
        1
        for char in sample
        if 0x3400 <= ord(char) <= 0x4DBF
        or 0x4E00 <= ord(char) <= 0x9FFF
        or 0xF900 <= ord(char) <= 0xFAFF
    )
    significant = max(sum(not char.isspace() for char in sample), 1)
    if kana / significant >= 0.01:
        return "ja"
    if han / significant >= 0.15:
        return "zh"
    if _latin_ratio(sample) >= 0.40:
        return "en"
    return "other"


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


def split_long_paragraph(p: str, max_chars: int = MAX_CHARS, target_chars: int = TARGET_CHARS) -> List[str]:
    """Split a long paragraph into chunks of roughly *target_chars*, never exceeding *max_chars*.

    Splits at sentence boundaries (punctuation) where possible.  Targeting
    a size below *max_chars* leaves headroom for ``merge_short_chunk_records``
    to absorb short neighbouring chunks (headers, captions, footnotes) into
    the surrounding text, producing more even final chunk sizes.
    """
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

    # Build parts up to target_chars (not max_chars) so neighbouring
    # short chunks can be merged in later.
    limit = max(target_chars, 1)
    parts: List[str] = []
    cur = ""
    for s in sentences:
        s = s.strip()
        if not s:
            continue

        add_len = len(joiner) if cur else 0
        if len(cur) + add_len + len(s) <= limit:
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
    boundary_key: Optional[Callable[[str, str, Dict[str, Any]], Hashable]] = None,
    union_keys: Optional[Iterable[str]] = None,
) -> List[Tuple[str, str, Dict[str, Any]]]:
    """Merge consecutive short chunks into balanced, evenly-sized chunks.

    Two-pass greedy approach:
    1. Forward pass: accumulate consecutive chunks into a buffer until it
       reaches min_chars (or next chunk would exceed max_chars).
    2. Backward pass: scan the output and merge any remaining short chunk
       into its preceding neighbour when the combined size fits within
       max_chars.  This eliminates isolated short chunks that are sandwiched
       between long chunks.

    - Never grows a merged chunk beyond *max_chars*.
    - Drops only *isolated* chunks below HARD_MIN_CHARS, after giving adjacent
      records with the same structural boundary a chance to merge.
    - ``union_keys`` names metadata keys whose list values are unioned
      (preserving source order, de-duplicated) into the surviving chunk's
      metadata whenever two chunks merge.  Used by the EPUB/HTML extractor to
      keep per-block ``element_ids`` / ``noteref_targets`` evidence across a
      merge so footnote/endnote body links (Part C, dev-notes/current/77) can
      still be resolved afterwards.
    """
    if not chunks:
        return []

    union_key_list = list(union_keys or [])

    def _union_into(dst_md: Any, src_md: Any) -> None:
        if not union_key_list or not isinstance(dst_md, dict) or not isinstance(src_md, dict):
            return
        for key in union_key_list:
            src_val = src_md.get(key)
            if not src_val:
                continue
            dst_val = dst_md.get(key)
            merged = list(dst_val) if isinstance(dst_val, list) else ([] if not dst_val else list(dst_val))
            seen = set(merged)
            for item in src_val:
                if item not in seen:
                    merged.append(item)
                    seen.add(item)
            dst_md[key] = merged

    def _key(cid: str, text: str, md: Dict[str, Any]) -> Hashable:
        return boundary_key(cid, text, md) if boundary_key is not None else "__all__"

    # --- Forward pass ---
    out: List[Tuple[str, str, Dict[str, Any]]] = []
    buf_id: Optional[str] = None
    buf_text: str = ""
    buf_md: Optional[Dict[str, Any]] = None
    merge_count: int = 0
    locator_end: Optional[str] = None
    buf_key: Hashable = "__all__"

    def _emit(
        cid: str, text: str, md: Dict[str, Any],
        count: int, loc_end: Optional[str],
    ) -> None:
        if count > 1:
            md["merged"] = True
            md["merge_count"] = int(count)
            if loc_end:
                md["locator_end"] = loc_end
        out.append((cid, text.strip(), md))

    sep = "\n\n"

    for cid, text, md in chunks:
        t = (text or "").strip()
        if not t:
            continue
        if buf_id is None:
            buf_id = cid
            buf_text = t
            buf_md = md
            merge_count = 1
            locator_end = md.get("locator") if isinstance(md, dict) else None
            buf_key = _key(cid, t, md)
            continue

        # If the buffer is short, try to grow it by appending the next chunk.
        if buf_key == _key(cid, t, md) and len(buf_text) < min_chars:
            if len(buf_text) + len(sep) + len(t) <= max_chars:
                buf_text = buf_text + sep + t
                _union_into(buf_md, md)
                merge_count += 1
                loc = md.get("locator") if isinstance(md, dict) else None
                if isinstance(loc, str) and loc:
                    locator_end = loc
                continue

        # Buffer is full enough — emit and start a new one.
        _emit(buf_id, buf_text, buf_md, merge_count, locator_end)
        buf_id = cid
        buf_text = t
        buf_md = md
        merge_count = 1
        locator_end = md.get("locator") if isinstance(md, dict) else None
        buf_key = _key(cid, t, md)

    # Final buffer
    if buf_id is not None and buf_md is not None:
        # Try backward merge first
        if len(buf_text) < min_chars and out:
            prev_id, prev_text, prev_md = out[-1]
            if _key(prev_id, prev_text, prev_md) == buf_key and len(prev_text) + len(sep) + len(buf_text) <= max_chars:
                merged_text = (prev_text + sep + buf_text).strip()
                prev_md["merged"] = True
                prev_md["merge_count"] = int(prev_md.get("merge_count", 1)) + 1
                if locator_end:
                    prev_md["locator_end"] = locator_end
                _union_into(prev_md, buf_md)
                out[-1] = (prev_id, merged_text, prev_md)
            else:
                _emit(buf_id, buf_text, buf_md, merge_count, locator_end)
        else:
            _emit(buf_id, buf_text, buf_md, merge_count, locator_end)

    # --- Backward pass: merge isolated short chunks ---
    # Some chunks are too short to stand alone but couldn't be merged in the
    # forward pass because they're sandwiched between long chunks.  Scan
    # right-to-left and merge any short chunk into its left neighbour when
    # the combined size fits within max_chars.
    merged_out: List[Tuple[str, str, Dict[str, Any]]] = []
    i = len(out) - 1
    while i >= 0:
        cid, text, md = out[i]
        if len(text) < min_chars and merged_out:
            # Try to prepend this short chunk to the RIGHT neighbour
            # (which is already in merged_out, in reverse order)
            nxt_id, nxt_text, nxt_md = merged_out[-1]
            combined = text + sep + nxt_text
            if _key(cid, text, md) == _key(nxt_id, nxt_text, nxt_md) and len(combined) <= max_chars:
                # Merge: short chunk's id/metadata becomes the primary,
                # right neighbour's locator becomes locator_end
                md["merged"] = True
                md["merge_count"] = int(md.get("merge_count", 1)) + int(nxt_md.get("merge_count", 1))
                md["locator_end"] = nxt_md.get("locator_end") or nxt_md.get("locator")
                _union_into(md, nxt_md)
                merged_out[-1] = (cid, combined.strip(), md)
                i -= 1
                continue
        merged_out.append((cid, text, md))
        i -= 1

    merged_out.reverse()
    # An isolated label/caption remains noise, but short chronology rows and
    # adjacent EPUB records are retained when the forward/backward pass joined
    # them into a useful searchable unit.
    return [row for row in merged_out if len(row[1]) >= HARD_MIN_CHARS]


def _control_char_ratio(text: str) -> float:
    """Ratio of non-whitespace control characters in text.

    Excludes tab (0x09), newline (0x0A), and carriage return (0x0D)
    which are legitimate text formatting, not encoding artifacts.
    """
    if not text:
        return 0.0
    count = 0
    for ch in text:
        o = ord(ch)
        if o < 0x20 and o not in (0x09, 0x0A, 0x0D):
            count += 1
        elif o == 0x7F:
            count += 1
    return count / len(text)


def _looks_like_structured_listing(text: str) -> bool:
    """Return True for readable TOC/index/table-like text.

    Such pages naturally contain few function words and many numbers or leader
    dots. That is a layout signal, not evidence of broken character mapping.
    """
    compact = " ".join(str(text or "").split())
    head = compact[:600].casefold()
    explicit_region = bool(re.search(
        r"\b(contents|index|appendix|appendices|list of (figures|tables))\b"
        r"|\b(figure|fig\.)\s*\d+\b|\bcourtesy of\b"
        r"|目次|図一覧|表一覧|索引",
        head,
    ))
    leader_dots = len(re.findall(r"\.{4,}|…{2,}", text)) >= 2
    table_signal = bool(re.search(
        r"(?:^|\s)(?:table|表)\s*[0-9０-９一二三四五六七八九十]+",
        head,
        flags=re.IGNORECASE,
    ))
    digit_ratio = sum(ch.isdigit() for ch in compact) / max(1, len(compact))
    return explicit_region or leader_dots or (table_signal and digit_ratio >= 0.08)


def analyze_text_quality(text: str) -> Dict[str, Any]:
    """
    Analyse page text quality and return scores + derived classifications.

    Returns a dict with:
      - scan_score: 0.0–1.0 likelihood that the page is image-only / unscanned
      - extraction_failure_score: 0.0–1.0 likelihood of font-encoding mismatch
        (PyMuPDF cannot decode custom-font glyphs → garbled extraction)
      - content_corruption_score: 0.0–1.0 likelihood of OCR / content corruption
        (extracted text is readable characters but linguistically nonsensical)
      - is_scanned: bool (derived from scan_score >= threshold)
      - is_corrupted: bool (derived from max of the two corruption scores)
      - corruption_type: "extraction_failure" | "content_corruption" | None

    Thresholds are configurable via environment variables:
      - TEXT_QUALITY_SCAN_THRESHOLD (default 0.8)
      - TEXT_QUALITY_CORRUPTION_THRESHOLD (default 0.6)
    """
    SCAN_THRESHOLD = float(os.environ.get("TEXT_QUALITY_SCAN_THRESHOLD", "0.8"))
    CORRUPTION_THRESHOLD = float(os.environ.get("TEXT_QUALITY_CORRUPTION_THRESHOLD", "0.6"))

    t = text.strip()

    # --- Scan score: how likely is this page unscanned (image-only)? ---
    if len(t) < 40:
        scan_score = 1.0
    elif len(t) < 80:
        scan_score = 0.8
    elif len(t) < 150:
        scan_score = 0.4
    else:
        scan_score = 0.0

    # --- Extraction failure score: font-encoding mismatch ---
    # Signals: high control-char ratio, very low alpha ratio
    ctrl_ratio = _control_char_ratio(t)
    letters_ratio = sum(ch.isalpha() for ch in t) / max(len(t), 1)
    structured_listing = _looks_like_structured_listing(t)

    extraction_score = 0.0
    # Dominated by non-whitespace control characters → font ToUnicode CMap missing
    if ctrl_ratio > 0.15:
        extraction_score = min(1.0, ctrl_ratio * 2.0)
    # Very few letters but text isn't empty → likely encoding issue.
    # The extracted bytes map to symbols/punctuation rather than readable text.
    elif letters_ratio < 0.15 and len(t) >= 80 and not structured_listing:
        extraction_score = 0.7

    # --- Content corruption score: linguistically nonsensical ---
    # Only evaluate when we have enough text AND extraction doesn't look broken
    content_score = 0.0
    if len(t) >= 200 and extraction_score < 0.5 and not structured_listing:
        is_cjk = is_no_space_language_document(t)

        if is_cjk:
            cjk = _cjk_ratio(t)
            if cjk >= 0.20:
                hira_chars = [ch for ch in t if 0x3040 <= ord(ch) <= 0x309F]
                hira_ratio = len(hira_chars) / len(t)
                if hira_ratio > 0.005:
                    particles = ["の", "に", "は", "を", "た", "て", "が"]
                    has_any = any(p in t for p in particles)
                    if not has_any:
                        # Score based on how many particles are missing
                        found = sum(1 for p in particles if p in t)
                        content_score = 1.0 - (found / len(particles))
        else:
            t_lower = t.lower()
            common_words = {"the", "of", "and", "to", "in", "is", "that", "for",
                           "it", "on", "with", "as", "this", "by", "an", "at"}
            found = sum(1 for w in common_words if f" {w} " in f" {t_lower} ")
            words = t_lower.split()
            # Higher word-count threshold avoids flagging bibliography entries,
            # editorial lists, and index sections that naturally lack common
            # English words due to proper-name / foreign-title dominance.
            if len(words) >= 30 and found == 0:
                content_score = 0.6
            elif len(words) >= 30 and found <= 1:
                content_score = 0.4
            elif len(words) >= 30 and found <= 2:
                content_score = 0.2

    # --- Combined classification ---
    corruption_score = max(extraction_score, content_score)

    if corruption_score >= CORRUPTION_THRESHOLD:
        if extraction_score >= content_score:
            corruption_type = "extraction_failure"
        else:
            corruption_type = "content_corruption"
    else:
        corruption_type = None

    return {
        "scan_score": round(scan_score, 3),
        "extraction_failure_score": round(extraction_score, 3),
        "content_corruption_score": round(content_score, 3),
        "corruption_score": round(corruption_score, 3),
        "structured_listing": structured_listing,
        "is_scanned": scan_score >= SCAN_THRESHOLD,
        "is_corrupted": corruption_score >= CORRUPTION_THRESHOLD,
        "corruption_type": corruption_type,
    }
