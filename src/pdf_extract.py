# src/pdf_extract.py
from __future__ import annotations

import os
import re
import shutil
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import signal

import fitz  # PyMuPDF

from text_utils import (
    HARD_MIN_CHARS,
    MAX_CHARS,
    TARGET_CHARS,
    MAX_CHARS_CJK,
    TARGET_CHARS_CJK,
    MIN_CHUNK_CHARS,
    MIN_CHUNK_CHARS_NO_SPACE,
    clean_extracted_text,
    is_no_space_language_document,
    joiner_for_text,
    looks_like_gibberish,
    merge_short_chunk_records,
    normalize_paragraphs,
    split_long_paragraph,
    analyze_text_quality,
)
from chapter_detect import (
    get_pdf_toc, build_pdf_page_chapter_lookup, build_pdf_page_structure_path_lookup,
)

PDF_DROP_REPEATED_LINES = (os.environ.get("PDF_DROP_REPEATED_LINES") or "1") == "1"
PDF_STRIP_REPEATED_PREFIX = (os.environ.get("PDF_STRIP_REPEATED_PREFIX") or "1") == "1"

# Per-page timeout (seconds). Set PAGE_TIMEOUT_SEC=0 to disable.
PAGE_TIMEOUT_SEC = int((os.environ.get("PAGE_TIMEOUT_SEC") or "30").strip())

# Tesseract OCR fallback for font-encoding mismatch pages.
# Requires tesseract binary on PATH. Set PDF_OCR_FALLBACK=0 to disable.
PDF_OCR_FALLBACK = (os.environ.get("PDF_OCR_FALLBACK") or "1") == "1"
PDF_OCR_DPI = int((os.environ.get("PDF_OCR_DPI") or "300").strip())


def _find_tesseract() -> Optional[str]:
    """Return the path to the tesseract binary, or None if not found.

    Checks PATH first, then common Homebrew locations which may not be
    on PATH when invoked via ``uv run``.
    """
    path = shutil.which("tesseract")
    if path:
        return path
    for candidate in ["/opt/homebrew/bin/tesseract", "/usr/local/bin/tesseract"]:
        if Path(candidate).exists():
            return candidate
    return None


def _resolve_ocr_lang() -> str:
    """Return the best available Tesseract language string.

    If PDF_OCR_LANG is explicitly set, use it verbatim.
    Otherwise, probe the installed Tesseract languages and prefer
    ``jpn+eng`` when Japanese data is available, falling back to
    ``eng`` with a one-time diagnostic message.

    To install Japanese Tesseract data:
      macOS:  brew install tesseract-lang
      Linux:  apt install tesseract-ocr-jpn
    """
    explicit = os.environ.get("PDF_OCR_LANG", "").strip()
    if explicit:
        return explicit

    tesseract_bin = _find_tesseract()
    if not tesseract_bin:
        return "eng"

    # Probe available languages once
    try:
        import subprocess
        result = subprocess.run(
            [tesseract_bin, "--list-langs"],
            capture_output=True, text=True, timeout=5,
        )
        installed = set(
            line.strip() for line in result.stdout.splitlines()
            if line.strip() and not line.startswith("List")
        )
    except Exception:
        return "eng"

    if "jpn" in installed:
        return "jpn+eng"
    else:
        # Warn once per process about missing Japanese OCR support
        if not getattr(_resolve_ocr_lang, "_warned", False):
            _resolve_ocr_lang._warned = True  # type: ignore[attr-defined]
            print(
                "[NOTE] Tesseract Japanese language data not found.\n"
                "       Extraction failures in Japanese PDFs will use English OCR,\n"
                "       which may produce garbled results.\n"
                "       Install Japanese support:\n"
                "         macOS:  brew install tesseract-lang\n"
                "         Linux:  apt install tesseract-ocr-jpn\n"
                "       Then set: PDF_OCR_LANG=jpn+eng",
                file=os.sys.__stderr__,
            )
        return "eng"


PDF_OCR_LANG = _resolve_ocr_lang()


def normalize_block_text_to_paragraph(text: str) -> str:
    lines = [ln.strip() for ln in (text or "").splitlines() if ln.strip()]
    if not lines:
        return ""

    parts: List[str] = []
    for ln in lines:
        if parts and parts[-1].endswith("-"):
            parts[-1] = parts[-1][:-1] + ln
        else:
            parts.append(ln)

    joiner = joiner_for_text("".join(parts))
    merged = joiner.join(parts)
    merged = re.sub(r"\s+", " ", merged).strip()
    return merged


def _detect_column_boundaries(
    blocks: List[Tuple[float, float, float, str]], page_width: float
) -> Optional[float]:
    """Detect a column boundary x-coordinate, or None if single-column.

    Uses block x0 positions. If blocks cluster into two distinct horizontal
    regions with a clear gap between the rightmost left block and the leftmost
    right block, returns the midpoint of the gap.
    """
    if len(blocks) < 3:
        return None

    # Collect x0 positions of blocks that are not spanning most of the page.
    # Wide blocks (title, full-width paragraphs) are excluded — they span
    # both columns and should not participate in column detection.
    x0s: List[float] = []
    for y0, y1, x0, txt in blocks:
        # Exclude blocks starting very close to the left margin or spanning
        # most of the page — these are full-width elements.
        if x0 > page_width * 0.15:
            x0s.append(x0)

    if len(x0s) < 3:
        return None

    x0s.sort()
    mid = page_width / 2

    left_x0s = [x for x in x0s if x < mid]
    right_x0s = [x for x in x0s if x >= mid]

    # Need at least one block on each side to detect columns
    if not left_x0s or not right_x0s:
        return None

    left_max = max(left_x0s)
    right_min = min(right_x0s)
    gap = right_min - left_max

    # Split if there's a meaningful gap (> 12% of page width)
    if gap > page_width * 0.12:
        return (left_max + right_min) / 2

    return None


def _merge_blocks_vertically(
    blocks: List[Tuple[float, float, float, str]],
) -> List[str]:
    """Merge vertically adjacent blocks into paragraphs (single-column)."""
    if not blocks:
        return []

    merged: List[str] = []
    cur_text = ""
    cur_y1: Optional[float] = None

    for y0, y1, _x0, txt in blocks:
        if not cur_text:
            cur_text = txt
            cur_y1 = y1
            continue

        gap = 0.0 if cur_y1 is None else (y0 - cur_y1)

        if gap >= 0 and gap <= 12.0:
            joiner = joiner_for_text(cur_text + txt)
            cur_text = (cur_text + joiner + txt) if joiner else (cur_text + txt)
            cur_y1 = max(cur_y1 or y1, y1)
        else:
            merged.append(cur_text.strip())
            cur_text = txt
            cur_y1 = y1

    if cur_text:
        merged.append(cur_text.strip())

    return [m for m in merged if m]


def extract_paragraphs_from_pdf_page(page: Any) -> List[str]:
    try:
        blocks = page.get_text("blocks") or []
        norm_blocks: List[Tuple[float, float, float, str]] = []  # (y0, y1, x0, text)

        for b in blocks:
            if not b or len(b) < 5:
                continue
            x0 = float(b[0])
            y0 = float(b[1])
            y1 = float(b[3]) if len(b) >= 4 else y0
            txt = b[4]
            btype = b[6] if len(b) >= 7 else 0

            if btype not in (0,):
                continue
            if not isinstance(txt, str):
                continue

            t = clean_extracted_text(txt)
            t = normalize_block_text_to_paragraph(t)
            if t:
                norm_blocks.append((y0, y1, x0, t))

        if norm_blocks:
            page_width = page.rect.width
            col_boundary = _detect_column_boundaries(norm_blocks, page_width)

            if col_boundary is not None:
                # Multi-column: split into left and right columns
                left_blocks = [(y0, y1, x0, txt) for (y0, y1, x0, txt) in norm_blocks
                               if x0 < col_boundary]
                right_blocks = [(y0, y1, x0, txt) for (y0, y1, x0, txt) in norm_blocks
                                if x0 >= col_boundary]

                left_blocks.sort(key=lambda t: (t[0], t[2]))
                right_blocks.sort(key=lambda t: (t[0], t[2]))

                left_paras = _merge_blocks_vertically(left_blocks)
                right_paras = _merge_blocks_vertically(right_blocks)

                return left_paras + right_paras
            else:
                # Single column
                norm_blocks.sort(key=lambda t: (t[0], t[2]))
                return _merge_blocks_vertically(norm_blocks)

    except TimeoutError:
        raise
    except Exception:
        pass

    # fallback
    try:
        raw = page.get_text("text") or ""
        raw = clean_extracted_text(raw)
        joiner = joiner_for_text(raw[:20000])
        return normalize_paragraphs(raw, joiner=joiner)
    except TimeoutError:
        raise
    except Exception:
        return []


def detect_repeated_lines(paras_by_page: List[List[str]]) -> set[str]:
    """
    Detect repeated lines across many pages (typical header/footer artefacts).
    Conservative heuristic: count exact paragraph strings that are short-ish.
    """
    from collections import Counter

    cnt: Counter[str] = Counter()
    pages_with_any = 0
    for paras in paras_by_page:
        if not paras:
            continue
        pages_with_any += 1
        uniq = set([p.strip() for p in paras if p and len(p.strip()) <= 180])
        for p in uniq:
            cnt[p] += 1

    if pages_with_any <= 3:
        return set()

    threshold = max(4, int(pages_with_any * 0.25))
    repeated = {s for s, c in cnt.items() if c >= threshold and len(s) <= 180}
    return repeated


def drop_repeated_lines_from_paras(paras: List[str], repeated_lines: set[str]) -> List[str]:
    out: List[str] = []
    for p in paras:
        if not p:
            continue
        if p.strip() in repeated_lines:
            continue
        out.append(p)
    return out


def detect_repeated_prefixes(paras_by_page: List[List[str]]) -> set[str]:
    """
    Detect repeated prefixes on the first paragraph (e.g., journal title + page number).
    Returns small prefix strings to strip if they reoccur frequently.
    """
    from collections import Counter

    cnt: Counter[str] = Counter()
    pages_with_any = 0
    for paras in paras_by_page:
        if not paras:
            continue
        pages_with_any += 1
        first = (paras[0] or "").strip()
        if not first:
            continue
        # Take first 40 chars as a candidate prefix (after collapsing spaces).
        cand = re.sub(r"\s+", " ", first)[:40].strip()
        if cand:
            cnt[cand] += 1

    if pages_with_any <= 3:
        return set()

    threshold = max(4, int(pages_with_any * 0.25))
    return {s for s, c in cnt.items() if c >= threshold and 5 <= len(s) <= 40}


def strip_repeated_prefix_from_first_para(paras: List[str], repeated_prefixes: set[str]) -> List[str]:
    if not paras:
        return paras
    first = paras[0]
    if not first:
        return paras
    norm_first = re.sub(r"\s+", " ", first).strip()
    for pref in sorted(repeated_prefixes, key=len, reverse=True):
        if norm_first.startswith(pref):
            # remove prefix from original string in a forgiving way
            cut = len(pref)
            new_first = norm_first[cut:].lstrip(" -–—:：\t")
            if new_first:
                paras = [new_first] + paras[1:]
            else:
                paras = paras[1:]
            break
    return paras


@contextmanager
def _capture_os_stderr():
    import os as _os

    saved_fd = _os.dup(2)
    r_fd, w_fd = _os.pipe()
    _os.dup2(w_fd, 2)
    _os.close(w_fd)
    try:
        yield r_fd
    finally:
        _os.dup2(saved_fd, 2)
        _os.close(saved_fd)


def _read_fd_text(fd: int) -> str:
    import os as _os

    chunks: List[bytes] = []
    try:
        while True:
            b = _os.read(fd, 8192)
            if not b:
                break
            chunks.append(b)
    finally:
        try:
            _os.close(fd)
        except Exception:
            pass
    return b"".join(chunks).decode("utf-8", errors="replace")


@contextmanager
def _page_timeout(seconds: int):
    """Context manager that raises TimeoutError if the block exceeds `seconds`.

    Uses SIGALRM (Unix/macOS only). If SIGALRM is unavailable or seconds <= 0,
    the block runs without a timeout.
    """
    if seconds <= 0 or not hasattr(signal, "SIGALRM"):
        yield
        return

    def _handler(signum, frame):
        raise TimeoutError(f"PyMuPDF page.get_text() timed out after {seconds}s")

    old_handler = signal.signal(signal.SIGALRM, _handler)
    signal.alarm(seconds)
    try:
        yield
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old_handler)


def _ocr_page_with_tesseract(page: Any) -> List[str]:
    """Render a PDF page to an image and OCR it with Tesseract.

    Used as a fallback when PyMuPDF text extraction fails due to
    custom font encodings without ToUnicode CMaps. The rendered
    image contains correct glyphs that Tesseract can read.

    Returns a list of paragraph strings, or an empty list on failure.
    """
    import subprocess
    import tempfile

    try:
        mat = fitz.Matrix(PDF_OCR_DPI / 72, PDF_OCR_DPI / 72)
        pix = page.get_pixmap(matrix=mat)
        img_bytes = pix.tobytes("png")
    except Exception:
        return []

    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            f.write(img_bytes)
            tmp_path = f.name

        tesseract_bin = _find_tesseract()
        if not tesseract_bin:
            return []

        result = subprocess.run(
            [tesseract_bin, tmp_path, "stdout", "-l", PDF_OCR_LANG, "--psm", "6"],
            capture_output=True,
            text=True,
            timeout=30,
        )
        if result.returncode != 0:
            stderr_msg = (result.stderr or "").strip()
            if stderr_msg:
                # Only log the first error per process to avoid spam
                if not getattr(_ocr_page_with_tesseract, "_logged_error", False):
                    _ocr_page_with_tesseract._logged_error = True  # type: ignore[attr-defined]
                    print(
                        f"[WARN] Tesseract OCR failed (lang={PDF_OCR_LANG}): {stderr_msg[:200]}\n"
                        f"       Install the language data or set PDF_OCR_LANG to an available language.",
                        file=os.sys.__stderr__,
                    )
            return []

        text = result.stdout.strip()
        if not text:
            return []

        # Parse OCR output into paragraphs
        joiner = joiner_for_text(text[:20000])
        paras = normalize_paragraphs(text, joiner=joiner)
        return [p for p in paras if p.strip()]
    except FileNotFoundError:
        # tesseract not installed — not an error, just unavailable
        if not getattr(_ocr_page_with_tesseract, "_logged_missing", False):
            _ocr_page_with_tesseract._logged_missing = True  # type: ignore[attr-defined]
            print(
                "[NOTE] Tesseract not found on PATH. OCR fallback unavailable.\n"
                "       Install: brew install tesseract (macOS) / apt install tesseract-ocr (Linux)",
                file=os.sys.__stderr__,
            )
        return []
    except Exception:
        return []
    finally:
        if tmp_path:
            try:
                os.unlink(tmp_path)
            except Exception:
                pass


def extract_chunks_from_pdf(
    pdf_path: Path,
    attachment_key: str,
    meta_base: Dict[str, Any],
) -> Tuple[List[Tuple[str, str, Dict[str, Any]]], Dict[str, Any]]:
    chunks: List[Tuple[str, str, Dict[str, Any]]] = []
    
    # Safe default quality info in case of early exit/failure
    quality_info = {
        "is_scanned": False,
        "is_corrupted": False,
        "scanned_pages": [],
        "corrupted_pages": [],
        "extraction_failure_pages": [],
        "content_corruption_pages": [],
        "ocr_fallback_pages": [],
        "low_text_pages": [],
        "empty_pages": [],
        "total_pages": 1,
        "scanned_ratio": 0.0,
        "corrupted_ratio": 0.0,
        "extraction_failure_ratio": 0.0,
        "ocr_fallback_ratio": 0.0,
        "avg_corruption_score": 0.0,
        "max_corruption_score": 0.0,
        "avg_extraction_failure_score": 0.0,
        "page_scores": {},
    }

    # 章構造を事前に取得（失敗しても処理続行）
    try:
        _toc = get_pdf_toc(str(pdf_path))
        _chapter_lookup = build_pdf_page_chapter_lookup(_toc) if _toc else None
        _structure_path_lookup = build_pdf_page_structure_path_lookup(_toc) if _toc else None
    except Exception:
        _chapter_lookup = None
        _structure_path_lookup = None

    captured_text = ""
    r_fd: Optional[int] = None
    try:
        with _capture_os_stderr() as _r_fd:
            r_fd = _r_fd
            doc = fitz.open(str(pdf_path))
            try:
                paras_by_page: List[List[str]] = []
                page_labels: Dict[int, str] = {}
                scanned_pages = []          # image-only pages (needs Docling)
                corrupted_pages = []        # above corruption threshold
                extraction_failure_pages = []  # font-encoding mismatch → Tesseract fallback
                content_corruption_pages = []  # OCR / linguistic corruption
                ocr_fallback_pages = []     # pages where Tesseract fallback was used
                ocr_notified = False        # one-time "OCR started" message
                low_text_pages = []         # very little text but not image-only
                empty_pages = []            # no text, no images
                page_scores: Dict[int, Dict[str, Any]] = {}  # per-page scores
                total_pages = doc.page_count

                SCAN_THRESHOLD = float(os.environ.get("TEXT_QUALITY_SCAN_THRESHOLD", "0.8"))
                CORRUPTION_THRESHOLD = float(os.environ.get("TEXT_QUALITY_CORRUPTION_THRESHOLD", "0.6"))

                for pi in range(doc.page_count):
                    try:
                        with _page_timeout(PAGE_TIMEOUT_SEC):
                            page = doc.load_page(pi)
                            # Capture PDF page label (book page number, e.g. "xii", "15")
                            try:
                                lbl = page.get_label()
                                if lbl:
                                    page_labels[pi] = lbl
                            except Exception:
                                pass
                            paras = extract_paragraphs_from_pdf_page(page)
                    except Exception as e:
                        print(
                            f"[WARN] Failed to extract page paragraphs: attachment={attachment_key} file={pdf_path} page={pi+1} err={e}",
                            file=os.sys.__stderr__,
                        )
                        paras_by_page.append([])
                        try:
                            has_images = len(page.get_images()) > 0
                        except Exception:
                            has_images = False
                        if has_images:
                            scanned_pages.append(pi + 1)
                        else:
                            empty_pages.append(pi + 1)
                        continue

                    if not paras:
                        paras_by_page.append([])
                        try:
                            has_images = len(page.get_images()) > 0
                        except Exception:
                            has_images = False
                        if has_images:
                            scanned_pages.append(pi + 1)
                        else:
                            empty_pages.append(pi + 1)
                        continue

                    joined = "\n\n".join(paras)

                    # Analyze text quality per page (score-based)
                    quality = analyze_text_quality(joined)

                    # Tesseract OCR fallback for font-encoding mismatch.
                    # When PyMuPDF can't decode custom fonts, we render the
                    # page to an image and OCR it — much lighter than Docling.
                    ocr_used = False
                    if (
                        PDF_OCR_FALLBACK
                        and quality["corruption_type"] == "extraction_failure"
                        and quality["extraction_failure_score"] >= 0.5
                    ):
                        if not ocr_notified:
                            ocr_notified = True
                            print(
                                f"[PROGRESS]   OCR fallback activated for "
                                f"attachment={attachment_key} "
                                f"(font-encoding mismatch detected). "
                                f"This may take ~1 sec per affected page...",
                                file=os.sys.__stderr__,
                            )
                        ocr_paras = _ocr_page_with_tesseract(page)
                        if ocr_paras:
                            paras = ocr_paras
                            joined = "\n\n".join(paras)
                            quality = analyze_text_quality(joined)
                            ocr_used = True

                    page_scores[pi + 1] = {
                        "scan_score": quality["scan_score"],
                        "extraction_failure_score": quality["extraction_failure_score"],
                        "content_corruption_score": quality["content_corruption_score"],
                        "corruption_score": quality["corruption_score"],
                    }
                    if ocr_used:
                        page_scores[pi + 1]["ocr_fallback"] = True
                        ocr_fallback_pages.append(pi + 1)
                        # Progress every 10 pages (after append so count is current)
                        if len(ocr_fallback_pages) % 10 == 0:
                            print(
                                f"[PROGRESS]   ↳ OCR fallback: {len(ocr_fallback_pages)} pages done "
                                f"(page {pi+1}/{total_pages})",
                                file=os.sys.__stderr__,
                            )

                    if quality["is_scanned"]:
                        try:
                            has_images = len(page.get_images()) > 0
                        except Exception:
                            has_images = False
                        if has_images:
                            scanned_pages.append(pi + 1)
                        else:
                            low_text_pages.append(pi + 1)
                    elif quality["is_corrupted"]:
                        corrupted_pages.append(pi + 1)
                        if quality["corruption_type"] == "extraction_failure":
                            extraction_failure_pages.append(pi + 1)
                        elif quality["corruption_type"] == "content_corruption":
                            content_corruption_pages.append(pi + 1)

                    if looks_like_gibberish(joined):
                        paras_by_page.append([])
                        continue

                    paras_by_page.append(paras)

                # Compute aggregate scores
                scanned_ratio = len(scanned_pages) / max(1, total_pages)
                corrupted_ratio = len(corrupted_pages) / max(1, total_pages)
                extraction_failure_ratio = len(extraction_failure_pages) / max(1, total_pages)

                # Aggregate per-page scores
                avg_corruption_score = 0.0
                max_corruption_score = 0.0
                avg_extraction_failure_score = 0.0
                if page_scores:
                    avg_corruption_score = sum(s["corruption_score"] for s in page_scores.values()) / len(page_scores)
                    max_corruption_score = max(s["corruption_score"] for s in page_scores.values())
                    avg_extraction_failure_score = sum(s["extraction_failure_score"] for s in page_scores.values()) / len(page_scores)

                if ocr_fallback_pages:
                    print(
                        f"[PROGRESS]   OCR fallback summary: "
                        f"{len(ocr_fallback_pages)}/{total_pages} pages processed with OCR",
                        file=os.sys.__stderr__,
                    )

                quality_info = {
                    "is_scanned": scanned_ratio >= SCAN_THRESHOLD,
                    "is_corrupted": corrupted_ratio >= CORRUPTION_THRESHOLD,
                    "scanned_pages": scanned_pages,
                    "corrupted_pages": corrupted_pages,
                    "extraction_failure_pages": extraction_failure_pages,
                    "content_corruption_pages": content_corruption_pages,
                    "ocr_fallback_pages": ocr_fallback_pages,
                    "low_text_pages": low_text_pages,
                    "empty_pages": empty_pages,
                    "total_pages": total_pages,
                    "scanned_ratio": round(scanned_ratio, 3),
                    "corrupted_ratio": round(corrupted_ratio, 3),
                    "extraction_failure_ratio": round(extraction_failure_ratio, 3),
                    "ocr_fallback_ratio": round(len(ocr_fallback_pages) / max(1, total_pages), 3),
                    "avg_corruption_score": round(avg_corruption_score, 3),
                    "max_corruption_score": round(max_corruption_score, 3),
                    "avg_extraction_failure_score": round(avg_extraction_failure_score, 3),
                    "page_scores": page_scores,
                }

                repeated_lines: set[str] = set()
                if PDF_DROP_REPEATED_LINES:
                    repeated_lines = detect_repeated_lines(paras_by_page)
                    if repeated_lines and os.environ.get("DEBUG_PDF_REPEAT") == "1":
                        ex = list(sorted(repeated_lines))[:10]
                        print(
                            f"[DEBUG] repeated header/footer lines detected: attachment={attachment_key} file={pdf_path} n={len(repeated_lines)} ex={ex}",
                            file=os.sys.__stderr__,
                        )

                repeated_prefixes: set[str] = set()
                if PDF_STRIP_REPEATED_PREFIX:
                    repeated_prefixes = detect_repeated_prefixes(paras_by_page)
                    if repeated_prefixes and os.environ.get("DEBUG_PDF_REPEAT") == "1":
                        ex = list(sorted(repeated_prefixes))[:10]
                        print(
                            f"[DEBUG] repeated header/footer prefixes detected: attachment={attachment_key} file={pdf_path} n={len(repeated_prefixes)} ex={ex}",
                            file=os.sys.__stderr__,
                        )

                for pi, paras in enumerate(paras_by_page):
                    if not paras:
                        continue

                    if repeated_lines:
                        paras = drop_repeated_lines_from_paras(paras, repeated_lines)
                        if not paras:
                            continue

                    if repeated_prefixes:
                        paras = strip_repeated_prefix_from_first_para(paras, repeated_prefixes)
                        if not paras:
                            continue

                    joined = "\n\n".join(paras)
                    is_cjk = is_no_space_language_document(joined)
                    local_min_chunk = MIN_CHUNK_CHARS_NO_SPACE if is_cjk else MIN_CHUNK_CHARS
                    local_max_chars = MAX_CHARS_CJK if is_cjk else MAX_CHARS
                    local_target_chars = TARGET_CHARS_CJK if is_cjk else TARGET_CHARS

                    page_chunks: List[Tuple[str, str, Dict[str, Any]]] = []
                    for para_index, para_text in enumerate(paras):
                        para_text = para_text.strip()
                        if not para_text:
                            continue

                        parts = split_long_paragraph(para_text, max_chars=local_max_chars, target_chars=local_target_chars)
                        for part_index, part in enumerate(parts):
                            part = part.strip()
                            if len(part) < HARD_MIN_CHARS:
                                continue

                            chunk_id = f"{attachment_key}:p{pi+1}:para{para_index}:part{part_index}"
                            md = dict(meta_base)
                            chapter_info: Dict[str, Any] = {}
                            if _chapter_lookup is not None:
                                try:
                                    _ch, _sec = _chapter_lookup(pi + 1)
                                    if _ch:
                                        chapter_info["chapter"] = _ch
                                    if _sec:
                                        chapter_info["section"] = _sec
                                except Exception:
                                    pass
                            if _structure_path_lookup is not None:
                                try:
                                    structure_path = _structure_path_lookup(pi + 1)
                                    if structure_path:
                                        chapter_info["structure_path"] = structure_path
                                except Exception:
                                    pass
                            md.update(
                                {
                                    "source_type": "pdf",
                                    "locator": f"p{pi+1}:para{para_index}",
                                    "page": int(pi + 1),
                                    "page_label": page_labels.get(pi, ""),
                                    "pdf_path": str(pdf_path),
                                    "path": str(pdf_path),
                                    "para_index": int(para_index),
                                    "part_index": int(part_index),
                                    **chapter_info,
                                }
                            )
                            page_chunks.append((chunk_id, part, md))

                    page_chunks = merge_short_chunk_records(page_chunks, min_chars=local_min_chunk, max_chars=local_max_chars)
                    chunks.extend(page_chunks)

            finally:
                try:
                    doc.close()
                except Exception:
                    pass

        captured_text = _read_fd_text(r_fd)
        r_fd = None

    except Exception as e:
        try:
            if r_fd is not None:
                captured_text = _read_fd_text(r_fd)
        except Exception:
            pass
        print(
            f"[WARN] Failed to open/extract PDF: attachment={attachment_key} file={pdf_path} err={e}",
            file=os.sys.__stderr__,
        )
        return [], quality_info

    finally:
        if captured_text and "MuPDF error" in captured_text:
            # Deduplicate: group identical messages, show each once with count
            from collections import Counter
            error_lines = [line.strip() for line in captured_text.splitlines()
                          if "MuPDF error" in line]
            counts = Counter(error_lines)
            for msg, count in counts.most_common():
                suffix = f" (x{count})" if count > 1 else ""
                print(
                    f"[WARN] PyMuPDF: {msg}{suffix} "
                    f"[attachment={attachment_key}]",
                    file=os.sys.__stderr__,
                )

    ids = [cid for (cid, _, _) in chunks]
    if len(ids) != len(set(ids)):
        dup = len(ids) - len(set(ids))
        raise RuntimeError(f"Duplicate chunk ids generated ({dup}). This should not happen.")

    return chunks, quality_info
