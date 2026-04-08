# src/pdf_extract.py
from __future__ import annotations

import os
import re
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import signal

import fitz  # PyMuPDF

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

PDF_DROP_REPEATED_LINES = (os.environ.get("PDF_DROP_REPEATED_LINES") or "1") == "1"
PDF_STRIP_REPEATED_PREFIX = (os.environ.get("PDF_STRIP_REPEATED_PREFIX") or "1") == "1"

# Per-page timeout (seconds). Set PAGE_TIMEOUT_SEC=0 to disable.
PAGE_TIMEOUT_SEC = int((os.environ.get("PAGE_TIMEOUT_SEC") or "30").strip())


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
            norm_blocks.sort(key=lambda t: (t[0], t[2]))

            merged: List[str] = []
            cur_text = ""
            cur_y1: Optional[float] = None

            for y0, y1, _x0, txt in norm_blocks:
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


def extract_chunks_from_pdf(
    pdf_path: Path,
    attachment_key: str,
    meta_base: Dict[str, Any],
) -> List[Tuple[str, str, Dict[str, Any]]]:
    chunks: List[Tuple[str, str, Dict[str, Any]]] = []

    captured_text = ""
    r_fd: Optional[int] = None
    try:
        with _capture_os_stderr() as _r_fd:
            r_fd = _r_fd
            doc = fitz.open(str(pdf_path))
            try:
                paras_by_page: List[List[str]] = []
                page_labels: Dict[int, str] = {}
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
                        continue

                    if not paras:
                        paras_by_page.append([])
                        continue

                    joined = "\n\n".join(paras)
                    if looks_like_gibberish(joined):
                        paras_by_page.append([])
                        continue

                    paras_by_page.append(paras)

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
                    local_min_chunk = MIN_CHUNK_CHARS_NO_SPACE if is_no_space_language_document(joined) else MIN_CHUNK_CHARS

                    page_chunks: List[Tuple[str, str, Dict[str, Any]]] = []
                    for para_index, para_text in enumerate(paras):
                        para_text = para_text.strip()
                        if not para_text:
                            continue

                        parts = split_long_paragraph(para_text, max_chars=MAX_CHARS)
                        for part_index, part in enumerate(parts):
                            part = part.strip()
                            if len(part) < HARD_MIN_CHARS:
                                continue

                            chunk_id = f"{attachment_key}:p{pi+1}:para{para_index}:part{part_index}"
                            md = dict(meta_base)
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
                                }
                            )
                            page_chunks.append((chunk_id, part, md))

                    page_chunks = merge_short_chunk_records(page_chunks, min_chars=local_min_chunk, max_chars=MAX_CHARS)
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
        return []

    finally:
        if captured_text and "MuPDF error" in captured_text:
            for line in captured_text.splitlines():
                if "MuPDF error" in line:
                    print(
                        f"[WARN] PyMuPDF reported error: attachment={attachment_key} file={pdf_path} {line}",
                        file=os.sys.__stderr__,
                    )

    ids = [cid for (cid, _, _) in chunks]
    if len(ids) != len(set(ids)):
        dup = len(ids) - len(set(ids))
        raise RuntimeError(f"Duplicate chunk ids generated ({dup}). This should not happen.")

    return chunks