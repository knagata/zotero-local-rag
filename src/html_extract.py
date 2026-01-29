# src/html_extract.py
from __future__ import annotations

import os
import re
from html import unescape
from pathlib import Path
from typing import Any, Dict, List, Tuple

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

# Optional: robust main-content extraction for Zotero Web Snapshots
try:
    import trafilatura  # type: ignore
except Exception:  # pragma: no cover
    trafilatura = None

# Optional: EPUB parsing
try:
    import ebooklib  # type: ignore
    from ebooklib import epub as ebooklib_epub  # type: ignore
    ITEM_DOCUMENT = getattr(ebooklib, "ITEM_DOCUMENT", None)
except Exception:  # pragma: no cover
    ebooklib_epub = None
    ITEM_DOCUMENT = None


HTML_SCRIPT_STYLE_RE = re.compile(r"(?is)<(script|style).*?>.*?(</\1>)")
MAX_HTML_BYTES = int(os.environ.get("MAX_HTML_BYTES", "10000000"))  # guard for huge snapshots


def _decode_html_bytes(raw: bytes) -> str:
    head = raw[:8192]
    m = re.search(br"charset\s*=\s*['\"]?\s*([A-Za-z0-9_\-]+)", head, flags=re.IGNORECASE)
    if m:
        enc = m.group(1).decode("ascii", errors="ignore")
        if enc:
            try:
                return raw.decode(enc, errors="replace")
            except Exception:
                pass
    return raw.decode("utf-8", errors="replace")


def _strip_tags_fast(s: str) -> str:
    out: List[str] = []
    in_tag = False
    for ch in s:
        if ch == "<":
            in_tag = True
            continue
        if ch == ">" and in_tag:
            in_tag = False
            out.append(" ")
            continue
        if not in_tag:
            out.append(ch)
    return "".join(out)


def html_to_text(html: str) -> str:
    if not html:
        return ""

    lower = html.lower()
    bi = lower.find("<body")
    if bi != -1:
        start = lower.find(">", bi)
        if start != -1:
            end = lower.find("</body", start)
            if end != -1:
                html = html[start + 1 : end]
            else:
                if os.environ.get("DEBUG_HTML") == "1":
                    print("[DEBUG] </body> not found; using truncated body remainder.", file=os.sys.__stderr__)
                html = html[start + 1 :]
        else:
            if os.environ.get("DEBUG_HTML") == "1":
                print("[DEBUG] Malformed <body> tag (no '>'); using full HTML.", file=os.sys.__stderr__)
    else:
        if os.environ.get("DEBUG_HTML") == "1":
            print("[DEBUG] No <body> tag found; using full HTML.", file=os.sys.__stderr__)

    html = HTML_SCRIPT_STYLE_RE.sub("", html)
    html = html.replace("</p>", "\n\n").replace("<br>", "\n").replace("<br/>", "\n").replace("<br />", "\n")
    text = _strip_tags_fast(html)
    text = unescape(text)
    return text


def extract_main_text_from_html(raw_html: str) -> str:
    if raw_html and trafilatura is not None:
        try:
            txt = trafilatura.extract(
                raw_html,
                output_format="txt",
                favor_precision=True,
                include_links=False,
                include_images=False,
                include_tables=False,
                include_comments=False,
                no_fallback=False,
            )
            if isinstance(txt, str) and txt.strip():
                return txt
        except Exception as e:
            if os.environ.get("DEBUG_HTML") == "1":
                print(f"[DEBUG] trafilatura.extract failed; falling back: {e}", file=os.sys.__stderr__)
    return html_to_text(raw_html)


def extract_chunks_from_html_snapshot(
    html_path: Path,
    attachment_key: str,
    meta_base: Dict[str, Any],
) -> List[Tuple[str, str, Dict[str, Any]]]:
    chunks: List[Tuple[str, str, Dict[str, Any]]] = []
    try:
        with open(html_path, "rb") as f:
            raw_bytes = f.read(MAX_HTML_BYTES + 1)
        truncated = len(raw_bytes) > MAX_HTML_BYTES
        raw_html = _decode_html_bytes(raw_bytes[:MAX_HTML_BYTES])
        if truncated and os.environ.get("DEBUG_HTML") == "1":
            print(
                f"[DEBUG] HTML snapshot truncated to {MAX_HTML_BYTES} bytes: attachment={attachment_key} file={html_path}",
                file=os.sys.__stderr__,
            )
    except Exception as e:
        print(
            f"[WARN] Failed to read HTML snapshot: attachment={attachment_key} file={html_path} err={e}",
            file=os.sys.__stderr__,
        )
        return []

    raw_text = clean_extracted_text(extract_main_text_from_html(raw_html))
    joiner = joiner_for_text(raw_text[:20000])
    paras = normalize_paragraphs(raw_text, joiner=joiner)
    if not paras:
        return []

    sample = "\n\n".join(paras[:20])[:5000]
    if looks_like_gibberish(sample):
        return []
    local_min_chunk = MIN_CHUNK_CHARS_NO_SPACE if is_no_space_language_document(sample) else MIN_CHUNK_CHARS

    for para_index, para_text in enumerate(paras):
        para_text = para_text.strip()
        if not para_text:
            continue
        parts = split_long_paragraph(para_text, max_chars=MAX_CHARS)
        for part_index, part in enumerate(parts):
            part = part.strip()
            if len(part) < HARD_MIN_CHARS:
                continue
            chunk_id = f"{attachment_key}:html:para{para_index}:part{part_index}"
            md = dict(meta_base)
            md.update(
                {
                    "source_type": "html",
                    "locator": f"html:para{para_index}",
                    "path": str(html_path),
                    "pdf_path": str(html_path),  # backward compatibility
                    "para_index": int(para_index),
                    "part_index": int(part_index),
                }
            )
            chunks.append((chunk_id, part, md))

    chunks = merge_short_chunk_records(chunks, min_chars=local_min_chunk, max_chars=MAX_CHARS)
    ids = [cid for (cid, _, _) in chunks]
    if len(ids) != len(set(ids)):
        dup = len(ids) - len(set(ids))
        raise RuntimeError(f"Duplicate chunk ids generated for HTML ({dup}).")
    return chunks


def extract_chunks_from_epub_snapshot(
    epub_path: Path,
    attachment_key: str,
    meta_base: Dict[str, Any],
) -> List[Tuple[str, str, Dict[str, Any]]]:
    if ebooklib_epub is None or ITEM_DOCUMENT is None:
        if os.environ.get("DEBUG_HTML") == "1":
            print("[DEBUG] EbookLib not installed; skipping EPUB.", file=os.sys.__stderr__)
        return []

    try:
        book = ebooklib_epub.read_epub(str(epub_path))
    except Exception as e:
        print(
            f"[WARN] Failed to read EPUB: attachment={attachment_key} file={epub_path} err={e}",
            file=os.sys.__stderr__,
        )
        return []

    all_paras: List[Tuple[int, str]] = []  # (chapter_index, paragraph_text)
    chap_idx = 0
    for item in book.get_items_of_type(ITEM_DOCUMENT):
        try:
            raw = item.get_content()  # bytes
            html = _decode_html_bytes(raw)
            txt = clean_extracted_text(extract_main_text_from_html(html))
            joiner = joiner_for_text(txt[:20000])
            paras = normalize_paragraphs(txt, joiner=joiner)
            for p in paras:
                if p and p.strip():
                    all_paras.append((chap_idx, p))
        except Exception as e:
            if os.environ.get("DEBUG_HTML") == "1":
                print(
                    f"[DEBUG] EPUB chapter parse failed; continuing: attachment={attachment_key} file={epub_path} err={e}",
                    file=os.sys.__stderr__,
                )
        finally:
            chap_idx += 1

    if not all_paras:
        return []

    sample = "\n\n".join([p for _ci, p in all_paras[:20]])[:5000]
    if looks_like_gibberish(sample):
        return []
    local_min_chunk = MIN_CHUNK_CHARS_NO_SPACE if is_no_space_language_document(sample) else MIN_CHUNK_CHARS

    chunks: List[Tuple[str, str, Dict[str, Any]]] = []
    global_para = 0
    for chapter_index, para_text in all_paras:
        para_text = para_text.strip()
        if not para_text:
            global_para += 1
            continue

        parts = split_long_paragraph(para_text, max_chars=MAX_CHARS)
        for part_index, part in enumerate(parts):
            part = part.strip()
            if len(part) < HARD_MIN_CHARS:
                continue

            chunk_id = f"{attachment_key}:epub:para{global_para}:part{part_index}"
            md = dict(meta_base)
            md.update(
                {
                    "source_type": "epub",
                    "locator": f"epub:para{global_para}",
                    "path": str(epub_path),
                    "pdf_path": str(epub_path),  # backward compatibility
                    "chapter_index": int(chapter_index),
                    "para_index": int(global_para),
                    "part_index": int(part_index),
                }
            )
            chunks.append((chunk_id, part, md))

        global_para += 1

    chunks = merge_short_chunk_records(chunks, min_chars=local_min_chunk, max_chars=MAX_CHARS)
    ids = [cid for (cid, _, _) in chunks]
    if len(ids) != len(set(ids)):
        dup = len(ids) - len(set(ids))
        raise RuntimeError(f"Duplicate chunk ids generated for EPUB ({dup}).")
    return chunks