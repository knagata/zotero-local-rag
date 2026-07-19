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

try:
    from chapter_detect import get_epub_chapter_index_to_path as _get_epub_toc_path_map
except Exception:  # pragma: no cover
    _get_epub_toc_path_map = None  # type: ignore


HTML_SCRIPT_STYLE_RE = re.compile(r"(?is)<(script|style).*?>.*?(</\1>)")
MAX_HTML_BYTES = int(os.environ.get("MAX_HTML_BYTES", "10000000"))  # guard for huge snapshots


def _strip_unclosed_script_style(html: str) -> str:
    """Remove unclosed <style> / <script> content caused by HTML truncation.

    When a large HTML file is truncated (MAX_HTML_BYTES), a closing
    </style> or </script> tag may be cut off.  This strips the orphaned
    opening tag and everything after it so that CSS / JS does not leak
    into extracted text.
    """
    for tag in ("style", "script"):
        open_marker = f"<{tag}"
        close_marker = f"</{tag}>"
        last_open = html.rfind(open_marker)
        if last_open >= 0:
            last_close = html.rfind(close_marker)
            if last_open > last_close:
                html = html[:last_open]
    return html


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
    html = _strip_unclosed_script_style(html)
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


def _read_html_skip_styles(path: Path) -> str:
    """Read an HTML file, skipping <style> and <script> blocks.

    Returns at most MAX_HTML_BYTES of cleaned HTML bytes (decoded to str).
    Large base64 font data inside <style> tags does not count toward the
    limit, so the actual page content is reached even when fonts are
    inlined by tools like SingleFile.
    """
    CHUNK = 1 << 20  # 1 MiB

    # Detect charset from the first read
    charset = "utf-8"
    cleaned_parts: List[str] = []
    cleaned_len = 0
    in_skip = False
    skip_close = b""
    leftover = b""

    with open(path, "rb") as f:
        while cleaned_len < MAX_HTML_BYTES:
            chunk = f.read(CHUNK)
            if not chunk:
                break
            chunk = leftover + chunk
            leftover = b""

            # Detect charset from the first few bytes
            if charset == "utf-8" and len(chunk) >= 512:
                m = re.search(
                    rb"charset\s*=\s*['\"]?\s*([A-Za-z0-9_\-]+)",
                    chunk[:8192], flags=re.IGNORECASE,
                )
                if m:
                    charset = m.group(1).decode("ascii", errors="ignore")

            i = 0
            while i < len(chunk):
                if in_skip:
                    end = chunk.find(skip_close, i)
                    if end >= 0:
                        i = end + len(skip_close)
                        in_skip = False
                    else:
                        leftover = chunk[-len(skip_close):] if len(chunk) > len(skip_close) else chunk
                        break
                else:
                    lt = chunk.find(b"<", i)
                    if lt < 0:
                        add = chunk[i:]
                        decoded = add.decode(charset, errors="replace")
                        cleaned_parts.append(decoded)
                        cleaned_len += len(decoded)
                        break

                    if lt > i:
                        add = chunk[i:lt]
                        decoded = add.decode(charset, errors="replace")
                        cleaned_parts.append(decoded)
                        cleaned_len += len(decoded)

                    tag_start = chunk[lt:lt + 7].lower()
                    if tag_start.startswith(b"<style") or tag_start.startswith(b"<scrip"):
                        gt = chunk.find(b">", lt)
                        if gt < 0:
                            leftover = chunk[lt:]
                            break
                        tag_name = b"style" if tag_start.startswith(b"<style") else b"script"
                        in_skip = True
                        skip_close = b"</" + tag_name + b">"
                        i = gt + 1
                    else:
                        gt = chunk.find(b">", lt)
                        if gt < 0:
                            leftover = chunk[lt:]
                            break
                        add = chunk[lt:gt + 1]
                        decoded = add.decode(charset, errors="replace")
                        cleaned_parts.append(decoded)
                        cleaned_len += len(decoded)
                        i = gt + 1

                if cleaned_len >= MAX_HTML_BYTES:
                    break

    return "".join(cleaned_parts)


def extract_chunks_from_html_snapshot(
    html_path: Path,
    attachment_key: str,
    meta_base: Dict[str, Any],
) -> Tuple[List[Tuple[str, str, Dict[str, Any]]], Dict[str, Any]]:
    default_quality = {
        "is_scanned": False,
        "is_corrupted": False,
        "scanned_pages": [],
        "corrupted_pages": [],
        "total_pages": 1,
    }
    chunks: List[Tuple[str, str, Dict[str, Any]]] = []
    try:
        raw_html = _read_html_skip_styles(html_path)
    except Exception as e:
        print(
            f"[WARN] Failed to read HTML snapshot: attachment={attachment_key} file={html_path} err={e}",
            file=os.sys.__stderr__,
        )
        return [], default_quality

    raw_text = clean_extracted_text(extract_main_text_from_html(raw_html))
    joiner = joiner_for_text(raw_text[:20000])
    paras = normalize_paragraphs(raw_text, joiner=joiner)
    if not paras:
        return [], default_quality

    sample = "\n\n".join(paras[:20])[:5000]
    if looks_like_gibberish(sample):
        return [], default_quality
    is_cjk = is_no_space_language_document(sample)
    local_min_chunk = MIN_CHUNK_CHARS_NO_SPACE if is_cjk else MIN_CHUNK_CHARS
    local_max_chars = MAX_CHARS_CJK if is_cjk else MAX_CHARS
    local_target_chars = TARGET_CHARS_CJK if is_cjk else TARGET_CHARS

    for para_index, para_text in enumerate(paras):
        para_text = para_text.strip()
        if not para_text:
            continue
        parts = split_long_paragraph(para_text, max_chars=local_max_chars, target_chars=local_target_chars)
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

    chunks = merge_short_chunk_records(chunks, min_chars=local_min_chunk, max_chars=local_max_chars)
    ids = [cid for (cid, _, _) in chunks]
    if len(ids) != len(set(ids)):
        dup = len(ids) - len(set(ids))
        raise RuntimeError(f"Duplicate chunk ids generated for HTML ({dup}).")
    return chunks, default_quality



def extract_chunks_from_epub_snapshot(
    epub_path: Path,
    attachment_key: str,
    meta_base: Dict[str, Any],
) -> Tuple[List[Tuple[str, str, Dict[str, Any]]], Dict[str, Any]]:
    default_quality = {
        "is_scanned": False,
        "is_corrupted": False,
        "scanned_pages": [],
        "corrupted_pages": [],
        "total_pages": 1,
    }
    if ebooklib_epub is None or ITEM_DOCUMENT is None:
        if os.environ.get("DEBUG_HTML") == "1":
            print("[DEBUG] EbookLib not installed; skipping EPUB.", file=os.sys.__stderr__)
        return [], default_quality

    try:
        book = ebooklib_epub.read_epub(str(epub_path))
    except Exception as e:
        print(
            f"[WARN] Failed to read EPUB: attachment={attachment_key} file={epub_path} err={e}",
            file=os.sys.__stderr__,
        )
        return [], default_quality

    # 章タイトルマップ（失敗しても処理続行）
    _epub_toc_path_map: Dict[int, List[str]] = {}
    if _get_epub_toc_path_map is not None:
        try:
            _epub_toc_path_map = _get_epub_toc_path_map(str(epub_path))
        except Exception:
            pass

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
        return [], default_quality

    sample = "\n\n".join([p for _ci, p in all_paras[:20]])[:5000]
    if looks_like_gibberish(sample):
        return [], default_quality
    is_cjk = is_no_space_language_document(sample)
    local_min_chunk = MIN_CHUNK_CHARS_NO_SPACE if is_cjk else MIN_CHUNK_CHARS
    local_max_chars = MAX_CHARS_CJK if is_cjk else MAX_CHARS
    local_target_chars = TARGET_CHARS_CJK if is_cjk else TARGET_CHARS

    chunks: List[Tuple[str, str, Dict[str, Any]]] = []
    global_para = 0
    for chapter_index, para_text in all_paras:
        para_text = para_text.strip()
        if not para_text:
            global_para += 1
            continue

        parts = split_long_paragraph(para_text, max_chars=local_max_chars, target_chars=local_target_chars)
        for part_index, part in enumerate(parts):
            part = part.strip()
            if len(part) < HARD_MIN_CHARS:
                continue

            chunk_id = f"{attachment_key}:epub:para{global_para}:part{part_index}"
            md = dict(meta_base)
            structure_path = _epub_toc_path_map.get(int(chapter_index), [])
            chapter_title = structure_path[-1] if structure_path else ""
            md.update(
                {
                    "source_type": "epub",
                    "locator": f"epub:para{global_para}",
                    "path": str(epub_path),
                    "pdf_path": str(epub_path),  # backward compatibility
                    "chapter_index": int(chapter_index),
                    "para_index": int(global_para),
                    "part_index": int(part_index),
                    **(({"structure_path": structure_path}) if structure_path else {}),
                    **(({"chapter": chapter_title}) if chapter_title else {}),
                }
            )
            chunks.append((chunk_id, part, md))

        global_para += 1

    chunks = merge_short_chunk_records(chunks, min_chars=local_min_chunk, max_chars=local_max_chars)
    ids = [cid for (cid, _, _) in chunks]
    if len(ids) != len(set(ids)):
        dup = len(ids) - len(set(ids))
        raise RuntimeError(f"Duplicate chunk ids generated for EPUB ({dup}).")
    return chunks, default_quality
