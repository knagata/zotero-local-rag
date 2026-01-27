#!/usr/bin/env python3
from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import shutil
import unicodedata
import sys
import time
import gc
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from html import unescape
# Optional: robust main-content extraction for Zotero Web Snapshots
try:
    import trafilatura  # type: ignore
except Exception:  # pragma: no cover
    trafilatura = None


# Optional: EPUB parsing (EPUB -> XHTML/HTML chapters)
# NOTE: EbookLib exposes item-type constants (e.g., ITEM_DOCUMENT) at the package root,
# not as `ebooklib.item`.
try:
    import ebooklib  # type: ignore
    from ebooklib import epub as ebooklib_epub  # type: ignore

    # ITEM_DOCUMENT is expected to exist in supported EbookLib versions, but keep this
    # defensive fallback to avoid crashing at import time.
    ITEM_DOCUMENT = getattr(ebooklib, "ITEM_DOCUMENT", None)
except Exception:  # pragma: no cover
    ebooklib_epub = None
    ITEM_DOCUMENT = None

# Optional: resolve cached Hugging Face model snapshots (offline-friendly)
try:
    from huggingface_hub import snapshot_download  # type: ignore
except Exception:  # pragma: no cover
    snapshot_download = None

import fitz  # PyMuPDF
import chromadb
from chromadb.utils import embedding_functions

from contextlib import contextmanager

from zotero_source_localapi import ZoteroLocalAPI, ZoteroAttachment


# ----------------------------
# Paths / Env
# ----------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
CHROMA_DIR = Path(os.environ.get("CHROMA_DIR", str(DATA_DIR / "chroma")))
PDF_CACHE_DIR = Path(os.environ.get("PDF_CACHE_DIR", str(DATA_DIR / "pdf_cache")))
MANIFEST_PATH = Path(os.environ.get("MANIFEST_PATH", str(DATA_DIR / "manifest.json")))

ZOTERO_DATA_DIR = os.environ.get("ZOTERO_DATA_DIR")  # required for local storage resolution in your pipeline
CHROMA_COLLECTION_ENV = os.environ.get("CHROMA_COLLECTION")
CHROMA_COLLECTION_DEFAULT = "zotero_paragraphs"

# Embedding model selection
# - If EMB_MODEL is explicitly set, use it.
# - Otherwise, pick a default based on EMB_PROFILE.
#   - fast: multilingual + lighter
#   - bge : bge-m3 (heavier; recommended to use a local path and cache offline)
def _resolve_embedder_settings() -> Tuple[str, str]:
    profile = (os.environ.get("EMB_PROFILE") or "fast").strip().lower()

    # Treat either flag as "offline".
    offline = (os.environ.get("HF_HUB_OFFLINE") == "1") or (os.environ.get("TRANSFORMERS_OFFLINE") == "1")

    def _pick_device(default: str = "cpu") -> str:
        return (os.environ.get("EMB_DEVICE") or default).strip()

    def _is_local_path(p: str) -> bool:
        try:
            return Path(p).expanduser().exists()
        except Exception:
            return False

    def _try_resolve_hf_cached_snapshot(model_id: str) -> Optional[str]:
        """Return local snapshot directory for a Hugging Face repo id if it is already cached.

        This is used to allow fully-offline execution even when EMB_MODEL is given as a remote
        repo id (e.g., `sentence-transformers/...`) as long as it has been cached previously.
        """
        if snapshot_download is None:
            return None
        try:
            p = snapshot_download(repo_id=model_id, local_files_only=True)
            if p and Path(p).exists():
                return p
        except Exception:
            return None
        return None

    def _offline_resolve_or_exit(model: str) -> str:
        """When offline mode is enabled, ensure we return a local path.

        Accept either:
        - an explicit local directory path (preferred), or
        - a Hugging Face repo id that is already present in the local cache.
        """
        if not offline:
            return model
        if _is_local_path(model):
            return model
        cached = _try_resolve_hf_cached_snapshot(model)
        if cached:
            return cached
        raise SystemExit(
            "ERROR: Offline mode is enabled (HF_HUB_OFFLINE=1 or TRANSFORMERS_OFFLINE=1), "
            f"but the requested embedding model is not available locally: {model}\n\n"
            "Fix options:\n"
            "  (1) Temporarily go online and cache it, then rerun offline:\n"
            "      HF_HUB_OFFLINE=0 TRANSFORMERS_OFFLINE=0 python -c \"from sentence_transformers import SentenceTransformer; SentenceTransformer('" + model + "')\"\n"
            "  (2) Or set EMB_MODEL to a local directory path containing the model files.\n"
        )

    # Explicit override wins.
    if "EMB_MODEL" in os.environ and (os.environ.get("EMB_MODEL") or "").strip():
        model = os.environ["EMB_MODEL"].strip()
        model = _offline_resolve_or_exit(model)
        return model, _pick_device("cpu")

    # Profile-based defaults.
    if profile == "bge":
        model = str(PROJECT_ROOT / "data" / "models" / "bge-m3")
        device_default = "mps" if sys.platform == "darwin" else "cpu"
        model = _offline_resolve_or_exit(model)
        return model, _pick_device(device_default)

    # fast (default): multilingual MiniLM
    remote_model = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    local_model = str(PROJECT_ROOT / "data" / "models" / "paraphrase-multilingual-MiniLM-L12-v2")

    if offline:
        # In offline mode, prefer a project-local on-disk model dir if present.
        if _is_local_path(local_model):
            return local_model, _pick_device("cpu")

        # Otherwise, if the model was cached in the Hugging Face cache already, use that snapshot directory.
        cached = _try_resolve_hf_cached_snapshot(remote_model)
        if cached:
            return cached, _pick_device("cpu")

        raise SystemExit(
            "ERROR: Offline mode is enabled (HF_HUB_OFFLINE=1 or TRANSFORMERS_OFFLINE=1) but the default fast model "
            "is not available locally.\n"
            f"Expected local model dir (project-local): {local_model}\n"
            "Also checked Hugging Face cache for: sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2\n\n"
            "Fix options:\n"
            "  A) Temporarily download/cache it (online):\n"
            "     HF_HUB_OFFLINE=0 TRANSFORMERS_OFFLINE=0 python -c \"from sentence_transformers import SentenceTransformer; SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')\"\n"
            "  B) Or download into the project-local directory and keep using offline mode afterwards (set EMB_MODEL to the local dir).\n"
        )

    return remote_model, _pick_device("cpu")

# Chunking controls
MAX_CHARS = int(os.environ.get("MAX_CHARS", "1200"))
MIN_CHUNK_CHARS = int(os.environ.get("MIN_CHUNK_CHARS", "200"))

# For languages that typically do not use whitespace word segmentation (e.g., Japanese/Chinese).
# This is applied per-document based on a simple heuristic.
MIN_CHUNK_CHARS_NO_SPACE = int(os.environ.get("MIN_CHUNK_CHARS_NO_SPACE", "120"))

# Batch sizing
# One knob: BATCH_SIZE
# - When pending chunks reach this size, we flush (delete+upsert) to Chroma.
# - We also use the same value as the sub-batch size for `col.upsert(...)` to reduce memory spikes.
BATCH_SIZE = int((os.environ.get("BATCH_SIZE") or "256").strip())


def _dedupe_by_id(
    ids: List[str],
    docs: List[str],
    metas: List[Dict[str, Any]],
) -> Tuple[List[str], List[str], List[Dict[str, Any]]]:
    """Dedupe records by id, keeping the last occurrence."""
    uniq: Dict[str, Tuple[str, Dict[str, Any]]] = {}
    for cid, doc, md in zip(ids, docs, metas):
        uniq[cid] = (doc, md)
    out_ids = list(uniq.keys())
    out_docs = [uniq[i][0] for i in out_ids]
    out_metas = [uniq[i][1] for i in out_ids]
    return out_ids, out_docs, out_metas


def _delete_by_attachment_keys(col: Any, attachment_keys: Iterable[str]) -> None:
    """Best-effort delete all chunks for each attachmentKey."""
    for dk in attachment_keys:
        try:
            col.delete(where={"attachmentKey": dk})
        except Exception:
            pass


def _upsert_in_subbatches(
    col: Any,
    ids: List[str],
    docs: List[str],
    metas: List[Dict[str, Any]],
    *,
    subbatch_size: int,
    show_progress: bool,
    label: str,
) -> None:
    """Upsert in smaller sub-batches to reduce memory spikes."""
    total = len(ids)
    if total == 0:
        return
    if show_progress:
        print(
            f"[PROGRESS] {label}: {total} chunks (embedding+write) | sub-batch={subbatch_size}",
            file=sys.__stderr__,
        )
    for start in range(0, total, subbatch_size):
        end = min(start + subbatch_size, total)
        if show_progress:
            print(
                f"[PROGRESS]   ↳ upsert sub-batch {start + 1}-{end}/{total}",
                file=sys.__stderr__,
            )
        col.upsert(
            ids=ids[start:end],
            documents=docs[start:end],
            metadatas=metas[start:end],
        )
        relieve_memory_pressure()


# Text cleaning / filtering
CONTROL_CHARS = re.compile(r"[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]")


def relieve_memory_pressure() -> None:
    """Best-effort memory pressure relief.

    This is primarily useful on unified-memory systems (e.g., Apple Silicon) where large
    embedding batches can cause system-wide sluggishness.
    """
    gc.collect()
    try:
        import torch  # type: ignore

        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            torch.mps.empty_cache()
    except Exception:
        pass


# HTML helpers (for Zotero Web Snapshots / Notes)
HTML_SCRIPT_STYLE_RE = re.compile(r"(?is)<(script|style).*?>.*?(</\1>)")

def _decode_html_bytes(raw: bytes) -> str:
    """Best-effort decode for Zotero Web Snapshot HTML."""
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


def extract_main_text_from_html(raw_html: str) -> str:
    """Extract main text from HTML.

    Primary path: trafilatura (if installed) for robust boilerplate removal.
    Fallback: lightweight tag stripping via html_to_text().
    """
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
                print(f"[DEBUG] trafilatura.extract failed; falling back: {e}", file=sys.__stderr__)

    return html_to_text(raw_html)

def _strip_tags_fast(s: str) -> str:
    """Fast-ish HTML tag stripper (state machine).

    Much faster than a broad regex like `<[^>]+>` on multi-megabyte snapshots.
    It is intentionally simple and not a full HTML parser.
    """
    out: List[str] = []
    in_tag = False
    for ch in s:
        if ch == "<":
            in_tag = True
            continue
        if ch == ">" and in_tag:
            in_tag = False
            # add a separator so words don't concatenate
            out.append(" ")
            continue
        if not in_tag:
            out.append(ch)
    return "".join(out)

MAX_HTML_BYTES = int(os.environ.get("MAX_HTML_BYTES", "10000000"))  # guard for huge snapshots


def html_to_text(html: str) -> str:
    """Very small HTML->text converter (no extra deps).

    Notes:
    - Web Snapshots can be very large; we try to reduce work by extracting <body>...</body> when present.
    - This is intentionally dependency-free; it is not a perfect boilerplate remover.
    """
    if not html:
        return ""

    # Prefer <body> when present to reduce boilerplate, but be permissive:
    # - Some snapshots may omit <body> (fragments)
    # - Large snapshots may be truncated before </body>
    lower = html.lower()
    bi = lower.find("<body")
    if bi != -1:
        start = lower.find(">", bi)
        if start != -1:
            end = lower.find("</body", start)
            if end != -1:
                html = html[start + 1 : end]
            else:
                # No closing tag (likely truncated). Use the remainder.
                if os.environ.get("DEBUG_HTML") == "1":
                    print("[DEBUG] </body> not found; using truncated body remainder.", file=sys.__stderr__)
                html = html[start + 1 :]
        else:
            if os.environ.get("DEBUG_HTML") == "1":
                print("[DEBUG] Malformed <body> tag (no '>'); using full HTML.", file=sys.__stderr__)
    else:
        if os.environ.get("DEBUG_HTML") == "1":
            print("[DEBUG] No <body> tag found; using full HTML.", file=sys.__stderr__)

    html = HTML_SCRIPT_STYLE_RE.sub("", html)
    html = html.replace("</p>", "\n\n").replace("<br>", "\n").replace("<br/>", "\n").replace("<br />", "\n")
    text = _strip_tags_fast(html)
    text = unescape(text)
    return text


# ----------------------------
# Manifest
# ----------------------------
def load_manifest() -> Dict[str, Any]:
    """
    Manifest format:
    {
      "version": 1,
      "files": {
         "<attachmentKey>": {"mtime": <float>, "size": <int>, "pdf_path": "<str>"}
      },
      "notes": {
         "<noteKey>": {"version": <int|null>}
      }
    }
    """
    if not MANIFEST_PATH.exists():
        return {"version": 1, "files": {}, "notes": {}}
    try:
        txt = MANIFEST_PATH.read_text(encoding="utf-8").strip()
        if not txt:
            return {"version": 1, "files": {}, "notes": {}}
        obj = json.loads(txt)
        if not isinstance(obj, dict):
            return {"version": 1, "files": {}, "notes": {}}
        obj.setdefault("version", 1)
        obj.setdefault("files", {})
        obj.setdefault("notes", {})
        if not isinstance(obj["files"], dict):
            obj["files"] = {}
        if not isinstance(obj["notes"], dict):
            obj["notes"] = {}
        return obj
    except Exception:
        backup = MANIFEST_PATH.with_suffix(".json.bak")
        try:
            MANIFEST_PATH.replace(backup)
        except Exception:
            pass
        return {"version": 1, "files": {}, "notes": {}}


def save_manifest(m: Dict[str, Any]) -> None:
    MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    tmp = MANIFEST_PATH.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(m, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(MANIFEST_PATH)


# ----------------------------
# Text helpers
# ----------------------------
def clean_extracted_text(s: str) -> str:
    s = CONTROL_CHARS.sub("", s)
    s = unicodedata.normalize("NFKC", s)
    s = s.replace("\uFFFD", "")
    return s


def normalize_paragraphs(raw: str) -> List[str]:
    lines = raw.splitlines()
    paras: List[str] = []
    buf: List[str] = []

    def flush():
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

        merged = " ".join(parts).strip()
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


def is_no_space_language_document(sample: str) -> bool:
    """Heuristic: treat as a no-whitespace language doc if CJK ratio is high and spaces are rare."""
    if not sample:
        return False
    s = sample.strip()
    if not s:
        return False
    # Measure on a bounded sample to avoid heavy loops.
    s = s[:20000]
    # If the text is mostly CJK and has very few ASCII spaces, treat it as no-space language.
    cjk = _cjk_ratio(s)
    space_ratio = s.count(" ") / max(len(s), 1)
    return (cjk >= 0.20) and (space_ratio <= 0.02)


def split_long_paragraph(p: str, max_chars: int = MAX_CHARS) -> List[str]:
    p = p.strip()
    if len(p) <= max_chars:
        return [p]

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
        if len(cur) + len(s) + 1 <= max_chars:
            cur = (cur + " " + s).strip()
        else:
            if cur:
                parts.append(cur)
            cur = s
    if cur:
        parts.append(cur)

    final_parts: List[str] = []
    for part in parts:
        if len(part) <= max_chars:
            final_parts.append(part)
        else:
            for i in range(0, len(part), max_chars):
                final_parts.append(part[i : i + max_chars].strip())
    return [x for x in final_parts if x]


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
                file=sys.__stderr__,
            )
    except Exception as e:
        print(
            f"[WARN] Failed to read HTML snapshot: attachment={attachment_key} file={html_path} err={e}",
            file=sys.__stderr__,
        )
        return []

    raw_text = clean_extracted_text(extract_main_text_from_html(raw_html))
    paras = normalize_paragraphs(raw_text)
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
            if len(part) < local_min_chunk:
                continue
            chunk_id = f"{attachment_key}:html:para{para_index}:part{part_index}"
            md = dict(meta_base)
            md.update(
                {
                    "source_type": "html",
                    "locator": f"html:para{para_index}",
                    "path": str(html_path),
                    "pdf_path": str(html_path),  # keep for backward compatibility
                    "para_index": int(para_index),
                    "part_index": int(part_index),
                }
            )
            chunks.append((chunk_id, part, md))

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
    """Extract paragraph chunks from an EPUB file.

    Uses EbookLib to read an EPUB and iterates document items (XHTML/HTML). Each chapter is decoded
    and passed through the same main-text extraction pipeline as HTML snapshots.

    Chunk id format:
      {attachmentKey}:epub:para{para}:part{part}

    Notes:
    - Paragraph indices are global across the EPUB to keep ids stable and make context windows work.
    - We store `chapter_index` in metadata when available.
    """
    if ebooklib_epub is None or ITEM_DOCUMENT is None:
        if os.environ.get("DEBUG_HTML") == "1":
            print("[DEBUG] EbookLib not installed; skipping EPUB.", file=sys.__stderr__)
        return []

    try:
        book = ebooklib_epub.read_epub(str(epub_path))
    except Exception as e:
        print(
            f"[WARN] Failed to read EPUB: attachment={attachment_key} file={epub_path} err={e}",
            file=sys.__stderr__,
        )
        return []

    all_paras: List[Tuple[int, str]] = []  # (chapter_index, paragraph_text)
    chap_idx = 0
    for item in book.get_items_of_type(ITEM_DOCUMENT):
        try:
            raw = item.get_content()  # bytes
            html = _decode_html_bytes(raw)
            txt = clean_extracted_text(extract_main_text_from_html(html))
            paras = normalize_paragraphs(txt)
            for p in paras:
                if p and p.strip():
                    all_paras.append((chap_idx, p))
        except Exception as e:
            if os.environ.get("DEBUG_HTML") == "1":
                print(
                    f"[DEBUG] EPUB chapter parse failed; continuing: attachment={attachment_key} file={epub_path} err={e}",
                    file=sys.__stderr__,
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
            if len(part) < local_min_chunk:
                continue

            chunk_id = f"{attachment_key}:epub:para{global_para}:part{part_index}"
            md = dict(meta_base)
            md.update(
                {
                    "source_type": "epub",
                    "locator": f"epub:para{global_para}",
                    "path": str(epub_path),
                    "pdf_path": str(epub_path),  # keep for backward compatibility
                    "chapter_index": int(chapter_index),
                    "para_index": int(global_para),
                    "part_index": int(part_index),
                }
            )
            chunks.append((chunk_id, part, md))

        global_para += 1

    ids = [cid for (cid, _, _) in chunks]
    if len(ids) != len(set(ids)):
        dup = len(ids) - len(set(ids))
        raise RuntimeError(f"Duplicate chunk ids generated for EPUB ({dup}).")

    return chunks


# ----------------------------
# PDF -> chunks
# ----------------------------

@contextmanager
def _capture_os_stderr():
    saved_fd = os.dup(2)
    r_fd, w_fd = os.pipe()
    os.dup2(w_fd, 2)
    os.close(w_fd)
    try:
        yield r_fd
    finally:
        os.dup2(saved_fd, 2)
        os.close(saved_fd)


def _read_fd_text(fd: int) -> str:
    chunks: List[bytes] = []
    try:
        while True:
            b = os.read(fd, 8192)
            if not b:
                break
            chunks.append(b)
    finally:
        try:
            os.close(fd)
        except Exception:
            pass
    return b"".join(chunks).decode("utf-8", errors="replace")


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
                for pi in range(doc.page_count):
                    try:
                        page = doc.load_page(pi)
                        raw = page.get_text("text") or ""
                    except Exception as e:
                        print(
                            f"[WARN] Failed to extract page text: attachment={attachment_key} file={pdf_path} page={pi+1} err={e}",
                            file=sys.__stderr__,
                        )
                        continue

                    raw = clean_extracted_text(raw)

                    paras = normalize_paragraphs(raw)
                    if not paras:
                        continue

                    joined = "\n\n".join(paras)
                    if looks_like_gibberish(joined):
                        continue
                    local_min_chunk = MIN_CHUNK_CHARS_NO_SPACE if is_no_space_language_document(joined) else MIN_CHUNK_CHARS

                    for para_index, para_text in enumerate(paras):
                        para_text = para_text.strip()
                        if not para_text:
                            continue

                        parts = split_long_paragraph(para_text, max_chars=MAX_CHARS)
                        for part_index, part in enumerate(parts):
                            part = part.strip()
                            if len(part) < local_min_chunk:
                                continue

                            chunk_id = f"{attachment_key}:p{pi+1}:para{para_index}:part{part_index}"

                            md = dict(meta_base)
                            md.update(
                                {
                                    "source_type": "pdf",
                                    "locator": f"p{pi+1}:para{para_index}",
                                    "page": int(pi + 1),
                                    "pdf_path": str(pdf_path),
                                    "path": str(pdf_path),
                                    "para_index": int(para_index),
                                    "part_index": int(part_index),
                                }
                            )

                            chunks.append((chunk_id, part, md))
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
            file=sys.__stderr__,
        )
        return []

    finally:
        if captured_text and "MuPDF error" in captured_text:
            for line in captured_text.splitlines():
                if "MuPDF error" in line:
                    print(
                        f"[WARN] PyMuPDF reported error: attachment={attachment_key} file={pdf_path} {line}",
                        file=sys.__stderr__,
                    )

    ids = [cid for (cid, _, _) in chunks]
    if len(ids) != len(set(ids)):
        dup = len(ids) - len(set(ids))
        raise RuntimeError(f"Duplicate chunk ids generated ({dup}). This should not happen.")

    return chunks


# ----------------------------
# Chroma
# ----------------------------
def get_collection():
    CHROMA_DIR.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))

    model_name, device = _resolve_embedder_settings()
    ef = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name=model_name,
        device=device,
        normalize_embeddings=True,
    )

    # Probe embedding dimension once (used for collection suffix + persisted config).
    probe_dim: Optional[int] = None
    try:
        probe_vecs = ef(["collection probe"])
        if isinstance(probe_vecs, list) and probe_vecs and isinstance(probe_vecs[0], (list, tuple)):
            probe_dim = len(probe_vecs[0])
    except Exception:
        probe_dim = None

    # If CHROMA_COLLECTION is not explicitly set, automatically suffix the default
    # collection name by embedding dimension to prevent dimension-mismatch when switching models.
    collection_name = (CHROMA_COLLECTION_ENV or "").strip() or CHROMA_COLLECTION_DEFAULT
    if not (CHROMA_COLLECTION_ENV or "").strip():
        if isinstance(probe_dim, int) and probe_dim > 0:
            collection_name = f"{CHROMA_COLLECTION_DEFAULT}_{probe_dim}"
        else:
            # If probing fails, fall back to the unsuffixed default.
            collection_name = CHROMA_COLLECTION_DEFAULT

    # Persist resolved embedder configuration for debugging / reproducibility.
    # (Written after collection_name is resolved so it matches what will be used.)
    try:
        cfg_path = CHROMA_DIR / "embedder_config.json"
        tmp_path = CHROMA_DIR / "embedder_config.json.tmp"
        cfg = {
            "python": sys.executable,
            "emb_model": model_name,
            "emb_device": device,
            "embedding_dim": probe_dim,
            "collection": collection_name,
        }
        tmp_path.write_text(json.dumps(cfg, ensure_ascii=False, indent=2), encoding="utf-8")
        tmp_path.replace(cfg_path)
    except Exception:
        pass

    # --- Optional embedder probe (helps verify which model is actually used) ---
    if os.environ.get("DEBUG_EMBEDDER") == "1":
        try:
            probe_text = "sanity check: embedder probe"
            t_probe0 = time.perf_counter()
            _ = ef([probe_text])
            t_probe1 = time.perf_counter()
            print(
                "[DEBUG] Embedder probe: "
                f"python={sys.executable} "
                f"EMB_MODEL={model_name} "
                f"EMB_DEVICE={device} "
                f"dim={probe_dim} "
                f"encode_time_s={(t_probe1 - t_probe0):.3f}",
                file=sys.__stderr__,
            )
        except Exception as e:
            print(f"[DEBUG] Embedder probe failed: {e}", file=sys.__stderr__)

    col = client.get_or_create_collection(
        name=collection_name,
        embedding_function=ef,
        metadata={"hnsw:space": "cosine"},
    )
    return col


# ----------------------------
# CLI
# ----------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Index Zotero local PDFs/HTML snapshots into Chroma (paragraph-level).")
    p.add_argument("--collection", help="Zotero collection key to restrict by (optional).", default=None)
    p.add_argument("--dump-attachments", action="store_true", help="Print resolved attachments list then proceed.")
    p.add_argument(
        "--progress",
        action="store_true",
        help="Print progress while indexing (also enabled by PROGRESS=1 env var).",
    )
    p.add_argument(
        "--require-data-dir",
        action="store_true",
        help="Fail fast if ZOTERO_DATA_DIR is not set or invalid.",
    )
    p.add_argument(
        "--rebuild",
        action="store_true",
        help="Force rebuild: delete Chroma DB and manifest, then re-index everything.",
    )
    return p.parse_args()


def _zotero_data_dir_is_valid(zotero_data_dir: Optional[str]) -> bool:
    if not zotero_data_dir:
        return False
    zdd = Path(zotero_data_dir).expanduser()
    return bool(zdd.exists() and (zdd / "storage").exists() and (zdd / "zotero.sqlite").exists())


def _validate_zotero_data_dir_or_exit():
    if not ZOTERO_DATA_DIR:
        raise SystemExit(
            "ERROR: ZOTERO_DATA_DIR is not set.\n"
            "Set it to your Zotero data directory (must contain 'storage/' and 'zotero.sqlite').\n"
        )
    zdd = Path(ZOTERO_DATA_DIR).expanduser()
    if not (zdd.exists() and (zdd / "storage").exists() and (zdd / "zotero.sqlite").exists()):
        raise SystemExit(
            f"ERROR: ZOTERO_DATA_DIR looks invalid: {zdd}\n"
            "Expected to find 'storage/' and 'zotero.sqlite' inside it.\n"
        )


async def main_async(args: argparse.Namespace) -> None:
    PDF_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    if args.rebuild:
        if CHROMA_DIR.exists():
            shutil.rmtree(CHROMA_DIR)
        if MANIFEST_PATH.exists():
            MANIFEST_PATH.unlink()

    manifest = load_manifest()
    files_manifest: Dict[str, Dict[str, Any]] = manifest.get("files", {})
    notes_manifest: Dict[str, Dict[str, Any]] = manifest.get("notes", {})
    if not isinstance(notes_manifest, dict):
        notes_manifest = {}
    manifest["notes"] = notes_manifest

    api = ZoteroLocalAPI()
    show_progress = bool(args.progress) or (os.environ.get("PROGRESS") == "1")
    t0 = time.perf_counter()

    if os.environ.get("TRACE_UNAWAITED") == "1":
        import tracemalloc
        tracemalloc.start(25)

    if os.environ.get("DEBUG_IMPORTS") == "1":
        import inspect
        import zotero_source_localapi as _zsl
        print(f"[DEBUG] zotero_source_localapi.__file__={_zsl.__file__}", file=sys.stderr)
        print(
            "[DEBUG] iscoroutinefunction(iter_normalized_attachments)="
            f"{inspect.iscoroutinefunction(ZoteroLocalAPI.iter_normalized_attachments)}",
            file=sys.stderr,
        )
        print(
            "[DEBUG] isasyncgenfunction(iter_normalized_attachments)="
            f"{inspect.isasyncgenfunction(ZoteroLocalAPI.iter_normalized_attachments)}",
            file=sys.stderr,
        )

    zotero_data_dir: Optional[str] = None
    if _zotero_data_dir_is_valid(ZOTERO_DATA_DIR):
        zotero_data_dir = ZOTERO_DATA_DIR
    else:
        if ZOTERO_DATA_DIR:
            if args.require_data_dir:
                _validate_zotero_data_dir_or_exit()
            else:
                print(
                    f"[WARN] ZOTERO_DATA_DIR looks invalid: {Path(ZOTERO_DATA_DIR).expanduser()}\n"
                    "      Falling back to Zotero Local API file download into PDF_CACHE_DIR.",
                    file=sys.__stderr__,
                )
        else:
            if args.require_data_dir:
                _validate_zotero_data_dir_or_exit()
            else:
                print(
                    "[WARN] ZOTERO_DATA_DIR is not set. Falling back to Zotero Local API file download into PDF_CACHE_DIR.",
                    file=sys.__stderr__,
                )

    attachments: List[ZoteroAttachment] = await api.list_normalized_attachments(
        zotero_data_dir=zotero_data_dir,
        pdf_cache_dir=str(PDF_CACHE_DIR),
        collection_key=args.collection,
    )
    attachments = [a for a in attachments if getattr(a, "pdf_path", None)]
    total_attachments = len(attachments)
    if show_progress:
        print(
            f"[PROGRESS] Attachments resolved: {total_attachments} (collection={args.collection or 'ALL'})",
            file=sys.__stderr__,
        )

    if args.dump_attachments:
        dump = []
        for a in attachments:
            d = asdict(a) if hasattr(a, "__dataclass_fields__") else dict(a.__dict__)
            dump.append(
                {
                    "attachmentKey": d.get("attachmentKey"),
                    "parentItemKey": d.get("parentItemKey"),
                    "title": d.get("title"),
                    "year": d.get("year"),
                    "creators": d.get("creators"),
                    "pdf_path": d.get("pdf_path"),
                    "source_type": d.get("source_type"),
                    "contentType": d.get("contentType"),
                    "filename": d.get("filename"),
                }
            )
        print(json.dumps(dump, ensure_ascii=False, indent=2))

    col = get_collection()

    # Delete stale attachment items
    current_keys = {a.attachmentKey for a in attachments}
    stale_keys = set(files_manifest.keys()) - current_keys

    deleted_stale = 0
    for stale_key in stale_keys:
        try:
            col.delete(where={"attachmentKey": stale_key})
            deleted_stale += 1
        except Exception:
            pass
        files_manifest.pop(stale_key, None)

    updated_pdf = updated_html = updated_epub = 0
    skipped_pdf = skipped_html = skipped_epub = 0

    pending_ids: List[str] = []
    pending_docs: List[str] = []
    pending_metas: List[Dict[str, Any]] = []

    pending_manifest_updates: Dict[str, Dict[str, Any]] = {}
    pending_delete_attachment_keys: set[str] = set()
    pending_source_types: Dict[str, str] = {}

    for idx, a in enumerate(attachments, start=1):
        file_path = Path(a.pdf_path).expanduser()
        # Zotero Web Snapshots can be stored as a directory containing an index.html.
        if file_path.is_dir():
            for name in ("index.html", "index.htm"):
                cand = file_path / name
                if cand.exists() and cand.is_file():
                    file_path = cand
                    break
            else:
                # Try a shallow search for any html file.
                htmls = sorted([p for p in file_path.iterdir() if p.is_file() and p.suffix.lower() in {".html", ".htm"}])
                if htmls:
                    file_path = htmls[0]
                else:
                    print(
                        f"[WARN] Web snapshot directory has no index.html: attachment={a.attachmentKey} dir={file_path}",
                        file=sys.__stderr__,
                    )
                    continue
        if not file_path.exists():
            continue

        # Derive a stable source type early (used for skip counters/logging).
        ctype = getattr(a, "contentType", None)
        stype = getattr(a, "source_type", None) or "pdf"
        if (ctype == "application/epub+zip") or (file_path.suffix.lower() == ".epub"):
            stype = "epub"
        elif file_path.suffix.lower() in {".html", ".htm"}:
            stype = "html"
        elif stype not in {"pdf", "html", "epub"}:
            stype = "pdf"

        st = file_path.stat()
        mtime = float(st.st_mtime)
        size = int(st.st_size)

        prev = files_manifest.get(a.attachmentKey)
        if prev and float(prev.get("mtime", -1)) == mtime and int(prev.get("size", -1)) == size:
            if stype == "html":
                skipped_html += 1
            elif stype == "epub":
                skipped_epub += 1
            else:
                skipped_pdf += 1
            if show_progress:
                print(
                    f"[PROGRESS]   ↳ skipped (unchanged): attachment={a.attachmentKey}",
                    file=sys.__stderr__,
                )
            continue

        creators_str = None
        if getattr(a, "creators", None):
            creators_str = "; ".join([c for c in a.creators if isinstance(c, str) and c.strip()]) or None

        # (stype/ctype computed above)

        meta_base = {
            "itemKey": a.parentItemKey,
            "attachmentKey": a.attachmentKey,
            "title": a.title,
            "year": a.year,
            "creators": creators_str,
            "source_type": stype,
            "contentType": ctype,
            "filename": getattr(a, "filename", None),
            "path": str(file_path),
            "locator": None,
        }

        if show_progress:
            short_title = (a.title or "").strip()
            if not short_title:
                short_title = (getattr(a, "filename", None) or "").strip()
            if not short_title:
                short_title = file_path.name

            if len(short_title) > 80:
                short_title = short_title[:77] + "..."

            parent_disp = a.parentItemKey or "-"
            if parent_disp == "-":
                parent_disp = "- (orphan?)"

            # stype already computed above
            print(
                f"[PROGRESS] ({idx}/{total_attachments}) attachment={a.attachmentKey} "
                f"item={parent_disp} type={stype} {short_title}",
                file=sys.__stderr__,
            )

        t_pdf = time.perf_counter()
        if stype == "html":
            chunks = extract_chunks_from_html_snapshot(file_path, a.attachmentKey, meta_base)
        elif stype == "epub":
            chunks = extract_chunks_from_epub_snapshot(file_path, a.attachmentKey, meta_base)
        else:
            chunks = extract_chunks_from_pdf(file_path, a.attachmentKey, meta_base)

        if show_progress:
            dt = time.perf_counter() - t_pdf
            print(
                f"[PROGRESS]   ↳ extracted {len(chunks)} chunks in {dt:.1f}s",
                file=sys.__stderr__,
            )

        pending_delete_attachment_keys.add(a.attachmentKey)

        for cid, text, md in chunks:
            pending_ids.append(cid)
            pending_docs.append(text)
            pending_metas.append(md)

        pending_manifest_updates[a.attachmentKey] = {"mtime": mtime, "size": size, "pdf_path": str(file_path)}
        pending_source_types[a.attachmentKey] = stype

        if len(pending_ids) >= BATCH_SIZE:
            ids, docs, metas = _dedupe_by_id(pending_ids, pending_docs, pending_metas)

            _delete_by_attachment_keys(col, pending_delete_attachment_keys)

            _upsert_in_subbatches(
                col,
                ids,
                docs,
                metas,
                subbatch_size=BATCH_SIZE,
                show_progress=show_progress,
                label="upsert batch",
            )

            for ak, entry in pending_manifest_updates.items():
                files_manifest[ak] = entry
            for t in pending_source_types.values():
                if t == "html":
                    updated_html += 1
                elif t == "epub":
                    updated_epub += 1
                else:
                    updated_pdf += 1

            pending_manifest_updates.clear()
            pending_delete_attachment_keys.clear()
            pending_source_types.clear()

            pending_ids.clear()
            pending_docs.clear()
            pending_metas.clear()

    if pending_delete_attachment_keys:
        for dk in list(pending_delete_attachment_keys):
            try:
                col.delete(where={"attachmentKey": dk})
            except Exception:
                pass

    if pending_ids:
        ids, docs, metas = _dedupe_by_id(pending_ids, pending_docs, pending_metas)
        _upsert_in_subbatches(
            col,
            ids,
            docs,
            metas,
            subbatch_size=BATCH_SIZE,
            show_progress=show_progress,
            label="final upsert",
        )

    if pending_manifest_updates:
        for ak, entry in pending_manifest_updates.items():
            files_manifest[ak] = entry
        for t in pending_source_types.values():
            if t == "html":
                updated_html += 1
            elif t == "epub":
                updated_epub += 1
            else:
                updated_pdf += 1
        pending_manifest_updates.clear()
        pending_delete_attachment_keys.clear()
        pending_source_types.clear()

    # ----------------------------
    # Notes -> chunks (indexed, but excluded from rag_search by default)
    # ----------------------------
    try:
        notes = await api.list_notes(collection_key=args.collection)
    except Exception as e:
        notes = []
        print(f"[WARN] Failed to list notes via Zotero Local API: err={e}", file=sys.__stderr__)

    current_note_keys = {n.get("noteKey") for n in notes if isinstance(n, dict) and n.get("noteKey")}
    stale_note_keys = set(notes_manifest.keys()) - set(current_note_keys)
    deleted_stale_notes = 0
    for nk in stale_note_keys:
        try:
            col.delete(where={"noteKey": nk})
            deleted_stale_notes += 1
        except Exception:
            pass
        notes_manifest.pop(nk, None)

    updated_notes = 0
    skipped_notes = 0

    pending_note_ids: List[str] = []
    pending_note_docs: List[str] = []
    pending_note_metas: List[Dict[str, Any]] = []

    def _flush_notes_batch() -> None:
        nonlocal pending_note_ids, pending_note_docs, pending_note_metas
        if not pending_note_ids:
            return
        ids, docs, metas = _dedupe_by_id(pending_note_ids, pending_note_docs, pending_note_metas)
        _upsert_in_subbatches(
            col,
            ids,
            docs,
            metas,
            subbatch_size=BATCH_SIZE,
            show_progress=show_progress,
            label="notes upsert",
        )
        pending_note_ids = []
        pending_note_docs = []
        pending_note_metas = []

    for n in notes:
        if not isinstance(n, dict):
            continue
        note_key = n.get("noteKey")
        if not isinstance(note_key, str) or not note_key:
            continue

        ver = n.get("version")
        prev = notes_manifest.get(note_key)
        prev_ver = prev.get("version") if isinstance(prev, dict) else None
        if prev is not None and prev_ver == ver:
            skipped_notes += 1
            continue

        try:
            col.delete(where={"noteKey": note_key})
        except Exception:
            pass

        creators_str = None
        creators = n.get("creators")
        if isinstance(creators, list):
            creators_str = "; ".join([c for c in creators if isinstance(c, str) and c.strip()]) or None

        meta_base = {
            "itemKey": n.get("parentItemKey"),
            "attachmentKey": None,
            "noteKey": note_key,
            "title": n.get("title"),
            "year": n.get("year"),
            "creators": creators_str,
            "source_type": "note",
            "path": None,
            "pdf_path": None,
        }

        note_html = n.get("note_html") or ""
        note_text = clean_extracted_text(extract_main_text_from_html(note_html if isinstance(note_html, str) else ""))
        paras = normalize_paragraphs(note_text)
        if not paras:
            notes_manifest[note_key] = {"version": ver}
            updated_notes += 1
            continue

        joined = "\n\n".join(paras)
        if looks_like_gibberish(joined):
            notes_manifest[note_key] = {"version": ver}
            updated_notes += 1
            continue
        local_min_chunk = MIN_CHUNK_CHARS_NO_SPACE if is_no_space_language_document(joined) else MIN_CHUNK_CHARS

        for para_index, para_text in enumerate(paras):
            para_text = para_text.strip()
            if not para_text:
                continue
            parts = split_long_paragraph(para_text, max_chars=MAX_CHARS)
            for part_index, part in enumerate(parts):
                part = part.strip()
                if len(part) < local_min_chunk:
                    continue
                cid = f"{note_key}:note:para{para_index}:part{part_index}"
                md = dict(meta_base)
                md.update(
                    {
                        "locator": f"note:para{para_index}",
                        "para_index": int(para_index),
                        "part_index": int(part_index),
                    }
                )
                pending_note_ids.append(cid)
                pending_note_docs.append(part)
                pending_note_metas.append(md)

                if len(pending_note_ids) >= BATCH_SIZE:
                    _flush_notes_batch()

        _flush_notes_batch()
        notes_manifest[note_key] = {"version": ver}
        updated_notes += 1

    manifest["files"] = files_manifest
    manifest["notes"] = notes_manifest
    save_manifest(manifest)

    print(
        f"Done. Updated PDFs={updated_pdf}, Updated HTML(WebClip)={updated_html}, Updated EPUB={updated_epub}, "
        f"Skipped PDFs={skipped_pdf}, Skipped HTML(WebClip)={skipped_html}, Skipped EPUB={skipped_epub}, "
        f"Deleted stale={deleted_stale}"
        f" | Updated Notes={updated_notes}, Skipped Notes={skipped_notes}, Deleted stale Notes={deleted_stale_notes}"
    )
    if show_progress:
        print(f"[PROGRESS] Total runtime: {time.perf_counter() - t0:.1f}s", file=sys.__stderr__)


if __name__ == "__main__":
    asyncio.run(main_async(parse_args()))