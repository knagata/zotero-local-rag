# src/chapter_detect.py
"""
PDF / EPUB から章構造（目次）を取得するユーティリティ。

PDF:
  - fitz (PyMuPDF) の get_toc() でアウトラインを取得
  - ページ番号をキーに (chapter_title, section_title) を返す lookup 関数を生成

EPUB:
  - ebooklib の book.toc でTOCツリーを取得
  - chapter_index (= get_items_of_type(ITEM_DOCUMENT) の反復順) を
    章タイトルにマッピングする辞書を返す
"""
from __future__ import annotations

from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

try:
    import fitz  # PyMuPDF
    _HAS_FITZ = True
except ImportError:  # pragma: no cover
    _HAS_FITZ = False

try:
    import ebooklib  # type: ignore
    from ebooklib import epub as _ebooklib_epub  # type: ignore
    _ITEM_DOCUMENT = getattr(ebooklib, "ITEM_DOCUMENT", None)
    _HAS_EBOOKLIB = True
except ImportError:  # pragma: no cover
    _ebooklib_epub = None
    _ITEM_DOCUMENT = None
    _HAS_EBOOKLIB = False


# ─── PDF ────────────────────────────────────────────────────────────────────

def get_pdf_toc(pdf_path: str) -> List[Tuple[int, str, int]]:
    """
    PDFのアウトライン（目次）を返す。

    Returns:
        [(level, title, page_1indexed), ...]  ページ順ソート済み
        アウトラインなし・エラー時は []
    """
    if not _HAS_FITZ:
        return []
    try:
        doc = fitz.open(str(pdf_path))
        raw = doc.get_toc()
        doc.close()
    except Exception:
        return []

    result: List[Tuple[int, str, int]] = []
    for entry in raw:
        if len(entry) < 3:
            continue
        level, title, page = int(entry[0]), str(entry[1]).strip(), int(entry[2])
        if page < 1:
            page = 1
        if title:
            result.append((level, title, page))

    result.sort(key=lambda x: x[2])
    return result


def build_pdf_page_chapter_lookup(
    toc: List[Tuple[int, str, int]],
) -> Optional[Callable[[int], Tuple[str, str]]]:
    """
    PDFのTOCから「ページ番号 → (chapter_title, section_title)」を返す関数を生成。

    Args:
        toc: get_pdf_toc() の戻り値

    Returns:
        lookup(page: int) -> (chapter: str, section: str)
        TOCが空なら None
    """
    if not toc:
        return None

    # 意味のある章構造かどうかをざっくり確認
    # ほぼすべてがファイル名パターンなら None を返す
    import re
    _RE_FILENAME = re.compile(r"\.(pdf|jpg|jpeg|png|tif|tiff)$", re.IGNORECASE)
    filename_count = sum(1 for _, t, _ in toc if _RE_FILENAME.search(t))
    if filename_count > len(toc) * 0.5:
        return None

    def lookup(page: int) -> Tuple[str, str]:
        chapter = ""
        section = ""
        for level, title, toc_page in toc:
            if toc_page > page:
                break
            if level == 1:
                chapter = title
                section = ""
            elif level == 2:
                section = title
        return chapter, section

    return lookup


# ─── EPUB ───────────────────────────────────────────────────────────────────

def get_epub_chapter_index_to_title(epub_path: str) -> Dict[int, str]:
    """
    EPUB の TOC から {chapter_index: chapter_title} を返す。

    chapter_index は extract_chunks_from_epub_snapshot() が使う
    book.get_items_of_type(ITEM_DOCUMENT) の反復順インデックスと一致する。

    Returns:
        {0: "表紙", 1: "第1章 ...", ...}
        ebooklib 未インストール・TOCなし・エラー時は {}
    """
    if not _HAS_EBOOKLIB or _ebooklib_epub is None or _ITEM_DOCUMENT is None:
        return {}

    try:
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            book = _ebooklib_epub.read_epub(str(epub_path))
    except Exception:
        return {}

    # ── TOC ツリーから href → タイトル の辞書を構築 ──
    href_to_title: Dict[str, str] = {}

    def _walk_toc(items: list, depth: int = 1) -> None:
        for item in items:
            if isinstance(item, tuple):
                # (Section, [children])
                section_obj, children = item
                title = (getattr(section_obj, "title", None) or "").strip()
                href = (getattr(section_obj, "href", None) or "").split("#")[0]
                if href and title and href not in href_to_title:
                    href_to_title[href] = title
                _walk_toc(children, depth + 1)
            else:
                # epub.Link
                title = (getattr(item, "title", None) or "").strip()
                href = (getattr(item, "href", None) or "").split("#")[0]
                if href and title and href not in href_to_title:
                    href_to_title[href] = title

    try:
        _walk_toc(book.toc)
    except Exception:
        pass

    if not href_to_title:
        return {}

    # ── chapter_index → タイトル の辞書を構築 ──
    try:
        items = list(book.get_items_of_type(_ITEM_DOCUMENT))
    except Exception:
        return {}

    result: Dict[int, str] = {}
    for idx, item in enumerate(items):
        name = item.get_name() or ""
        # 完全一致を優先、なければファイル名部分で照合
        title = href_to_title.get(name)
        if not title:
            basename = name.split("/")[-1]
            for href_key, t in href_to_title.items():
                if href_key.split("/")[-1] == basename:
                    title = t
                    break
        if title:
            result[idx] = title

    return result
