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

import unicodedata
from typing import Any, Callable, Dict, List, Optional, Tuple

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
        title = "".join(
            character for character in unicodedata.normalize("NFKC", title)
            if unicodedata.category(character) != "Cf"
        ).strip()
        if page < 1:
            page = 1
        if title:
            result.append((level, title, page))

    result.sort(key=lambda x: x[2])
    return result


def build_pdf_outline_tree(toc: List[Tuple[int, str, int]]) -> List[Dict[str, Any]]:
    """Return every PDF outline level as a nested, source-order-preserving tree.

    ``get_pdf_toc`` remains a compatibility API.  New consumers must use this
    tree instead of discarding levels deeper than two.
    """
    roots: List[Dict[str, Any]] = []
    stack: List[Dict[str, Any]] = []
    for ordinal, entry in enumerate(toc):
        if len(entry) < 3:
            continue
        level, title, page = int(entry[0]), str(entry[1]).strip(), max(1, int(entry[2]))
        if not title:
            continue
        level = max(1, level)
        while len(stack) >= level:
            stack.pop()
        # Malformed PDFs sometimes jump from level 1 to level 4.  Preserve the
        # entry, but attach it to the nearest actual parent rather than inventing
        # phantom headings.
        parent = stack[-1] if stack else None
        node = {
            "title": title, "level": level, "page": page, "ordinal": ordinal,
            "children": [], "parent_title": parent["title"] if parent else None,
        }
        if parent is None:
            roots.append(node)
        else:
            parent["children"].append(node)
        stack.append(node)
    return roots


def get_pdf_outline_tree(pdf_path: str) -> List[Dict[str, Any]]:
    """Read a PDF outline without flattening its hierarchy."""
    if not _HAS_FITZ:
        return []
    try:
        doc = fitz.open(str(pdf_path))
        raw = doc.get_toc()
        doc.close()
    except Exception:
        return []
    entries: List[Tuple[int, str, int]] = []
    for entry in raw:
        if len(entry) < 3:
            continue
        try:
            level, title, page = int(entry[0]), str(entry[1]).strip(), max(1, int(entry[2]))
        except (TypeError, ValueError):
            continue
        if title:
            entries.append((level, title, page))
    return build_pdf_outline_tree(entries)


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


def build_pdf_page_structure_path_lookup(
    toc: List[Tuple[int, str, int]],
) -> Optional[Callable[[int], List[str]]]:
    """Map a page to the active full PDF outline path."""
    if not toc:
        return None
    usable = [
        (max(1, int(level)), str(title).strip(), max(1, int(page)))
        for level, title, page in toc if str(title).strip()
    ]
    if not usable:
        return None

    def lookup(page: int) -> List[str]:
        active: List[str] = []
        for level, title, toc_page in usable:
            if toc_page > page:
                break
            active = active[: level - 1]
            active.append(title)
        return active

    return lookup


# ─── EPUB ───────────────────────────────────────────────────────────────────

def _epub_toc_node(item: Any, *, depth: int, ordinal: int) -> Optional[Dict[str, Any]]:
    title = (getattr(item, "title", None) or "").strip()
    href = (getattr(item, "href", None) or "").strip()
    if not title and not href:
        return None
    path, _, fragment = href.partition("#")
    return {
        "title": title or path or "(untitled)", "href": path, "fragment": fragment,
        "depth": depth, "ordinal": ordinal, "children": [],
    }


def build_epub_toc_tree(toc: List[Any]) -> List[Dict[str, Any]]:
    """Normalise EbookLib's mixed Link/(Section, children) TOC representation."""
    roots: List[Dict[str, Any]] = []

    def walk(items: List[Any], depth: int) -> List[Dict[str, Any]]:
        output: List[Dict[str, Any]] = []
        for ordinal, item in enumerate(items):
            source = item[0] if isinstance(item, tuple) and item else item
            children = item[1] if isinstance(item, tuple) and len(item) > 1 else []
            node = _epub_toc_node(source, depth=depth, ordinal=ordinal)
            if node is None:
                # Keep descendants even when a malformed container has no title.
                output.extend(walk(list(children or []), depth))
                continue
            node["children"] = walk(list(children or []), depth + 1)
            output.append(node)
        return output

    roots.extend(walk(list(toc or []), 1))
    return roots


def get_epub_toc_tree(epub_path: str) -> List[Dict[str, Any]]:
    """Read an EPUB navigation tree with depth, href and fragment intact."""
    if not _HAS_EBOOKLIB or _ebooklib_epub is None:
        return []
    try:
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            book = _ebooklib_epub.read_epub(str(epub_path))
        return build_epub_toc_tree(list(book.toc or []))
    except Exception:
        return []


def get_epub_chapter_index_to_path(epub_path: str) -> Dict[int, List[str]]:
    """Map EPUB spine document indexes to their complete TOC title paths."""
    if not _HAS_EBOOKLIB or _ebooklib_epub is None or _ITEM_DOCUMENT is None:
        return {}
    try:
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            book = _ebooklib_epub.read_epub(str(epub_path))
        tree = build_epub_toc_tree(list(book.toc or []))
        path_by_href: Dict[str, List[str]] = {}

        def walk(nodes: List[Dict[str, Any]], prefix: List[str]) -> None:
            for node in nodes:
                current = [*prefix, str(node["title"])]
                href = str(node.get("href") or "")
                if href and href not in path_by_href:
                    path_by_href[href] = current
                walk(list(node.get("children") or []), current)

        walk(tree, [])
        result: Dict[int, List[str]] = {}
        for index, item in enumerate(book.get_items_of_type(_ITEM_DOCUMENT)):
            name = str(item.get_name() or "")
            path = path_by_href.get(name)
            if path is None:
                basename = name.split("/")[-1]
                path = next(
                    (candidate for href, candidate in path_by_href.items() if href.split("/")[-1] == basename),
                    None,
                )
            if path:
                result[index] = path
        return result
    except Exception:
        return {}

def get_epub_chapter_index_to_title(epub_path: str) -> Dict[int, str]:
    """
    EPUB の TOC から {chapter_index: chapter_title} を返す。

    chapter_index は extract_chunks_from_epub_snapshot() が使う
    book.get_items_of_type(ITEM_DOCUMENT) の反復順インデックスと一致する。

    Returns:
        {0: "表紙", 1: "第1章 ...", ...}
        ebooklib 未インストール・TOCなし・エラー時は {}
    """
    return {
        index: path[-1]
        for index, path in get_epub_chapter_index_to_path(epub_path).items()
        if path
    }
