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

import re
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

_PDF_FILENAME_BOOKMARK = re.compile(
    r"\.(?:pdf|jpg|jpeg|png|tif|tiff)$", re.IGNORECASE,
)
_PDF_GENERATED_PAGE_BOOKMARK = re.compile(
    r"^\d{5}___[0-9a-f]{32,64}$", re.IGNORECASE,
)


def _is_usable_pdf_toc(toc: List[Tuple[int, str, int]]) -> bool:
    """Reject machine navigation indexes that contain no semantic headings."""
    if not toc:
        return False
    suspicious = sum(
        bool(_PDF_FILENAME_BOOKMARK.search(title))
        or bool(_PDF_GENERATED_PAGE_BOOKMARK.fullmatch(title))
        for _level, title, _page in toc
    )
    return suspicious <= len(toc) * 0.5


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
            if unicodedata.category(character) not in {"Cf", "Cc"}
        ).strip()
        if page < 1:
            page = 1
        if title:
            result.append((level, title, page))

    result.sort(key=lambda x: x[2])
    recovered = _recover_flat_pdf_toc(result)
    return recovered if _is_usable_pdf_toc(recovered) else []


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
    return build_pdf_outline_tree(get_pdf_toc(pdf_path))


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
    if not _is_usable_pdf_toc(toc):
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
    if not _is_usable_pdf_toc(toc):
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
    entries = get_epub_chapter_index_to_toc_entries(epub_path)
    return {
        index: list(records[0]["path"])
        for index, records in entries.items()
        if records
    }


_PART_RE = re.compile(
    r"^(?:第\s*(?:[0-9一二三四五六七八九十百]+|[IVXLCDM]+)\s*部|"
    r"PART\s+(?:[0-9]+|[IVXLCDM]+|ONE|TWO|THREE|FOUR|FIVE|SIX|SEVEN|EIGHT|NINE|TEN)|"
    r"[IVXLCDM]+\s+THE\s+)", re.IGNORECASE,
)
_CHAPTER_RE = re.compile(
    r"^(?:第?\s*[0-9一二三四五六七八九十百]+\s*(?:章|講)|"
    r"序章|終章|序論|結論|序\s|はじめに|おわりに|補考(?:\s|$)|[注註](?:\s|$))",
    re.IGNORECASE,
)
_EXPLICIT_SECTION_RE = re.compile(
    r"^第?\s*[0-9一二三四五六七八九十百]+\s*節(?:\s|$)",
)
_NUMBERED_HEADING_RE = re.compile(
    r"^(?:[0-9]+[.)．。:]?|[一二三四五六七八九十百]+)(?:\s|　)",
)
_JAPANESE_NOTE_CHILD_RE = re.compile(
    r"^[［\[](?:第\s*[0-9０-９一二三四五六七八九十百]+\s*章|序章|終章)[］\]]$",
)
_BACK_MATTER_RE = re.compile(
    r"^(?:あとがき|謝辞|参考文献|主要参考文献|索引|奥付|著者略歴|脚注|"
    r"ACKNOWLEDGMENTS?|CREDITS?|GLOSSARY|INDEX|NOTES?|ENDNOTES?|FOOTNOTES?|"
    r"BIBLIOGRAPHY|REFERENCES|FURTHER READING|ABOUT THE AUTHOR|BACK MATTER)$",
    re.IGNORECASE,
)
_ROMAN_CONTAINER_RE = re.compile(r"^[IVXLCDM]+[.)]\s+", re.IGNORECASE)


def _recover_flat_pdf_toc(
    toc: List[Tuple[int, str, int]],
) -> List[Tuple[int, str, int]]:
    """Recover explicit Part/chapter relations from a level-one PDF outline."""
    if not toc or any(int(level) != 1 for level, _title, _page in toc):
        return list(toc)
    nodes = [
        {"title": title, "children": []}
        for _level, title, _page in toc
    ]
    kinds = [_toc_title_kind(title) for _level, title, _page in toc]
    for index, node in enumerate(nodes[:-1]):
        title = str(node["title"])
        if _ROMAN_CONTAINER_RE.match(title) and kinds[index + 1] == "numbered":
            node["title"] = f"PART {title}"
    recovered = _recover_flat_toc_paths(nodes)
    output: List[Tuple[int, str, int]] = []
    for node, (_level, original_title, page) in zip(nodes, toc):
        path = recovered.get(id(node)) or [str(node["title"])]
        output.append((len(path), original_title, page))
    return output


def _toc_title_kind(title: str) -> str:
    value = " ".join(unicodedata.normalize("NFKC", title).split())
    if _PART_RE.match(value):
        return "part"
    if _CHAPTER_RE.match(value):
        return "chapter"
    if _EXPLICIT_SECTION_RE.match(value):
        return "section"
    if _NUMBERED_HEADING_RE.match(value):
        return "numbered"
    if _BACK_MATTER_RE.match(value):
        return "back_matter"
    return "other"


def _recover_flat_toc_paths(tree: List[Dict[str, Any]]) -> Dict[int, List[str]]:
    """Recover only unambiguous chapter/section relations in a flat TOC.

    Some Japanese EPUB producers emit every navigation point at depth one even
    though chapter and section labels are explicit.  Keeping this deliberately
    narrow avoids inventing hierarchy for ordinary unnumbered navigation lists.
    Keys are object identities so repeated titles remain independent.
    """
    if not tree or any(node.get("children") for node in tree):
        return {}
    recovered: Dict[int, List[str]] = {}
    part: str | None = None
    chapter: str | None = None
    section: str | None = None
    kinds = [_toc_title_kind(str(node.get("title") or "").strip()) for node in tree]
    for index, node in enumerate(tree):
        title = str(node.get("title") or "").strip()
        kind = kinds[index]
        next_kind = kinds[index + 1] if index + 1 < len(kinds) else ""
        if kind == "part":
            part, chapter, section = title, None, None
            path = [title]
        elif kind == "chapter":
            if "講" in unicodedata.normalize("NFKC", title):
                part = None
            chapter = title
            section = None
            path = [*([part] if part else []), title]
        elif kind == "section" and chapter:
            section = title
            path = [*([part] if part else []), chapter, title]
        elif kind == "numbered":
            base = [*([part] if part else []), *([chapter] if chapter else [])]
            path = [*base, title] if base else [title]
            # A plain numbered heading is a leaf unless an explicit lower
            # level follows; keeping it out of state prevents the next sibling
            # from being nested beneath it.
            section = None
        elif chapter == "注" and _JAPANESE_NOTE_CHILD_RE.match(title):
            path = [*([part] if part else []), chapter, title]
        elif part and next_kind == "numbered":
            # Flat English contents commonly express Part -> unnumbered
            # chapter/layer -> numbered subsection solely through ordering.
            chapter, section = title, None
            path = [part, title]
        elif kind == "back_matter":
            part = chapter = section = None
            path = [title]
        elif section:
            path = [*([part] if part else []), *([chapter] if chapter else []), section, title]
        elif chapter:
            path = [*([part] if part else []), chapter, title]
        else:
            # An unnumbered item following a numbered leaf can begin a new
            # chapter only when the next item proves it by being numbered.
            # Otherwise it is independent paratext rather than invented
            # ancestry.
            if chapter and next_kind == "numbered":
                chapter, section = title, None
                path = [*([part] if part else []), title]
            else:
                part = chapter = section = None
                path = [title]
        recovered[id(node)] = path
    return recovered


def get_epub_chapter_index_to_toc_entries(
    epub_path: str,
) -> Dict[int, List[Dict[str, Any]]]:
    """Map each EPUB document to all TOC paths and fragment anchors it owns.

    Unlike :func:`get_epub_chapter_index_to_path`, this retains multiple
    navigation points targeting one XHTML document.  Extractors can therefore
    switch from a chapter path to its section path at the matching element id.
    """
    if not _HAS_EBOOKLIB or _ebooklib_epub is None or _ITEM_DOCUMENT is None:
        return {}
    try:
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            book = _ebooklib_epub.read_epub(str(epub_path))
        entries_by_href = _toc_entries_by_href(build_epub_toc_tree(list(book.toc or [])))
        result: Dict[int, List[Dict[str, Any]]] = {}
        # OCR/fixed-layout locators use the OPF spine ordinal. Enumerating all
        # document manifest items shifts that ordinal whenever a nav or other
        # non-spine XHTML item exists.
        for index, spine_entry in enumerate(book.spine or []):
            idref = spine_entry[0] if isinstance(spine_entry, (list, tuple)) else spine_entry
            item = book.get_item_with_id(str(idref))
            if item is None:
                continue
            name = str(item.get_name() or "")
            records = entries_by_href.get(name)
            if records is None:
                basename = name.split("/")[-1]
                matches = [
                    candidate for href, candidate in entries_by_href.items()
                    if href.split("/")[-1] == basename
                ]
                records = matches[0] if len(matches) == 1 else None
            if records:
                result[index] = records
        return result
    except Exception:
        return {}


def _toc_entries_by_href(
    tree: List[Dict[str, Any]],
) -> Dict[str, List[Dict[str, Any]]]:
    flat_recovery = _recover_flat_toc_paths(tree)
    root_part_recovery: Dict[int, List[str]] = {}
    active_root_part: str | None = None
    for node in tree:
        title = str(node.get("title") or "").strip()
        kind = _toc_title_kind(title)
        if kind == "part":
            active_root_part = title
            root_part_recovery[id(node)] = [title]
        elif active_root_part and kind in {"numbered", "chapter"}:
            root_part_recovery[id(node)] = [active_root_part, title]
        else:
            active_root_part = None
    entries: Dict[str, List[Dict[str, Any]]] = {}
    roman_root_candidates = [
        node for node in tree
        if re.match(r"^[IVXLCDM]+\s+[A-Z]", str(node.get("title") or ""))
    ]
    roman_part_root_ids = {
        id(node) for node in roman_root_candidates
        if any(
            _toc_title_kind(str(child.get("title") or "")) == "numbered"
            for child in node.get("children") or []
        )
    }

    def walk(nodes: List[Dict[str, Any]], prefix: List[str], prefix_roles: List[str]) -> None:
        for node in nodes:
            recovered = flat_recovery.get(id(node)) or root_part_recovery.get(id(node))
            current = recovered or [*prefix, str(node["title"])]
            if recovered:
                current_roles = _roles_for_toc_path(current)
            else:
                kind = _toc_title_kind(str(node.get("title") or ""))
                if not prefix_roles:
                    role = "part" if kind == "part" or id(node) in roman_part_root_ids else "chapter"
                elif prefix_roles[-1] == "part":
                    role = "chapter"
                elif prefix_roles[-1] == "chapter":
                    role = "section"
                else:
                    role = "subsection"
                current_roles = [*prefix_roles, role]
            href = str(node.get("href") or "")
            if href:
                entries.setdefault(href, []).append({
                    "fragment": str(node.get("fragment") or ""),
                    "path": current,
                    "roles": current_roles,
                })
            walk(list(node.get("children") or []), current, current_roles)

    walk(tree, [], [])
    return entries


def _roles_for_toc_path(path: List[str]) -> List[str]:
    """Describe semantic heading roles retained alongside a title path."""
    if not path:
        return []
    kinds = [_toc_title_kind(title) for title in path]
    roles = ["chapter"] * len(path)
    first_is_part = kinds[0] == "part"
    if (
        len(path) >= 2 and _ROMAN_CONTAINER_RE.match(
            " ".join(unicodedata.normalize("NFKC", path[0]).split())
        ) and kinds[1] == "numbered"
    ):
        first_is_part = True
    # A nested root with numbered children is also a publisher-level Part even
    # when its title is not literally ``Part`` (for example Cultural Analytics
    # uses ``I Studying Culture at Scale``).
    if (
        len(path) >= 3 and kinds[1] in {"numbered", "chapter"}
        and re.match(r"^[IVXLCDM]+\s+[A-Z]", path[0])
    ):
        first_is_part = True
    if first_is_part:
        roles[0] = "part"
    for index in range(1, len(path)):
        if index == 1 and first_is_part:
            title = unicodedata.normalize("NFKC", path[index])
            japanese_direct_section = (
                kinds[0] == "part" and bool(re.match(
                    r"^第\s*(?:[0-9一二三四五六七八九十百]+|[IVXLCDM]+)\s*部",
                    unicodedata.normalize("NFKC", path[0]), re.IGNORECASE,
                )) and kinds[index] == "numbered"
                and "章" not in title and len(path) == 2
            )
            roles[index] = "section" if japanese_direct_section else "chapter"
        elif index == 1:
            roles[index] = "section"
        else:
            roles[index] = "section" if roles[index - 1] == "chapter" else "subsection"
    return roles


def infer_structure_roles(path: List[str]) -> List[str]:
    """Infer stable Part/chapter/section roles for a canonical heading path."""
    return _roles_for_toc_path([str(value) for value in path if str(value).strip()])


def get_epub_href_to_toc_entries(epub_path: str) -> Dict[str, List[Dict[str, Any]]]:
    """Return fragment-aware TOC entries keyed by their archive document href."""
    if not _HAS_EBOOKLIB or _ebooklib_epub is None:
        return {}
    try:
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            book = _ebooklib_epub.read_epub(str(epub_path))
        return _toc_entries_by_href(build_epub_toc_tree(list(book.toc or [])))
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
