# src/html_extract.py
from __future__ import annotations

import os
import posixpath
import re
from html import unescape
from pathlib import Path
from typing import Any, Dict, List, Tuple

from bs4 import BeautifulSoup  # type: ignore

try:
    from .text_utils import (
        HARD_MIN_CHARS, MAX_CHARS, TARGET_CHARS, MAX_CHARS_CJK, TARGET_CHARS_CJK,
        MIN_CHUNK_CHARS, MIN_CHUNK_CHARS_NO_SPACE, clean_extracted_text,
        is_no_space_language_document, joiner_for_text, looks_like_gibberish,
        merge_short_chunk_records, normalize_paragraphs, split_long_paragraph,
    )
except ImportError:  # pragma: no cover - direct src entrypoint
    from text_utils import (
        HARD_MIN_CHARS, MAX_CHARS, TARGET_CHARS, MAX_CHARS_CJK, TARGET_CHARS_CJK,
        MIN_CHUNK_CHARS, MIN_CHUNK_CHARS_NO_SPACE, clean_extracted_text,
        is_no_space_language_document, joiner_for_text, looks_like_gibberish,
        merge_short_chunk_records, normalize_paragraphs, split_long_paragraph,
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
    from .chapter_detect import get_epub_chapter_index_to_path as _get_epub_toc_path_map
except Exception:  # pragma: no cover
    try:
        from chapter_detect import get_epub_chapter_index_to_path as _get_epub_toc_path_map
    except Exception:
        _get_epub_toc_path_map = None  # type: ignore


HTML_SCRIPT_STYLE_RE = re.compile(r"(?is)<(script|style).*?>.*?(</\1>)")
MAX_HTML_BYTES = int(os.environ.get("MAX_HTML_BYTES", "10000000"))  # guard for huge snapshots

_BLOCK_TAGS = {"p", "li", "blockquote", "table", "figcaption", "dt", "dd", "aside"}
_HEADING_TAGS = {f"h{level}" for level in range(1, 7)}
_LEAF_CONTAINER_OWNERS = {"div", "section", "article"}
_LEAF_CONTAINER_EXCLUDED_TOKENS = {
    "nav", "navigation", "masthead", "header", "footer", "toolbar",
    "breadcrumb", "breadcrumbs", "menu", "pagination", "share", "sharing",
    "social", "advertisement", "advertising",
}
# EPUB conversion tools sometimes apply an ``hN`` CSS class to ordinary
# paragraph prose.  A real heading longer than this is not credible; retaining
# it as body text prevents a malformed tag from deleting an entire chapter.
_MALFORMED_HEADING_BODY_CHARS = 240
# "Sources" was in none of these, which is why a 72,150-character source list
# stayed classified as body text (GI85JWZH, 2026-07-28). The trailing "to
# Chapter 3" form matters as much as the bare word: a fullmatch against
# "Notes" alone misses every per-chapter note section in a book that has them.
_BIBLIOGRAPHY_RE = re.compile(
    r"^(?:references?|bibliography|select bibliography|further reading|sources|"
    r"works cited|参考文献|引用文献|文献一覧|文献目録|参考資料)"
    r"(?:\s*(?:to|for)\s+.+)?$", re.I)
_NOTE_RE = re.compile(
    r"^(?:notes?|endnotes?|footnotes?|注|注釈|註|巻末注|後注|原注|訳注|注記)"
    r"(?:\s*(?:to|for)\s+.+)?$", re.I)
# In these zones a block often packs many entries (one reference / one note each)
# into a single element separated by <br/>.  We split on <br/> so each entry
# becomes its own block, keeping the boundary intact through chunking (Part A of
# dev-notes/current/77).  Body prose stays paragraph-merged.
_ENTRY_ZONES = {"bibliography", "endnote", "footnote"}
_MIN_ENTRY_CHARS = 8
_TOC_RE = re.compile(r"^(?:目次|contents?|tableofcontents)$", re.I)
_INDEX_RE = re.compile(
    r"^(?:索引|人名索引|事項索引|地名索引|subject\s*index|name\s*index|general\s*index|index)$", re.I)
_COLOPHON_RE = re.compile(r"^(?:奥付|colophon|copyright)$", re.I)
_FRONT_MATTER_RE = re.compile(r"^(?:凡例|謝辞|序文|まえがき|acknowledg(?:e)?ments?)$", re.I)
_BACK_MATTER_RE = re.compile(
    r"^(?:あとがき|後書き|付録|付表|appendix|appendices|afterword|glossary|用語集)$", re.I)
_ZONE_TERMS = {
    "footnote": "footnote", "doc-footnote": "footnote",
    "endnote": "endnote", "doc-endnote": "endnote", "rearnote": "endnote",
    "bibliography": "bibliography", "references": "bibliography", "reference": "bibliography",
    "toc": "toc", "table-of-contents": "toc", "index": "index", "colophon": "colophon",
    "acknowledgments": "front_matter", "acknowledgements": "front_matter",
}


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


#: The walk stops *before* these, so their own attributes never classify the
#: content they contain. ``<main>`` is by definition the document's principal
#: content, so a zone term on it applies to everything inside and therefore
#: distinguishes nothing -- while the zones it can select are exclusionary.
#: A Squarespace page carries ``<main class="Index">`` (their name for a
#: landing-page layout, not a book index), which marked all 790 elements of a
#: 65,000-character essay ``zone="index"`` and so ``retrieval_policy="exclude"``:
#: the document vanished from search entirely (TA2PTL9B, 2026-07-28).
_CONTENT_ROOTS = {"html", "body", "main", None}


def _semantic_tokens(tag: Any) -> set[str]:
    """Collect EPUB/ARIA/class semantics from a tag and its ancestors."""
    tokens: set[str] = set()
    while tag is not None and getattr(tag, "name", None) not in _CONTENT_ROOTS:
        for attribute in ("epub:type", "role", "class", "id"):
            value = tag.get(attribute)
            for item in value if isinstance(value, list) else [value]:
                tokens.update(re.findall(r"[a-z0-9-]+", str(item or "").casefold()))
        tag = tag.parent
    return tokens


def _zone_for_element(tag: Any, heading_path: List[str]) -> str:
    tokens = _semantic_tokens(tag)
    for token, zone in _ZONE_TERMS.items():
        if token in tokens:
            return zone
    heading = heading_path[-1] if heading_path else ""
    heading = re.sub(r"[\s　]+", "", heading).strip("[]［］()（）")
    if _BIBLIOGRAPHY_RE.match(heading):
        return "bibliography"
    if _NOTE_RE.match(heading):
        return "endnote"
    if _TOC_RE.match(heading):
        return "toc"
    if _INDEX_RE.match(heading):
        return "index"
    if _COLOPHON_RE.match(heading):
        return "colophon"
    if _FRONT_MATTER_RE.match(heading):
        return "front_matter"
    if _BACK_MATTER_RE.match(heading):
        return "back_matter"
    return "body"


def _collect_element_ids(element: Any) -> set[str]:
    """Return the ``id`` of *element* and of every descendant.

    Note bodies carry the ``id`` that noterefs target (``id="note1"``), and the
    id may sit on the block itself or on a child, so descendant ids are included
    (Part C, dev-notes/current/77).
    """
    ids: set[str] = set()
    getter = getattr(element, "get", None)
    if callable(getter):
        own = getter("id")
        if own:
            ids.add(str(own))
    finder = getattr(element, "find_all", None)
    if callable(finder):
        for child in finder(id=True):
            child_id = child.get("id")
            if child_id:
                ids.add(str(child_id))
    return ids


def _noteref_targets(element: Any) -> List[str]:
    """Return the href fragment ids of noterefs inside *element*, in order.

    Only explicit ``epub:type~=noteref`` / ``role~=doc-noteref`` anchors count,
    so ordinary inline links are never mistaken for note references
    (Part C, dev-notes/current/77).
    """
    targets: List[str] = []
    links: List[Any] = []
    if getattr(element, "name", None) == "a":
        links.append(element)
    finder = getattr(element, "find_all", None)
    if callable(finder):
        links.extend(finder("a"))
    for link in links:
        href = link.get("href")
        if not href or "#" not in str(href):
            continue
        tokens: set[str] = set()
        for attr in ("epub:type", "role"):
            value = link.get(attr)
            for item in value if isinstance(value, list) else [value]:
                tokens.update(re.findall(r"[a-z0-9-]+", str(item or "").casefold()))
        if not (tokens & {"noteref", "doc-noteref"}):
            continue
        fragment = str(href).split("#", 1)[1].strip()
        if fragment and fragment not in targets:
            targets.append(fragment)
    return targets


def _noteref_hrefs(element: Any) -> List[str]:
    """Return raw hrefs for explicit note references, in document order."""
    hrefs: List[str] = []
    links: List[Any] = []
    if getattr(element, "name", None) == "a":
        links.append(element)
    finder = getattr(element, "find_all", None)
    if callable(finder):
        links.extend(finder("a"))
    for link in links:
        href = str(link.get("href") or "")
        if "#" not in href:
            continue
        tokens: set[str] = set()
        for attr in ("epub:type", "role"):
            value = link.get(attr)
            for item in value if isinstance(value, list) else [value]:
                tokens.update(re.findall(r"[a-z0-9-]+", str(item or "").casefold()))
        if tokens & {"noteref", "doc-noteref"} and href not in hrefs:
            hrefs.append(href)
    return hrefs


def _entry_records(element: Any, zone: str) -> List[Dict[str, Any]]:
    """Return one record per entry in a block element.

    Each record is ``{"text", "element_ids", "noteref_targets"}``.  Body blocks
    yield a single space-joined record (unchanged text behaviour).  Reference/note
    zones split into one record per ``<li>``/``<dd>``/``<p>`` entry or ``<br/>``
    boundary so each reference or note stays its own block; inline elements
    (``<em>``/``<a>``) never introduce a boundary, so a single reference is never
    over-split.  ``element_ids`` / ``noteref_targets`` carry the evidence needed
    to link footnote/endnote bodies back to their citing body chunk (Part C,
    dev-notes/current/77).
    """
    def _rec(text: str, source: Any) -> Dict[str, Any] | None:
        collapsed = " ".join(str(text or "").split())
        if not collapsed:
            return None
        return {
            "text": collapsed,
            "element_ids": _collect_element_ids(source) if source is not None else set(),
            "noteref_targets": _noteref_targets(source) if source is not None else [],
            "noteref_hrefs": _noteref_hrefs(source) if source is not None else [],
        }

    if zone not in _ENTRY_ZONES:
        rec = _rec(element.get_text(" ", strip=True), element)
        return [rec] if rec else []

    # Work on a detached copy so mutations never touch the shared tree.
    clone = BeautifulSoup(str(element), "html.parser")
    # Prefer explicit per-entry elements: a bibliography/notes container often
    # wraps one reference/note per <li> (or <dd>, or one-<p>-each) inside an
    # <aside>/<div>/<ol>.  extract_dom_blocks would otherwise emit the wrapper
    # as one space-joined blob and skip the items (Part A, dev-notes/current/77).
    for tag in ("li", "dd"):
        items = clone.find_all(tag)
        if items:
            recs = []
            for item in items:
                if len(" ".join(item.get_text(" ", strip=True).split())) < _MIN_ENTRY_CHARS:
                    continue
                rec = _rec(item.get_text(" ", strip=True), item)
                if rec:
                    recs.append(rec)
            if recs:
                return recs
    paragraphs = clone.find_all("p")
    if len(paragraphs) >= 2:
        recs = []
        for item in paragraphs:
            if len(" ".join(item.get_text(" ", strip=True).split())) < _MIN_ENTRY_CHARS:
                continue
            rec = _rec(item.get_text(" ", strip=True), item)
            if rec:
                recs.append(rec)
        if recs:
            return recs
    # Otherwise split on <br/> boundaries; inline text nodes join with a space so
    # a single reference is never over-split.  Per-segment id/noteref mapping is
    # not recoverable after a <br/> split, so the container's ids/noterefs are
    # kept at container granularity (allowed by Part C, dev-notes/current/77).
    container_ids = _collect_element_ids(clone)
    container_refs = _noteref_targets(clone)
    container_hrefs = _noteref_hrefs(clone)
    sentinel = "\x00"
    for br in clone.find_all("br"):
        br.replace_with(sentinel)
    raw = clone.get_text(" ")
    segments = [" ".join(segment.split()) for segment in raw.split(sentinel)]
    kept = [segment for segment in segments if len(segment) >= _MIN_ENTRY_CHARS]
    if not kept:
        collapsed = " ".join(raw.split())
        kept = [collapsed] if collapsed else []
    return [
        {
            "text": segment,
            "element_ids": set(container_ids),
            "noteref_targets": list(container_refs),
            "noteref_hrefs": list(container_hrefs),
        }
        for segment in kept
    ]


def extract_dom_blocks(raw_html: str, *, initial_path: List[str] | None = None) -> List[Dict[str, Any]]:
    """Return source-order blocks without flattening HTML/EPUB semantics.

    The result is intentionally small and JSON-like so EPUB and HTML callers
    can use the same canonical pre-chunk representation.  It preserves
    heading paths, tables, and zone evidence needed by document_structure.
    """
    soup = BeautifulSoup(raw_html, "html.parser")
    root = soup.body or soup
    heading_stack: List[str] = []
    heading_levels: List[int] = []
    seed = [part.strip() for part in (initial_path or []) if str(part).strip()]
    blocks: List[Dict[str, Any]] = []
    for element in root.find_all(list(_HEADING_TAGS | _BLOCK_TAGS)):
        if element.find_parent(["script", "style", "nav"]):
            continue
        if element.name in _HEADING_TAGS:
            title = " ".join(element.get_text(" ", strip=True).split())
            if not title:
                continue
            if len(title) > _MALFORMED_HEADING_BODY_CHARS:
                path = list(heading_stack)
                if seed and (not path or path[0] != seed[-1]):
                    path = seed + path
                zone = _zone_for_element(element, path)
                for record in _entry_records(element, zone):
                    blocks.append({
                        "text": record["text"],
                        "structure_path": path,
                        "zone": zone,
                        "block_type": "malformed_heading_body",
                        "element_ids": record["element_ids"],
                        "noteref_targets": record["noteref_targets"],
                        "noteref_hrefs": record["noteref_hrefs"],
                    })
                continue
            level = int(element.name[1])
            while heading_levels and heading_levels[-1] >= level:
                heading_levels.pop()
                heading_stack.pop()
            heading_levels.append(level)
            heading_stack.append(title)
            continue
        if element.find_parent(list(_BLOCK_TAGS)):
            # A table owns its cell text; a list item owns nested paragraphs.
            continue
        path = list(heading_stack)
        if seed and (not path or path[0] != seed[-1]):
            path = seed + path
        zone = _zone_for_element(element, path)
        block_type = "table" if element.name == "table" else element.name
        # Reference/note blocks are split on <br/> so each entry is its own block
        # and keeps its boundary through chunking (Part A, dev-notes/current/77).
        for record in _entry_records(element, zone):
            blocks.append({
                "text": record["text"],
                "structure_path": path,
                "zone": zone,
                "block_type": block_type,
                "element_ids": record["element_ids"],
                "noteref_targets": record["noteref_targets"],
                "noteref_hrefs": record["noteref_hrefs"],
            })
    return blocks


def extract_leaf_container_blocks(
    raw_html: str,
    *,
    initial_path: List[str] | None = None,
) -> List[Dict[str, Any]]:
    """Extract text owned by leaf ``div`` containers.

    Some publisher-generated EPUBs use a ``div`` for every paragraph and have
    no semantic paragraph elements.  This fallback deliberately accepts only
    leaf ownership containers, so ancestor text is never emitted a second
    time.  Callers should use it only when the normal semantic extractor
    produced no usable body blocks.
    """
    soup = BeautifulSoup(raw_html, "html.parser")
    root = soup.body or soup
    seed = [part.strip() for part in (initial_path or []) if str(part).strip()]
    blocks: List[Dict[str, Any]] = []
    heading_stack: List[str] = []
    heading_levels: List[int] = []

    for element in root.find_all(list(_HEADING_TAGS | {"div"})):
        if element.find_parent(["script", "style", "nav"]):
            continue
        if element.name in _HEADING_TAGS:
            title = " ".join(element.get_text(" ", strip=True).split())
            if not title:
                continue
            level = int(element.name[1])
            while heading_levels and heading_levels[-1] >= level:
                heading_levels.pop()
                heading_stack.pop()
            heading_levels.append(level)
            heading_stack.append(title)
            continue

        # The closest ownership container emits the text.  Semantic children
        # are excluded because the primary extractor, not this fallback,
        # should own those documents.
        if element.find(list(_LEAF_CONTAINER_OWNERS)):
            continue
        if element.find(list(_BLOCK_TAGS | _HEADING_TAGS)):
            continue
        tokens = _semantic_tokens(element)
        if tokens & _LEAF_CONTAINER_EXCLUDED_TOKENS:
            continue
        path = list(heading_stack)
        if seed and (not path or path[0] != seed[-1]):
            path = seed + path
        zone = _zone_for_element(element, path)
        for record in _entry_records(element, zone):
            if len(record["text"]) < 2:
                continue
            blocks.append({
                "text": record["text"],
                "structure_path": path,
                "zone": zone,
                "block_type": "div",
                "extraction_engine": "epub_dom_leaf_fallback",
                "element_ids": record["element_ids"],
                "noteref_targets": record["noteref_targets"],
                "noteref_hrefs": record["noteref_hrefs"],
            })
    return blocks


def _extract_epub_document_blocks(
    raw_html: str,
    *,
    initial_path: List[str] | None = None,
) -> List[Dict[str, Any]]:
    """Extract one EPUB spine document without retaining a token fragment.

    Publisher EPUBs occasionally put full prose in nested ``div`` elements but
    leave a table, caption, or quotation in a semantic tag.  In that shape the
    primary semantic extractor is non-empty, so a collection-wide ``no blocks``
    fallback silently loses the actual chapter.  Compare the two ownership
    strategies only for a small semantic result and replace that *document*
    when leaf containers provide decisively more text.  Documents that already
    have substantive semantic blocks keep their richer tag/zone semantics.
    """
    semantic_blocks = extract_dom_blocks(raw_html, initial_path=initial_path)
    semantic_chars = sum(len(str(block.get("text") or "")) for block in semantic_blocks)
    # Avoid parsing ordinary substantive chapters twice.  The value is only a
    # probe threshold; replacement still requires a large relative advantage.
    if semantic_chars >= 2_000:
        return semantic_blocks

    leaf_blocks = extract_leaf_container_blocks(raw_html, initial_path=initial_path)
    leaf_chars = sum(len(str(block.get("text") or "")) for block in leaf_blocks)
    if leaf_chars >= 500 and leaf_chars > max(semantic_chars * 3, semantic_chars + 1_000):
        return leaf_blocks
    return semantic_blocks


_ELEMENT_IDS_TEMP_KEY = "_element_ids"
_NOTEREF_TEMP_KEY = "_noteref_targets"
_NOTEREF_HREFS_TEMP_KEY = "_noteref_hrefs"


def _resolve_note_citations(
    chunks: List[Tuple[str, str, Dict[str, Any]]],
) -> List[Tuple[str, str, Dict[str, Any]]]:
    """Deterministically link footnote/endnote chunks to their citing body chunk.

    Builds ``(scope, note_id) -> citing_chunk_id`` from every chunk's noteref
    targets (the href fragments of ``<a epub:type="noteref">`` markers; first
    occurrence in document order wins), then stamps a scalar ``citing_chunk_id``
    on any chunk whose element ids include a linked note id.

    The scope is the spine document (``chapter_index``) so that EPUBs which reuse
    note ids per chapter (``note1``/``note2`` restart every chapter) resolve a
    same-file noteref to the note body in that same chapter, never to another
    chapter's identically-numbered note.  Cross-file endnotes (note body in a
    separate spine document) are intentionally left unlinked rather than linked
    wrongly.  HTML snapshots are a single document (scope ``None``).

    The intermediate ``_element_ids`` / ``_noteref_targets`` lists are stripped so
    only scalar metadata reaches Chroma (Part C, dev-notes/current/77).  EPUBs
    without any noteref get no ``citing_chunk_id`` (behaviour unchanged).
    """
    def _document_key(md: Dict[str, Any]) -> str:
        spine_path = md.get("_spine_path")
        if spine_path:
            return str(spine_path)
        chapter_index = md.get("chapter_index")
        return str(chapter_index) if chapter_index is not None else ""

    note_to_citing: Dict[Tuple[str, str], str] = {}
    for chunk_id, _text, md in chunks:
        document = _document_key(md)
        for href in md.get(_NOTEREF_HREFS_TEMP_KEY) or ():
            target_file, sep, fragment = str(href).partition("#")
            if not sep or not fragment:
                continue
            target_document = (
                posixpath.normpath(posixpath.join(posixpath.dirname(document), target_file))
                if target_file else document
            )
            note_to_citing.setdefault((target_document, fragment), chunk_id)

    for chunk_id, _text, md in chunks:
        if note_to_citing:
            document = _document_key(md)
            for element_id in md.get(_ELEMENT_IDS_TEMP_KEY) or []:
                citing = note_to_citing.get((document, str(element_id)))
                if citing and citing != chunk_id:
                    md["citing_chunk_id"] = citing
                    break
        md.pop(_ELEMENT_IDS_TEMP_KEY, None)
        md.pop(_NOTEREF_TEMP_KEY, None)
        md.pop(_NOTEREF_HREFS_TEMP_KEY, None)
        md.pop("_spine_path", None)
    return chunks


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

    blocks = extract_dom_blocks(raw_html)
    if not blocks:
        return [], default_quality

    sample = "\n\n".join(str(block["text"]) for block in blocks[:20])[:5000]
    if looks_like_gibberish(sample):
        return [], default_quality
    is_cjk = is_no_space_language_document(sample)
    local_min_chunk = MIN_CHUNK_CHARS_NO_SPACE if is_cjk else MIN_CHUNK_CHARS
    local_max_chars = MAX_CHARS_CJK if is_cjk else MAX_CHARS
    local_target_chars = TARGET_CHARS_CJK if is_cjk else TARGET_CHARS

    for block_index, block in enumerate(blocks):
        para_text = str(block["text"]).strip()
        if not para_text:
            continue
        structure_path = list(block.get("structure_path") or [])
        zone = str(block.get("zone") or "body")
        if structure_path and block.get("block_type") != "table":
            para_text = f"{structure_path[-1]}\n\n{para_text}"
        parts = split_long_paragraph(para_text, max_chars=local_max_chars, target_chars=local_target_chars)
        for part_index, part in enumerate(parts):
            part = part.strip()
            # Keep short records until merge_short_chunk_records has had a
            # chance to join adjacent body rows (notably EPUB chronologies).
            # It drops any genuinely isolated sub-HARD_MIN label afterwards.
            if len(part) < 1:
                continue
            chunk_id = f"{attachment_key}:html:block{block_index}:part{part_index}"
            md = dict(meta_base)
            md.update(
                {
                    "source_type": "html",
                    "locator": f"html:block{block_index}",
                    "path": str(html_path),
                    "pdf_path": str(html_path),  # backward compatibility
                    "block_index": int(block_index),
                    "part_index": int(part_index),
                    "zone": zone,
                    "block_type": block.get("block_type"),
                    "extraction_engine": "html_dom",
                    "extraction_version": "3",
                    **(({"structure_path": structure_path}) if structure_path else {}),
                }
            )
            block_element_ids = block.get("element_ids") or ()
            if block_element_ids:
                md[_ELEMENT_IDS_TEMP_KEY] = sorted(str(x) for x in block_element_ids)
            block_noterefs = block.get("noteref_targets") or ()
            if block_noterefs:
                md[_NOTEREF_TEMP_KEY] = [str(x) for x in block_noterefs]
            block_noteref_hrefs = block.get("noteref_hrefs") or ()
            if block_noteref_hrefs:
                md[_NOTEREF_HREFS_TEMP_KEY] = [str(x) for x in block_noteref_hrefs]
            chunks.append((chunk_id, part, md))

    chunks = merge_short_chunk_records(
        chunks, min_chars=local_min_chunk, max_chars=local_max_chars,
        boundary_key=lambda _cid, _text, md: (
            tuple(md.get("structure_path") or []), md.get("zone"), md.get("block_type"),
        ),
        union_keys=(_ELEMENT_IDS_TEMP_KEY, _NOTEREF_TEMP_KEY, _NOTEREF_HREFS_TEMP_KEY),
    )
    chunks = _resolve_note_citations(chunks)
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

    all_blocks: List[Dict[str, Any]] = []
    chap_idx = 0
    for item in book.get_items_of_type(ITEM_DOCUMENT):
        try:
            raw = item.get_content()  # bytes
            html = _decode_html_bytes(raw)
            document_blocks = _extract_epub_document_blocks(
                html, initial_path=_epub_toc_path_map.get(int(chap_idx), []),
            )
            for block_index, block in enumerate(document_blocks):
                block.update({
                    "chapter_index": chap_idx,
                    "block_index": block_index,
                    "spine_path": posixpath.normpath(str(item.get_name() or "")),
                })
                all_blocks.append(block)
        except Exception as e:
            if os.environ.get("DEBUG_HTML") == "1":
                print(
                    f"[DEBUG] EPUB chapter parse failed; continuing: attachment={attachment_key} file={epub_path} err={e}",
                    file=os.sys.__stderr__,
                )
        finally:
            chap_idx += 1

    if not all_blocks:
        try:
            from .epub_fallback import profile_epub
        except ImportError:  # pragma: no cover - direct src entrypoint
            from epub_fallback import profile_epub
        try:
            profile = profile_epub(epub_path)
            default_quality.update({
                "total_pages": int(profile["spine_document_count"]),
                "epub_profile": profile,
                "is_scanned": bool(profile["classification"] == "fixed_layout_image"),
                "scanned_pages": (
                    list(range(1, int(profile["spine_document_count"]) + 1))
                    if profile["classification"] == "fixed_layout_image" else []
                ),
            })
        except Exception as e:
            if os.environ.get("DEBUG_HTML") == "1":
                print(f"[DEBUG] EPUB profiling failed: {e}", file=os.sys.__stderr__)
        return [], default_quality

    sample = "\n\n".join(str(block["text"]) for block in all_blocks[:20])[:5000]
    if looks_like_gibberish(sample):
        return [], default_quality
    is_cjk = is_no_space_language_document(sample)
    local_min_chunk = MIN_CHUNK_CHARS_NO_SPACE if is_cjk else MIN_CHUNK_CHARS
    local_max_chars = MAX_CHARS_CJK if is_cjk else MAX_CHARS
    local_target_chars = TARGET_CHARS_CJK if is_cjk else TARGET_CHARS

    chunks: List[Tuple[str, str, Dict[str, Any]]] = []
    for global_block, block in enumerate(all_blocks):
        para_text = str(block["text"]).strip()
        if not para_text:
            continue

        structure_path = list(block.get("structure_path") or [])
        zone = str(block.get("zone") or "body")
        if structure_path and block.get("block_type") != "table":
            para_text = f"{structure_path[-1]}\n\n{para_text}"
        parts = split_long_paragraph(para_text, max_chars=local_max_chars, target_chars=local_target_chars)
        for part_index, part in enumerate(parts):
            part = part.strip()
            # See the HTML path above: merge before discarding short body rows.
            if len(part) < 1:
                continue

            chunk_id = f"{attachment_key}:epub:block{global_block}:part{part_index}"
            md = dict(meta_base)
            chapter_title = structure_path[-1] if structure_path else ""
            md.update(
                {
                    "source_type": "epub",
                    "locator": f"epub:spine{block['chapter_index']}:block{block['block_index']}",
                    "path": str(epub_path),
                    "pdf_path": str(epub_path),  # backward compatibility
                    "chapter_index": int(block["chapter_index"]),
                    "block_index": int(block["block_index"]),
                    "part_index": int(part_index),
                    "zone": zone,
                    "block_type": block.get("block_type"),
                    "extraction_engine": str(block.get("extraction_engine") or "epub_dom"),
                    "extraction_version": "3",
                    **(({"structure_path": structure_path}) if structure_path else {}),
                    **(({"chapter": chapter_title}) if chapter_title else {}),
                }
            )
            block_element_ids = block.get("element_ids") or ()
            if block_element_ids:
                md[_ELEMENT_IDS_TEMP_KEY] = sorted(str(x) for x in block_element_ids)
            block_noterefs = block.get("noteref_targets") or ()
            if block_noterefs:
                md[_NOTEREF_TEMP_KEY] = [str(x) for x in block_noterefs]
            block_noteref_hrefs = block.get("noteref_hrefs") or ()
            if block_noteref_hrefs:
                md[_NOTEREF_HREFS_TEMP_KEY] = [str(x) for x in block_noteref_hrefs]
            md["_spine_path"] = str(block.get("spine_path") or "")
            chunks.append((chunk_id, part, md))

    chunks = merge_short_chunk_records(
        chunks, min_chars=local_min_chunk, max_chars=local_max_chars,
        boundary_key=lambda _cid, _text, md: (
            tuple(md.get("structure_path") or []), md.get("zone"), md.get("block_type"),
        ),
        union_keys=(_ELEMENT_IDS_TEMP_KEY, _NOTEREF_TEMP_KEY, _NOTEREF_HREFS_TEMP_KEY),
    )
    chunks = _resolve_note_citations(chunks)
    ids = [cid for (cid, _, _) in chunks]
    if len(ids) != len(set(ids)):
        dup = len(ids) - len(set(ids))
        raise RuntimeError(f"Duplicate chunk ids generated for EPUB ({dup}).")
    return chunks, default_quality
