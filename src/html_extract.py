# src/html_extract.py
from __future__ import annotations

import os
import posixpath
import re
import unicodedata
import zipfile
from html import unescape
from pathlib import Path
from typing import Any, Dict, List, Tuple
from urllib.parse import unquote

from bs4 import BeautifulSoup  # type: ignore

try:
    from .text_utils import (
        HARD_MIN_CHARS, HARD_MIN_CHARS_CJK, MAX_CHARS, TARGET_CHARS, MAX_CHARS_CJK, TARGET_CHARS_CJK,
        MIN_CHUNK_CHARS, MIN_CHUNK_CHARS_NO_SPACE,
        is_no_space_language_document, looks_like_gibberish,
        merge_short_chunk_records, split_long_paragraph,
    )
    from . import heading_zone
    from .heading_zone import classify_heading_path
    from .source_coverage import make_source_coverage
except ImportError:  # pragma: no cover - direct src entrypoint
    from text_utils import (
        HARD_MIN_CHARS, HARD_MIN_CHARS_CJK, MAX_CHARS, TARGET_CHARS, MAX_CHARS_CJK, TARGET_CHARS_CJK,
        MIN_CHUNK_CHARS, MIN_CHUNK_CHARS_NO_SPACE, is_no_space_language_document, looks_like_gibberish,
        merge_short_chunk_records, split_long_paragraph,
    )
    import heading_zone
    from heading_zone import classify_heading_path
    from source_coverage import make_source_coverage

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
    from .chapter_detect import get_epub_href_to_toc_entries as _get_epub_href_toc_entries
except Exception:  # pragma: no cover
    try:
        from chapter_detect import get_epub_href_to_toc_entries as _get_epub_href_toc_entries
    except Exception:
        _get_epub_href_toc_entries = None  # type: ignore


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
_INDEPENDENT_HEADING_RE = re.compile(
    r"^(?:第?\s*[0-9０-９一二三四五六七八九十百]+\s*(?:章|講)|序章|終章|"
    r"PART\s+(?:[IVXLCDM]+|ONE|TWO|THREE|FOUR|FIVE)|CHAPTER\s+\d+|"
    r"[IVXLCDM]+\s+THE\s+)", re.IGNORECASE,
)
_SUBORDINATE_HEADING_RE = re.compile(
    r"^(?:▼|[0-9０-９]+[.)．。:]?\s|[一二三四五六七八九十百]+\s)",
)
# Vocabulary moved to heading_zone (the single definition, shared with
# pdf_extract.py and docling_extract.py). Aliased here rather than deleted so
# existing call sites and tests keep working; _zone_for_element itself now
# calls classify_heading_path directly.
_BIBLIOGRAPHY_RE = heading_zone.BIBLIOGRAPHY_RE
_NOTE_RE = heading_zone.ENDNOTE_RE
_TOC_RE = heading_zone.TOC_RE
_INDEX_RE = heading_zone.INDEX_RE
_COLOPHON_RE = heading_zone.COLOPHON_RE
_FRONT_MATTER_RE = heading_zone.FRONT_MATTER_RE
_BACK_MATTER_RE = heading_zone.BACK_MATTER_RE
# In these zones a block often packs many entries (one reference / one note each)
# into a single element separated by <br/>.  We split on <br/> so each entry
# becomes its own block, keeping the boundary intact through chunking (Part A of
# dev-notes/current/77).  Body prose stays paragraph-merged.
_ENTRY_ZONES = {"bibliography", "endnote", "footnote"}
_MIN_ENTRY_CHARS = 8
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
    # Checking only the last heading missed the ordinary case of a subsection
    # with no paratext vocabulary of its own nested under one that has: "Notes"
    # containing "Chapter 3" classified as body, because only "Chapter 3" was
    # checked. 2,715 chunks were affected (2026-07-28). classify_heading_path
    # checks the whole ancestor chain, outermost first.
    cleaned_path = [
        re.sub(r"[\s　]+", "", title).strip("[]［］()（）") for title in heading_path
    ]
    return classify_heading_path(cleaned_path)


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


def _heading_identity(value: str) -> str:
    value = unicodedata.normalize("NFKC", str(value or "")).casefold()
    plain = re.sub(r"[^\w\u3040-\u30ff\u3400-\u9fff]+", "", value)
    without_number = re.sub(r"^\s*(?:chapter\s+)?\d+[.:：]?\s*", "", value)
    without_number = re.sub(r"[^\w\u3040-\u30ff\u3400-\u9fff]+", "", without_number)
    return without_number or plain


def _clean_heading_title(value: str) -> str:
    """Remove publisher decoration that is not part of the visible title."""
    return re.sub(r"^▼\s*", "", " ".join(str(value or "").split())).strip()


def _same_heading(left: str, right: str) -> bool:
    return bool(_heading_identity(left)) and _heading_identity(left) == _heading_identity(right)


def _merge_heading_paths(seed: List[str], headings: List[str]) -> List[str]:
    """Combine canonical TOC ancestry with DOM headings without duplicates."""
    if not seed:
        return list(headings)
    if not headings:
        return list(seed)
    limit = min(len(seed), len(headings))
    for overlap in range(limit, 0, -1):
        if all(
            _same_heading(seed[len(seed) - overlap + index], headings[index])
            for index in range(overlap)
        ):
            return [*seed, *headings[overlap:]]
    # A publisher can retain an ancestor in the DOM stack after the TOC has
    # advanced at an inline/non-heading anchor.  Align the longest DOM prefix
    # represented anywhere in the canonical TOC path, then append only new
    # descendants.
    represented = 0
    for seed_index in range(len(seed)):
        matched = 0
        while (
            seed_index + matched < len(seed)
            and matched < len(headings)
            and _same_heading(seed[seed_index + matched], headings[matched])
        ):
            matched += 1
        represented = max(represented, matched)
    if represented:
        return [*seed, *headings[represented:]]
    return [*seed, *headings]


def _toc_paths_by_position(
    root: Any,
    toc_entries: List[Dict[str, Any]] | None,
) -> tuple[List[str], List[tuple[int, List[str]]]]:
    """Resolve TOC fragments to source-order DOM positions."""
    default_path: List[str] = []
    anchors: List[tuple[int, int, List[str]]] = []
    positions: Dict[str, int] = {}
    for position, element in enumerate(root.find_all(True)):
        element_id = str(element.get("id") or "")
        if element_id and element_id not in positions:
            positions[element_id] = position
    for ordinal, entry in enumerate(toc_entries or []):
        path = [str(part).strip() for part in entry.get("path") or [] if str(part).strip()]
        fragment = unquote(str(entry.get("fragment") or "").lstrip("#"))
        if not path:
            continue
        if not fragment and not default_path:
            default_path = path
        elif fragment in positions:
            anchors.append((positions[fragment], ordinal, path))
    anchors.sort(key=lambda value: (value[0], value[1]))
    return default_path, [(position, path) for position, _ordinal, path in anchors]


def _matching_toc_entries(
    entries_by_href: Dict[str, List[Dict[str, Any]]],
    document_path: str,
) -> List[Dict[str, Any]]:
    """Match a spine path exactly, using basename only when unambiguous."""
    normalised = posixpath.normpath(unquote(document_path))
    exact = [
        records for href, records in entries_by_href.items()
        if posixpath.normpath(unquote(href)) == normalised
    ]
    if exact:
        return list(exact[0])
    basename = posixpath.basename(normalised)
    basename_matches = [
        records for href, records in entries_by_href.items()
        if posixpath.basename(posixpath.normpath(unquote(href))) == basename
    ]
    return list(basename_matches[0]) if len(basename_matches) == 1 else []


def _structure_roles_for_path(
    path: List[str], entries: List[Dict[str, Any]],
) -> List[str]:
    best_path: List[str] = []
    best_roles: List[str] = []
    for entry in entries:
        candidate = [str(value) for value in entry.get("path") or []]
        if len(candidate) <= len(path) and path[:len(candidate)] == candidate and len(candidate) > len(best_path):
            best_path = candidate
            best_roles = [str(value) for value in entry.get("roles") or []]
    if not best_path:
        return []
    roles = best_roles[:len(best_path)]
    while len(roles) < len(path):
        roles.append("section" if roles and roles[-1] == "chapter" else "subsection")
    return roles


def _safe_carried_path(
    raw_html: str, carried_path: List[str], carried_roles: List[str] | None = None,
) -> List[str]:
    """Carry a TOC parent only across a plausible continuation document."""
    if not carried_path:
        return []
    soup = BeautifulSoup(raw_html, "html.parser")
    if soup.title:
        document_title = " ".join(soup.title.get_text(" ", strip=True).split())
        if document_title and classify_heading_path([document_title]) != "body":
            return []
    first_heading = soup.find(list(_HEADING_TAGS))
    if first_heading:
        raw_title = " ".join(first_heading.get_text(" ", strip=True).split())
        title = _clean_heading_title(raw_title)
        subordinate = bool(_SUBORDINATE_HEADING_RE.match(
            unicodedata.normalize("NFKC", raw_title),
        ))
        if classify_heading_path([title]) != "body" or _INDEPENDENT_HEADING_RE.match(
            unicodedata.normalize("NFKC", title),
        ):
            return []
        carried_titles = {
            _heading_identity(value) for value in carried_path if _heading_identity(value)
        }
        if (
            first_heading.name == "h1"
            and _heading_identity(title) not in carried_titles
            and not subordinate
        ):
            return []
        carried_leaf_is_detail = bool(
            carried_roles and carried_roles[-1:] in (["section"], ["subsection"])
        )
        if (
            first_heading.name == "h1" and subordinate and len(carried_path) > 1
            and carried_leaf_is_detail
        ):
            return list(carried_path[:-1])
        # A heading at the start of the next spine normally replaces the
        # previous heading at the same HTML level.  Retain only its ancestors;
        # otherwise Section 2 becomes a child of the carried Section 1.
        level = int(first_heading.name[1])
        if level > 1 and len(carried_path) > 1 and carried_leaf_is_detail:
            return list(carried_path[:-1])
    return list(carried_path)


def _remove_linked_noteref_suffixes(
    raw_html: str, entries: List[Dict[str, Any]],
) -> None:
    """Correct a TOC title only when its target DOM proves a linked suffix."""
    soup = BeautifulSoup(raw_html, "html.parser")
    for entry in entries:
        fragment = unquote(str(entry.get("fragment") or "").lstrip("#"))
        path = list(entry.get("path") or [])
        if not fragment or not path:
            continue
        target = soup.find(id=fragment)
        if target is None:
            continue
        links = target.find_all("a")
        if not links:
            continue
        suffix = "".join(link.get_text(" ", strip=True) for link in links[-1:]).strip()
        title = str(path[-1])
        if suffix.isdigit() and title.endswith(suffix):
            visible_without_suffix = title[:-len(suffix)].rstrip()
            if visible_without_suffix.endswith(("?", "!", "？", "！")):
                path[-1] = visible_without_suffix
                entry["path"] = path


def extract_dom_blocks(
    raw_html: str,
    *,
    initial_path: List[str] | None = None,
    toc_entries: List[Dict[str, Any]] | None = None,
) -> List[Dict[str, Any]]:
    """Return source-order blocks without flattening HTML/EPUB semantics.

    The result is intentionally small and JSON-like so EPUB and HTML callers
    can use the same canonical pre-chunk representation.  It preserves
    heading paths, tables, and zone evidence needed by document_structure.
    """
    soup = BeautifulSoup(raw_html, "html.parser")
    root = soup.body or soup
    heading_stack: List[str] = []
    heading_levels: List[int] = []
    toc_default, toc_anchors = _toc_paths_by_position(root, toc_entries)
    seed = toc_default or [part.strip() for part in (initial_path or []) if str(part).strip()]
    active_seed = list(seed)
    anchor_index = 0
    element_positions = {id(element): position for position, element in enumerate(root.find_all(True))}
    blocks: List[Dict[str, Any]] = []
    for element in root.find_all(list(_HEADING_TAGS | _BLOCK_TAGS)):
        position = element_positions.get(id(element), -1)
        while anchor_index < len(toc_anchors) and toc_anchors[anchor_index][0] <= position:
            active_seed = list(toc_anchors[anchor_index][1])
            anchor_index += 1
        if element.find_parent(["script", "style", "nav"]):
            continue
        if element.name in _HEADING_TAGS:
            title = _clean_heading_title(element.get_text(" ", strip=True))
            if not title:
                continue
            if len(title) > _MALFORMED_HEADING_BODY_CHARS:
                path = _merge_heading_paths(active_seed, heading_stack)
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
        path = _merge_heading_paths(active_seed, heading_stack)
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
    toc_entries: List[Dict[str, Any]] | None = None,
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
    toc_default, toc_anchors = _toc_paths_by_position(root, toc_entries)
    seed = toc_default or [part.strip() for part in (initial_path or []) if str(part).strip()]
    active_seed = list(seed)
    anchor_index = 0
    element_positions = {id(element): position for position, element in enumerate(root.find_all(True))}
    blocks: List[Dict[str, Any]] = []
    heading_stack: List[str] = []
    heading_levels: List[int] = []

    for element in root.find_all(list(_HEADING_TAGS | {"div"})):
        position = element_positions.get(id(element), -1)
        while anchor_index < len(toc_anchors) and toc_anchors[anchor_index][0] <= position:
            active_seed = list(toc_anchors[anchor_index][1])
            anchor_index += 1
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
        path = _merge_heading_paths(active_seed, heading_stack)
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
    toc_entries: List[Dict[str, Any]] | None = None,
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
    semantic_blocks = extract_dom_blocks(
        raw_html, initial_path=initial_path, toc_entries=toc_entries,
    )
    semantic_chars = sum(len(str(block.get("text") or "")) for block in semantic_blocks)
    # Avoid parsing ordinary substantive chapters twice.  The value is only a
    # probe threshold; replacement still requires a large relative advantage.
    if semantic_chars >= 2_000:
        return semantic_blocks

    leaf_blocks = extract_leaf_container_blocks(
        raw_html, initial_path=initial_path, toc_entries=toc_entries,
    )
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


def _read_html_skip_styles(path: Path) -> Tuple[str, bool]:
    """Read an HTML file, skipping <style> and <script> blocks.

    Return ``(html, reached_eof)``.  ``reached_eof`` is false when the
    cleaned-content guard stopped the read before EOF was observed.
    Callers must not treat that prefix as a complete document.

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
    reached_eof = False

    with open(path, "rb") as f:
        while cleaned_len < MAX_HTML_BYTES:
            chunk = f.read(CHUNK)
            if not chunk:
                reached_eof = True
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
                    # HTML tag names are ASCII-case-insensitive.  Searching
                    # the original bytes made ``</STYLE>`` / ``</SCRIPT>``
                    # appear unclosed, which discarded the rest of an
                    # otherwise complete snapshot.
                    end = chunk.lower().find(skip_close, i)
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

    return "".join(cleaned_parts), reached_eof


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
        raw_html, reached_eof = _read_html_skip_styles(html_path)
    except Exception as e:
        print(
            f"[WARN] Failed to read HTML snapshot: attachment={attachment_key} file={html_path} err={e}",
            file=os.sys.__stderr__,
        )
        # default_quality otherwise looks like a healthy, empty one-page
        # document -- indistinguishable from a document that genuinely has no
        # content. failure_reason is the only trace of what was actually
        # discarded (2026-07-29).
        return [], {**default_quality, "failure_reason": "html_read_failed"}

    if not reached_eof:
        # A prefix can contain enough valid DOM to produce chunks, but indexing
        # it would silently certify a truncated snapshot as complete.
        return [], {
            **default_quality,
            "failure_reason": "html_size_truncated",
            "source_read_complete": False,
        }

    blocks = extract_dom_blocks(raw_html)
    if not blocks:
        return [], {**default_quality, "failure_reason": "no_dom_blocks"}

    sample = "\n\n".join(str(block["text"]) for block in blocks[:20])[:5000]
    if looks_like_gibberish(sample):
        return [], {**default_quality, "failure_reason": "gibberish_sample"}
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
        hard_min_chars=HARD_MIN_CHARS_CJK if is_cjk else HARD_MIN_CHARS,
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
    # The OPF spine, not EbookLib's all-document iterator, defines the source
    # we promise to cover.  The latter includes navigation and other manifest
    # documents in some books and can omit the evidence needed to detect a
    # skipped spine item.
    try:
        try:
            from .epub_fallback import profile_epub
        except ImportError:  # pragma: no cover - direct src entrypoint
            from epub_fallback import profile_epub
        profile = profile_epub(epub_path)
    except Exception as e:
        print(
            f"[WARN] Failed to profile EPUB: attachment={attachment_key} file={epub_path} err={e}",
            file=os.sys.__stderr__,
        )
        return [], {
            **default_quality,
            "failure_reason": "epub_profile_failed",
            "source_coverage": make_source_coverage(
                unit_kind="spine", expected_units=[], attempted_units=[], text_units=[],
                failed_units=[], expected_known=False,
            ),
        }

    pages = list(profile.get("pages") or [])
    expected_spines = list(range(1, len(pages) + 1))
    default_quality.update({
        "total_pages": len(expected_spines),
        "spine_document_count": len(expected_spines),
        "expected_spines": expected_spines,
        "epub_profile": profile,
        "attempted_spines": [],
        "text_spines": [],
        "blank_spines": [],
        "ignored_image_spines": [],
        "failed_spines": [],
    })

    # A fixed-layout EPUB can have a tiny amount of wrapper/alt text.  That
    # text is not canonical source text: return no DOM chunks so the existing
    # fixed-layout OCR fallback below the extractor can run.
    if profile.get("classification") == "fixed_layout_image":
        default_quality.update({
            "is_scanned": True,
            "scanned_pages": expected_spines,
            "attempted_spines": expected_spines,
            # No wrapper DOM text is canonical in this classification.  OCR
            # must therefore account for every OPF spine, including a wrapper
            # that happens to contain a comparatively long caption/alt value.
            "failed_spines": expected_spines,
            "failure_reason": "fixed_layout_epub_requires_ocr",
            "source_coverage": make_source_coverage(
                unit_kind="spine", expected_units=expected_spines,
                attempted_units=expected_spines, text_units=[],
                failed_units=expected_spines,
            ),
        })
        return [], default_quality

    # Fragment-aware TOC paths (failure is non-fatal; DOM headings remain).
    _epub_href_entries: Dict[str, List[Dict[str, Any]]] = {}
    if _get_epub_href_toc_entries is not None:
        try:
            _epub_href_entries = _get_epub_href_toc_entries(str(epub_path))
        except Exception:
            pass
    all_toc_role_entries = [
        entry for records in _epub_href_entries.values() for entry in records
    ]

    def toc_entries_for(document_path: str) -> List[Dict[str, Any]]:
        return _matching_toc_entries(_epub_href_entries, document_path)

    all_blocks: List[Dict[str, Any]] = []
    block_spines: set[int] = set()
    attempted_spines: list[int] = []
    blank_spines: list[int] = []
    ignored_image_spines: list[int] = []
    failed_spines: list[int] = []
    image_primary = {
        int(index) + 1 for index in profile.get("image_primary_spine_indices") or []
    }
    carried_toc_path: List[str] = []
    carried_toc_roles: List[str] = []
    try:
        with zipfile.ZipFile(epub_path, "r") as archive:
            names = set(archive.namelist())
            for chap_idx, page in enumerate(pages):
                spine_unit = chap_idx + 1
                attempted_spines.append(spine_unit)
                document_path = str(page.get("document_href") or "")
                try:
                    if not document_path or document_path not in names:
                        raise FileNotFoundError(document_path or "missing OPF spine href")
                    raw = archive.read(document_path)
                    html = _decode_html_bytes(raw)
                    document_toc_entries = toc_entries_for(document_path)
                    _remove_linked_noteref_suffixes(html, document_toc_entries)
                    initial_toc_path = (
                        list(document_toc_entries[0].get("path") or [])
                        if document_toc_entries else _safe_carried_path(
                            html, carried_toc_path, carried_toc_roles,
                        )
                    )
                    if not document_toc_entries and carried_toc_path and not initial_toc_path:
                        carried_toc_path = []
                        carried_toc_roles = []
                    document_blocks = _extract_epub_document_blocks(
                        html,
                        initial_path=initial_toc_path,
                        toc_entries=document_toc_entries,
                    )
                    for block in document_blocks:
                        structure_path = [str(value) for value in block.get("structure_path") or []]
                        roles = _structure_roles_for_path(structure_path, all_toc_role_entries)
                        if roles:
                            block["structure_roles"] = roles
                    if document_toc_entries:
                        carried_toc_path = list(document_toc_entries[-1].get("path") or [])
                        carried_toc_roles = list(document_toc_entries[-1].get("roles") or [])
                    elif document_blocks:
                        # Preserve a DOM-derived chapter across later unlinked
                        # continuation spines.  The next document still passes
                        # through _safe_carried_path before this is accepted.
                        for block in reversed(document_blocks):
                            candidate = [
                                str(value) for value in block.get("structure_path") or []
                                if str(value).strip()
                            ]
                            if candidate:
                                carried_toc_path = candidate
                                carried_toc_roles = [
                                    str(value) for value in block.get("structure_roles") or []
                                ]
                                break
                    if not document_blocks:
                        # Blank is a positive finding only for a document with
                        # neither visible text nor an image wrapper.  Navigation
                        # and image-only spine documents otherwise remain an
                        # explicit failure for the coverage gate to handle.
                        if (
                            profile.get("classification") == "html_or_mixed"
                            and bool(page.get("has_image_content"))
                        ):
                            # For structured EPUBs the DOM is canonical.
                            # Image-only cover/logo/illustration spines are
                            # intentionally non-searchable and never OCRed.
                            blank_spines.append(spine_unit)
                            ignored_image_spines.append(spine_unit)
                        elif (
                            spine_unit not in image_primary
                            and int(page.get("visible_text_chars") or 0) == 0
                            and not bool(page.get("has_nontext_content"))
                        ):
                            blank_spines.append(spine_unit)
                        else:
                            failed_spines.append(spine_unit)
                        continue
                    for block_index, block in enumerate(document_blocks):
                        block.update({
                            "chapter_index": chap_idx,
                            "block_index": block_index,
                            "spine_path": posixpath.normpath(document_path),
                        })
                        all_blocks.append(block)
                        block_spines.add(spine_unit)
                except Exception as e:
                    failed_spines.append(spine_unit)
                    if os.environ.get("DEBUG_HTML") == "1":
                        print(
                            f"[DEBUG] EPUB chapter parse failed; continuing: attachment={attachment_key} file={epub_path} err={e}",
                            file=os.sys.__stderr__,
                        )
    except Exception as e:
        # An archive-level failure leaves all expected spines unaccounted; do
        # not turn any partial DOM output into a successful extraction.
        failed_spines = expected_spines
        if os.environ.get("DEBUG_HTML") == "1":
            print(f"[DEBUG] EPUB archive read failed: {e}", file=os.sys.__stderr__)

    if not all_blocks:
        default_quality.update({
            "attempted_spines": sorted(set(attempted_spines)),
            "blank_spines": sorted(set(blank_spines)),
            "ignored_image_spines": sorted(set(ignored_image_spines)),
            "failed_spines": sorted(set(failed_spines)),
            "failure_reason": "no_epub_blocks",
            "source_coverage": make_source_coverage(
                unit_kind="spine", expected_units=expected_spines,
                attempted_units=attempted_spines, text_units=[],
                blank_units=blank_spines, failed_units=failed_spines,
            ),
        })
        return [], default_quality

    sample = "\n\n".join(str(block["text"]) for block in all_blocks[:20])[:5000]
    if looks_like_gibberish(sample):
        default_quality.update({
            "attempted_spines": sorted(set(attempted_spines)),
            "blank_spines": sorted(set(blank_spines)),
            "ignored_image_spines": sorted(set(ignored_image_spines)),
            "failed_spines": sorted(set(failed_spines) | set(expected_spines)),
            "failure_reason": "gibberish_sample",
            "source_coverage": make_source_coverage(
                unit_kind="spine", expected_units=expected_spines,
                attempted_units=attempted_spines, text_units=[],
                blank_units=blank_spines, failed_units=set(failed_spines) | set(expected_spines),
            ),
        })
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
                    **(({"structure_roles": list(block.get("structure_roles") or [])})
                       if block.get("structure_roles") else {}),
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

    premerge_chunks = list(chunks)
    chunks = merge_short_chunk_records(
        premerge_chunks, min_chars=local_min_chunk, max_chars=local_max_chars,
        boundary_key=lambda _cid, _text, md: (
            md.get("chapter_index"), tuple(md.get("structure_path") or []),
            md.get("zone"), md.get("block_type"),
        ),
        union_keys=(_ELEMENT_IDS_TEMP_KEY, _NOTEREF_TEMP_KEY, _NOTEREF_HREFS_TEMP_KEY),
        hard_min_chars=HARD_MIN_CHARS_CJK if is_cjk else HARD_MIN_CHARS,
    )
    # Chunk-size policy must not erase an entire OPF spine. Preserve its
    # original short records only when merging would otherwise remove every
    # searchable trace of that source unit. This commonly covers title and
    # colophon pages without weakening the normal chunk-size policy.
    merged_spines = {
        int(metadata["chapter_index"])
        for _chunk_id, text, metadata in chunks
        if str(text or "").strip() and isinstance(metadata.get("chapter_index"), int)
    }
    premerge_spines = {
        int(metadata["chapter_index"])
        for _chunk_id, text, metadata in premerge_chunks
        if str(text or "").strip() and isinstance(metadata.get("chapter_index"), int)
    }
    missing_spines = premerge_spines - merged_spines
    if missing_spines:
        chunks.extend(
            row for row in premerge_chunks
            if row[2].get("chapter_index") in missing_spines
        )
        chunks.sort(key=lambda row: (
            int(row[2].get("chapter_index") or 0),
            int(row[2].get("block_index") or 0),
            int(row[2].get("part_index") or 0),
            row[0],
        ))
    chunks = _resolve_note_citations(chunks)
    ids = [cid for (cid, _, _) in chunks]
    if len(ids) != len(set(ids)):
        dup = len(ids) - len(set(ids))
        raise RuntimeError(f"Duplicate chunk ids generated for EPUB ({dup}).")
    text_spines = sorted({
        int(metadata["chapter_index"]) + 1
        for _chunk_id, text, metadata in chunks
        if str(text or "").strip() and isinstance(metadata.get("chapter_index"), int)
    })
    # An image-primary wrapper's short DOM text is normally a caption, alt
    # value, folio label, or accessibility description.  Preserve it as a
    # useful chunk, but never let it stand in for OCR of the page image.
    failed_spines = sorted(
        set(failed_spines)
        | (block_spines - set(text_spines))
        | image_primary
    )
    default_quality.update({
        "attempted_spines": sorted(set(attempted_spines)),
        "text_spines": text_spines,
        "blank_spines": sorted(set(blank_spines)),
        "ignored_image_spines": sorted(set(ignored_image_spines)),
        "failed_spines": sorted(set(failed_spines)),
        "source_coverage": make_source_coverage(
            unit_kind="spine", expected_units=expected_spines,
            attempted_units=attempted_spines, text_units=text_spines,
            blank_units=blank_spines, failed_units=failed_spines,
        ),
    })
    if failed_spines:
        default_quality["failure_reason"] = "epub_spine_coverage_incomplete"
    return chunks, default_quality
