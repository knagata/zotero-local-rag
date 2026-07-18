import os
import sys
import zipfile
import re
from typing import List, Dict, Any
from bs4 import BeautifulSoup


REFERENCE_HEADING_RE = re.compile(
    r"^\s*(?:references?|bibliography|works cited|参考文献|引用文献|文献一覧)\s*$",
    re.IGNORECASE,
)
NOTE_HEADING_RE = re.compile(
    r"^\s*(?:notes?|endnotes?|footnotes?|注|注釈|註)\s*$",
    re.IGNORECASE,
)
NOTE_TERMS = {"noteref", "doc-noteref", "footnote", "doc-footnote", "endnote", "doc-endnote", "rearnote"}
BIBLIOGRAPHY_TERMS = {"bibliography", "doc-bibliography", "references", "reference", "works-cited"}


def _pbar(current: int, total: int, label: str = "", width: int = 30) -> None:
    """1行でインプレース更新するプログレスバー。完了時は改行を出す。"""
    filled = int(width * current / total) if total else width
    bar = "█" * filled + "░" * (width - filled)
    pct = current / total * 100 if total else 100
    end = "\n" if current >= total else "\r"
    print(f"        [{bar}] {current:>5}/{total}  {pct:5.1f}%  {label}",
          end=end, file=sys.stderr, flush=True)


def _semantic_tokens(tag: Any) -> set[str]:
    tokens: set[str] = set()
    while tag is not None and getattr(tag, "name", None) not in {"html", "body", None}:
        for attribute in ("epub:type", "role", "class", "id"):
            value = tag.get(attribute)
            values = value if isinstance(value, list) else [value]
            for item in values:
                tokens.update(re.findall(r"[a-z0-9-]+", str(item or "").casefold()))
        tag = tag.parent
    return tokens


def _reference_zone(
    target: Any, target_file: str, *, explicit_noteref: bool, has_backlink: bool,
) -> str | None:
    tokens = _semantic_tokens(target)
    filename_tokens = set(re.findall(r"[a-z0-9-]+", target_file.casefold()))
    combined = tokens | filename_tokens
    if combined & BIBLIOGRAPHY_TERMS or any(
        token.startswith(("bib", "ref")) for token in combined
    ):
        return "bibliography"
    if any("endnote" in token or "rearnote" in token for token in combined):
        return "endnote"
    if any("footnote" in token for token in combined):
        return "footnote"
    if combined & NOTE_TERMS or any(token.startswith("note") for token in combined):
        return "endnote"
    heading = target.find_previous(["h1", "h2", "h3", "h4", "h5", "h6"])
    if heading and REFERENCE_HEADING_RE.match(heading.get_text(" ", strip=True)):
        return "bibliography"
    if heading and NOTE_HEADING_RE.match(heading.get_text(" ", strip=True)):
        return "endnote"
    block = (
        target if target.name in {"p", "div", "li", "aside", "td", "tr"}
        else target.find_parent(["p", "div", "li", "aside", "td", "tr"])
    )
    previous = block.find_previous_sibling() if block is not None else None
    for _ in range(200):
        if previous is None or previous.name in {"h1", "h2", "h3", "h4", "h5", "h6", "hr"}:
            break
        label = previous.get_text(" ", strip=True)
        if REFERENCE_HEADING_RE.match(label):
            return "bibliography"
        if NOTE_HEADING_RE.match(label):
            return "endnote"
        previous = previous.find_previous_sibling()
    if has_backlink:
        return "endnote"
    return "footnote" if explicit_noteref else None


def _bibliography_elements(soup: BeautifulSoup) -> list[Any]:
    results: list[Any] = []
    for heading in soup.find_all(["h1", "h2", "h3", "h4", "h5", "h6"]):
        if not REFERENCE_HEADING_RE.match(heading.get_text(" ", strip=True)):
            continue
        level = int(heading.name[1])
        container = heading.find_parent(["section", "article"])
        if container is not None:
            scope = list(container.find_all(["li", "p"], recursive=True))
        else:
            scope = []
            for sibling in heading.find_next_siblings():
                if sibling.name in {f"h{index}" for index in range(1, level + 1)}:
                    break
                if sibling.name in {"li", "p"}:
                    scope.append(sibling)
                scope.extend(sibling.find_all(["li", "p"], recursive=True))
        # Prefer list items when the section uses an ordered/unordered bibliography.
        list_items = [element for element in scope if element.name == "li"]
        results.extend(list_items or [element for element in scope if element.name == "p"])
    return results


def extract_epub_reference_candidates(epub_path: str) -> List[Dict[str, Any]]:
    """Extract only structurally supported bibliography/note candidates from an EPUB."""
    if not os.path.exists(epub_path):
        return []
    html_files: dict[str, str] = {}
    with zipfile.ZipFile(epub_path, "r") as archive:
        for info in archive.infolist():
            if info.filename.casefold().endswith((".xhtml", ".html")):
                html_files[info.filename] = archive.read(info.filename).decode("utf-8", errors="replace")
    parsed_files = {
        filename: BeautifulSoup(content, "html.parser")
        for filename, content in html_files.items()
    }

    from collections import Counter
    target_link_counts: Counter = Counter()
    for filename, soup in parsed_files.items():
        for link in soup.find_all("a", href=True):
            if "#" not in link["href"]:
                continue
            relative_file, target_id = link["href"].split("#", 1)
            target_file = filename
            if relative_file:
                target_file = os.path.normpath(
                    os.path.join(os.path.dirname(filename), relative_file)
                )
            target_link_counts[(target_file, target_id)] += 1

    candidates: list[dict[str, Any]] = []
    for filename, soup in parsed_files.items():
        for link in soup.find_all("a", href=True):
            href = link["href"]
            if "#" not in href:
                continue
            link_tokens = _semantic_tokens(link)
            explicit_noteref = bool(link_tokens & {"noteref", "doc-noteref"})
            link_text = link.get_text(strip=True)
            if not explicit_noteref and not re.match(r"^\[?\d+\]?$", link_text):
                continue
            parent = link.find_parent(["p", "div", "li", "td", "tr"])
            if parent is None:
                continue
            context = parent.get_text(separator=" ", strip=True)
            relative_file, target_id = href.split("#", 1)
            target_file = filename
            if relative_file:
                target_file = os.path.normpath(os.path.join(os.path.dirname(filename), relative_file))
            target_soup = parsed_files.get(target_file)
            if target_soup is None or not target_id:
                continue
            target = target_soup.find(id=target_id) or target_soup.find("a", attrs={"name": target_id})
            if target is None:
                continue
            target_block = (
                target if target.name in {"p", "div", "li", "aside", "td", "tr"}
                else target.find_parent(["p", "div", "li", "aside", "td", "tr"]) or target
            )
            has_backlink = any(
                backlink.get_text(" ", strip=True).casefold() in {"↩", "↵", "back", "return"}
                or "backlink" in _semantic_tokens(backlink)
                for backlink in target_block.find_all("a", href=True)
            )
            zone = _reference_zone(
                target, target_file,
                explicit_noteref=explicit_noteref, has_backlink=has_backlink,
            )
            if zone is None:
                continue
            raw = target_block.get_text(separator=" ", strip=True)
            if raw == context or len(raw) < 10:
                continue
            candidates.append({
                "context_snippet": context, "raw_reference_text": raw,
                "target_id": f"{target_file}#{target_id}", "source_zone": zone,
                "cite_count": target_link_counts.get((target_file, target_id), 1),
            })

    for filename, soup in parsed_files.items():
        for index, element in enumerate(_bibliography_elements(soup), start=1):
            raw = element.get_text(separator=" ", strip=True)
            if len(raw) < 10:
                continue
            candidates.append({
                "context_snippet": raw, "raw_reference_text": raw,
                "target_id": f"{filename}#bibliography-{index}",
                "source_zone": "bibliography", "cite_count": 1,
            })

    # Prefer candidates linked from the body; otherwise retain one bibliography entry per raw text.
    deduplicated: dict[str, dict[str, Any]] = {}
    for candidate in candidates:
        raw_key = " ".join(candidate["raw_reference_text"].casefold().split())
        existing = deduplicated.get(raw_key)
        if existing is None or candidate["context_snippet"] != candidate["raw_reference_text"]:
            deduplicated[raw_key] = candidate
    priority = {"bibliography": 0, "endnote": 1, "footnote": 2}
    return sorted(
        deduplicated.values(),
        key=lambda row: (priority.get(row["source_zone"], 9), -row.get("cite_count", 1)),
    )


def extract_epub_references(epub_path: str, item_key: str) -> List[Dict[str, Any]]:
    """
    Parses an EPUB file to extract footnotes and endnotes, and maps them to local chunks.
    Returns a list of references with:
    - context_snippet: the paragraph text containing the footnote link
    - raw_reference_text: the actual footnote text
    - citing_chunk_id: the local chunk ID that best matches the context
    - similarity_distance: score of the match
    """
    if not os.path.exists(epub_path):
        print(f"EPUB not found: {epub_path}")
        return []

    references = []
    candidates = extract_epub_reference_candidates(epub_path)

    # Phase 2: チャンク検索（重い処理）をプログレスバー付きで実行
    from citation_mapper import search_chunks
    total_cands = len(candidates)
    for i, candidate in enumerate(candidates):
        _pbar(i + 1, total_cands, "epub ref  ")
        context_snippet = candidate["context_snippet"]
        raw_reference_text = candidate["raw_reference_text"]
        hits = search_chunks(context_snippet, item_key, n_results=1)
        if hits and hits[0]["distance"] < 0.4:
            references.append({
                    "context_snippet": context_snippet,
                    "raw_reference_text": raw_reference_text,
                    "citing_chunk_id": hits[0]["id"],
                    "similarity_distance": hits[0]["distance"],
                    "cite_count": candidate.get("cite_count", 1),
                    "source_zone": candidate["source_zone"],
                })

    return references


__all__ = ["extract_epub_reference_candidates", "extract_epub_references"]
