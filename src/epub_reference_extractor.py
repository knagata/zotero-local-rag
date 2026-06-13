import os
import sys
import zipfile
import re
from typing import List, Dict, Any, Optional
from bs4 import BeautifulSoup


def _pbar(current: int, total: int, label: str = "", width: int = 30) -> None:
    """1行でインプレース更新するプログレスバー。完了時は改行を出す。"""
    filled = int(width * current / total) if total else width
    bar = "█" * filled + "░" * (width - filled)
    pct = current / total * 100 if total else 100
    end = "\n" if current >= total else "\r"
    print(f"        [{bar}] {current:>5}/{total}  {pct:5.1f}%  {label}",
          end=end, file=sys.stderr, flush=True)


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
    
    # We will read all XHTML/HTML files from the EPUB
    html_files = {}
    with zipfile.ZipFile(epub_path, 'r') as z:
        for info in z.infolist():
            if info.filename.endswith('.xhtml') or info.filename.endswith('.html'):
                html_files[info.filename] = z.read(info.filename).decode('utf-8')

    # Parse each file
    parsed_files = {}
    for filename, content in html_files.items():
        parsed_files[filename] = BeautifulSoup(content, 'html.parser')

    # Phase 1: 脚注候補を収集（HTMLスキャン、高速）
    # target_id ごとのリンク出現回数を数える（= 本文中での引用頻度 = 重要度の代理指標）
    from collections import Counter
    target_link_counts: Counter = Counter()
    for soup in parsed_files.values():
        for a_tag in soup.find_all('a', href=True):
            href = a_tag['href']
            if '#' in href:
                target_link_counts[href.split('#', 1)[1]] += 1

    candidates: List[tuple] = []  # (context_snippet, raw_reference_text, target_id)
    for filename, soup in parsed_files.items():
        for a_tag in soup.find_all('a', href=True):
            href = a_tag['href']
            text = a_tag.get_text(strip=True)
            if not re.match(r'^\[?\d+\]?$', text):
                continue
            parent_p = a_tag.find_parent(['p', 'div'])
            if not parent_p:
                continue
            context_snippet = parent_p.get_text(separator=' ', strip=True)
            target_file = filename
            target_id = None
            if '#' in href:
                parts = href.split('#', 1)
                if parts[0]:
                    base_dir = os.path.dirname(filename)
                    target_file = os.path.normpath(os.path.join(base_dir, parts[0]))
                target_id = parts[1]
            else:
                continue
            if target_file not in parsed_files:
                continue
            target_soup = parsed_files[target_file]
            target_elem = target_soup.find(id=target_id)
            if not target_elem:
                target_elem = target_soup.find('a', attrs={'name': target_id})
            if not target_elem:
                continue
            target_p = target_elem.find_parent(['p', 'div', 'li']) if target_elem.name not in ['p', 'div', 'li'] else target_elem
            if not target_p:
                target_p = target_elem
            raw_reference_text = target_p.get_text(separator=' ', strip=True)
            if raw_reference_text == context_snippet or len(raw_reference_text) < 10:
                continue
            candidates.append((context_snippet, raw_reference_text, target_id))

    # Phase 2: チャンク検索（重い処理）をプログレスバー付きで実行
    from citation_mapper import search_chunks
    total_cands = len(candidates)
    for i, (context_snippet, raw_reference_text, target_id) in enumerate(candidates):
        _pbar(i + 1, total_cands, "epub ref  ")
        hits = search_chunks(context_snippet, item_key, n_results=1)
        if hits and hits[0]["distance"] < 0.4:
            references.append({
                    "context_snippet": context_snippet,
                    "raw_reference_text": raw_reference_text,
                    "citing_chunk_id": hits[0]["id"],
                    "similarity_distance": hits[0]["distance"],
                    "cite_count": target_link_counts.get(target_id, 1),
                })

    return references
