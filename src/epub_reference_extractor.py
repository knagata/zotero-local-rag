import os
import zipfile
import re
from typing import List, Dict, Any, Optional
from bs4 import BeautifulSoup


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

    # Scan for footnote links in the body.
    # Typically, a footnote link looks like <a href="some_file.xhtml#note1">1</a> or is wrapped in <sup>
    for filename, soup in parsed_files.items():
        # Find all anchors that might be footnotes
        for a_tag in soup.find_all('a', href=True):
            href = a_tag['href']
            text = a_tag.get_text(strip=True)
            
            # Simple heuristic: if the link text is just a number or wrapped in brackets like [1]
            if not re.match(r'^\[?\d+\]?$', text):
                continue
                
            # It's likely a footnote link. Let's get the context (the parent paragraph)
            parent_p = a_tag.find_parent(['p', 'div'])
            if not parent_p:
                continue
                
            # We don't want the footnote section itself. The body usually has the link.
            # We skip if the parent paragraph itself looks like a footnote list item.
            # But let's just grab the text for now.
            context_snippet = parent_p.get_text(separator=' ', strip=True)
            
            # Now let's follow the href to find the footnote text
            target_file = filename
            target_id = None
            if '#' in href:
                parts = href.split('#', 1)
                if parts[0]: # link to another file
                    # resolve relative path
                    base_dir = os.path.dirname(filename)
                    target_file = os.path.normpath(os.path.join(base_dir, parts[0]))
                target_id = parts[1]
            else:
                continue # If no anchor, it might just link to a whole bibliography file, harder to parse exactly

            if target_file not in parsed_files:
                continue

            target_soup = parsed_files[target_file]
            target_elem = target_soup.find(id=target_id)
            if not target_elem:
                # Sometimes the id is on an <a> tag with name="target_id"
                target_elem = target_soup.find('a', attrs={'name': target_id})
                
            if not target_elem:
                continue
                
            # The actual footnote text is usually the parent paragraph of the target element, or the element itself.
            target_p = target_elem.find_parent(['p', 'div', 'li']) if target_elem.name not in ['p', 'div', 'li'] else target_elem
            if not target_p:
                target_p = target_elem

            raw_reference_text = target_p.get_text(separator=' ', strip=True)
            
            # Skip if the raw_reference_text is identical to the context_snippet (circular link)
            if raw_reference_text == context_snippet or len(raw_reference_text) < 10:
                continue

            print(f"Found footnote reference. Searching for matching chunk...", flush=True)
            # Use lightweight search that bypasses ChromaDB (avoids MCP server lock conflict)
            from citation_mapper import search_chunks
            hits = search_chunks(context_snippet, item_key, n_results=1)
            print(f"Search complete.", flush=True)
            
            if hits and hits[0]["distance"] < 0.4:
                references.append({
                    "context_snippet": context_snippet,
                    "raw_reference_text": raw_reference_text,
                    "citing_chunk_id": hits[0]["id"],
                    "similarity_distance": hits[0]["distance"]
                })

    return references
