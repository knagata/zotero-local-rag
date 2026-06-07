import urllib.request
import urllib.parse
import json
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["OMP_NUM_THREADS"] = "1"
import sys
import re
import time
import fcntl
import sqlite3
import difflib
import numpy as np
from typing import Dict, Any, Optional, List, Tuple

from db_relations import insert_citation, update_item_citation_status
from pathlib import Path

ROOT = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CHROMA_DIR = Path(os.environ.get("CHROMA_DIR", str(ROOT / "data" / "chroma")))

# Cross-process S2 rate-limit coordination via shared file lock.
# Both rag_mcp_server.py and update_citations.py import this module,
# so _LAST_REQUEST_TIME was previously per-process and caused 429s when
# both processes sent requests close together.  We now use a lock file
# so any process waits for the global (cross-process) cooldown.
_S2_RATE_FILE = str(ROOT / "data" / "s2_rate.lock")

# ---------------------------------------------------------------------------
# Lightweight vector search that bypasses ChromaDB's PersistentClient.
#
# ChromaDB 1.5+ no longer persists index_metadata.pickle; the id→label
# mapping lives in-memory only.  We therefore:
#   1. Read document texts and chunk IDs from ChromaDB's chroma.sqlite3
#   2. Embed them on the fly with the same SentenceTransformer model
#   3. Compute cosine similarity with numpy (results cached per item_key)
# This avoids both the hnswlib lock problem and the missing pickle issue.
# ---------------------------------------------------------------------------

_EMB_FN_CACHE = None
_SEGMENT_META: Optional[Dict[str, Any]] = None  # cached segment info
_ITEM_CHUNKS_CACHE: Dict[str, List[Tuple[str, np.ndarray]]] = {}  # item_key → [(emb_id, vec)]

# Path for debug logs (relative to project root, not CWD)
_DEBUG_LOG = str(ROOT / "data" / "mapping_debug.log")
_DEBUG_REF_LOG = str(ROOT / "data" / "mapping_debug_ref.log")

# Cosine distance threshold for accepting a chunk match.
# Cosine distance range is [0, 2]; 0 = identical, 2 = opposite.
# 0.4 corresponds to cosine similarity ≥ 0.6 — a meaningful match.
_MAX_COSINE_DISTANCE = 0.4



def _get_emb_fn():
    """Get or create the SentenceTransformer embedding function (cached)."""
    global _EMB_FN_CACHE
    if _EMB_FN_CACHE is not None:
        return _EMB_FN_CACHE

    from chromadb.utils import embedding_functions as chroma_ef
    from embedder import _resolve_embedder_settings
    model_name, device = _resolve_embedder_settings(ROOT)
    print(f"[citation_mapper] Loading embedding model '{model_name}' on {device}...", file=sys.stderr)
    _EMB_FN_CACHE = chroma_ef.SentenceTransformerEmbeddingFunction(
        model_name=model_name,
        device=device,
        normalize_embeddings=True,
    )
    # Warmup
    _ = _EMB_FN_CACHE(["warmup"])
    print(f"[citation_mapper] Embedding model ready.", file=sys.stderr)
    return _EMB_FN_CACHE


def _get_segment_meta() -> Dict[str, Any]:
    """
    Discover the correct metadata segment and vector segment from ChromaDB's SQLite.
    Returns dict with keys: metadata_segment_id, vector_segment_id, collection_name, chunk_count.
    """
    global _SEGMENT_META
    if _SEGMENT_META is not None:
        return _SEGMENT_META

    db_path = str(CHROMA_DIR / "chroma.sqlite3")
    conn = sqlite3.connect(db_path, timeout=10)

    # Find all metadata segments and their embedding counts
    cursor = conn.execute("""
        SELECT s.id, c.name, COUNT(e.id) as cnt
        FROM segments s
        JOIN collections c ON s.collection = c.id
        LEFT JOIN embeddings e ON e.segment_id = s.id
        WHERE s.scope = 'METADATA'
        GROUP BY s.id
        ORDER BY cnt DESC
    """)
    rows = cursor.fetchall()
    if not rows or rows[0][2] == 0:
        conn.close()
        raise RuntimeError("No embeddings found in ChromaDB. Run the indexer first.")

    meta_seg_id = rows[0][0]
    col_name = rows[0][1]
    chunk_count = rows[0][2]

    # Find the corresponding vector segment
    cursor = conn.execute("""
        SELECT s2.id
        FROM segments s2
        JOIN segments s1 ON s1.collection = s2.collection
        WHERE s1.id = ? AND s2.scope = 'VECTOR'
    """, (meta_seg_id,))
    vec_row = cursor.fetchone()
    conn.close()

    if not vec_row:
        raise RuntimeError(f"No vector segment found for collection '{col_name}'")

    _SEGMENT_META = {
        "metadata_segment_id": meta_seg_id,
        "vector_segment_id": vec_row[0],
        "collection_name": col_name,
        "chunk_count": chunk_count,
    }
    print(f"[citation_mapper] Using collection '{col_name}' ({chunk_count} chunks)", file=sys.stderr)
    return _SEGMENT_META


def _load_chunks_for_item(item_key: str) -> List[Tuple[str, np.ndarray]]:
    """
    Load chunk vectors for a given item_key by embedding their text from ChromaDB SQLite.

    ChromaDB 1.5+ no longer persists index_metadata.pickle; the id→label mapping
    lives in-memory only.  We read document texts directly from chroma.sqlite3 and
    embed them with the cached model.  Results are cached per item_key so repeated
    calls from search_chunks() within the same process incur no extra embedding cost.
    """
    if item_key in _ITEM_CHUNKS_CACHE:
        return _ITEM_CHUNKS_CACHE[item_key]

    seg = _get_segment_meta()
    db_path = str(CHROMA_DIR / "chroma.sqlite3")
    conn = sqlite3.connect(db_path, timeout=10)
    try:
        cursor = conn.execute("""
            SELECT e.embedding_id, doc.string_value
            FROM embeddings e
            JOIN embedding_metadata ikey ON ikey.id = e.id
                AND ikey.key = 'itemKey' AND ikey.string_value = ?
            JOIN embedding_metadata doc ON doc.id = e.id
                AND doc.key = 'chroma:document' AND doc.string_value IS NOT NULL
            WHERE e.segment_id = ?
        """, (item_key, seg["metadata_segment_id"]))
        rows = cursor.fetchall()
    finally:
        conn.close()

    if not rows:
        _ITEM_CHUNKS_CACHE[item_key] = []
        return []

    ef = _get_emb_fn()
    embedding_ids = [r[0] for r in rows]
    texts = [r[1] for r in rows]

    try:
        vectors = ef(texts)
    except Exception as e:
        print(f"[citation_mapper] Embedding failed for {item_key}: {e}", file=sys.stderr)
        return []

    result = [
        (emb_id, np.array(vec, dtype=np.float32))
        for emb_id, vec in zip(embedding_ids, vectors)
    ]
    _ITEM_CHUNKS_CACHE[item_key] = result
    print(f"[citation_mapper] Embedded {len(result)} chunks for item {item_key}", file=sys.stderr)
    return result


def search_chunks(query_text: str, item_key: str, n_results: int = 1) -> List[Dict[str, Any]]:
    """
    Search for the most similar chunks to query_text within a specific item.
    Bypasses ChromaDB entirely - uses SQLite + hnswlib files + numpy cosine similarity.
    """
    ef = _get_emb_fn()

    # Embed the query
    query_emb = np.array(ef([query_text])[0], dtype=np.float32)
    norm = np.linalg.norm(query_emb)
    if norm > 0:
        query_emb = query_emb / norm

    # Load chunks for item
    chunks = _load_chunks_for_item(item_key)
    if not chunks:
        return []

    # Compute cosine similarity (embeddings are already normalized by the model)
    results = []
    for emb_id, vec in chunks:
        vec_norm = np.linalg.norm(vec)
        if vec_norm > 0:
            vec = vec / vec_norm
        similarity = float(np.dot(query_emb, vec))
        distance = 1.0 - similarity  # cosine distance
        results.append({"id": emb_id, "distance": distance})

    results.sort(key=lambda x: x["distance"])
    return results[:n_results]


# ---------------------------------------------------------------------------
# Semantic Scholar helpers
# ---------------------------------------------------------------------------

def _s2_wait_and_claim() -> None:
    """Cross-process rate limiter for the S2 API.

    Uses an exclusive file lock on _S2_RATE_FILE so that both
    rag_mcp_server.py and update_citations.py share the same cooldown
    window even when running simultaneously.

    Protocol:
      1. Acquire exclusive lock on the rate file.
      2. Read the last-request timestamp written by any process.
      3. Sleep if the required interval hasn't elapsed yet.
      4. Write the current time to claim our slot.
      5. Release the lock (other processes may now proceed).
      6. Caller sends the actual HTTP request (outside the lock).
    """
    s2_api_key = os.environ.get("S2_API_KEY", "")
    delay_required = 1.1 if s2_api_key else 3.1

    os.makedirs(str(ROOT / "data"), exist_ok=True)
    with open(_S2_RATE_FILE, "a+") as fd:
        fcntl.flock(fd, fcntl.LOCK_EX)
        try:
            fd.seek(0)
            raw = fd.read().strip()
            last_time = float(raw) if raw else 0.0

            elapsed = time.time() - last_time
            if elapsed < delay_required:
                time.sleep(delay_required - elapsed)

            # Claim this slot
            fd.seek(0)
            fd.truncate()
            fd.write(str(time.time()))
            fd.flush()
        finally:
            fcntl.flock(fd, fcntl.LOCK_UN)


def s2_request(url: str, max_retries: int = 3) -> Optional[Dict[str, Any]]:
    """Make a rate-limited S2 API request with exponential backoff on 429.

    Each attempt:
      1. Waits for the cross-process rate-limit slot (_s2_wait_and_claim).
      2. Sends the request.
      3. On HTTP 429: waits an exponentially increasing delay (5s, 10s, 20s)
         before the next attempt, giving the server time to recover.
    """
    import urllib.error

    s2_api_key = os.environ.get("S2_API_KEY", "")
    req_headers = {"User-Agent": "ZoteroLocalRAG/1.0"}
    if s2_api_key:
        req_headers["x-api-key"] = s2_api_key

    for attempt in range(max_retries):
        _s2_wait_and_claim()
        req = urllib.request.Request(url, headers=req_headers)
        try:
            with urllib.request.urlopen(req, timeout=15) as response:
                return json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            if e.code == 429:
                wait = 5 * (2 ** attempt)  # 5s → 10s → 20s
                print(
                    f"[S2] 429 Too Many Requests (attempt {attempt + 1}/{max_retries}), "
                    f"waiting {wait}s before retry...",
                    file=sys.stderr,
                )
                time.sleep(wait)
                continue
            print(f"S2 API HTTP Error {e.code} on {url}: {e}", file=sys.stderr)
            return None
        except Exception as e:
            print(f"S2 API Request Error on {url}: {e}", file=sys.stderr)
            return None

    print(f"[S2] Gave up after {max_retries} retries: {url}", file=sys.stderr)
    return None

def find_s2_paper_id(title: str, year: Optional[int] = None, creators: str = "", doi: str = "", isbn: str = "") -> Optional[Dict[str, Any]]:
    # 1. Try DOI/ISBN exact lookup first
    identifier = doi or isbn
    if identifier:
        prefix = "DOI:" if doi else "ISBN:"
        url = f"https://api.semanticscholar.org/graph/v1/paper/{prefix}{identifier}?fields=paperId,title,authors,year,citationCount"
        print(f"        -> Querying S2 by {prefix}{identifier}...", file=sys.stderr)
        res = s2_request(url)
        if res and "paperId" in res:
            return res
        print(f"        -> S2 exact lookup failed, falling back to title search...", file=sys.stderr)

    # 2. Fallback to Title + Author search
    import re
    
    # Use only the main title (before the colon) because S2 often omits subtitles.
    main_title = title.split(':')[0]
    
    # Strip special characters
    clean_title = re.sub(r'[^\w\s]', ' ', main_title).strip()
    
    author = creators.split(',')[0].strip() if creators else ""
    clean_author = re.sub(r'[^\w\s]', ' ', author).strip()
    
    query_parts = [clean_title]
    if clean_author:
        query_parts.append(clean_author)
    if year:
        query_parts.append(str(year))

    query = " ".join(query_parts)
    # Collapse multiple spaces
    query = re.sub(r'\s+', ' ', query).strip()
    encoded_query = urllib.parse.quote(query)

    url = f"https://api.semanticscholar.org/graph/v1/paper/search?query={encoded_query}&limit=20&fields=paperId,title,authors,year,citationCount"

    res = s2_request(url)
    if not res or not res.get("data"):
        return None

    results = res["data"]

    # Filter by title similarity before ranking by citation count.
    # This prevents matching a famous paper with a similar but different title.
    query_title_lower = clean_title.lower()
    def _sim(paper: Dict[str, Any]) -> float:
        s2_title = (paper.get("title") or "").lower()
        return difflib.SequenceMatcher(None, query_title_lower, s2_title).ratio()

    similar = [p for p in results if _sim(p) >= 0.5]
    if not similar:
        print(f"        -> No S2 results passed title similarity threshold; returning None.", file=sys.stderr)
        return None

    best_match = max(similar, key=lambda x: x.get("citationCount", 0))
    print(
        f"        -> Best Match: '{best_match.get('title')}' "
        f"(similarity={_sim(best_match):.2f}, Citations: {best_match.get('citationCount', 0)})",
        file=sys.stderr,
    )
    return best_match


# ---------------------------------------------------------------------------
# Main citation mapping functions
# ---------------------------------------------------------------------------

def map_item_global_citations(item_key: str, title: str = "", year: str = "", creators: str = "", doi: str = "", isbn: str = "", max_citations: int = 50) -> Dict[str, Any]:
    """
    Fetches global citations from S2 for a given Zotero item and maps them to local chunks.
    """
    # 1. Use title and year provided
    if not title:
        update_item_citation_status(item_key, "error")
        return {"status": "error", "message": "Item has no title."}

    # 2. Find paper on S2
    print(f"[{time.time()}] Calling find_s2_paper_id...", file=sys.stderr)
    s2_paper = find_s2_paper_id(title, year, creators, doi, isbn)
    print(f"[{time.time()}] find_s2_paper_id returned.", file=sys.stderr)
    if not s2_paper:
        update_item_citation_status(item_key, "not_found")
        return {"status": "success", "message": "Item not found on Semantic Scholar.", "mapped_count": 0}

    paper_id = s2_paper["paperId"]

    # 3. Fetch citations (with pagination)
    data_items = []
    offset = 0
    limit = 1000  # Max limit per request for S2 graph API
    
    while True:
        citations_url = f"https://api.semanticscholar.org/graph/v1/paper/{paper_id}/citations?fields=title,year,contexts,intents,citationCount,influentialCitationCount&limit={limit}&offset={offset}"
        citations_res = s2_request(citations_url)
        
        if citations_res is None:
            if not data_items: # If it fails on the first page
                update_item_citation_status(item_key, "error")
                return {"status": "error", "message": "S2 API Error while fetching citations.", "mapped_count": 0, "s2_paper": s2_paper}
            break # Otherwise just stop fetching more
            
        page_data = citations_res.get("data", [])
        if not page_data:
            break
            
        data_items.extend(page_data)
        
        if len(data_items) >= max_citations:
            data_items = data_items[:max_citations]
            break
            
        next_offset = citations_res.get("next")
        if not next_offset:
            break
            
        offset = next_offset
        time.sleep(1) # Be gentle to the API

    if not data_items:
        update_item_citation_status(item_key, "not_found")
        return {"status": "success", "message": "No citations with context found.", "mapped_count": 0, "s2_paper": s2_paper}

    mapped_count = 0
    total_contexts = 0

    print(f"        -> Found {len(data_items)} citing papers on Semantic Scholar. Mapping to local chunks...", file=sys.stderr)

    for i, item in enumerate(data_items):
        print(f"          -> Analyzing citing paper {i+1}/{len(data_items)}...", file=sys.stderr)

        citing_paper = item.get("citingPaper", {})
        contexts = item.get("contexts", [])
        if not contexts:
            continue

        c_paper_id = citing_paper.get("paperId", "")
        c_title = citing_paper.get("title", "")
        c_year = citing_paper.get("year")
        c_citation_count = citing_paper.get("citationCount", 0)
        c_influential_count = citing_paper.get("influentialCitationCount", 0)

        for ctx in contexts:
            total_contexts += 1
            page_hint = None
            page_match = re.search(r'\b(?:p\.|page|pp\.|:)\s*(\d+)\b', ctx, re.IGNORECASE)
            if page_match:
                page_hint = page_match.group(1)

            # Use our lightweight search instead of ChromaDB
            hits = search_chunks(ctx, item_key, n_results=1)

            if hits:
                best_dist = hits[0]["distance"]
                with open(_DEBUG_LOG, "a") as f:
                    f.write(f"Global Context: {ctx[:100]}...\n")
                    f.write(f"  -> Best Hit Distance: {best_dist:.4f}\n")

                if best_dist < _MAX_COSINE_DISTANCE:
                    insert_citation(
                        citing_paper_id=c_paper_id,
                        citing_title=c_title,
                        citing_year=c_year,
                        context_snippet=ctx,
                        cited_item_key=item_key,
                        cited_chunk_id=hits[0]["id"],
                        similarity_distance=best_dist,
                        page_hint=page_hint,
                        citing_citation_count=c_citation_count,
                        citing_influential_count=c_influential_count
                    )
                    mapped_count += 1
            else:
                with open(_DEBUG_LOG, "a") as f:
                    f.write(f"Global Context: {ctx[:100]}...\n")
                    f.write("  -> No chunks found in DB for this item.\n")

    # 4. Fetch References (Outgoing, with pagination)
    from db_relations import insert_reference
    r_data_items = []
    offset = 0
    
    while True:
        references_url = f"https://api.semanticscholar.org/graph/v1/paper/{paper_id}/references?fields=title,year,contexts,citationCount,influentialCitationCount&limit={limit}&offset={offset}"
        r_res = s2_request(references_url)
        
        if r_res is None:
            break
            
        page_data = r_res.get("data", [])
        if not page_data:
            break
            
        r_data_items.extend(page_data)
        
        if len(r_data_items) >= max_citations:
            r_data_items = r_data_items[:max_citations]
            break
            
        next_offset = r_res.get("next")
        if not next_offset:
            break
            
        offset = next_offset
        time.sleep(1)

    ref_mapped_count = 0
    ref_total_contexts = 0

    if r_data_items:
        print(f"        -> Found {len(r_data_items)} referenced papers on Semantic Scholar. Mapping to local chunks...", file=sys.stderr)
    else:
        print("        -> No referenced papers found on Semantic Scholar (data was empty).", file=sys.stderr)

    if r_data_items:
        for i, item in enumerate(r_data_items):
            print(f"          -> Analyzing referenced paper {i+1}/{len(r_data_items)}...", file=sys.stderr)

            cited_paper = item.get("citedPaper", {})
            contexts = item.get("contexts", [])
            if not contexts:
                continue

            c_paper_id = cited_paper.get("paperId", "")
            c_title = cited_paper.get("title", "")
            c_year = cited_paper.get("year")
            c_citation_count = cited_paper.get("citationCount", 0)
            c_influential_count = cited_paper.get("influentialCitationCount", 0)

            for ctx in contexts:
                ref_total_contexts += 1
                page_hint = None
                page_match = re.search(r'\b(?:p\.|page|pp\.|:)\s*(\d+)\b', ctx, re.IGNORECASE)
                if page_match:
                    page_hint = page_match.group(1)

                # Use our lightweight search instead of ChromaDB
                hits = search_chunks(ctx, item_key, n_results=1)

                if hits:
                    best_dist = hits[0]["distance"]
                    with open(_DEBUG_REF_LOG, "a") as f:
                        f.write(f"Local Context: {ctx[:100]}...\n")
                        f.write(f"  -> Best Hit Distance: {best_dist:.4f}\n")

                    if best_dist < _MAX_COSINE_DISTANCE:
                        insert_reference(
                            cited_paper_id=c_paper_id,
                            cited_title=c_title,
                            cited_year=c_year,
                            context_snippet=ctx,
                            citing_item_key=item_key,
                            citing_chunk_id=hits[0]["id"],
                            similarity_distance=best_dist,
                            page_hint=page_hint,
                            source='s2',
                            cited_citation_count=c_citation_count,
                            cited_influential_count=c_influential_count
                        )
                        ref_mapped_count += 1
                else:
                    with open(_DEBUG_REF_LOG, "a") as f:
                        f.write(f"Context: {ctx[:100]}...\n")
                        f.write("  -> No chunks found in DB for this item.\n")

    update_item_citation_status(item_key, "mapped")
    
    msg = f"Global Citations: {mapped_count}/{total_contexts} contexts mapped. Local References: {ref_mapped_count}/{ref_total_contexts} contexts mapped."
    with open(_DEBUG_LOG, "a") as f:
        f.write(f"Result for {item_key}: {msg}\n")
        
    return {
        "status": "success",
        "message": msg,
        "s2_paper": s2_paper,
        "total_contexts_analyzed": total_contexts,
        "mapped_count": mapped_count,
        "references_contexts_analyzed": ref_total_contexts,
        "references_mapped_count": ref_mapped_count
    }

def map_item_local_references(item_key: str, epub_path: str) -> Dict[str, Any]:
    """
    Parses local EPUB to extract footnotes/endnotes, attempts to resolve them to Semantic Scholar,
    and saves to global_references.
    """
    from epub_reference_extractor import extract_epub_references
    from db_relations import insert_reference

    print(f"        -> Extracting local EPUB references for {item_key}...", file=sys.stderr)
    local_refs = extract_epub_references(epub_path, item_key)

    if not local_refs:
        return {"status": "success", "message": "No EPUB references found.", "mapped_count": 0}

    mapped_count = 0
    total = len(local_refs)
    print(f"        -> Found {total} references in EPUB. Resolving via Semantic Scholar...", file=sys.stderr)

    for i, ref in enumerate(local_refs):
        print(f"          -> Resolving reference {i+1}/{total}...", file=sys.stderr)

        raw_text = ref["raw_reference_text"]
        context_snippet = ref["context_snippet"]
        chunk_id = ref["citing_chunk_id"]
        dist = ref["similarity_distance"]

        # Try to resolve via Semantic Scholar
        query = raw_text[:100]
        query = re.sub(r'[^\w\s]', ' ', query)

        s2_paper = None
        s2_status = 'not_found'

        if len(query.strip()) > 10:
            encoded_query = urllib.parse.quote(query.strip())
            search_url = f"https://api.semanticscholar.org/graph/v1/paper/search?query={encoded_query}&limit=1&fields=title,year"
            res = s2_request(search_url)

            if res is None:
                s2_status = 'error'
            elif res.get("data"):
                s2_paper = res["data"][0]
                s2_status = 'mapped'
            else:
                s2_status = 'not_found'

        c_paper_id = None
        c_title = None
        c_year = None

        if s2_paper:
            c_paper_id = s2_paper.get("paperId")
            c_title = s2_paper.get("title")
            c_year = s2_paper.get("year")

        insert_reference(
            cited_paper_id=c_paper_id,
            cited_title=c_title,
            cited_year=c_year,
            context_snippet=context_snippet,
            citing_item_key=item_key,
            citing_chunk_id=chunk_id,
            similarity_distance=dist,
            page_hint=None,
            source='epub',
            raw_reference_text=raw_text,
            s2_status=s2_status
        )
        mapped_count += 1

    return {
        "status": "success",
        "total_extracted": len(local_refs),
        "mapped_count": mapped_count,
        "message": f"Extracted {len(local_refs)} references, mapped {mapped_count}."
    }
