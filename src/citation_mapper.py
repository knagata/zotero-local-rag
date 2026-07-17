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

from db_relations import get_s2_lookup_candidates, insert_citation, update_item_citation_status
from pathlib import Path

try:
    from .reference_text import is_short_form_reference, s2_candidate_is_supported
except ImportError:  # pragma: no cover - direct script imports
    from reference_text import is_short_form_reference, s2_candidate_is_supported


class S2RetryExhaustedError(Exception):
    """Raised by s2_request when all retries are exhausted (persistent rate-limiting).
    Distinct from returning None, which means the paper was not found."""


ROOT = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CHROMA_DIR = Path(os.environ.get("CHROMA_DIR", str(ROOT / "data" / "chroma")))

from env_utils import load_dotenv_native as _load_dotenv

_load_dotenv()
_s2_key_at_import = os.environ.get("S2_API_KEY", "")
print(
    f"[citation_mapper] S2_API_KEY: "
    + ("SET (length=%d)" % len(_s2_key_at_import) if _s2_key_at_import else "NOT SET — shared 1000 RPS pool"),
    file=sys.stderr,
)


def _fmt_authors(authors_list: list, max_authors: int = 5) -> str:
    """S2 authors リスト（[{name: ...}, ...]）を表示用文字列に整形する。"""
    if not authors_list:
        return ""
    names = [a.get("name", "") for a in authors_list if a.get("name")]
    if not names:
        return ""
    if len(names) > max_authors:
        return ", ".join(names[:max_authors]) + " et al."
    return ", ".join(names)


def _select_supported_s2_candidate(raw_text: str, result: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """Return the only defensible S2 search hit; never trust result ordering."""
    supported = [
        paper for paper in (result or {}).get("data", [])
        if s2_candidate_is_supported(raw_text, paper)
    ]
    unique = {paper.get("paperId"): paper for paper in supported if paper.get("paperId")}
    return next(iter(unique.values())) if len(unique) == 1 else None


def _pbar(current: int, total: int, label: str = "", width: int = 30, file=sys.stderr) -> None:
    """1行でインプレース更新するプログレスバー。完了時は改行を出す。"""
    filled = int(width * current / total) if total else width
    bar = "█" * filled + "░" * (width - filled)
    pct = current / total * 100 if total else 100
    end = "\n" if current >= total else "\r"
    print(f"        [{bar}] {current:>5}/{total}  {pct:5.1f}%  {label}", end=end, file=file, flush=True)

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
    """Get or create the embedding function (cached)."""
    global _EMB_FN_CACHE
    if _EMB_FN_CACHE is not None:
        return _EMB_FN_CACHE

    from embedder import resolve_embedder_settings, create_embedding_function
    cfg = resolve_embedder_settings(ROOT)
    if cfg.provider == "gemini":
        provider_label = f"Gemini API ({cfg.model_name})"
    else:
        provider_label = f"'{cfg.model_name}' on {cfg.device}"
    print(f"[citation_mapper] Loading embedding function: {provider_label} ...", file=sys.stderr)
    _EMB_FN_CACHE = create_embedding_function(cfg)
    # Warmup
    _ = _EMB_FN_CACHE(["warmup"])
    print(f"[citation_mapper] Embedding function ready.", file=sys.stderr)
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
    delay_required = 2.5 if s2_api_key else 3.5

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


# Circuit breaker: when S2 keeps returning 429 despite backoff, stop burning
# time on in-place retries — fail fast, let callers record s2_status='error',
# and rely on the end-of-run / next-run retry passes instead.
_S2_FAILFAST_THRESHOLD = 3      # consecutive retry-exhaustions to trip the breaker
_S2_FAILFAST_WINDOW = 300.0     # seconds to fail fast once tripped
_s2_consec_exhausted = 0
_s2_failfast_until = 0.0


def s2_request(url: str, max_retries: int = 10) -> Optional[Dict[str, Any]]:
    """Make a rate-limited S2 API request with adaptive retry on 429.

    429 responses fall into two categories with different strategies:

    1. AWS API Gateway throttle (x-amzn-ErrorType=TooManyRequestsException,
       no Retry-After/X-RateLimit-* headers): The shared token bucket is
       temporarily exhausted by global traffic. Backoff does not help since
       other users consume refilled tokens. Strategy: short fixed wait (2s)
       and retry up to max_retries times to catch a gap in traffic.
       P(success in 10 tries) ≈ 99% at observed 35% per-attempt success rate.

    2. Application-level rate limit (Retry-After header present): Respect
       the server-specified wait time with exponential backoff.

    After max_retries exhaustion, raises S2RetryExhaustedError. Failed
    lookups are recorded as s2_status='error' and retried on the next run.
    A circuit breaker skips S2 entirely for _S2_FAILFAST_WINDOW seconds
    after _S2_FAILFAST_THRESHOLD consecutive exhaustions.
    """
    import urllib.error
    global _s2_consec_exhausted, _s2_failfast_until

    if time.time() < _s2_failfast_until:
        raise S2RetryExhaustedError(f"circuit breaker open ({url})")

    s2_api_key = os.environ.get("S2_API_KEY", "")
    req_headers = {"User-Agent": "ZoteroLocalRAG/1.0"}
    if s2_api_key:
        req_headers["x-api-key"] = s2_api_key

    for attempt in range(max_retries):
        _s2_wait_and_claim()
        req = urllib.request.Request(url, headers=req_headers)
        try:
            with urllib.request.urlopen(req, timeout=15) as response:
                _s2_consec_exhausted = 0
                return json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            if e.code == 429:
                retry_after   = e.headers.get("Retry-After")           if e.headers else None
                x_remaining   = e.headers.get("X-RateLimit-Remaining") if e.headers else None
                x_limit       = e.headers.get("X-RateLimit-Limit")     if e.headers else None
                x_reset       = e.headers.get("X-RateLimit-Reset")     if e.headers else None
                amzn_type     = e.headers.get("x-amzn-ErrorType")      if e.headers else None

                header_info = ", ".join(
                    f"{k}={v}" for k, v in [
                        ("Retry-After", retry_after),
                        ("X-RateLimit-Remaining", x_remaining),
                        ("X-RateLimit-Limit", x_limit),
                        ("X-RateLimit-Reset", x_reset),
                        ("x-amzn-ErrorType", amzn_type),
                    ] if v is not None
                )

                import random
                is_gateway_throttle = (
                    amzn_type == "TooManyRequestsException"
                    and not retry_after and not x_remaining
                )
                if is_gateway_throttle:
                    # AWS Gateway shared bucket: short fixed wait, no backoff
                    wait = 2.0 + random.uniform(0, 1.0)
                    kind = "gateway"
                elif retry_after:
                    try:
                        wait = max(int(retry_after), 1)
                    except ValueError:
                        wait = 10.0
                    kind = "rate-limit"
                else:
                    s2_key = os.environ.get("S2_API_KEY", "")
                    base = 10 if s2_key else 15
                    wait = min(base * (2 ** attempt) + random.uniform(0, base * 0.5), 45)
                    kind = "backoff"

                print(
                    f"[S2] 429 {kind} (attempt {attempt + 1}/{max_retries})"
                    + (f" [{header_info}]" if header_info else " [no headers]")
                    + f", waiting {wait:.1f}s...",
                    file=sys.stderr,
                )
                try:
                    with open(_S2_RATE_FILE, "w") as _rf:
                        _rf.write(str(time.time() + wait))
                except OSError:
                    pass
                time.sleep(wait)
                continue
            print(f"S2 API HTTP Error {e.code} on {url}: {e}", file=sys.stderr)
            return None
        except Exception as e:
            print(f"S2 API Request Error on {url}: {e}", file=sys.stderr)
            return None

    # Give the S2 token bucket time to recover before the next request,
    # otherwise subsequent calls immediately re-trigger 429s.
    try:
        with open(_S2_RATE_FILE, "w") as _rf:
            _rf.write(str(time.time() + 30))
    except OSError:
        pass
    _s2_consec_exhausted += 1
    if _s2_consec_exhausted >= _S2_FAILFAST_THRESHOLD:
        _s2_failfast_until = time.time() + _S2_FAILFAST_WINDOW
        _s2_consec_exhausted = 0
        print(
            f"[S2] {_S2_FAILFAST_THRESHOLD} consecutive retry-exhaustions — "
            f"circuit breaker open: skipping S2 lookups for {int(_S2_FAILFAST_WINDOW)}s "
            f"(failures recorded as 'error' and retried later).",
            file=sys.stderr,
        )
    print(f"[S2] Gave up after {max_retries} retries: {url}", file=sys.stderr)
    raise S2RetryExhaustedError(url)

def find_s2_paper_id(title: str, year: Optional[int] = None, creators: str = "", doi: str = "", isbn: str = "") -> Optional[Dict[str, Any]]:
    # 1. Try DOI/ISBN exact lookup first
    # Zotero's ISBN field may contain multiple ISBNs separated by spaces; use only the first.
    identifier = doi or (isbn.split()[0] if isbn else "")
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

    # S2 indexes primarily English papers; skip search if title is mostly non-ASCII (e.g. Japanese)
    non_ascii_ratio = sum(1 for c in title if ord(c) > 0x7F) / max(len(title), 1)
    if non_ascii_ratio > 0.3:
        return None

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
    try:
        s2_paper = find_s2_paper_id(title, year, creators, doi, isbn)
        if not s2_paper:
            for candidate in get_s2_lookup_candidates(item_key):
                candidate_title = candidate.get("title") or ""
                if not candidate_title or candidate_title == title:
                    continue
                print(
                    f"        -> Retrying S2 through equivalent/chapter work: {candidate_title}",
                    file=sys.stderr,
                )
                s2_paper = find_s2_paper_id(
                    candidate_title, candidate.get("year"), candidate.get("authors") or "",
                    candidate.get("doi") or "", candidate.get("isbn") or "",
                )
                if s2_paper:
                    break
        print(f"[{time.time()}] find_s2_paper_id returned.", file=sys.stderr)
    except S2RetryExhaustedError as exc:
        print(f"[S2] Gave up after retries: {exc}", file=sys.stderr)
        update_item_citation_status(item_key, "error")
        return {"status": "error", "message": "S2 rate limit exhausted while finding paper ID.", "mapped_count": 0}
    if not s2_paper:
        # S2 に見つからない（非英語資料・書籍・人文系などカバレッジ外で頻発）。
        # DOI があれば Crossref で書誌情報（年・被引用数）をフォールバック補完する。
        # 注: 引用関係そのものは S2 のみで取得し、ここでは書誌情報だけを補う。
        cr_year = cr_cc = None
        if doi:
            try:
                from crossref_client import fetch_crossref_by_doi
                meta = fetch_crossref_by_doi(doi)
            except Exception as exc:  # CrossrefError 含む。フォールバックは best-effort。
                print(f"        -> Crossref fallback failed: {exc}", file=sys.stderr)
                meta = None
            if meta:
                cr_year = meta.get("year")
                cr_cc = meta.get("citation_count")
                print(f"        -> Crossref fallback: year={cr_year}, citations={cr_cc}", file=sys.stderr)
        update_item_citation_status(
            item_key, "s2_done",
            s2_year=cr_year,            # S2 列だが Crossref 由来の年を補完保存
            s2_citation_count=cr_cc,    # 同上（is-referenced-by-count）
        )
        suffix = " (Crossref 書誌フォールバック適用)" if (cr_year or cr_cc) else ""
        return {"status": "success",
                "message": "Item not found on Semantic Scholar." + suffix,
                "mapped_count": 0}

    # S2 メタ情報を DB に保存（以降の処理でも参照できるよう早めに記録）
    # 注: アブストラクトは Citation-Update では取得しない（S2 依存・API 制限回避）。
    #     概要は概要パネルの取得ボタンから Zotero local API 等でオンデマンド取得する。
    update_item_citation_status(
        item_key, "mapped",
        s2_paper_id=s2_paper.get("paperId"),
        s2_year=s2_paper.get("year"),
        s2_citation_count=s2_paper.get("citationCount"),
    )

    paper_id = s2_paper["paperId"]
    limit = 1000  # Max limit per request for S2 graph API

    # 3. Fetch citations
    data_items = []
    offset = 0
    while True:
        citations_url = (
            f"https://api.semanticscholar.org/graph/v1/paper/{paper_id}/citations"
            f"?fields=title,year,authors,contexts,intents,citationCount,influentialCitationCount,externalIds"
            f"&limit={limit}&offset={offset}"
        )
        try:
            citations_res = s2_request(citations_url)
        except S2RetryExhaustedError:
            citations_res = None

        if citations_res is None:
            if not data_items:
                update_item_citation_status(item_key, "error")
                return {"status": "error", "message": "S2 API Error while fetching citations.", "mapped_count": 0, "s2_paper": s2_paper}
            break

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

    if not data_items:
        update_item_citation_status(item_key, "s2_done")
        return {"status": "success", "message": "No citations with context found.", "mapped_count": 0, "s2_paper": s2_paper}

    mapped_count = 0
    total_contexts = 0

    n_cit = len(data_items)
    print(f"        -> Found {n_cit} citing papers on Semantic Scholar. Mapping to local chunks...", file=sys.stderr)

    for i, item in enumerate(data_items):
        _pbar(i + 1, n_cit, "  citing  ", file=sys.stderr)

        citing_paper = item.get("citingPaper", {})
        contexts = item.get("contexts", [])
        if not contexts:
            continue

        c_paper_id = citing_paper.get("paperId", "")
        c_title = citing_paper.get("title", "")
        c_year = citing_paper.get("year")
        c_citation_count = citing_paper.get("citationCount", 0)
        c_influential_count = citing_paper.get("influentialCitationCount", 0)
        c_doi = (citing_paper.get("externalIds") or {}).get("DOI")
        c_authors = _fmt_authors(citing_paper.get("authors") or [])

        for ctx in contexts:
            total_contexts += 1
            page_hint = None
            page_match = re.search(r'\b(?:p\.|page|pp\.|:)\s*(\d+)\b', ctx, re.IGNORECASE)
            if page_match:
                page_hint = page_match.group(1)

            # チャンクマッチを試みる（失敗しても引用自体は必ず記録する）
            hits = search_chunks(ctx, item_key, n_results=1)
            matched_chunk_id = None
            matched_dist = None
            cit_status = 'no_chunk'

            if hits:
                best_dist = hits[0]["distance"]
                with open(_DEBUG_LOG, "a") as f:
                    f.write(f"Global Context: {ctx[:100]}...\n")
                    f.write(f"  -> Best Hit Distance: {best_dist:.4f}\n")
                if best_dist < _MAX_COSINE_DISTANCE:
                    matched_chunk_id = hits[0]["id"]
                    matched_dist = best_dist
                    cit_status = 'matched'
                    mapped_count += 1
            else:
                with open(_DEBUG_LOG, "a") as f:
                    f.write(f"Global Context: {ctx[:100]}...\n")
                    f.write("  -> No chunks found in DB for this item.\n")

            # チャンクマッチの成否に関わらず引用を記録
            insert_citation(
                citing_paper_id=c_paper_id,
                citing_title=c_title,
                citing_year=c_year,
                context_snippet=ctx,
                cited_item_key=item_key,
                cited_chunk_id=matched_chunk_id,
                similarity_distance=matched_dist,
                page_hint=page_hint,
                citing_citation_count=c_citation_count,
                citing_influential_count=c_influential_count,
                chunk_status=cit_status,
                citing_doi=c_doi,
                citing_authors=c_authors or None,
            )

    # 4. Fetch References
    from db_relations import insert_reference
    r_data_items = []
    offset = 0
    while True:
        references_url = (
            f"https://api.semanticscholar.org/graph/v1/paper/{paper_id}/references"
            f"?fields=title,year,authors,contexts,citationCount,influentialCitationCount,externalIds"
            f"&limit={limit}&offset={offset}"
        )
        try:
            r_res = s2_request(references_url)
        except S2RetryExhaustedError:
            r_res = None

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

    ref_mapped_count = 0
    ref_total_contexts = 0

    if r_data_items:
        n_ref = len(r_data_items)
        print(f"        -> Found {n_ref} referenced papers on Semantic Scholar. Mapping to local chunks...", file=sys.stderr)
    else:
        print("        -> No referenced papers found on Semantic Scholar.", file=sys.stderr)

    if r_data_items:
        for i, item in enumerate(r_data_items):
            _pbar(i + 1, n_ref, "  reference", file=sys.stderr)

            cited_paper = item.get("citedPaper", {})
            contexts = item.get("contexts", [])

            c_paper_id = cited_paper.get("paperId", "")
            c_title = cited_paper.get("title", "")
            c_year = cited_paper.get("year")
            c_citation_count = cited_paper.get("citationCount", 0)
            c_influential_count = cited_paper.get("influentialCitationCount", 0)
            c_doi = (cited_paper.get("externalIds") or {}).get("DOI")
            c_authors = _fmt_authors(cited_paper.get("authors") or [])

            if not contexts:
                # S2にコンテキストなし → 論文情報だけ記録・AI解析候補としてマーク
                insert_reference(
                    cited_paper_id=c_paper_id,
                    cited_title=c_title,
                    cited_year=c_year,
                    context_snippet=None,
                    citing_item_key=item_key,
                    citing_chunk_id=None,
                    similarity_distance=None,
                    page_hint=None,
                    source='s2',
                    s2_status='no_context',
                    cited_citation_count=c_citation_count,
                    cited_influential_count=c_influential_count,
                    cited_doi=c_doi,
                    cited_authors=c_authors or None,
                )
                continue

            for ctx in contexts:
                ref_total_contexts += 1
                page_hint = None
                page_match = re.search(r'\b(?:p\.|page|pp\.|:)\s*(\d+)\b', ctx, re.IGNORECASE)
                if page_match:
                    page_hint = page_match.group(1)

                # チャンクマッチを試みる（失敗しても参照自体は必ず記録する）
                hits = search_chunks(ctx, item_key, n_results=1)
                matched_chunk_id = None
                matched_dist = None
                ref_status = 'no_chunk'

                if hits:
                    best_dist = hits[0]["distance"]
                    with open(_DEBUG_REF_LOG, "a") as f:
                        f.write(f"Local Context: {ctx[:100]}...\n")
                        f.write(f"  -> Best Hit Distance: {best_dist:.4f}\n")
                    if best_dist < _MAX_COSINE_DISTANCE:
                        matched_chunk_id = hits[0]["id"]
                        matched_dist = best_dist
                        ref_status = 'matched'
                        ref_mapped_count += 1
                else:
                    with open(_DEBUG_REF_LOG, "a") as f:
                        f.write(f"Context: {ctx[:100]}...\n")
                        f.write("  -> No chunks found in DB for this item.\n")

                # チャンクマッチの成否に関わらず参照を記録
                insert_reference(
                    cited_paper_id=c_paper_id,
                    cited_title=c_title,
                    cited_year=c_year,
                    context_snippet=ctx,
                    citing_item_key=item_key,
                    citing_chunk_id=matched_chunk_id,
                    similarity_distance=matched_dist,
                    page_hint=page_hint,
                    source='s2',
                    s2_status=ref_status,
                    cited_citation_count=c_citation_count,
                    cited_influential_count=c_influential_count,
                    cited_doi=c_doi,
                    cited_authors=c_authors or None,
                )

    update_item_citation_status(item_key, "s2_done")

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

def map_item_local_references(item_key: str, epub_path: str, epub_budget: int = 50) -> Dict[str, Any]:
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

    # S2 lookups per item are capped to avoid 429 storms on EPUBs with hundreds of references.
    # Sort by cite_count desc (document-level citation frequency) then by chunk match distance.
    local_refs.sort(key=lambda r: (-r.get("cite_count", 1), r.get("similarity_distance", 1.0)))

    mapped_count = 0
    total = len(local_refs)
    s2_budget = min(total, epub_budget)
    print(
        f"        -> Found {total} references in EPUB."
        f" Resolving top {s2_budget} via Semantic Scholar...",
        file=sys.stderr,
    )

    lookups_used = 0
    for i, ref in enumerate(local_refs):
        _pbar(i + 1, total, "  epub ref ", file=sys.stderr)

        raw_text = ref["raw_reference_text"]
        context_snippet = ref["context_snippet"]
        chunk_id = ref["citing_chunk_id"]
        dist = ref["similarity_distance"]

        s2_paper = None
        s2_status = 'not_found'

        if is_short_form_reference(raw_text):
            s2_status = 'short_form'
        elif lookups_used < s2_budget:
            lookups_used += 1
            # Try to resolve via Semantic Scholar (60 chars keeps title, avoids journal/page noise)
            query = re.sub(r'[^\w\s]', ' ', raw_text[:60])
            query = re.sub(r'\s+', ' ', query).strip()

            _q = query.strip()
            _non_ascii = sum(1 for c in _q if ord(c) > 0x7F) / max(len(_q), 1)
            if len(_q) > 10 and _non_ascii <= 0.3:
                encoded_query = urllib.parse.quote(_q)
                search_url = (
                    "https://api.semanticscholar.org/graph/v1/paper/search"
                    f"?query={encoded_query}&limit=5"
                    "&fields=paperId,title,year,citationCount,influentialCitationCount,authors,externalIds"
                )
                try:
                    res = s2_request(search_url)
                except S2RetryExhaustedError:
                    res = None

                if res is None:
                    s2_status = 'error'
                elif res.get("data"):
                    s2_paper = _select_supported_s2_candidate(raw_text, res)
                    s2_status = 'mapped' if s2_paper else 'unverified'
                else:
                    s2_status = 'not_found'
        else:
            s2_status = 'skipped'

        c_paper_id = None
        c_title = None
        c_year = None
        c_cc = 0
        c_ic = 0
        c_doi = None
        c_authors = None

        if s2_paper:
            c_paper_id = s2_paper.get("paperId")
            c_title = s2_paper.get("title")
            c_year = s2_paper.get("year")
            c_cc = s2_paper.get("citationCount", 0)
            c_ic = s2_paper.get("influentialCitationCount", 0)
            c_doi = (s2_paper.get("externalIds") or {}).get("DOI")
            c_authors = _fmt_authors(s2_paper.get("authors") or [])

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
            raw_reference_text=raw_text, s2_status=s2_status,
            cited_citation_count=c_cc, cited_influential_count=c_ic,
            cited_doi=c_doi, cited_authors=c_authors,
        )
        if s2_paper:
            mapped_count += 1

    return {
        "status": "success",
        "total_extracted": len(local_refs),
        "mapped_count": mapped_count,
        "message": f"Extracted {len(local_refs)} references, mapped {mapped_count}."
    }


def resolve_skipped_epub_refs(item_key: str, budget: int = 200, statuses: tuple = ('skipped',)) -> Dict[str, Any]:
    """DB に残っている指定 s2_status（既定 'skipped'）の EPUB 参照を S2 で解決する。
    EPUB の再解析不要。DB 内の行を直接 UPDATE する。
    statuses=('error',) を渡すと 429 リトライ枯渇などで失敗した行を再試行できる。
    """
    from db_relations import get_skipped_epub_refs, update_reference_s2_data

    skipped = get_skipped_epub_refs(item_key, statuses=statuses)
    if not skipped:
        return {"status": "success", "message": "No skipped references.", "resolved": 0, "total": 0}

    to_process = skipped[:budget]
    total = len(skipped)
    resolved = 0

    print(
        f"        -> Resolving {len(to_process)}/{total} skipped EPUB refs via S2...",
        file=sys.stderr,
    )

    for i, ref in enumerate(to_process):
        _pbar(i + 1, len(to_process), "  epub res ", file=sys.stderr)

        raw_text = ref.get("raw_reference_text") or ""
        if is_short_form_reference(raw_text):
            update_reference_s2_data(
                ref_id=ref["id"], cited_paper_id=None, cited_title=None, cited_year=None,
                cited_citation_count=0, cited_influential_count=0, cited_doi=None,
                cited_authors=None, s2_status='short_form',
            )
            continue
        query = re.sub(r'[^\w\s]', ' ', raw_text[:60])
        query = re.sub(r'\s+', ' ', query).strip()

        s2_paper = None
        s2_status = 'not_found'

        _q = query.strip()
        _non_ascii = sum(1 for c in _q if ord(c) > 0x7F) / max(len(_q), 1)
        if len(_q) > 10 and _non_ascii <= 0.3:
            encoded_query = urllib.parse.quote(_q)
            search_url = (
                f"https://api.semanticscholar.org/graph/v1/paper/search"
                f"?query={encoded_query}&limit=5"
                f"&fields=paperId,title,year,citationCount,influentialCitationCount,externalIds,authors"
            )
            try:
                res = s2_request(search_url)
            except S2RetryExhaustedError:
                res = None
            if res is None:
                s2_status = 'error'
            elif res.get("data"):
                s2_paper = _select_supported_s2_candidate(raw_text, res)
                if not s2_paper:
                    s2_status = 'unverified'
            if s2_paper:
                s2_status = 'mapped'

        c_paper_id = s2_paper.get("paperId") if s2_paper else None
        c_title    = s2_paper.get("title")    if s2_paper else None
        c_year     = s2_paper.get("year")     if s2_paper else None
        c_cc       = s2_paper.get("citationCount", 0)            if s2_paper else 0
        c_ic       = s2_paper.get("influentialCitationCount", 0) if s2_paper else 0
        c_doi      = (s2_paper.get("externalIds") or {}).get("DOI") if s2_paper else None
        c_authors  = _fmt_authors(s2_paper.get("authors") or []) if s2_paper else None

        update_reference_s2_data(
            ref_id=ref["id"],
            cited_paper_id=c_paper_id,
            cited_title=c_title,
            cited_year=c_year,
            cited_citation_count=c_cc,
            cited_influential_count=c_ic,
            cited_doi=c_doi,
            cited_authors=c_authors,
            s2_status=s2_status,
        )
        if s2_paper:
            resolved += 1

    remaining = total - len(to_process)
    msg = f"Resolved {resolved}/{len(to_process)} skipped refs."
    if remaining:
        msg += f" {remaining} still skipped (increase --epub-budget to resolve more)."
    return {"status": "success", "resolved": resolved, "processed": len(to_process),
            "total_skipped": total, "message": msg}
