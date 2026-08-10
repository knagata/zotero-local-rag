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
import unicodedata
import numpy as np
from typing import Callable, Dict, Any, Optional, List, Tuple

from pathlib import Path

if __package__:
    from .db_relations import (
        clear_s2_relations_for_item,
        get_item_s2_paper_id,
        get_s2_lookup_candidates,
        insert_citation,
        update_item_citation_status,
    )
    from .reference_text import is_short_form_reference, s2_candidate_is_supported
    from .v3_data_plane import (
        collection_name as v3_collection_name,
        chroma_dir as _chroma_dir,
    )
    from .env_utils import load_dotenv_native as _load_dotenv
else:  # pragma: no cover - direct script imports
    from db_relations import (
        clear_s2_relations_for_item,
        get_item_s2_paper_id,
        get_s2_lookup_candidates,
        insert_citation,
        update_item_citation_status,
    )
    from reference_text import is_short_form_reference, s2_candidate_is_supported
    from v3_data_plane import (
        collection_name as v3_collection_name,
        chroma_dir as _chroma_dir,
    )
    from env_utils import load_dotenv_native as _load_dotenv


class S2RetryExhaustedError(Exception):
    """Raised by s2_request when all retries are exhausted (persistent rate-limiting).
    Distinct from returning None, which means the paper was not found."""


ROOT = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# .env を読むのは CHROMA_DIR を読む *前* でなければならない。
# load_dotenv_native は既に設定済みの変数を上書きしないので、順序が逆だと
# 「.env にしか書いていない CHROMA_DIR」をこのモジュールだけが無視し、他の
# モジュールとは別の場所を見にいく（2026-08-04）。
_load_dotenv()

# 解決規則は v3_data_plane.resolve_configured_path が唯一の正典（~展開と
# プロジェクトルート基準の相対解決）。ここで独自に Path() するとその2つが
# 抜け、CWD次第で別のディレクトリを指す。
CHROMA_DIR = _chroma_dir(ROOT)
_s2_key_at_import = os.environ.get("S2_API_KEY", "")
print(
    "[citation_mapper] S2_API_KEY: "
    + ("SET (length=%d)" % len(_s2_key_at_import) if _s2_key_at_import else "NOT SET — Citation Network disabled"),
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
    """1行でインプレース更新するプログレスバー。完了時は改行を出す。

    The in-place ``\\r`` update only makes sense on a real terminal: a file
    (including a log captured via ``tee``, as Maintenance-Widget.command does)
    has no concept of "overwrite the current line", so every intermediate
    frame was previously kept verbatim -- hundreds of ``\\r``-joined updates
    per item piling onto what looks like one endless line, and a citation
    run over a library with many EPUB references pushing a log past 10MB
    (found 2026-08-02). When ``file`` is not a tty, fall back to a handful of
    real, newline-terminated milestone lines instead.
    """
    if not file.isatty():
        step = max(1, total // 10)
        if current != 1 and current != total and current % step != 0:
            return
        pct = current / total * 100 if total else 100
        print(f"        {label.strip()}: {current}/{total} ({pct:.0f}%)", file=file, flush=True)
        return
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

#: Chunk vectors for the *one* item being processed: its ids, and their unit
#: vectors as a single (N, dim) matrix.
#:
#: This used to be a dict keyed by item_key that was never evicted, so a run
#: over the library accumulated every item it had already finished with. At
#: 513,683 chunks held as individual small arrays that reached a 26 GB
#: footprint and drove the machine into swap. Items are processed one after
#: another and a finished item is never revisited, so one item's worth is all
#: that is ever needed.
_ITEM_CHUNKS_CACHE_KEY: Optional[str] = None
_ITEM_CHUNK_IDS: List[str] = []
_ITEM_CHUNK_MATRIX: Optional[np.ndarray] = None

# Debug logs live beside the relations database they describe, resolved when
# they are written rather than at import. A run pointed at another database --
# a test with its own RELATIONS_DB_PATH, an audit against a copy -- was still
# appending its log into the real data directory, which is both wrong and how
# a test came to touch the user's data at all (2026-08-09).
def _debug_log_path(name: str) -> str:
    from db_relations import DB_PATH

    return str(Path(DB_PATH).resolve().parent / name)


def _debug_log() -> str:
    return _debug_log_path("mapping_debug.log")


def _debug_ref_log() -> str:
    return _debug_log_path("mapping_debug_ref.log")

# Cosine distance threshold for accepting a chunk match.
# Cosine distance range is [0, 2]; 0 = identical, 2 = opposite.
# 0.4 corresponds to cosine similarity ≥ 0.6 — a meaningful match.
_MAX_COSINE_DISTANCE = 0.4



def _get_emb_fn():
    """Get or create the embedding function (cached)."""
    global _EMB_FN_CACHE
    if _EMB_FN_CACHE is not None:
        return _EMB_FN_CACHE

    if __package__:
        from .embedder import resolve_embedder_settings, create_embedding_function
    else:  # pragma: no cover - direct script imports
        from embedder import resolve_embedder_settings, create_embedding_function
    cfg = resolve_embedder_settings(ROOT)
    provider_label = f"'{cfg.model_name}' on {cfg.device}"
    print(f"[citation_mapper] Loading embedding function: {provider_label} ...", file=sys.stderr)
    _EMB_FN_CACHE = create_embedding_function(cfg)
    # Warmup
    _ = _EMB_FN_CACHE(["warmup"])
    print("[citation_mapper] Embedding function ready.", file=sys.stderr)
    return _EMB_FN_CACHE


def _configured_collection_name() -> str:
    """Return the one Chroma collection citation mapping is allowed to read.

    Citation matching must never infer the active collection from the largest
    SQLite segment: a previous rebuild can leave a larger legacy collection in
    the same Chroma directory.  Match the indexer's configuration instead.
    """
    return v3_collection_name()


def _chroma_db_stat() -> tuple[tuple[int, int, int, int], ...]:
    """Return a cheap generation hint including SQLite's WAL sidecar.

    Chroma commonly writes in WAL mode, where ``chroma.sqlite3`` itself can
    remain unchanged while collection rows are replaced.  Looking only at the
    main file would therefore keep item-vector caches from an old generation.
    """
    stats: list[tuple[int, int, int, int]] = []
    for path in (CHROMA_DIR / "chroma.sqlite3", CHROMA_DIR / "chroma.sqlite3-wal"):
        try:
            stat = path.stat()
        except OSError:
            continue
        stats.append((
            int(stat.st_dev), int(stat.st_ino), int(stat.st_mtime_ns), int(stat.st_size),
        ))
    return tuple(stats)


def _segment_snapshot(collection_name: str) -> Dict[str, Any]:
    """Read an exact collection's segment identity and current row count.

    The collection UUID and count form a generation token.  The SQLite stat is
    included so a delete/recreate with the same name (or a database replacement)
    cannot reuse vectors embedded from the old collection.
    """
    db_file = CHROMA_DIR / "chroma.sqlite3"
    if not db_file.exists():
        raise RuntimeError(f"ChromaDB database is missing: {db_file}")
    db_stat = _chroma_db_stat()
    conn = sqlite3.connect(str(db_file), timeout=10)
    try:
        rows = conn.execute("""
            SELECT c.id, s.id, COUNT(e.id) AS cnt
            FROM collections c
            JOIN segments s ON s.collection = c.id AND s.scope = 'METADATA'
            LEFT JOIN embeddings e ON e.segment_id = s.id
            WHERE c.name = ?
            GROUP BY c.id, s.id
        """, (collection_name,)).fetchall()
        if len(rows) != 1:
            if not rows:
                raise RuntimeError(
                    f"Configured Chroma collection '{collection_name}' was not found or has no metadata segment."
                )
            raise RuntimeError(
                f"Configured Chroma collection '{collection_name}' is ambiguous ({len(rows)} metadata segments)."
            )
        collection_id, meta_seg_id, chunk_count = rows[0]
        if int(chunk_count) <= 0:
            raise RuntimeError(
                f"Configured Chroma collection '{collection_name}' has no embeddings."
            )
        vector_rows = conn.execute("""
            SELECT s.id
            FROM segments s
            WHERE s.collection = ? AND s.scope = 'VECTOR'
        """, (collection_id,)).fetchall()
        if len(vector_rows) != 1:
            if not vector_rows:
                raise RuntimeError(f"No vector segment found for collection '{collection_name}'")
            raise RuntimeError(
                f"Configured Chroma collection '{collection_name}' is ambiguous ({len(vector_rows)} vector segments)."
            )
    finally:
        conn.close()

    return {
        "metadata_segment_id": meta_seg_id,
        "vector_segment_id": vector_rows[0][0],
        "collection_id": collection_id,
        "collection_name": collection_name,
        "chunk_count": int(chunk_count),
        "db_stat": db_stat,
        "generation": (
            str(collection_id), int(chunk_count), *db_stat,
        ),
    }


def _get_segment_meta() -> Dict[str, Any]:
    """
    Discover the correct metadata segment and vector segment from ChromaDB's SQLite.
    Returns dict with keys: metadata_segment_id, vector_segment_id, collection_name, chunk_count.
    """
    global _SEGMENT_META
    configured_name = _configured_collection_name()
    db_file = CHROMA_DIR / "chroma.sqlite3"
    if _SEGMENT_META is not None and db_file.exists():
        db_stat = _chroma_db_stat()
        if (
            _SEGMENT_META.get("collection_name") == configured_name
            and _SEGMENT_META.get("db_stat") == db_stat
        ):
            return _SEGMENT_META
    snapshot = _segment_snapshot(configured_name)
    if _SEGMENT_META is not None and _SEGMENT_META.get("generation") == snapshot["generation"]:
        return _SEGMENT_META

    # Item vectors are tied to both the collection UUID and the exact set of
    # chunks.  Clear them before accepting a changed generation.
    global _ITEM_CHUNKS_CACHE_KEY, _ITEM_CHUNK_IDS, _ITEM_CHUNK_MATRIX
    _ITEM_CHUNKS_CACHE_KEY, _ITEM_CHUNK_IDS, _ITEM_CHUNK_MATRIX = None, [], None
    _SEGMENT_META = snapshot
    print(
        f"[citation_mapper] Using collection '{snapshot['collection_name']}' "
        f"({snapshot['chunk_count']} chunks)", file=sys.stderr,
    )
    return _SEGMENT_META


def _unit_rows(vectors) -> np.ndarray:
    """Stack vectors into one (N, dim) matrix whose rows are unit length.

    Always copies: np.asarray would hand back the caller's own array when it is
    already float32, and the normalisation below would then rewrite it in place.
    """
    matrix = np.array(vectors, dtype=np.float32, copy=True)
    if matrix.ndim == 1:
        matrix = matrix.reshape(1, -1)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    # A zero vector has no direction. Dividing by its norm would make every
    # component inf or NaN, which then propagates through the dot product and
    # corrupts the ranking of *every* chunk; leaving it at zero keeps its
    # similarity at 0, which is what "no direction" should score.
    #
    # Substituting 1 rather than masking with ``where=``: the masked lanes are
    # still evaluated in SIMD, so the divide-by-zero raises the FPU flag even
    # though its result is discarded, and numpy reports it against whatever
    # operation runs next -- a RuntimeWarning apparently coming from the matmul.
    np.divide(matrix, np.where(norms > 0, norms, np.float32(1.0)), out=matrix)
    return matrix


def _stored_item_vectors(item_key: str) -> Optional[Tuple[List[str], np.ndarray]]:
    """Read one item's already-indexed vectors out of Chroma, or None."""
    try:
        import chromadb

        collection = chromadb.PersistentClient(
            path=str(CHROMA_DIR),
        ).get_collection(_configured_collection_name())
        stored = collection.get(where={"itemKey": item_key}, include=["embeddings"])
    except Exception as exc:
        print(
            f"[citation_mapper] Could not read stored vectors for {item_key} "
            f"({type(exc).__name__}: {str(exc)[:120]}); re-embedding instead.",
            file=sys.stderr,
        )
        return None
    ids = list(stored.get("ids") or [])
    vectors = stored.get("embeddings")
    if not ids or vectors is None or len(vectors) != len(ids):
        return None
    return ids, _unit_rows(vectors)


def _load_chunks_for_item(item_key: str) -> Tuple[List[str], Optional[np.ndarray]]:
    """Chunk ids and unit vectors for one item, as ``(ids, (N, dim) matrix)``.

    The vectors are read back from Chroma, which already holds them: this used
    to re-embed every chunk's text from scratch on the belief that ChromaDB 1.5+
    no longer persists the id→label mapping. It does -- index_metadata.pickle is
    present with all 513,683 entries -- and re-embedding cost 134s and 3.8 GB for
    one 2,861-chunk item where reading takes 0.9s and 2.6 GB. Embedding remains
    as a fallback so a missing or stale index still yields an answer.

    Only the item currently being processed is held. Caching every item, as this
    used to, accumulated the whole library: 513,683 chunks as individual small
    arrays reached a 26 GB footprint and pushed the machine into swap. Items are
    processed one after another and never revisited, so the previous item's
    vectors are dead weight the moment the next one starts.
    """
    global _ITEM_CHUNKS_CACHE_KEY, _ITEM_CHUNK_IDS, _ITEM_CHUNK_MATRIX

    seg = _get_segment_meta()
    if item_key == _ITEM_CHUNKS_CACHE_KEY:
        return _ITEM_CHUNK_IDS, _ITEM_CHUNK_MATRIX

    # Drop the previous item before loading the next one, so the two are never
    # resident at once.
    _ITEM_CHUNKS_CACHE_KEY, _ITEM_CHUNK_IDS, _ITEM_CHUNK_MATRIX = None, [], None

    stored = _stored_item_vectors(item_key)
    if stored is not None:
        _ITEM_CHUNK_IDS, _ITEM_CHUNK_MATRIX = stored
        _ITEM_CHUNKS_CACHE_KEY = item_key
        return _ITEM_CHUNK_IDS, _ITEM_CHUNK_MATRIX

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
        _ITEM_CHUNKS_CACHE_KEY = item_key
        return [], None

    ef = _get_emb_fn()
    embedding_ids = [r[0] for r in rows]
    texts = [r[1] for r in rows]

    try:
        vectors = ef(texts)
    except Exception as e:
        print(f"[citation_mapper] Embedding failed for {item_key}: {e}", file=sys.stderr)
        return [], None

    # Normalized once here rather than on every search: the vectors do not
    # change, and re-normalizing them per query context allocated one temporary
    # array per chunk per context.
    _ITEM_CHUNK_IDS = embedding_ids
    _ITEM_CHUNK_MATRIX = _unit_rows(vectors)
    _ITEM_CHUNKS_CACHE_KEY = item_key
    print(f"[citation_mapper] Embedded {len(embedding_ids)} chunks for item {item_key}", file=sys.stderr)
    return _ITEM_CHUNK_IDS, _ITEM_CHUNK_MATRIX


def search_chunks(query_text: str, item_key: str, n_results: int = 1) -> List[Dict[str, Any]]:
    """
    Search for the most similar chunks to query_text within a specific item.
    Bypasses ChromaDB entirely - uses SQLite + hnswlib files + numpy cosine similarity.
    """
    ef = _get_emb_fn()

    chunk_ids, matrix = _load_chunks_for_item(item_key)
    if matrix is None or not chunk_ids:
        return []

    query_emb = _unit_rows(ef([query_text])[0])[0]

    # One matmul over the item's whole matrix. This was a Python loop that
    # re-normalized every chunk on every call -- for an item with a thousand
    # chunks and several hundred citation contexts that is a million throwaway
    # arrays, and it dominated the runtime of a full citation refresh.
    # numpy 2.2 on macOS/Accelerate raises divide-by-zero, overflow and invalid
    # RuntimeWarnings from matmul even for two arrays that are entirely finite
    # -- reproducible with freshly generated random unit vectors and no code of
    # ours involved. The results are correct; only the FPU flags are spurious,
    # and left alone they print three warnings per citation context.
    with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
        similarities = matrix @ query_emb
    if n_results >= len(chunk_ids):
        order = np.argsort(-similarities)
    else:
        # Only the top n need to be in order; the rest do not need sorting.
        top = np.argpartition(-similarities, n_results - 1)[:n_results]
        order = top[np.argsort(-similarities[top])]
    return [
        {"id": chunk_ids[i], "distance": float(1.0 - similarities[i])}
        for i in order
    ]


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

def _clean_query_text(value: str) -> str:
    """Strip punctuation and collapse spaces so S2's tokenizer sees plain words."""
    return re.sub(r'\s+', ' ', re.sub(r'[^\w\s]', ' ', value)).strip()


def _fold_diacritics(value: str) -> str:
    """Drop combining marks so accented and plain spellings of a name compare equal.

    Zotero keeps the author's own orthography while S2 and OpenAlex routinely
    strip it, so "Żylinska"/"Zylinska", "Fáber"/"Faber" and "Ōmura"/"Omura" are
    the same person written two ways. Without folding, the authorship check
    rejects a work in favour of nobody.
    """
    return "".join(
        char for char in unicodedata.normalize("NFKD", value)
        if not unicodedata.combining(char)
    )


#: Single letters are initials ("B." in "B. Bratton") and carry no identity, but
#: two-letter tokens are real romanized surnames (Xu, Li, Ng, Wu). Dropping those
#: emptied the token set for such names, which silently disabled the whole
#: review-rejection check below.
_MIN_NAME_TOKEN_CHARS = 2


def _creator_name_tokens(creators: str) -> set:
    """Every name token in Zotero's ``"Last First, Last First"`` creator string.

    Deliberately not just the leading surname: compound and particled surnames
    ("Hartman Davies", "Von Essen") are split differently by Zotero and by S2,
    and a first-token-only comparison misses those. Reviewers' names are wholly
    disjoint from the author's, so the extra tokens do not weaken the check.
    """
    return {
        token.casefold()
        for token in _clean_query_text(_fold_diacritics(creators)).split()
        if len(token) >= _MIN_NAME_TOKEN_CHARS
    }


#: Generational suffixes sit after the surname and must not be mistaken for it.
_NAME_SUFFIXES = {"jr", "sr", "ii", "iii", "iv"}


def _external_surnames(names) -> set:
    """Surname of each externally-listed author (``"B. Bratton"`` -> ``{"bratton"}``).

    Only the surname is taken, because a shared *given* name is not evidence of
    identity: a review of Pratt's "Imperial Eyes" written by Mary Baine Campbell
    shares "Mary" with the Zotero creator "Pratt Mary Louise" and would pass an
    any-token comparison. External sources usually render names "First Last", so
    the surname is the trailing token.

    A bare two-word name is the exception: S2 carries CJK authors both ways
    ("Keiichi Omura" and "Omura Keiichi"), and there is nothing in the string to
    say which token is the surname, so both are kept and the comparison becomes
    order-independent. An initial ("B." in "B. Bratton") does mark the given-name
    position, and three-or-more-token names keep the trailing token alone, so
    neither of those loses the precision the check depends on.
    """
    surnames = set()
    for name in names:
        raw = _clean_query_text(_fold_diacritics(str(name or ""))).split()
        tokens = [token.casefold() for token in raw if len(token) >= _MIN_NAME_TOKEN_CHARS]
        while len(tokens) > 1 and tokens[-1] in _NAME_SUFFIXES:
            tokens.pop()
        if not tokens:
            continue
        surnames.add(tokens[-1])
        if len(tokens) == 2 and len(raw) == 2:
            surnames.add(tokens[0])
    return surnames


def _s2_name_tokens(paper: Dict[str, Any]) -> set:
    """Surnames S2 lists for a paper, comparable to Zotero's creator tokens."""
    return _external_surnames(
        (author.get("name") or "") for author in paper.get("authors") or []
    )


def _record_names_a_creator(paper: Dict[str, Any], creators: str) -> bool:
    """Whether an S2 record credits any of the item's Zotero creators.

    Missing evidence counts as a pass: an item with no creators recorded, or an
    S2 record listing no author, cannot be judged either way and must not be
    rejected on that basis. Only a record naming *different* people is refused.
    """
    wanted = _creator_name_tokens(creators)
    listed = _s2_name_tokens(paper)
    if not wanted or not listed:
        return True
    return bool(wanted & listed)


#: How S2 marks a review record: a parenthesised "(review)" in the title. Only
#: that marking counts. Matching the phrase "review of" as well looked more
#: thorough and refused "Annual Review of Anthropology" and "The Review of
#: English Studies" -- journal names a humanities library is full of.
_REVIEW_TITLE_RE = re.compile(r"\(review\)", re.IGNORECASE)


def _titles_a_review(paper: Dict[str, Any]) -> bool:
    """Whether the record is a review *of* a work rather than the work."""
    return bool(_REVIEW_TITLE_RE.search(paper.get("title") or ""))


def _select_s2_title_match(
    results: List[Dict[str, Any]], full_title: str, main_title: str, creators: str,
) -> Optional[Dict[str, Any]]:
    """Choose the S2 record that is the same work, or None when none qualifies."""
    def _sim(paper: Dict[str, Any]) -> float:
        s2_title = (paper.get("title") or "").casefold()
        # Compare against both forms: S2 stores some works with their subtitle
        # and some without, and either shape is a legitimate match.
        return max(
            difflib.SequenceMatcher(None, full_title.casefold(), s2_title).ratio(),
            difflib.SequenceMatcher(None, main_title.casefold(), s2_title).ratio(),
        )

    similar = [p for p in results if _sim(p) >= 0.5]
    if not similar:
        return None

    # A title alone cannot separate a work from a same-titled different work, or
    # from a *review* of the work written by someone else -- and adopting a
    # review's paperId would attach its whole citation graph to the book. This is
    # the same rule the DOI/ISBN path applies, so both go through one helper:
    # each is the only guard on its own path, and a rule change that reached just
    # one of them would make the two paths disagree about the same record.
    similar = [p for p in similar if _record_names_a_creator(p, creators)]
    if not similar:
        return None

    # That guard passes a record listing no author at all, because missing
    # evidence cannot convict -- which is right on the DOI/ISBN path, where the
    # identifier already establishes identity, and leaves this path with nothing
    # checking it. A review indexed without its reviewer is then accepted on
    # title similarity alone: "Camera Geologica" (Angus) matched "Contact: Art
    # and the Pull of Print ... and: Camera Geologica ... by Siobhan Angus
    # (review)", a record with an empty author list, and took its paperId.
    # S2 marks these in the title, so they can be refused by name. Measured
    # across all 274 currently mapped items, none carries "(review)" in its
    # title, so nothing correct is lost. update_citations applies the same rule
    # to OpenAlex results; only this path lacked it.
    similar = [p for p in similar if not _titles_a_review(p)]
    if not similar:
        return None

    # Rank by similarity tier first so a near-exact title beats a merely close
    # one, then by citation count so the canonical record wins over stub
    # duplicates of the same work.
    return max(similar, key=lambda p: (round(_sim(p), 1), p.get("citationCount", 0) or 0))


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
            # An identifier is strong evidence, but not proof of *whose* work
            # this is. A DOI can point at a review of the book rather than the
            # book (Pratt "Imperial Eyes" carried 10.1086/600773, Lorimer's
            # review), and returning it unchecked imported the review's
            # citations. Only a record crediting different people is refused;
            # the title is not compared, because a legitimate identifier match
            # may carry a translated or differently-subtitled title.
            if _record_names_a_creator(res, creators):
                return res
            print(
                f"        -> {prefix}{identifier} resolves to "
                f"'{(res.get('title') or '')[:60]}' by "
                f"{', '.join(sorted(_s2_name_tokens(res)))}, which does not name "
                f"this item's creators; falling back to title search...",
                file=sys.stderr,
            )
        else:
            print("        -> S2 exact lookup failed, falling back to title search...", file=sys.stderr)

    # 2. Fallback to title search.
    # S2 indexes primarily English papers; skip search if title is mostly non-ASCII (e.g. Japanese)
    non_ascii_ratio = sum(1 for c in title if ord(c) > 0x7F) / max(len(title), 1)
    if non_ascii_ratio > 0.3:
        return None

    full_title = _clean_query_text(title)
    main_title = _clean_query_text(title.split(':')[0])

    # Query with the title alone first. Appending the author and the year --
    # which this used to do as its *only* query -- poisons S2's relevance
    # ranking: the year matches thousands of unrelated papers, and an author can
    # empty the result set outright. The year is never a query term nor a
    # filter, because Zotero records the edition year, which for reprints and
    # translations differs from S2's original by decades.
    queries = [full_title]
    if main_title and main_title != full_title:
        queries.append(main_title)
    # Last resort: a short or generic title ("Biopiracy") gives S2's ranking
    # nothing to discriminate on, and there the author is the only usable
    # signal. Reached only when the title-only queries found nothing, so the
    # extra call is spent on items that would otherwise stay unresolved.
    lead_author = _clean_query_text(creators.split(',')[0]) if creators else ""
    if lead_author:
        queries.append(f"{full_title} {lead_author}".strip())

    seen_queries = set()

    for query in queries:
        if len(query) < 4 or query in seen_queries:
            continue
        seen_queries.add(query)
        url = (
            "https://api.semanticscholar.org/graph/v1/paper/search"
            f"?query={urllib.parse.quote(query)}&limit=20"
            "&fields=paperId,title,authors,year,citationCount"
        )
        res = s2_request(url)
        best_match = _select_s2_title_match(
            (res or {}).get("data") or [], full_title, main_title, creators,
        )
        if best_match:
            print(
                f"        -> Best Match: '{best_match.get('title')}' "
                f"(query='{query}', Citations: {best_match.get('citationCount', 0)})",
                file=sys.stderr,
            )
            return best_match

    print("        -> No S2 results passed title/author verification; returning None.", file=sys.stderr)
    return None


# ---------------------------------------------------------------------------
# Main citation mapping functions
# ---------------------------------------------------------------------------


def _fetch_s2_relation_pages(
    paper_id: str,
    relation: str,
    max_items: int,
    limit: int = 1000,
) -> Tuple[List[Dict[str, Any]], bool, bool]:
    """Fetch one complete S2 relation list, retaining retry/limit state.

    Citations and references use the same S2 pagination contract.  Keeping the
    partial rows is intentional: they are useful to the user, while the
    ``incomplete`` flag prevents a partial refresh from being reported as
    complete.  The caller handles the historical difference that a failed
    first citations page is an immediate error.
    """
    fields = "title,year,authors,contexts,citationCount,influentialCitationCount,externalIds"
    if relation == "citations":
        fields = "title,year,authors,contexts,intents,citationCount,influentialCitationCount,externalIds"

    data_items: List[Dict[str, Any]] = []
    offset = 0
    incomplete = False
    limit_reached = False
    while True:
        url = (
            f"https://api.semanticscholar.org/graph/v1/paper/{paper_id}/{relation}"
            f"?fields={fields}&limit={limit}&offset={offset}"
        )
        try:
            response = s2_request(url)
        except S2RetryExhaustedError:
            response = None

        if response is None:
            incomplete = True
            break

        page_data = response.get("data", [])
        if not page_data:
            break
        data_items.extend(page_data)

        if len(data_items) >= max_items:
            data_items = data_items[:max_items]
            limit_reached = bool(response.get("next"))
            break

        next_offset = response.get("next")
        if not next_offset:
            break
        offset = next_offset

    return data_items, incomplete, limit_reached


def _map_s2_relation_contexts(
    data_items: List[Dict[str, Any]],
    item_key: str,
    relation: str,
    save_relation: Callable[..., None],
) -> Tuple[int, int]:
    """Match relation contexts to local chunks and pass rows to a saver.

    The matching, page-hint extraction, diagnostics, and no-context behavior
    are deliberately shared.  ``save_relation`` remains relation-specific so
    the asymmetric citation/reference database columns stay explicit.
    """
    paper_key = "citingPaper" if relation == "citations" else "citedPaper"
    debug_path = _debug_log() if relation == "citations" else _debug_ref_log()
    debug_label = "Global Context" if relation == "citations" else "Local Context"
    no_hit_label = "Global Context" if relation == "citations" else "Context"
    progress_label = "  citing  " if relation == "citations" else "  reference"
    mapped_count = 0
    total_contexts = 0

    for index, item in enumerate(data_items):
        _pbar(index + 1, len(data_items), progress_label, file=sys.stderr)
        paper = item.get(paper_key, {})
        metadata = {
            "paper_id": paper.get("paperId", ""),
            "title": paper.get("title", ""),
            "year": paper.get("year"),
            "citation_count": paper.get("citationCount", 0),
            "influential_count": paper.get("influentialCitationCount", 0),
            "doi": (paper.get("externalIds") or {}).get("DOI"),
            "authors": _fmt_authors(paper.get("authors") or []),
        }
        contexts = item.get("contexts", [])

        if not contexts:
            save_relation(
                item_key=item_key, metadata=metadata, context="", page_hint=None,
                matched_chunk_id=None, matched_distance=None, status="no_context",
            )
            continue

        for context in contexts:
            total_contexts += 1
            page_match = re.search(
                r"\b(?:p\.|page|pp\.|:)\s*(\d+)\b", context, re.IGNORECASE,
            )
            page_hint = page_match.group(1) if page_match else None
            hits = search_chunks(context, item_key, n_results=1)
            matched_chunk_id = None
            matched_distance = None
            status = "no_chunk"

            if hits:
                best_distance = hits[0]["distance"]
                with open(debug_path, "a") as log_file:
                    log_file.write(f"{debug_label}: {context[:100]}...\n")
                    log_file.write(f"  -> Best Hit Distance: {best_distance:.4f}\n")
                if best_distance < _MAX_COSINE_DISTANCE:
                    matched_chunk_id = hits[0]["id"]
                    matched_distance = best_distance
                    status = "matched"
                    mapped_count += 1
            else:
                with open(debug_path, "a") as log_file:
                    log_file.write(f"{no_hit_label}: {context[:100]}...\n")
                    log_file.write("  -> No chunks found in DB for this item.\n")

            save_relation(
                item_key=item_key, metadata=metadata, context=context,
                page_hint=page_hint, matched_chunk_id=matched_chunk_id,
                matched_distance=matched_distance, status=status,
            )

    return mapped_count, total_contexts


def _save_global_citation(
    *, item_key: str, metadata: Dict[str, Any], context: str,
    page_hint: Optional[str], matched_chunk_id: Optional[str],
    matched_distance: Optional[float], status: str,
) -> None:
    """Persist one incoming citation row."""
    insert_citation(
        citing_paper_id=metadata["paper_id"], citing_title=metadata["title"],
        citing_year=metadata["year"], context_snippet=context,
        cited_item_key=item_key, cited_chunk_id=matched_chunk_id,
        similarity_distance=matched_distance, page_hint=page_hint,
        citing_citation_count=metadata["citation_count"],
        citing_influential_count=metadata["influential_count"],
        chunk_status=status, citing_doi=metadata["doi"],
        citing_authors=metadata["authors"] or None,
    )


def _save_global_reference(
    *, item_key: str, metadata: Dict[str, Any], context: str,
    page_hint: Optional[str], matched_chunk_id: Optional[str],
    matched_distance: Optional[float], status: str,
) -> None:
    """Persist one outgoing reference row (whose columns differ from citations)."""
    if __package__:
        from .db_relations import insert_reference
    else:  # pragma: no cover - direct script imports
        from db_relations import insert_reference
    insert_reference(
        cited_paper_id=metadata["paper_id"], cited_title=metadata["title"],
        cited_year=metadata["year"], context_snippet=context,
        citing_item_key=item_key, citing_chunk_id=matched_chunk_id,
        similarity_distance=matched_distance, page_hint=page_hint,
        source="s2", s2_status=status,
        cited_citation_count=metadata["citation_count"],
        cited_influential_count=metadata["influential_count"],
        cited_doi=metadata["doi"], cited_authors=metadata["authors"] or None,
    )


def _finish_global_mapping(
    item_key: str,
    s2_paper: Dict[str, Any],
    mapped_count: int,
    total_contexts: int,
    ref_mapped_count: int,
    ref_total_contexts: int,
    citation_incomplete: bool,
    reference_incomplete: bool,
    citation_limited: bool,
    reference_limited: bool,
) -> Dict[str, Any]:
    """Write the final mapping diagnostic and preserve retryable statuses."""
    message = (
        f"Global Citations: {mapped_count}/{total_contexts} contexts mapped. "
        f"Local References: {ref_mapped_count}/{ref_total_contexts} contexts mapped."
    )
    with open(_debug_log(), "a") as log_file:
        log_file.write(f"Result for {item_key}: {message}\n")

    incomplete_parts = [
        name for name, incomplete in (
            ("citations", citation_incomplete), ("references", reference_incomplete),
        ) if incomplete
    ]
    common = {
        "s2_paper": s2_paper,
        "total_contexts_analyzed": total_contexts,
        "mapped_count": mapped_count,
        "references_contexts_analyzed": ref_total_contexts,
        "references_mapped_count": ref_mapped_count,
    }
    if incomplete_parts:
        update_item_citation_status(item_key, "error")
        return {
            "status": "error",
            "message": (
                f"S2 pagination incomplete for {', '.join(incomplete_parts)}; "
                "partial relations were saved and will be retried."
            ),
            "retryable": True, "incomplete_parts": incomplete_parts, **common,
        }

    limited_parts = [
        name for name, limited in (
            ("citations", citation_limited), ("references", reference_limited),
        ) if limited
    ]
    if limited_parts:
        update_item_citation_status(item_key, "limited")
        return {
            "status": "error",
            "message": (
                "S2 pagination reached max_citations for "
                f"{', '.join(limited_parts)}; increase the limit before retrying."
            ),
            "retryable": False, "incomplete_parts": limited_parts, **common,
        }

    update_item_citation_status(item_key, "s2_done")
    return {"status": "success", "s2_resolved": True, "message": message, **common}

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
                if __package__:
                    from .crossref_client import fetch_crossref_by_doi
                else:  # pragma: no cover - direct script imports
                    from crossref_client import fetch_crossref_by_doi
                meta = fetch_crossref_by_doi(doi)
            except Exception as exc:  # CrossrefError 含む。フォールバックは best-effort。
                print(f"        -> Crossref fallback failed: {exc}", file=sys.stderr)
                meta = None
            if meta:
                cr_year = meta.get("year")
                cr_cc = meta.get("citation_count")
                print(f"        -> Crossref fallback: year={cr_year}, citations={cr_cc}", file=sys.stderr)
        # "not_found", not "s2_done": s2_done means the S2 step *identified* the
        # work and saved its relations, which is what the resume path relies on.
        update_item_citation_status(
            item_key, "not_found",
            s2_year=cr_year,            # S2 列だが Crossref 由来の年を補完保存
            s2_citation_count=cr_cc,    # 同上（is-referenced-by-count）
        )
        suffix = " (Crossref 書誌フォールバック適用)" if (cr_year or cr_cc) else ""
        # "success" here means the step ran to completion, not that the work was
        # identified. s2_resolved is what callers must branch on: recording this
        # outcome as "mapped" would claim an S2 identity the item does not have.
        return {"status": "success",
                "s2_resolved": False,
                "message": "Item not found on Semantic Scholar." + suffix,
                "mapped_count": 0}

    # 旧行はこの後それぞれの取得が終わってから落とす（下の二箇所）。ここで消すと、
    # 最初のページが 429 で枯渇したときに、置き換えるものが無いまま正常だった
    # 被引用ごと消えてしまう。同定先が変わったかどうかはログ用に保持しておく。
    previous_paper_id = get_item_s2_paper_id(item_key)

    def _clear_previous(**scope) -> None:
        """Drop the previous run's rows now that their replacement is in hand."""
        if not previous_paper_id:
            return
        removed = clear_s2_relations_for_item(item_key, **scope)
        if removed["global_citations"] or removed["global_references"]:
            changed = (
                f"identity changed ({previous_paper_id} -> {s2_paper.get('paperId')}); "
                if previous_paper_id != s2_paper.get("paperId") else "re-fetching; "
            )
            print(
                f"        -> S2 {changed}cleared {removed['global_citations']} citation "
                f"and {removed['global_references']} S2 reference rows from the previous run.",
                file=sys.stderr,
            )

    # S2 メタ情報を DB に保存（以降の処理でも参照できるよう早めに記録）
    # 注: Citation Network更新ではアブストラクトを取得しない（S2依存・API制限回避）。
    #     概要は概要パネルの取得ボタンから Zotero local API 等でオンデマンド取得する。
    update_item_citation_status(
        item_key, "mapped",
        s2_paper_id=s2_paper.get("paperId"),
        s2_year=s2_paper.get("year"),
        s2_citation_count=s2_paper.get("citationCount"),
    )

    paper_id = s2_paper["paperId"]
    limit = 1000  # Max limit per request for S2 graph API

    data_items, citation_incomplete, citation_limited = _fetch_s2_relation_pages(
        paper_id, "citations", max_citations, limit,
    )
    if citation_incomplete and not data_items:
        update_item_citation_status(item_key, "error")
        return {
            "status": "error",
            "message": "S2 API Error while fetching citations.",
            "mapped_count": 0,
            "s2_paper": s2_paper,
        }

    # A complete, uncapped response is the only safe replacement for old rows.
    if not (citation_incomplete or citation_limited):
        _clear_previous(references=False)
    if data_items:
        print(
            f"        -> Found {len(data_items)} citing papers on Semantic Scholar. "
            "Mapping to local chunks...", file=sys.stderr,
        )
    else:
        print("        -> No citing papers found on Semantic Scholar.", file=sys.stderr)
    mapped_count, total_contexts = _map_s2_relation_contexts(
        data_items, item_key, "citations", _save_global_citation,
    )

    r_data_items, reference_incomplete, reference_limited = _fetch_s2_relation_pages(
        paper_id, "references", max_citations, limit,
    )
    if not (reference_incomplete or reference_limited):
        _clear_previous(citations=False)
    if r_data_items:
        print(
            f"        -> Found {len(r_data_items)} referenced papers on Semantic Scholar. "
            "Mapping to local chunks...", file=sys.stderr,
        )
    else:
        print("        -> No referenced papers found on Semantic Scholar.", file=sys.stderr)
    ref_mapped_count, ref_total_contexts = _map_s2_relation_contexts(
        r_data_items, item_key, "references", _save_global_reference,
    )

    return _finish_global_mapping(
        item_key, s2_paper, mapped_count, total_contexts,
        ref_mapped_count, ref_total_contexts,
        citation_incomplete, reference_incomplete,
        citation_limited, reference_limited,
    )

def map_item_local_references(item_key: str, epub_path: str = "", epub_budget: int = 50) -> Dict[str, Any]:
    """
    Extract references/notes from the item's canonical V3 chunks (bibliography,
    endnote, footnote zones), resolve them to Semantic Scholar, and save to
    global_references.  ``epub_path`` is accepted for backward compatibility but
    no longer parsed: reference boundaries and body ``noteref`` links are already
    preserved at ingestion, so the second EPUB parse is retired (R6).
    """
    if __package__:
        from .chunk_reference_extractor import extract_references_from_chunks
        from .db_relations import insert_reference
    else:  # pragma: no cover - direct script imports
        from chunk_reference_extractor import extract_references_from_chunks
        from db_relations import insert_reference

    print(f"        -> Extracting local references for {item_key} from V3 chunks...", file=sys.stderr)
    local_refs = extract_references_from_chunks(item_key)

    if not local_refs:
        return {"status": "success", "message": "No EPUB references found.", "mapped_count": 0}

    # S2 lookups per item are capped to avoid 429 storms on EPUBs with hundreds of references.
    # Resolve explicit bibliography/endnote evidence before footnotes, then rank by use frequency.
    zone_priority = {"bibliography": 0, "endnote": 1, "footnote": 2}
    local_refs.sort(key=lambda r: (
        zone_priority.get(r.get("source_zone"), 9),
        -r.get("cite_count", 1), r.get("similarity_distance", 1.0),
    ))

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
    if __package__:
        from .db_relations import get_skipped_epub_refs, update_reference_s2_data
    else:  # pragma: no cover - direct script imports
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
