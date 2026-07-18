# MCP server for paragraph-level Zotero RAG (local Chroma).
from __future__ import annotations

import os
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import json
import logging
import sys
import time
import traceback
import gc
from datetime import datetime, timezone, timedelta
from typing import Any, Dict, Optional, List
from typing_extensions import TypedDict

from pathlib import Path

from env_utils import load_dotenv_native

load_dotenv_native()

import chromadb
from fastmcp import FastMCP
from zotero_source_localapi import ZoteroLocalAPI
from manifest import load_manifest
from chapter_detect import get_pdf_toc, get_epub_chapter_index_to_title
from query_expansion import expand_queries
from search_fusion import language_balanced_order
from db_relations import get_case_annotations, get_item_summary as load_item_summary


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CHROMA_DIR = os.environ.get("CHROMA_DIR", os.path.join(ROOT, "data", "chroma"))
MANIFEST_PATH = os.environ.get("MANIFEST_PATH", os.path.join(ROOT, "data", "manifest.json"))
INDEXING_LOCK_PATH = os.path.join(ROOT, "data", "indexing.lock")
_LOCK_STALE_HOURS = 4
_LOG_PATH = os.path.join(ROOT, "data", "zotero-rag.log")


def _setup_logger() -> logging.Logger:
    logger = logging.getLogger("zotero-rag")
    if logger.handlers:
        return logger
    logger.setLevel(logging.DEBUG)
    try:
        os.makedirs(os.path.join(ROOT, "data"), exist_ok=True)
        fh = logging.FileHandler(_LOG_PATH, encoding="utf-8")
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
        logger.addHandler(fh)
    except Exception:
        pass
    return logger


_log = _setup_logger()


# ---------------------------------------------------------------------------
# Indexing-lock guard — prevents queries during active ChromaDB writes.
# ---------------------------------------------------------------------------

def _check_indexing_lock() -> tuple:
    """Check whether the indexer is currently writing to ChromaDB.

    Returns ``(is_blocked: bool, message: str | None)``.
    When *is_blocked* is ``True``, the caller should abort and return
    *message* to the client.
    """
    if not os.path.exists(INDEXING_LOCK_PATH):
        return False, None

    # Read lock metadata
    try:
        lock_data = json.loads(Path(INDEXING_LOCK_PATH).read_text(encoding="utf-8"))
    except Exception:
        _log.warning("Corrupt indexing.lock — treating as stale")
        return False, None

    pid = lock_data.get("pid")

    # --- PID-based staleness check ---
    if pid is not None:
        try:
            os.kill(pid, 0)  # signal 0 = existence check only
        except OSError:
            _log.info("Indexer PID %s is dead — treating lock as stale", pid)
            return False, None
    else:
        # No PID in lock file — treat as stale
        _log.warning("indexing.lock has no PID — treating as stale")
        return False, None

    # --- Time-based staleness check (safety net) ---
    started_at = lock_data.get("started_at")
    if started_at:
        try:
            started = datetime.fromisoformat(started_at)
            age = datetime.now(timezone.utc) - started
            if age > timedelta(hours=_LOCK_STALE_HOURS):
                _log.warning(
                    "Indexing lock is %.1f h old — treating as stale",
                    age.total_seconds() / 3600,
                )
                return False, None
        except Exception:
            pass

    return True, (
        "現在インデックス更新中です。"
        "ChromaDB を使用する検索・参照機能は一時的に利用できません。\n"
        "更新完了までしばらくお待ちください（通常数分〜数十分）。\n"
        "Zotero 書誌検索（search_zotero_items）や get_item_details など "
        "ChromaDB 非依存の機能は通常通りご利用いただけます。"
    )


# Collection name is intentionally configurable.
# IMPORTANT: Chroma collections are dimension-fixed. If you switch embedding models
# (e.g., 384-d MiniLM <-> 1024-d bge-m3), use a different collection name or rebuild.
COLLECTION_NAME_DEFAULT = "zotero_paragraphs"

# Embedding model selection — delegated to embedder module (single source of truth).
from embedder import (
    resolve_embedder_settings,
    create_embedding_function,
    resolve_collection_name,
    open_chroma_collection,
)


mcp = FastMCP("zotero-paragraph-rag")

# ----------------------------
# Tool I/O shapes (for Claude)
# ----------------------------


class RagMeta(TypedDict, total=False):
    """Metadata stored alongside each paragraph chunk."""

    title: Optional[str]
    year: Optional[int]
    creators: Optional[str]
    page: Optional[int]
    page_label: Optional[str]  # book page label from PDF page label dictionary (e.g. "xii", "15")
    pdf_path: Optional[str]
    path: Optional[str]
    itemKey: Optional[str]
    attachmentKey: Optional[str]
    noteKey: Optional[str]
    source_type: Optional[str]  # "pdf" | "html" | "epub" | "note"
    locator: Optional[str]
    contentType: Optional[str]
    filename: Optional[str]
    chapter: Optional[str]  # NEW: Level-1 chapter title
    section: Optional[str]  # NEW: Level-2 section title (PDF only)
    lang: Optional[str]  # "ja" | "zh" | "en" | "other"


class RagContextChunk(TypedDict, total=False):
    """Neighboring paragraph (context window) around a hit."""

    id: str
    page: Optional[int]
    citation: str
    text: str


class RagHit(TypedDict, total=False):
    """One semantic-search hit (paragraph chunk)."""

    id: str
    distance: Optional[float]
    rrf_score: Optional[float]
    citation: str
    text: str
    context: List[RagContextChunk]
    meta: RagMeta


class RagSearchResponse(TypedDict):
    """Response returned by rag_search."""

    results: List[RagHit]


_COL = None
# Embedding function is cached separately so that collection resets do not force
# a full model reload.
_EMB_FN = None
_EMB_COLLECTION_NAME: Optional[str] = None
# Combined mtime of chroma.sqlite3 and manifest.json at the time _COL was last initialized.
# Used to proactively detect when the indexer has written new data.
_COL_INIT_MTIME: float = 0.0


def _db_mtime_sum() -> float:
    """
    Return the sum of mtimes for chroma.sqlite3 and manifest.json.
    Watching both ensures we pick up the absolute final state of the indexer.
    """
    m = 0.0
    try:
        m += os.path.getmtime(os.path.join(CHROMA_DIR, "chroma.sqlite3"))
    except OSError:
        pass
    try:
        m += os.path.getmtime(MANIFEST_PATH)
    except OSError:
        pass
    return m


def _reset_col() -> None:
    """Invalidate the cached collection and fully tear down the ChromaDB System.

    chromadb caches one System per persist path in
    ``SharedSystemClient._identifier_to_system``.  Dropping the collection and
    creating a "fresh" PersistentClient reuses that cached System — including
    the stale in-memory HNSW segment written before the indexer ran — which is
    why reloads used to require a full process restart.  Closing the client
    (and force-evicting any leaked System) makes the next _col() call re-read
    everything from disk.
    """
    global _COL
    client = getattr(_COL, "_chroma_client", None) if _COL is not None else None
    _COL = None
    if client is not None:
        try:
            client.close()
        except Exception as e:
            _log.warning("ChromaDB client close() failed: %s", e)

    # Belt and braces: if any client for this path was never closed (leaked
    # refcount), the System survives close().  Stop and evict it explicitly.
    try:
        from chromadb.api.shared_system_client import SharedSystemClient

        chroma_real = os.path.realpath(CHROMA_DIR)
        for ident in list(SharedSystemClient._identifier_to_system.keys()):
            try:
                if os.path.realpath(ident) != chroma_real:
                    continue
            except OSError:
                continue
            system = SharedSystemClient._identifier_to_system.pop(ident, None)
            with SharedSystemClient._refcount_lock:
                SharedSystemClient._identifier_to_refcount.pop(ident, None)
            if system is not None:
                _log.info("Force-stopping leaked ChromaDB System for %s", ident)
                try:
                    system.stop()
                except Exception as e:
                    _log.warning("ChromaDB System stop() failed: %s", e)
    except Exception as e:
        _log.warning("ChromaDB system-cache eviction failed: %s", e)

    gc.collect()


def _col():
    global _COL, _EMB_FN, _EMB_COLLECTION_NAME, _COL_INIT_MTIME

    # --- Proactive staleness check ---
    # The HNSW index is memory-mapped, so when the indexer rewrites it the new
    # entries become visible in the stale _COL without updating the label map,
    # which causes "Error finding id".  We detect this by comparing the mtime of
    # both the SQLite DB and the manifest (the latter is updated last).
    # If it changed, invalidate before any query runs.
    current_mtime = _db_mtime_sum()
    if _COL is not None and current_mtime > _COL_INIT_MTIME:
        msg = f"ChromaDB/Manifest modified since last init (prev={_COL_INIT_MTIME:.3f}, new={current_mtime:.3f}) — reloading collection"
        print(f"[zotero-rag] {msg}", file=sys.stderr)
        _log.info(msg)
        _reset_col()

    if _COL is not None:
        return _COL

    # --- Embedding function (cached across resets) ---
    if _EMB_FN is None:
        cfg = resolve_embedder_settings(Path(ROOT))
        try:
            provider_label = f"local model '{cfg.model_name}'"
            print(f"[PROGRESS] Initializing {provider_label} (this may take a moment)...", file=sys.stderr)
            _EMB_FN = create_embedding_function(cfg, task_type="RETRIEVAL_QUERY")
        except Exception as e:
            raise RuntimeError(
                "Failed to initialize local embedding model.\n"
                f"EMB_MODEL={cfg.model_name}\n"
                f"EMB_DEVICE={cfg.device}\n"
                "If you are running offline, ensure the model is already cached for this Python environment.\n"
                "Try once online (example): python -c 'from sentence_transformers import SentenceTransformer; SentenceTransformer(\"sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2\")'\n"
                f"Original error: {e}"
            )

    emb_fn = _EMB_FN

    # --- Collection name (cached; dimension probe is skipped after first run) ---
    if _EMB_COLLECTION_NAME is None:
        _EMB_COLLECTION_NAME = resolve_collection_name(
            emb_fn,
            env_value=os.environ.get("CHROMA_COLLECTION"),
            default=COLLECTION_NAME_DEFAULT,
        )
    collection_name = _EMB_COLLECTION_NAME

    # --- Fresh PersistentClient on every (re-)initialization ---
    _COL = open_chroma_collection(
        CHROMA_DIR, collection_name, emb_fn,
        metadata={"hnsw:space": "cosine"},
    )

    # Dimension compatibility check — uses ChromaDB get() with explicit
    # None checks (not truthiness) because the Rust backend returns numpy
    # arrays that cannot be evaluated as booleans.
    try:
        probe = emb_fn(["dimension probe"])
        probe_dim = None
        if probe is not None and len(probe) > 0 and probe[0] is not None:
            probe_dim = len(probe[0])

        if isinstance(probe_dim, int) and probe_dim > 0 and hasattr(_COL, "count") and _COL.count() > 0:
            # Use peek to get a sample ID, then get() to fetch the stored vector.
            # We avoid truthiness checks on the returned dicts/arrays.
            peek = _COL.peek(1)
            peek_ids = peek.get("ids") if peek is not None else None
            if peek_ids is not None and len(peek_ids) > 0:
                got = _COL.get(ids=[peek_ids[0]], include=["embeddings"])
                stored = got.get("embeddings") if got is not None else None
                stored_dim = None
                if stored is not None and hasattr(stored, "shape"):
                    stored_dim = stored.shape[1] if len(stored.shape) > 1 else stored.shape[0]

                if (
                    isinstance(stored_dim, int)
                    and stored_dim > 0
                    and stored_dim != probe_dim
                ):
                    raise RuntimeError(
                        "Embedding dimension mismatch for existing Chroma collection.\n"
                        f"CHROMA_DIR={CHROMA_DIR}\n"
                        f"COLLECTION={collection_name}\n"
                        f"Stored dimension={stored_dim}, embedder dimension={probe_dim}\n\n"
                        "Fix options:\n"
                        "  (1) Set EMB_PROFILE to match the model used when indexing, OR\n"
                        "  (2) Use a different CHROMA_COLLECTION name for this embedding model, OR\n"
                        "  (3) Rebuild the index for this collection (delete Chroma dir / run index_from_zotero.py --rebuild).\n"
                    )
    except Exception as e:
        if isinstance(e, RuntimeError) and "Embedding dimension mismatch" in str(e):
            raise
        # Other errors (e.g. collection not fully initialized) are non-fatal;
        # the query itself will fail with a specific error message.

    # Record the DB mtime so we can detect future indexer writes.
    _COL_INIT_MTIME = _db_mtime_sum()
    try:
        _count = _COL.count()
    except Exception:
        _count = "?"
    _log.info("Collection initialized: name=%s count=%s mtime=%.3f", collection_name, _count, _COL_INIT_MTIME)
    return _COL


_HNSW_ERROR_MSG = (
    "検索インデックスの状態が不整合です（HNSWラベルマップとバイナリの不一致）。"
    "force_reload_index ツールを実行してから再検索してください。"
    "それでも解消しない場合は Claude Desktop を再起動してください。"
)


def _hydrate_chunks_from_sqlite(chunk_ids: List[str]) -> tuple[Dict[str, str], Dict[str, dict]]:
    """Fetch chunk text and metadata directly from ChromaDB's SQLite store.

    Bypasses ChromaDB PersistentClient to avoid potential hangs/timeouts on
    the Rust HNSW index when many chunks are requested.
    """
    import sqlite3
    doc_map: Dict[str, str] = {}
    meta_map: Dict[str, dict] = {}
    db_path = Path(CHROMA_DIR) / "chroma.sqlite3"
    if not db_path.exists():
        return doc_map, meta_map
    try:
        conn = sqlite3.connect(str(db_path))
        try:
            for i in range(0, len(chunk_ids), 500):
                batch = chunk_ids[i : i + 500]
                placeholders = ",".join(["?"] * len(batch))
                cursor = conn.execute(
                    f"""
                    SELECT e.embedding_id, em.key, em.string_value, em.int_value, em.float_value
                    FROM embeddings e
                    JOIN embedding_metadata em ON em.id = e.id
                    WHERE e.embedding_id IN ({placeholders})
                    """,
                    tuple(batch),
                )
                for emb_id, key, s_val, i_val, f_val in cursor.fetchall():
                    if emb_id not in meta_map:
                        meta_map[emb_id] = {}
                    if key == "chroma:document":
                        doc_map[emb_id] = s_val
                    elif s_val is not None:
                        meta_map[emb_id][key] = s_val
                    elif i_val is not None:
                        meta_map[emb_id][key] = i_val
                    elif f_val is not None:
                        meta_map[emb_id][key] = f_val
        finally:
            conn.close()
    except Exception as e:
        _log.error("SQLite chunk hydration error: %s", e)
    return doc_map, meta_map


_Z_API = None

def parse_id(chunk_id: str):
    try:
        a0, seg1, perseg, partseg = chunk_id.split(":")
        if seg1.startswith("p"):
            source_type = "pdf"
            page = int(seg1[1:])
        elif seg1 == "html":
            source_type = "html"
            page = None
        elif seg1 == "epub":
            source_type = "epub"
            page = None
        elif seg1 == "note":
            source_type = "note"
            page = None
        else:
            return None
        para = int(perseg[4:])
        part = int(partseg[4:])
        return a0, source_type, page, para, part
    except Exception:
        return None

def neighbor_ids(chunk_id: str, w: int) -> List[str]:
    parsed = parse_id(chunk_id)
    if not parsed or w <= 0:
        return []
    a0, stype, page, para, part = parsed
    out_ids: List[str] = []
    # Long paragraphs are split into multiple parts; probe up to a
    # reasonable limit so we don't miss split-para neighbors.
    max_parts = 15
    for dp in range(-w, w + 1):
        pidx = para + dp
        if pidx < 0:
            continue
        for pi in range(max_parts):
            if stype == "pdf" and page is not None:
                cid = f"{a0}:p{page}:para{pidx}:part{pi}"
            elif stype == "html":
                cid = f"{a0}:html:para{pidx}:part{pi}"
            elif stype == "epub":
                cid = f"{a0}:epub:para{pidx}:part{pi}"
            elif stype == "note":
                cid = f"{a0}:note:para{pidx}:part{pi}"
            else:
                continue
            if cid != chunk_id:
                out_ids.append(cid)
    return out_ids


RRF_K = 60

def _where_requests_notes(w: Optional[Dict[str, Any]]) -> bool:
    """Detect whether a Chroma `where` filter explicitly includes Notes."""
    def _positive_note_stype(val: Any) -> bool:
        if val == "note":
            return True
        if isinstance(val, dict):
            if val.get("$eq") == "note":
                return True
            if "note" in (val.get("$in") or []):
                return True
        return False

    def _walk(node: Any, negated: bool = False, depth: int = 0) -> bool:
        if depth > 50 or node is None:
            return False
        if isinstance(node, dict):
            if "$not" in node:
                return _walk(node.get("$not"), not negated, depth + 1)
            if "$and" in node and isinstance(node.get("$and"), list):
                for sub in node.get("$and"):
                    if _walk(sub, negated, depth + 1):
                        return True
            if "$or" in node and isinstance(node.get("$or"), list):
                for sub in node.get("$or"):
                    if _walk(sub, negated, depth + 1):
                        return True
            if "source_type" in node:
                if _positive_note_stype(node.get("source_type")):
                    return not negated
            for v in node.values():
                if isinstance(v, (dict, list)) and _walk(v, negated, depth + 1):
                    return True
            return False
        if isinstance(node, list):
            for sub in node:
                if _walk(sub, negated, depth + 1):
                    return True
        return False
    if not w or not isinstance(w, dict):
        return False
    return _walk(w)


def _z_api():
    global _Z_API
    if _Z_API is None:
        _Z_API = ZoteroLocalAPI()
    return _Z_API


def _make_citation(md: dict) -> str:
    title = md.get("title") or ""
    year = md.get("year")
    page = md.get("page")
    page_label = (md.get("page_label") or "").strip()
    chapter = (md.get("chapter") or "").strip()

    # Prefer the book's own page label (e.g. "xii", "15") over the sequential PDF page number.
    page_display = page_label if page_label else (str(page) if page is not None else None)
    
    # [Chapter name] p.X style
    loc_part = ""
    if chapter:
        loc_part = f"[{chapter}] "
    if page_display:
        loc_part += f"p.{page_display}"
    
    if title and loc_part and year:
        return f"{title} ({year}) {loc_part}"
    if title and loc_part:
        return f"{title} {loc_part}"
    if title and year:
        return f"{title} ({year})"
    return title or ""


@mcp.tool()
def force_reload_index() -> Dict[str, Any]:
    """
    Forcefully reload the ChromaDB index and metadata.
    Use this if you have just run the indexer and are seeing 'Error finding id'
    or missing results, and the automatic reload didn't seem to trigger.
    """
    blocked, msg = _check_indexing_lock()
    if blocked:
        return {"status": "blocked", "message": msg}

    prev_mtime = _COL_INIT_MTIME
    _reset_col()
    try:
        _col()
        new_mtime = _COL_INIT_MTIME
        return {
            "status": "reloaded",
            "prev_mtime": prev_mtime,
            "new_mtime": new_mtime,
            "count": _COL.count() if _COL else None
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}


@mcp.tool()
def server_status() -> Dict[str, Any]:
    """
    Check the health and configuration of the zotero-rag MCP server.
    Use this to diagnose connection issues or verify the server is running correctly.

    Returns a status report including:
    - overall status ("ok" or "error")
    - ChromaDB path and whether it exists on disk
    - Collection name(s) found and document count
    - Embedding model profile and resolved model path
    - Whether the embedding model is already loaded in memory
    """
    report: Dict[str, Any] = {
        "status": "ok",
        "chroma_dir": CHROMA_DIR,
        "chroma_dir_exists": os.path.isdir(CHROMA_DIR),
        "emb_profile": (os.environ.get("EMB_PROFILE") or "fast"),
        "emb_model_loaded": _COL is not None,
        "collections": [],
        "errors": [],
    }

    # Surface indexing-lock state for observability
    blocked, lock_msg = _check_indexing_lock()
    report["indexing_lock"] = blocked
    if blocked:
        report["indexing_lock_message"] = lock_msg

    try:
        cfg = resolve_embedder_settings(Path(ROOT))
        report["emb_model"] = cfg.model_name
        report["emb_device"] = cfg.device
        report["emb_provider"] = cfg.provider
    except Exception as e:
        report["emb_model"] = None
        report["emb_provider"] = None
        report["errors"].append(f"EMB resolve error: {e}")

    if not report["chroma_dir_exists"]:
        report["status"] = "error"
        report["errors"].append(
            f"CHROMA_DIR does not exist: {CHROMA_DIR}. Run the indexer first."
        )
        return report

    try:
        client = chromadb.PersistentClient(path=CHROMA_DIR)
        try:
            cols = client.list_collections()
            for c in cols:
                try:
                    count = client.get_collection(c.name).count()
                except Exception:
                    count = None
                report["collections"].append({"name": c.name, "count": count})
        finally:
            # Unclosed clients leak a refcount on the shared System, which
            # would keep the stale System alive across _reset_col().
            client.close()
    except Exception as e:
        report["status"] = "error"
        report["errors"].append(f"ChromaDB error: {e}")

    if not report["collections"]:
        report["status"] = "error"
        report["errors"].append(
            "No collections found in ChromaDB. Run the indexer to build the index."
        )

    return report


def _build_effective_where(
    where: Any, *, include_notes: bool, include_item_keys: Any
) -> Any:
    """Merge note-exclusion and item-key filters into a Chroma ``where`` clause."""
    effective = where
    if (not include_notes) and (not _where_requests_notes(where)):
        note_excl = {"source_type": {"$ne": "note"}}
        effective = note_excl if effective is None else {"$and": [effective, note_excl]}
    if include_item_keys:
        item_filter = {"itemKey": {"$in": include_item_keys}}
        effective = item_filter if effective is None else {"$and": [effective, item_filter]}
    return effective


def _coerce_json(value: Any) -> Any:
    """If *value* is a JSON string, parse it; otherwise return as-is.

    MCP clients may serialize list/dict parameters as JSON strings. This
    helper lets us accept ``Any`` in the tool signature while still getting
    the correct Python object at runtime.
    """
    if value is not None and isinstance(value, str):
        stripped = value.strip()
        if stripped.startswith(("[", "{")):
            try:
                return json.loads(stripped)
            except (json.JSONDecodeError, TypeError):
                pass
    return value


@mcp.tool()
def rag_search(
    query: str | List[str],
    k: int = 5,
    where: Any = None,
    context_window: int = 0,
    include_notes: bool = False,
    include_item_keys: Any = None,
    exclude_chunk_ids: Any = None,
    auto_expand: bool = True,
    search_mode: str = "default",
    hybrid: bool = True,
    language_balance: bool = False,
) -> RagSearchResponse:
    """
    Paragraph-level semantic search over local Zotero PDFs/HTML snapshots (+ optionally Notes).
    Args:
        query:
            Natural-language query string OR a list of strings.
            Prefer a list containing Japanese and English queries plus useful synonyms so
            material in either language can be retrieved. For proper names, include both
            the original spelling and its Japanese transliteration where applicable, for
            example ["贈与論 互酬性", "gift exchange reciprocity Mauss", "モース 贈与"].
            Multi-query results are fused with Reciprocal Rank Fusion and deduplicated by
            chunk ID, saving tokens by avoiding redundant context.
        k:
            Number of results to return (after filtering short fragments and deduplication). Default: 5.
        where:
            Optional metadata filter (Chroma `where` filter). Use this to restrict eligible chunks.

            Indexed metadata keys (this project):
                - title: str
                - year: int
                - creators: str (authors joined by '; ')
                - page: int (PDF only; sequential 1-based PDF page number)
                - page_label: str (PDF only; book page label from PDF page label dictionary, e.g. "xii", "15"; empty string if not defined)
                - pdf_path: str (PDF/HTML; kept for compatibility)
                - path: str (PDF/HTML)
                - itemKey: str (parent Zotero item key)
                - attachmentKey: str (attachments)
                - noteKey: str (notes)
                - source_type: "pdf" | "html" | "epub" | "note"
                - lang: "ja" | "zh" | "en" | "other"
                - locator: str (e.g., "p12:para3" / "html:para10" / "note:para2")
                - chapter: str (Chapter title, e.g., "Chapter 1", "第一章")
                - section: str (Section title, e.g., "1.1 Introduction")
            Examples:
              - Restrict to one Zotero item:
                {"itemKey": "BGZ9UFUJ"}
              - Only HTML snapshots:
                {"source_type": "html"}
              - Only EPUB:
                {"source_type": "epub"}
              - Only Notes:
                {"source_type": "note"}   (or set include_notes=True)
              - Notes OR a specific item:
                {"$or": [{"source_type": "note"}, {"itemKey": "BGZ9UFUJ"}]}
        context_window:
            Neighbor paragraphs to fetch around a hit. Default: 0 (saves tokens).
            For PDF: neighbors are within the same page by para index.
            For HTML/Notes: neighbors are within the same doc by para index.
        include_notes:
            If True, include Zotero Notes chunks in the search space. Default: False.
            Notes are indexed but excluded by default.
        include_item_keys:
            Optional list of Zotero item keys (e.g. ['ABCDEF12', 'GHIJKL34']) to restrict the search to.
        exclude_chunk_ids:
            Optional list of chunk IDs to exclude from the results. Use this to avoid
            seeing the same paragraphs across multiple turns.
        auto_expand:
            If True (default), add cached Japanese/English query variants. Expansion
            failures fall back to the original query without failing the search.
        search_mode:
            "default" for ordinary semantic search or "case" for cross-topic case
            retrieval. Case mode adds hypothetical ethnographic passages and broader/
            narrower concepts, and uses one neighboring paragraph of context by default.
        hybrid:
            If True (default), fuse semantic results with the local FTS5 trigram index.
            Arbitrary `where` filters temporarily disable lexical fusion because they cannot
            be translated safely to SQL; semantic search still runs normally.
        language_balance:
            If True, reserve up to two final-result slots each for Japanese and English
            chunks when both are present in the candidate pool. Requires a reindex that
            includes the `lang` metadata. Default: False.
    Returns:
        {"results": [ ... ]}
    """

    blocked, msg = _check_indexing_lock()
    if blocked:
        return {"results": [], "warning": msg}

    where = _coerce_json(where)
    include_item_keys = _coerce_json(include_item_keys)
    exclude_chunk_ids = _coerce_json(exclude_chunk_ids)

    if search_mode not in {"default", "case"}:
        return {"results": [], "warning": "search_mode must be 'default' or 'case'."}
    queries = [query] if isinstance(query, str) else list(query)
    queries = [item for item in queries if isinstance(item, str) and item.strip()]
    if not queries:
        return {"results": [], "warning": "query must contain at least one non-empty string."}
    if auto_expand:
        queries = expand_queries(queries, mode=search_mode, timeout=5.0, logger=_log)
    if search_mode == "case" and context_window == 0:
        context_window = 1

    col = _col()
    if k <= 0:
        return {"results": []}


    effective_where = _build_effective_where(
        where, include_notes=include_notes, include_item_keys=include_item_keys
    )

    internal_k = max(k * 5, k)
    if exclude_chunk_ids:
        internal_k += len(exclude_chunk_ids)

    for _attempt in range(2):
        try:
            col = _col()
            if not col:
                raise ValueError("Chroma collection is empty or not initialized. Run the indexer first.")

            # Manually compute embeddings to avoid Chroma FFI deadlock
            query_embeddings = col._embedding_function(queries)

            res = col.query(
                query_embeddings=query_embeddings,
                n_results=internal_k,
                where=effective_where,
                include=["documents", "metadatas", "distances"],
            )
            break
        except Exception as _exc:
            _log.warning("rag_search query failed (attempt %d): %s\n%s", _attempt + 1, _exc, traceback.format_exc())
            if _attempt == 0:
                # Collection state may be stale (indexer ran while server was live).
                # Wait briefly to let the indexer finish a write cycle, then
                # reset the cache and re-initialize with a fresh PersistentClient.
                _reset_col()
                time.sleep(1)
                col = _col()
            else:
                _log.error("rag_search: both attempts failed — returning error message")
                return {"results": [], "warning": _HNSW_ERROR_MSG, "error": str(_exc)}

    # Consolidated hits map: id -> {distance, rrf_score, document, metadata}
    hits_combined = {}
    all_q_ids = res.get("ids") or []
    all_q_docs = res.get("documents") or []
    all_q_metas = res.get("metadatas") or []
    all_q_dists = res.get("distances") or []

    for q_idx in range(len(all_q_ids)):
        q_ids = all_q_ids[q_idx]
        q_docs = all_q_docs[q_idx] if q_idx < len(all_q_docs) else []
        q_metas = all_q_metas[q_idx] if q_idx < len(all_q_metas) else []
        q_dists = all_q_dists[q_idx] if q_idx < len(all_q_dists) else []

        for h_idx in range(len(q_ids)):
            hid = q_ids[h_idx]
            hdoc = q_docs[h_idx] if h_idx < len(q_docs) else ""
            hmd = q_metas[h_idx] if h_idx < len(q_metas) else {}
            hdist = q_dists[h_idx] if h_idx < len(q_dists) else 1.0
            
            # Reciprocal Rank Fusion contribution
            # rank is h_idx + 1
            rrf_val = 1.0 / (RRF_K + (h_idx + 1))

            if hid not in hits_combined:
                hits_combined[hid] = {
                    "distance": hdist,
                    "rrf_score": rrf_val,
                    "document": hdoc,
                    "metadata": hmd,
                }
            else:
                # Keep the smallest distance (highest similarity)
                if hdist < hits_combined[hid]["distance"]:
                    hits_combined[hid]["distance"] = hdist
                # Accumulate RRF score
                hits_combined[hid]["rrf_score"] += rrf_val

    # Lexical BM25 results are fused by rank, never by incomparable raw scores.
    # Restrictive arbitrary Chroma filters remain semantic-only; item-key and note
    # restrictions are supported directly by the lexical index.
    if hybrid and where is None:
        try:
            from lexical_index import search_chunks as lexical_search

            lexical_rankings = [
                lexical_search(
                    q,
                    k=internal_k,
                    include_notes=include_notes,
                    item_keys=include_item_keys,
                )
                for q in queries
            ]
            lexical_ids = list(dict.fromkeys(
                row["chunk_id"] for ranking in lexical_rankings for row in ranking
            ))
            lexical_docs: dict[str, str] = {}
            lexical_metas: dict[str, dict[str, Any]] = {}
            if lexical_ids:
                hydrated = col.get(ids=lexical_ids, include=["documents", "metadatas"])
                for idx, chunk_id in enumerate(hydrated.get("ids") or []):
                    documents = hydrated.get("documents") or []
                    metadatas = hydrated.get("metadatas") or []
                    lexical_docs[chunk_id] = documents[idx] if idx < len(documents) else ""
                    metadata = metadatas[idx] if idx < len(metadatas) else {}
                    lexical_metas[chunk_id] = metadata if isinstance(metadata, dict) else {}
            for ranking in lexical_rankings:
                for rank, row in enumerate(ranking, start=1):
                    chunk_id = row["chunk_id"]
                    if chunk_id not in lexical_docs:
                        continue
                    contribution = 1.0 / (RRF_K + rank)
                    if chunk_id in hits_combined:
                        hits_combined[chunk_id]["rrf_score"] += contribution
                    else:
                        hits_combined[chunk_id] = {
                            "distance": None,
                            "rrf_score": contribution,
                            "document": lexical_docs[chunk_id],
                            "metadata": lexical_metas[chunk_id],
                        }
        except Exception as exc:
            _log.warning("Lexical search unavailable; using semantic results: %s", exc)

    # Sort all consolidated hits by RRF score descending (highest first).
    sorted_hits = sorted(hits_combined.items(), key=lambda x: x[1]["rrf_score"], reverse=True)

    # Filtering out excluded IDs
    if exclude_chunk_ids:
        exclude_set = set(exclude_chunk_ids)
        sorted_hits = [h for h in sorted_hits if h[0] not in exclude_set]

    min_return_chars = int(os.environ.get("MIN_RETURN_CHARS", "200"))
    sorted_hits = [
        hit for hit in sorted_hits
        if len(str(hit[1].get("document") or "").strip()) >= min_return_chars
    ]
    if language_balance:
        sorted_hits = language_balanced_order(sorted_hits, k)

    ids0 = [x[0] for x in sorted_hits]
    docs0 = [x[1]["document"] for x in sorted_hits]
    metas0 = [x[1]["metadata"] for x in sorted_hits]
    dists0 = [x[1]["distance"] for x in sorted_hits]
    rrfs0 = [x[1]["rrf_score"] for x in sorted_hits]



    out: List[RagHit] = []
    for i in range(len(ids0)):
        md = metas0[i] if i < len(metas0) and isinstance(metas0[i], dict) else {}
        dist = dists0[i] if i < len(dists0) else None
        text = docs0[i] if i < len(docs0) else ""

        citation = _make_citation(md)

        ctx: List[RagContextChunk] = []
        if context_window and context_window > 0:
            # Defensive: never call `col.get()` with an empty IDs list (Chroma raises),
            # and tolerate missing neighbors / partial failures.
            nids = [nid for nid in neighbor_ids(ids0[i], context_window) if isinstance(nid, str) and nid]
            if nids:
                try:
                    got = col.get(ids=nids, include=["documents", "metadatas"])
                except Exception:
                    got = {"ids": [], "documents": [], "metadatas": []}

                got_ids = got.get("ids", [])
                got_docs = got.get("documents", [])
                got_metas = got.get("metadatas", [])

                tmp = []
                for j in range(len(got_ids)):
                    gid = got_ids[j]
                    gdoc = got_docs[j] if j < len(got_docs) else ""
                    gmd = got_metas[j] if j < len(got_metas) and isinstance(got_metas[j], dict) else {}
                    parsed = parse_id(gid)
                    gpara = parsed[3] if parsed else None
                    tmp.append(
                        (
                            gpara,
                            {
                                "id": gid,
                                "page": gmd.get("page"),
                                "citation": _make_citation(gmd),
                                "text": gdoc,
                            },
                        )
                    )
                tmp.sort(key=lambda x: (x[0] is None, x[0]))
                ctx = [x[1] for x in tmp if isinstance(x[1].get("text", ""), str) and x[1]["text"].strip()]

        out.append(
            {
                "id": ids0[i],
                "distance": dists0[i],
                "rrf_score": rrfs0[i],
                "citation": citation,
                "text": text,
                "context": ctx,
                "meta": {
                    "title": md.get("title"),
                    "year": md.get("year"),
                    "creators": md.get("creators"),
                    "page": md.get("page"),
                    "page_label": md.get("page_label"),
                    "pdf_path": md.get("pdf_path"),
                    "path": md.get("path"),
                    "itemKey": md.get("itemKey"),
                    "attachmentKey": md.get("attachmentKey"),
                    "noteKey": md.get("noteKey"),
                    "source_type": md.get("source_type"),
                    "locator": md.get("locator"),
                    "contentType": md.get("contentType"),
                    "filename": md.get("filename"),
                    "chapter": md.get("chapter"),
                    "section": md.get("section"),
                    "lang": md.get("lang"),
                },
            }
        )

        if len(out) >= k:
            break

    return {"results": out}


@mcp.tool()
def get_item_summary(item_key: str) -> Dict[str, Any]:
    """Return a local search-index summary; verify research claims against source chunks."""
    summary = load_item_summary((item_key or "").strip())
    return {"item_key": item_key, "summary": summary}


@mcp.tool()
def hierarchical_search(
    query: str | List[str],
    k: int = 8,
    k_items: int = 12,
    where: Any = None,
    include_direct: bool = True,
    return_summaries: bool = True,
    auto_expand: bool = True,
) -> Dict[str, Any]:
    """Search item/section summaries, then retrieve evidence paragraphs.

    Use this for overview questions, literature discovery, or comparisons across
    documents. For an exact quotation or an out-of-theme ethnographic case, use
    ``rag_search`` directly. Global paragraph retrieval remains enabled by default
    as a recall safeguard when a summary misses the relevant detail.
    """
    if k <= 0 or k_items <= 0:
        return {"results": [], "candidate_items": []}
    queries = [query] if isinstance(query, str) else list(query)
    queries = [value.strip() for value in queries if isinstance(value, str) and value.strip()]
    if not queries:
        return {"results": [], "candidate_items": [], "warning": "query is empty"}
    if auto_expand:
        queries = expand_queries(queries, mode="default", timeout=5.0, logger=_log)

    paragraph_collection = _col()
    embeddings = paragraph_collection._embedding_function(queries)
    candidate_scores: dict[str, float] = {}
    candidate_meta: dict[str, dict[str, Any]] = {}
    warnings: list[str] = []
    client = getattr(paragraph_collection, "_chroma_client", None)
    base_name = _EMB_COLLECTION_NAME or COLLECTION_NAME_DEFAULT
    for suffix, result_count in (("sum_item", k_items), ("sum_section", max(k_items * 2, 20))):
        try:
            summary_collection = client.get_collection(f"{base_name}__{suffix}")
            response = summary_collection.query(
                query_embeddings=embeddings, n_results=result_count,
                include=["metadatas", "documents", "distances"],
            )
        except Exception as exc:
            warnings.append(f"{suffix} collection unavailable: {exc}")
            continue
        for ranking in response.get("metadatas") or []:
            for rank, metadata in enumerate(ranking, start=1):
                if not isinstance(metadata, dict) or not metadata.get("itemKey"):
                    continue
                item_key = str(metadata["itemKey"])
                candidate_scores[item_key] = candidate_scores.get(item_key, 0.0) + 1.0 / (RRF_K + rank)
                candidate_meta[item_key] = metadata

    candidate_keys = sorted(candidate_scores, key=lambda key: (-candidate_scores[key], key))[:k_items]
    candidate_response = rag_search(
        queries, k=max(k * 3, k), where=where, include_item_keys=candidate_keys or None,
        auto_expand=False, hybrid=True,
    ) if candidate_keys else {"results": []}
    direct_response = rag_search(
        queries, k=max(k * 3, k), where=where, auto_expand=False, hybrid=True,
    ) if include_direct else {"results": []}

    fused: dict[str, dict[str, Any]] = {}
    scores: dict[str, float] = {}
    for weight, response in ((1.5, candidate_response), (1.0, direct_response)):
        for rank, hit in enumerate(response.get("results") or [], start=1):
            chunk_id = hit.get("id")
            if not chunk_id:
                continue
            fused[chunk_id] = hit
            scores[chunk_id] = scores.get(chunk_id, 0.0) + weight / (RRF_K + rank)
    ordered_ids = sorted(scores, key=lambda chunk_id: (-scores[chunk_id], chunk_id))[:k]
    results = []
    for chunk_id in ordered_ids:
        hit = dict(fused[chunk_id])
        hit["hierarchical_rrf_score"] = scores[chunk_id]
        if return_summaries:
            item_key = (hit.get("meta") or {}).get("itemKey")
            summary = load_item_summary(item_key) if item_key else None
            hit["item_summary_snippet"] = (summary.get("summary") or "")[:120] if summary else ""
        results.append(hit)

    candidate_items = []
    for item_key in candidate_keys:
        metadata = candidate_meta.get(item_key, {})
        summary = load_item_summary(item_key) if return_summaries else None
        candidate_items.append({
            "item_key": item_key, "title": metadata.get("title"), "year": metadata.get("year"),
            "score": candidate_scores[item_key],
            "summary_snippet": (summary.get("summary") or "")[:120] if summary else "",
        })
    response: Dict[str, Any] = {"results": results, "candidate_items": candidate_items}
    if warnings:
        response["warnings"] = warnings
    return response


@mcp.tool()
def search_cases(
    query: str | List[str], region: Optional[str] = None, k: int = 10,
    auto_expand: bool = True,
) -> Dict[str, Any]:
    """Find empirical/ethnographic cases using annotations plus direct paragraphs.

    The case annotation index may cover only summarized items, so results are always
    fused with global ``rag_search(search_mode='case')`` as a coverage safeguard.
    ``region`` is a case-insensitive partial-match filter on structured annotations.
    """
    if k <= 0:
        return {"results": []}
    queries = [query] if isinstance(query, str) else list(query)
    queries = [value.strip() for value in queries if isinstance(value, str) and value.strip()]
    if not queries:
        return {"results": [], "warning": "query is empty"}
    if auto_expand:
        queries = expand_queries(queries, mode="case", timeout=5.0, logger=_log)

    paragraph_collection = _col()
    case_rows = {int(row["case_id"]): row for row in get_case_annotations()}
    structured_by_id: dict[int, dict[str, Any]] = {}
    structured_scores: dict[int, float] = {}
    warning = None
    try:
        client = getattr(paragraph_collection, "_chroma_client", None)
        case_collection = client.get_collection(f"{_EMB_COLLECTION_NAME or COLLECTION_NAME_DEFAULT}__cases")
        embeddings = paragraph_collection._embedding_function(queries)
        response = case_collection.query(
            query_embeddings=embeddings, n_results=max(k * 5, k),
            include=["metadatas", "documents", "distances"],
        )
        for ranking in response.get("metadatas") or []:
            for rank, metadata in enumerate(ranking, start=1):
                case_id = int((metadata or {}).get("case_id") or 0)
                row = case_rows.get(case_id)
                if not row:
                    continue
                if region and region.casefold() not in str(row.get("region") or "").casefold():
                    continue
                structured_by_id[case_id] = row
                structured_scores[case_id] = (
                    structured_scores.get(case_id, 0.0) + 1.0 / (RRF_K + rank)
                )
    except Exception as exc:
        warning = f"case annotation index unavailable: {exc}"

    structured = [
        {"kind": "case", **structured_by_id[case_id], "rrf_score": score}
        for case_id, score in structured_scores.items()
    ]
    direct = rag_search(
        queries, k=max(k * 3, k), search_mode="case", auto_expand=False, hybrid=True,
    ).get("results") or []
    combined: list[dict[str, Any]] = structured + [
        {
            "kind": "chunk", "chunk_id": hit.get("id"),
            "item_key": (hit.get("meta") or {}).get("itemKey"),
            "description": hit.get("text"), "citation": hit.get("citation"),
            "region": None, "rrf_score": 1.0 / (RRF_K + rank), "meta": hit.get("meta"),
        }
        for rank, hit in enumerate(direct, start=1)
    ]
    combined.sort(key=lambda row: (-float(row.get("rrf_score") or 0), str(row.get("chunk_id") or row.get("case_id"))))
    result: Dict[str, Any] = {"results": combined[:k]}
    if warning:
        result["warning"] = warning
    return result


@mcp.tool()
def extract_references_for_item(
    item_key: str, dry_run: bool = True, use_llm: bool = True,
) -> Dict[str, Any]:
    """Extract and resolve one item's bibliography; preview by default.

    When ``use_llm`` is true, candidate bibliography text is sent to the configured
    ``LLM_EXTRACT`` provider. ``EXTRACT_EXCLUDE_TAGS`` is enforced fail-closed.
    Set ``dry_run=False`` only after reviewing the preview.
    """
    from reference_agent import extract_references_for_item as run_extraction
    try:
        return run_extraction(item_key, dry_run=dry_run, use_llm=use_llm)
    except Exception as exc:
        _log.exception("Reference extraction failed for %s", item_key)
        return {"item_key": item_key, "status": "error", "error": str(exc), "references": []}


@mcp.tool()
def confirm_reference_match(edge_id: int, work_id: Optional[int] = None) -> Dict[str, Any]:
    """Accept a replacement canonical work, or reject an edge by omitting work_id."""
    from db_relations import confirm_work_edge
    changed = confirm_work_edge(edge_id, work_id)
    return {
        "status": "updated" if changed else "not_found",
        "edge_id": edge_id, "work_id": work_id,
        "action": "reassigned" if work_id is not None else "rejected",
    }


@mcp.tool()
def promote_chapters(item_key: str, dry_run: bool = True) -> Dict[str, Any]:
    """Promote clearly multi-author sections to child works; preview by default."""
    from work_identity import promote_chapters as run_promotion
    return run_promotion(item_key, dry_run=dry_run)


@mcp.tool()
def detect_translation(item_key: str, dry_run: bool = True) -> Dict[str, Any]:
    """Detect an original work from explicit Zotero metadata or an NDL record.

    This intentionally does not create links from an unverified LLM-only guess.
    """
    from work_identity import detect_translation as run_detection
    try:
        return run_detection(item_key, dry_run=dry_run)
    except Exception as exc:
        return {"item_key": item_key, "status": "error", "error": str(exc)}


@mcp.tool()
async def get_item_details(item_key: str) -> Dict[str, Any]:
    """
    Fetch full bibliographic metadata for a specific Zotero item.

    Args:
        item_key: The Zotero item key (e.g., 'ABCDEFGH'). Found in search results as itemKey.
    """
    api = _z_api()
    return await api.get_item(item_key)


@mcp.tool()
async def list_recent_items(limit: int = 20) -> List[Dict[str, Any]]:
    """
    List the most recently modified items in the Zotero library.

    Args:
        limit: Number of items to return. Default is 20.
    """
    api = _z_api()
    # Fetch recent items. We use the internal _get_json to avoid adding too much boilerplate.
    raw = await api._get_json(
        "items", params={"limit": limit, "direction": "desc", "sort": "dateModified"}
    )

    out = []
    if isinstance(raw, list):
        for item in raw:
            try:
                # Use the existing unwrap logic from the API class
                _, data = api._unwrap_item(item)
                # Skip attachments to focus on top-level library items
                if data.get("itemType") == "attachment":
                    continue
                out.append(
                    {
                        "key": data.get("key"),
                        "itemType": data.get("itemType"),
                        "title": data.get("title"),
                        "creators": data.get("creators"),
                        "date": data.get("date"),
                        "dateModified": data.get("dateModified"),
                    }
                )
            except Exception:
                continue
    return out


@mcp.tool()
async def search_zotero_items(
    query: str, limit: int = 20, qmode: str = "titleCreatorYear"
) -> List[Dict[str, Any]]:
    """
    Search the Zotero library directly for items using Zotero's quick search API.
    This queries the Zotero API directly (without relying on Chroma vector indices)
    and is perfect for finding items by exact or partial title, author, or year.

    Args:
        query: The search term or phrase (e.g. title, author name, or year).
        limit: Maximum number of results to return. Default is 20.
        qmode: The quick search mode. Must be 'titleCreatorYear' (searches only title, creator/author, and year fields - default)
               or 'everything' (searches all fields including indexed PDF attachments).
    """
    api = _z_api()
    params = {
        "q": query,
        "qmode": qmode,
        "limit": limit,
    }
    raw = await api._get_json("items", params=params)

    out = []
    if isinstance(raw, list):
        for item in raw:
            try:
                # Use the existing unwrap logic from the API class
                _, data = api._unwrap_item(item)
                # Skip attachments to focus on top-level library items
                if data.get("itemType") == "attachment":
                    continue
                out.append(
                    {
                        "key": data.get("key"),
                        "itemType": data.get("itemType"),
                        "title": data.get("title"),
                        "creators": data.get("creators"),
                        "date": data.get("date"),
                        "dateModified": data.get("dateModified"),
                    }
                )
            except Exception:
                continue
    return out



@mcp.tool()
def search_items(
    query: str | List[str],
    k: int = 10,
    where: Any = None,
    include_notes: bool = False,
    include_item_keys: Any = None,
) -> Dict[str, Any]:
    """
    Search for relevant Zotero documents (items) without returning full paragraph text.
    Returns a list of unique items with their bibliographic metadata and relevance scores.
    Items that match in multiple places or for multiple keywords will have higher RRF scores.

    Args:
        query:
            Natural-language query string OR a list of strings.
        k:
            Number of unique materials to return. Default: 10.
        where:
            Optional metadata filter (Chroma `where` filter).
        include_notes:
            If True, include Zotero Notes chunks in the search space. Default: False.
        include_item_keys:
            Optional list of Zotero item keys to restrict the search to.
    """

    blocked, msg = _check_indexing_lock()
    if blocked:
        return {"items": [], "warning": msg}

    where = _coerce_json(where)
    include_item_keys = _coerce_json(include_item_keys)

    col = _col()
    if k <= 0:
        return {"items": []}

    # Internally fetch more chunks to ensure we can find 'k' unique library items.
    k_internal = max(k * 10, 100)



    effective_where = _build_effective_where(
        where, include_notes=include_notes, include_item_keys=include_item_keys
    )

    queries = [query] if isinstance(query, str) else query
    for _attempt in range(2):
        try:
            res = col.query(
                query_texts=queries,
                n_results=k_internal,
                where=effective_where,
                include=["metadatas", "distances"],
            )
            break
        except Exception as _exc:
            _log.warning("search_items query failed (attempt %d): %s\n%s", _attempt + 1, _exc, traceback.format_exc())
            if _attempt == 0:
                _reset_col()
                time.sleep(1)
                col = _col()
            else:
                _log.error("search_items: both attempts failed — returning error message")
                return {"items": [], "warning": _HNSW_ERROR_MSG, "error": str(_exc)}

    # itemKey -> {distance, rrf_score, title, year, creators, itemKey, source_type}
    items_map = {}

    all_q_ids = res.get("ids") or []
    all_q_metas = res.get("metadatas") or []
    all_q_dists = res.get("distances") or []

    for q_idx in range(len(all_q_ids)):
        q_ids = all_q_ids[q_idx]
        q_metas = all_q_metas[q_idx] if q_idx < len(all_q_metas) else []
        q_dists = all_q_dists[q_idx] if q_idx < len(all_q_dists) else []

        for h_idx in range(len(q_ids)):
            md = q_metas[h_idx] if h_idx < len(q_metas) else {}
            dist = q_dists[h_idx] if h_idx < len(q_dists) else 1.0
            ikey = md.get("itemKey")

            if not ikey:
                continue

            # RRF contribution based on rank in THIS query's result list
            rrf_contrib = 1.0 / (RRF_K + (h_idx + 1))

            if ikey not in items_map:
                items_map[ikey] = {
                    "distance": dist,
                    "rrf_score": rrf_contrib,
                    "title": md.get("title"),
                    "year": md.get("year"),
                    "creators": md.get("creators"),
                    "itemKey": ikey,
                    "source_type": md.get("source_type"),
                }
            else:
                # Minimum distance (best hit)
                if dist < items_map[ikey]["distance"]:
                    items_map[ikey]["distance"] = dist
                # Accumulate RRF scores (density boost)
                items_map[ikey]["rrf_score"] += rrf_contrib

    # Sort items by accumulated RRF score descending
    sorted_items = sorted(
        items_map.values(), key=lambda x: x["rrf_score"], reverse=True
    )

    return {"items": sorted_items[:k]}


@mcp.tool()
def get_chunk_context(chunk_id: str, window: int = 2) -> Dict[str, Any]:
    """
    Fetch the surrounding paragraphs for a specific chunk ID to understand its context.
    This avoids re-running a semantic search when you already have a relevant chunk ID.

    Args:
        chunk_id: The ID of the chunk (e.g., 'ABCDEFGH:p12:para3:part0') found in search results.
        window: The number of paragraphs to fetch before and after the chunk. Default: 2 (fetches up to 5 paragraphs total).
    
    Returns:
        A dictionary containing the combined text and metadata of the context region.
    """

    blocked, msg = _check_indexing_lock()
    if blocked:
        return {"error": msg}

    col = _col()
    nids = neighbor_ids(chunk_id, window)
    if not nids:
        return {"error": "Invalid chunk_id format or window <= 0"}
    
    res = col.get(ids=nids, include=["documents", "metadatas"])
    found_ids = res.get("ids") or []
    
    if not found_ids:
        return {"error": "Chunk ID not found in database"}

    # Sort results by the chunk index so they read chronologically
    def _para_idx(cid: str) -> int:
        parsed = parse_id(cid)
        return parsed[3] if parsed else 0

    combined = []
    docs = res.get("documents") or []
    metas = res.get("metadatas") or []
    
    for _, doc, meta in sorted(zip(found_ids, docs, metas), key=lambda x: _para_idx(x[0])):
        combined.append(doc)

    # Use the metadata of the requested chunk (or the first available if not found)
    base_meta = {}
    if chunk_id in found_ids:
        idx = found_ids.index(chunk_id)
        base_meta = metas[idx]
    elif metas:
        base_meta = metas[0]

    return {
        "context_text": "\n\n".join(combined),
        "metadata": base_meta,
        "chunk_ids_included": sorted(found_ids, key=lambda x: _para_idx(x))
    }

@mcp.tool()
def get_document_outline(attachment_key: str) -> Dict[str, Any]:
    """
    Retrieve the table of contents (chapter structure) of a PDF or EPUB document.
    
    Args:
        attachment_key: The Zotero attachment key (e.g., 'ABCDEFGH').
    """
    manifest = load_manifest(Path(MANIFEST_PATH))
    
    file_info = manifest.get("files", {}).get(attachment_key)
    if not file_info:
        return {"error": f"Attachment key '{attachment_key}' not found in manifest."}
    
    path = file_info.get("pdf_path")
    if not path or not os.path.exists(path):
        return {"error": f"File not found at path: {path}"}
    
    path_lower = path.lower()
    if path_lower.endswith(".pdf"):
        toc = get_pdf_toc(path)
        return {
            "source_type": "pdf",
            "outline": [{"level": l, "title": t, "page": p} for l, t, p in toc]
        }
    elif path_lower.endswith(".epub"):
        chap_map = get_epub_chapter_index_to_title(path)
        # Convert dict to sorted list of objects
        outline = []
        for idx in sorted(chap_map.keys()):
            outline.append({"index": idx, "title": chap_map[idx]})
        return {
            "source_type": "epub",
            "outline": outline
        }
    else:
        return {"error": f"Unsupported file type for outline extraction: {path}"}


@mcp.tool()
def get_debug_logs(lines: int = 100) -> Dict[str, Any]:
    """
    Return the last N lines of the zotero-rag server log file.
    Use this to diagnose issues such as ChromaDB errors, collection reload events,
    or query failures with full tracebacks.

    Args:
        lines: Number of log lines to return (most recent). Default: 100.

    Returns:
        A dict with keys: log_path, total_lines, returned_lines, log (the text).
    """
    try:
        with open(_LOG_PATH, "r", encoding="utf-8") as f:
            all_lines = f.readlines()
        tail = all_lines[-lines:] if len(all_lines) > lines else all_lines
        return {
            "log_path": _LOG_PATH,
            "total_lines": len(all_lines),
            "returned_lines": len(tail),
            "log": "".join(tail),
        }
    except FileNotFoundError:
        return {
            "log_path": _LOG_PATH,
            "total_lines": 0,
            "returned_lines": 0,
            "log": "",
            "error": "Log file not found — server may not have written any logs yet.",
        }
    except Exception as e:
        return {"log_path": _LOG_PATH, "total_lines": 0, "returned_lines": 0, "log": "", "error": str(e)}


@mcp.resource("docs://zotero_rag_guide")
def get_zotero_rag_guide_resource() -> str:
    """
    Zotero Local RAG MCP Reference Guide for AI Assistants.
    Provides the best practices and tool usage instructions for querying the Zotero library as a resource.
    """
    guide_path = os.path.join(ROOT, "docs", "claude-guide.md")
    try:
        with open(guide_path, "r", encoding="utf-8") as f:
            return f.read()
    except Exception as e:
        return f"Guide file not found or could not be read: {e}"

@mcp.prompt()
def zotero_rag_guide() -> str:
    """
    Zotero Local RAG MCP Reference Guide for AI Assistants.
    Provides the best practices and tool usage instructions for querying the Zotero library.
    """
    guide_path = os.path.join(ROOT, "docs", "claude-guide.md")
    try:
        with open(guide_path, "r", encoding="utf-8") as f:
            return f.read()
    except Exception as e:
        return f"Guide file not found or could not be read: {e}"

@mcp.tool()
def get_chunk_citations(chunk_id: str) -> Dict[str, Any]:
    """
    Get the global Semantic Scholar citations for a specific Zotero paragraph chunk.
    This shows you WHICH external papers cited this specific paragraph and in WHAT context.
    
    Args:
        chunk_id: The ID of the chunk to query (e.g., 'BGZ9UFUJ:p12:para3:part0')
    """
    try:
        from db_relations import get_citations_for_chunk
        citations = get_citations_for_chunk(chunk_id)
        return {
            "chunk_id": chunk_id,
            "citation_count": len(citations),
            "citations": citations
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}

@mcp.tool()
def get_cited_chunks_for_item(item_key: str, max_citations_per_chunk: int = 3) -> Dict[str, Any]:
    """
    Given a Zotero item key, returns all paragraphs (chunks) from that item
    that have been cited by external papers, along with the citation context.

    Args:
        item_key: The parent Zotero item key (e.g., 'BGZ9UFUJ')
        max_citations_per_chunk: Maximum number of representative citations to return per chunk.
    """

    blocked, msg = _check_indexing_lock()
    if blocked:
        return {"status": "blocked", "message": msg}

    try:
        from db_relations import get_cited_chunks_for_item as get_chunks, get_citations_for_chunk
        chunks = get_chunks(item_key)
        
        # Hydrate with actual chunk text via direct SQLite (avoids ChromaDB lock/hang)
        if chunks:
            chunk_ids = [c["cited_chunk_id"] for c in chunks]
            doc_map, meta_map = _hydrate_chunks_from_sqlite(chunk_ids)

            for c in chunks:
                cid = c["cited_chunk_id"]
                c["text"] = doc_map.get(cid, "")
                meta = meta_map.get(cid, {})
                c["page"] = meta.get("page")
                c["page_label"] = meta.get("page_label")
                
                # Fetch top citation snippets for context
                c["top_citations"] = get_citations_for_chunk(cid, limit=max_citations_per_chunk)
                
        return {
            "item_key": item_key,
            "cited_chunks_count": len(chunks),
            "cited_chunks": chunks
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}

# Internal function (formerly exposed as MCP tool)
async def extract_local_epub_references(item_key: str) -> Dict[str, Any]:
    """
    Extracts footnote and endnote references from an EPUB attachment in Zotero,
    resolves them to Semantic Scholar where possible, and saves them to the global_references DB.
    
    Args:
        item_key: The parent Zotero item key (e.g., 'BGZ9UFUJ')
    """
    try:
        # Get children to find attachment
        children = await _z_api()._get_json(f"items/{item_key}/children")
        if not isinstance(children, list):
            children = []
        
        epub_key = None
        epub_data = None
        for child in children:
            _, att_data = ZoteroLocalAPI._unwrap_item(child)
            filename = att_data.get("filename", "")
            if filename.lower().endswith(".epub"):
                epub_key = att_data.get("key")
                epub_data = att_data
                break
                
        if not epub_key:
            return {"status": "error", "message": "No EPUB attachment found for this item."}
            
        zotero_data_dir = os.environ.get("ZOTERO_DATA_DIR", os.path.expanduser("~/Zotero"))
        epub_path = _z_api().resolve_pdf_path_from_attachment(epub_key, epub_data, zotero_data_dir)
        
        if not epub_path or not os.path.exists(epub_path):
            return {"status": "error", "message": f"EPUB file not found on disk for attachment {epub_key}."}
            
        from citation_mapper import map_item_local_references
        return map_item_local_references(item_key, epub_path)
    except Exception as e:
        return {"status": "error", "message": str(e)}

@mcp.tool()
def get_chunk_references(chunk_id: str) -> Dict[str, Any]:
    """
    Get the external references (outgoing citations) for a specific Zotero paragraph chunk.
    This shows you WHICH external papers this specific paragraph is citing.
    
    Args:
        chunk_id: The ID of the chunk to query (e.g., 'BGZ9UFUJ:p12:para3:part0')
    """
    try:
        from db_relations import get_references_for_chunk
        references = get_references_for_chunk(chunk_id)
        return {
            "chunk_id": chunk_id,
            "reference_count": len(references),
            "references": references
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}

@mcp.tool()
def get_references_for_item(item_key: str) -> Dict[str, Any]:
    """
    List all paragraph chunks within a specific Zotero item that cite external papers.
    Use this to see which parts of a document rely most heavily on external literature.

    Args:
        item_key: The parent Zotero item key (e.g., 'BGZ9UFUJ')
    """

    blocked, msg = _check_indexing_lock()
    if blocked:
        return {"status": "blocked", "message": msg}

    try:
        from db_relations import get_references_for_item as get_ref_chunks, get_references_for_chunk
        chunks = get_ref_chunks(item_key)

        # Hydrate with chunk text via direct SQLite query (avoids ChromaDB lock/hang)
        if chunks:
            chunk_ids = [c["citing_chunk_id"] for c in chunks]
            doc_map, meta_map = _hydrate_chunks_from_sqlite(chunk_ids)

            for c in chunks:
                cid = c["citing_chunk_id"]
                c["text"] = doc_map.get(cid, "")
                meta = meta_map.get(cid, {})
                c["page"] = meta.get("page")
                c["page_label"] = meta.get("page_label")
                c["top_references"] = get_references_for_chunk(cid)[:3]

        return {
            "item_key": item_key,
            "citing_chunks_count": len(chunks),
            "citing_chunks": chunks
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}


@mcp.tool()
def suggest_unowned_works(
    scope_item_keys: Any = None,
    direction: str = "references",
    k: int = 20,
    min_citing_items: int = 2,
) -> Dict[str, Any]:
    """Rank external works adjacent to the library but not already owned.

    Args:
        scope_item_keys: Optional Zotero item-key list. If omitted, aggregate the whole library.
        direction: "references" finds works cited by owned items; "citations" finds external
            papers that cite owned items.
        k: Maximum number of suggestions (default 20, maximum 100).
        min_citing_items: Require adjacency to at least this many distinct owned items.
    """
    scope = _coerce_json(scope_item_keys)
    if scope is not None and not isinstance(scope, list):
        return {"status": "error", "message": "scope_item_keys must be a list or null."}
    if direction not in {"references", "citations"}:
        return {"status": "error", "message": "direction must be 'references' or 'citations'."}
    if k <= 0 or min_citing_items <= 0:
        return {"status": "error", "message": "k and min_citing_items must be positive."}

    try:
        from db_relations import aggregate_unowned_works, normalize_work_title

        manifest = load_manifest(Path(MANIFEST_PATH))
        owned_titles = {
            normalize_work_title(info.get("title"))
            for info in manifest.get("files", {}).values()
            if isinstance(info, dict) and info.get("title")
        }
        suggestions = aggregate_unowned_works(
            scope,
            direction=direction,
            min_citing_items=min_citing_items,
            limit=min(k, 100),
            normalized_owned_titles=owned_titles,
        )
        count_key = "cited_by_n_items" if direction == "references" else "cites_n_items"
        item_keys_key = "citing_item_keys" if direction == "references" else "cited_item_keys"
        for suggestion in suggestions:
            suggestion[count_key] = suggestion.pop("adjacent_item_count")
            suggestion[item_keys_key] = suggestion.pop("adjacent_item_keys")
        return {
            "direction": direction,
            "scope_item_keys": scope,
            "suggestion_count": len(suggestions),
            "suggestions": suggestions,
            "identity_note": "Provisional S2/DOI/title grouping; canonical works IDs are planned.",
        }
    except Exception as exc:
        return {"status": "error", "message": str(exc)}


@mcp.tool()
def related_items(item_key: str, method: str = "hybrid", k: int = 10) -> Dict[str, Any]:
    """Find owned Zotero items related by citations or semantic content.

    Args:
        item_key: Source Zotero item key.
        method: "coupling", "cocitation", "semantic", or equal-weight "hybrid".
        k: Maximum number of related items (default 10, maximum 100).
    """
    if not item_key.strip():
        return {"status": "error", "message": "item_key is required."}
    if k <= 0:
        return {"status": "error", "message": "k must be positive."}
    try:
        from recommendations import related_items as find_related_items

        results = find_related_items(item_key, method=method, k=min(k, 100))
        return {
            "item_key": item_key,
            "method": method,
            "related_count": len(results),
            "related_items": results,
        }
    except Exception as exc:
        return {"status": "error", "message": str(exc)}

@mcp.tool()
async def build_citation_network(item_key: str) -> Dict[str, Any]:
    """
    Builds the complete citation network for a Zotero item.
    This runs BOTH:
    1. Retrieval of incoming citations (Citations) from Semantic Scholar.
    2. Extraction of outgoing references (References) from local EPUBs.

    Call this single tool before querying for cited/citing chunks to ensure
    the database is fully populated with both incoming and outgoing citation data.

    Args:
        item_key: The parent Zotero item key (e.g., 'BGZ9UFUJ')
    """

    if not os.environ.get("S2_API_KEY", "").strip():
        return {
            "status": "error",
            "message": "S2_API_KEY is required. Run Setup.command and configure Citation Network.",
        }

    blocked, msg = _check_indexing_lock()
    if blocked:
        return {"status": "blocked", "message": msg}

    res: Dict[str, Any] = {}

    # Fetch Zotero item metadata so the mapper can look up the paper on S2.
    title, year, creators, doi, isbn = "", "", "", "", ""
    try:
        raw = await _z_api().get_item(item_key)
        _, item_data = ZoteroLocalAPI._unwrap_item(raw)
        title = item_data.get("title") or ""
        date = item_data.get("date") or ""
        year = date[:4] if len(date) >= 4 and date[:4].isdigit() else ""
        doi = item_data.get("DOI") or ""
        isbn = item_data.get("ISBN") or ""
        creators_list = item_data.get("creators") or []
        creators = ", ".join(
            (
                (c.get("lastName", "") + " " + c.get("firstName", "")).strip().strip(", ")
                if "lastName" in c
                else c.get("name", "")
            )
            for c in creators_list
            if isinstance(c, dict)
        )
        res["item_metadata"] = {"title": title, "year": year, "doi": doi, "isbn": isbn}
    except Exception as e:
        res["metadata_fetch_warning"] = str(e)

    # 1. Global citation mapping (Citations from Semantic Scholar)
    try:
        from citation_mapper import map_item_global_citations as _mapper
        res["citations_import"] = _mapper(
            item_key, title=title, year=year, creators=creators, doi=doi, isbn=isbn,
            max_citations=5000,
        )
    except Exception as e:
        res["citations_import"] = {"status": "error", "message": str(e)}

    # 2. Local EPUB reference extraction (outgoing References)
    try:
        res["references_import"] = await extract_local_epub_references(item_key)
    except Exception as e:
        res["references_import"] = {"status": "error", "message": str(e)}

    # Both steps complete → mark as fully mapped
    try:
        from db_relations import update_item_citation_status
        update_item_citation_status(item_key, "mapped")
    except Exception as e:
        _log.warning("build_citation_network: failed to set mapped status: %s", e)

    res["status"] = "success"
    res["message"] = "Completed citation network build process."
    return res

def main():
    try:
        cfg = resolve_embedder_settings(Path(ROOT))
        _probe_fn = create_embedding_function(cfg, task_type="RETRIEVAL_QUERY")
        _coll = resolve_collection_name(
            _probe_fn,
            env_value=os.environ.get("CHROMA_COLLECTION"),
            default=COLLECTION_NAME_DEFAULT,
        )
        _startup_msg = f"starting (CHROMA_DIR={CHROMA_DIR}, COLLECTION={_coll}, EMB_MODEL={cfg.model_name}, EMB_DEVICE={cfg.device}, PROVIDER={cfg.provider})"
        print(f"[zotero-rag] {_startup_msg}", file=sys.stderr)
        _log.info(_startup_msg)
        mcp.run()
        print("[zotero-rag] mcp.run() returned (unexpected). Exiting.", file=sys.stderr)
        _log.warning("mcp.run() returned unexpectedly")
    except KeyboardInterrupt:
        raise
    except Exception as e:
        print(f"[zotero-rag] FATAL: {e}", file=sys.stderr)
        _log.critical("FATAL: %s\n%s", e, traceback.format_exc())
        traceback.print_exc(file=sys.stderr)
        raise

if __name__ == "__main__":
    main()
