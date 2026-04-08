# MCP server for paragraph-level Zotero RAG (local Chroma).
from __future__ import annotations

import os
import sys
from typing import Any, Dict, Optional, List
from typing_extensions import TypedDict

import chromadb
from chromadb.utils import embedding_functions
from fastmcp import FastMCP
from zotero_source_localapi import ZoteroLocalAPI


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CHROMA_DIR = os.environ.get("CHROMA_DIR", os.path.join(ROOT, "data", "chroma"))

# Collection name is intentionally configurable.
# IMPORTANT: Chroma collections are dimension-fixed. If you switch embedding models
# (e.g., 384-d MiniLM <-> 1024-d bge-m3), use a different collection name or rebuild.
COLLECTION_NAME_ENV = os.environ.get("CHROMA_COLLECTION")
COLLECTION_NAME_DEFAULT = "zotero_paragraphs"

# Embedding model selection
# - If EMB_MODEL is explicitly set, use it.
# - Otherwise, pick a default based on EMB_PROFILE.
#   - fast: multilingual + lighter
#   - bge : bge-m3 (heavier; recommended to use a local path and cache offline)
def _resolve_embedder_settings():
    profile = (os.environ.get("EMB_PROFILE") or "fast").strip().lower()
    offline = (os.environ.get("HF_HUB_OFFLINE") == "1") or (os.environ.get("TRANSFORMERS_OFFLINE") == "1")

    def _pick_device(default: str = "cpu") -> str:
        return (os.environ.get("EMB_DEVICE") or default).strip()

    def _is_local_path(p: str) -> bool:
        try:
            return os.path.exists(os.path.expanduser(p))
        except Exception:
            return False

    # Optional: resolve cached Hugging Face model snapshots (offline-friendly)
    try:
        from huggingface_hub import snapshot_download  # type: ignore
    except Exception:  # pragma: no cover
        snapshot_download = None

    def _try_resolve_hf_cached_snapshot(model_id: str):
        if snapshot_download is None:
            return None
        try:
            p = snapshot_download(repo_id=model_id, local_files_only=True)
            if p and os.path.exists(p):
                return p
        except Exception:
            return None
        return None

    def _offline_resolve_or_exit(model: str) -> str:
        if not offline:
            return model
        if _is_local_path(model):
            return os.path.expanduser(model)
        cached = _try_resolve_hf_cached_snapshot(model)
        if cached:
            return cached
        raise RuntimeError(
            "Offline mode is enabled (HF_HUB_OFFLINE=1 or TRANSFORMERS_OFFLINE=1), "
            f"but the requested embedding model is not available locally: {model}\n\n"
            "Fix options:\n"
            "  (1) Temporarily go online and cache it, then rerun offline:\n"
            "      HF_HUB_OFFLINE=0 TRANSFORMERS_OFFLINE=0 python -c \"from sentence_transformers import SentenceTransformer; SentenceTransformer('" + model + "')\"\n"
            "  (2) Or set EMB_MODEL to a local directory path containing the model files.\n"
        )

    # Explicit override wins.
    if (os.environ.get("EMB_MODEL") or "").strip():
        model = os.environ["EMB_MODEL"].strip()
        model = _offline_resolve_or_exit(model)
        return model, _pick_device("cpu")

    # Profile-based defaults.
    if profile == "bge":
        model = os.path.join(ROOT, "data", "models", "bge-m3")
        device_default = "mps" if sys.platform == "darwin" else "cpu"
        model = _offline_resolve_or_exit(model)
        return model, _pick_device(device_default)

    # fast (default): multilingual MiniLM
    remote_model = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    local_model = os.path.join(ROOT, "data", "models", "paraphrase-multilingual-MiniLM-L12-v2")

    if offline:
        if _is_local_path(local_model):
            return local_model, _pick_device("cpu")
        cached = _try_resolve_hf_cached_snapshot(remote_model)
        if cached:
            return cached, _pick_device("cpu")
        raise RuntimeError(
            "Offline mode is enabled (HF_HUB_OFFLINE=1 or TRANSFORMERS_OFFLINE=1) but the default fast model "
            "is not available locally.\n"
            f"Expected local model dir (project-local): {local_model}\n"
            "Also checked Hugging Face cache for: sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2\n\n"
            "Fix options:\n"
            "  A) Temporarily download/cache it (online):\n"
            "     HF_HUB_OFFLINE=0 TRANSFORMERS_OFFLINE=0 python -c \"from sentence_transformers import SentenceTransformer; SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')\"\n"
            "  B) Or download into the project-local directory and keep using offline mode afterwards (set EMB_MODEL to the local dir).\n"
        )

    return remote_model, _pick_device("cpu")


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


def _col():
    global _COL
    if _COL is not None:
        return _COL

    model_name, device = _resolve_embedder_settings()
    try:
        emb_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=model_name,
            device=device,
            normalize_embeddings=True,
        )
    except Exception as e:
        raise RuntimeError(
            "Failed to initialize embedding model.\n"
            f"EMB_MODEL={model_name}\n"
            f"EMB_DEVICE={device}\n"
            "If you are running offline, ensure the model is already cached for this Python environment.\n"
            "Try once online (example): python -c 'from sentence_transformers import SentenceTransformer; SentenceTransformer(\"sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2\")'\n"
            f"Original error: {e}"
        )

    # If CHROMA_COLLECTION is not explicitly set, automatically suffix the default
    # collection name by embedding dimension to prevent dimension-mismatch when switching models.
    collection_name = (COLLECTION_NAME_ENV or "").strip() or COLLECTION_NAME_DEFAULT
    if not (COLLECTION_NAME_ENV or "").strip():
        try:
            probe_vecs = emb_fn(["collection probe"])
            dim = None
            if isinstance(probe_vecs, list) and probe_vecs and isinstance(probe_vecs[0], (list, tuple)):
                dim = len(probe_vecs[0])
            if isinstance(dim, int) and dim > 0:
                collection_name = f"{COLLECTION_NAME_DEFAULT}_{dim}"
        except Exception:
            collection_name = COLLECTION_NAME_DEFAULT

    client = chromadb.PersistentClient(path=CHROMA_DIR)
    _COL = client.get_or_create_collection(
        name=collection_name,
        embedding_function=emb_fn,
        metadata={"hnsw:space": "cosine"},
    )

    # Dimension compatibility check:
    # If the collection already contains vectors, ensure the current embedder's dimension matches.
    # This avoids confusing runtime errors like "expecting embedding with dimension X, got Y".
    try:
        if hasattr(_COL, "count") and _COL.count() > 0 and hasattr(_COL, "peek"):
            peek = _COL.peek(1)
            peek_ids = (peek or {}).get("ids") or []
            if peek_ids:
                first_id = peek_ids[0]
                got = _COL.get(ids=[first_id], include=["embeddings"])
                stored = (got or {}).get("embeddings") or []
                stored_dim = len(stored[0]) if stored and stored[0] is not None else None

                probe = emb_fn(["dimension probe"])
                probe_dim = len(probe[0]) if probe and probe[0] is not None else None

                if (
                    isinstance(stored_dim, int)
                    and isinstance(probe_dim, int)
                    and stored_dim > 0
                    and probe_dim > 0
                    and stored_dim != probe_dim
                ):
                    raise RuntimeError(
                        "Embedding dimension mismatch for existing Chroma collection.\n"
                        f"CHROMA_DIR={CHROMA_DIR}\n"
                        f"COLLECTION={collection_name}\n"
                        f"Stored dimension={stored_dim}, embedder dimension={probe_dim}\n\n"
                        "Fix options:\n"
                        "  (1) Use a different CHROMA_COLLECTION name for this embedding model, OR\n"
                        "  (2) Unset CHROMA_COLLECTION to enable auto-suffix by dimension (recommended), OR\n"
                        "  (3) Rebuild the index for this collection (delete Chroma dir / run index_from_zotero.py --rebuild).\n"
                    )
    except Exception as e:
        # If it's our deliberate mismatch error, re-raise; otherwise ignore (best-effort check).
        if isinstance(e, RuntimeError) and "Embedding dimension mismatch" in str(e):
            raise

    return _COL


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
    a0, stype, page, para, _part = parsed
    out_ids: List[str] = []
    for dp in range(-w, w + 1):
        pidx = para + dp
        if pidx < 0:
            continue
        if stype == "pdf" and page is not None:
            out_ids.append(f"{a0}:p{page}:para{pidx}:part0")
        elif stype == "html":
            out_ids.append(f"{a0}:html:para{pidx}:part0")
        elif stype == "epub":
            out_ids.append(f"{a0}:epub:para{pidx}:part0")
        elif stype == "note":
            out_ids.append(f"{a0}:note:para{pidx}:part0")
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
    # Prefer the book's own page label (e.g. "xii", "15") over the sequential PDF page number.
    page_display = page_label if page_label else (str(page) if page is not None else None)
    if title and page_display and year:
        return f"{title} ({year}) p.{page_display}"
    if title and page_display:
        return f"{title} p.{page_display}"
    if title and year:
        return f"{title} ({year})"
    return title or ""


@mcp.tool()
def rag_search(
    query: str | List[str],
    k: int = 5,
    where: Optional[Dict[str, Any]] = None,
    context_window: int = 0,
    include_notes: bool = False,
    include_item_keys: Optional[List[str]] = None,
    exclude_chunk_ids: Optional[List[str]] = None,
) -> RagSearchResponse:
    """
    Paragraph-level semantic search over local Zotero PDFs/HTML snapshots (+ optionally Notes).
    Args:
        query:
            Natural-language query string OR a list of strings.
            Providing a list (e.g. synonyms, different languages) allows for broader semantic
            matching in a single call. Results are deduplicated by chunk ID (keeping the
            best distance hit), saving tokens by avoiding redundant context.
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
                - locator: str (e.g., "p12:para3" / "html:para10" / "note:para2")
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
    Returns:
        {"results": [ ... ]}
    """

    col = _col()
    if k <= 0:
        return {"results": []}


    effective_where = where
    if (not include_notes) and (not _where_requests_notes(where)):
        note_excl = {"source_type": {"$ne": "note"}}
        if effective_where is None:
            effective_where = note_excl
        else:
            effective_where = {"$and": [effective_where, note_excl]}

    if include_item_keys:
        item_filter = {"itemKey": {"$in": include_item_keys}}
        if effective_where is None:
            effective_where = item_filter
        else:
            effective_where = {"$and": [effective_where, item_filter]}

    queries = [query] if isinstance(query, str) else query
    
    internal_k = max(k * 5, k)
    if exclude_chunk_ids:
        internal_k += len(exclude_chunk_ids)

    res = col.query(
        query_texts=queries,
        n_results=internal_k,
        where=effective_where,
        include=["documents", "metadatas", "distances"],
    )

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

    # Sort all consolidated hits by RRF score descending (highest first)
    sorted_hits = sorted(hits_combined.items(), key=lambda x: x[1]["rrf_score"], reverse=True)

    # Filtering out excluded IDs
    if exclude_chunk_ids:
        exclude_set = set(exclude_chunk_ids)
        sorted_hits = [h for h in sorted_hits if h[0] not in exclude_set]

    ids0 = [x[0] for x in sorted_hits]
    docs0 = [x[1]["document"] for x in sorted_hits]
    metas0 = [x[1]["metadata"] for x in sorted_hits]
    dists0 = [x[1]["distance"] for x in sorted_hits]
    rrfs0 = [x[1]["rrf_score"] for x in sorted_hits]



    MIN_RETURN_CHARS = int(os.environ.get("MIN_RETURN_CHARS", "200"))

    out: List[RagHit] = []
    for i in range(len(ids0)):
        md = metas0[i] if i < len(metas0) and isinstance(metas0[i], dict) else {}
        dist = dists0[i] if i < len(dists0) else None
        text = docs0[i] if i < len(docs0) else ""

        if len(text.strip()) < MIN_RETURN_CHARS:
            continue

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
                },
            }
        )

        if len(out) >= k:
            break

    return {"results": out}


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
def search_items(
    query: str | List[str],
    k: int = 10,
    where: Optional[Dict[str, Any]] = None,
    include_notes: bool = False,
    include_item_keys: Optional[List[str]] = None,
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
    col = _col()
    if k <= 0:
        return {"items": []}

    # Internally fetch more chunks to ensure we can find 'k' unique library items.
    k_internal = max(k * 10, 100)



    effective_where = where
    if (not include_notes) and (not _where_requests_notes(where)):
        note_excl = {"source_type": {"$ne": "note"}}
        if effective_where is None:
            effective_where = note_excl
        else:
            effective_where = {"$and": [effective_where, note_excl]}

    if include_item_keys:
        item_filter = {"itemKey": {"$in": include_item_keys}}
        if effective_where is None:
            effective_where = item_filter
        else:
            effective_where = {"$and": [effective_where, item_filter]}

    queries = [query] if isinstance(query, str) else query
    res = col.query(
        query_texts=queries,
        n_results=k_internal,
        where=effective_where,
        include=["metadatas", "distances"],
    )

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

@mcp.resource("docs://zotero_rag_guide")
def get_zotero_rag_guide_resource() -> str:
    """
    Zotero Local RAG MCP Reference Guide for AI Assistants.
    Provides the best practices and tool usage instructions for querying the Zotero library as a resource.
    """
    guide_path = os.path.join(ROOT, "ZOTERO_RAG_GUIDE.md")
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
    guide_path = os.path.join(ROOT, "ZOTERO_RAG_GUIDE.md")
    try:
        with open(guide_path, "r", encoding="utf-8") as f:
            return f.read()
    except Exception as e:
        return f"Guide file not found or could not be read: {e}"

def main():
    try:
        model_name, device = _resolve_embedder_settings()
        # Match the same auto-suffix logic used in _col()
        _coll = (COLLECTION_NAME_ENV or "").strip() or COLLECTION_NAME_DEFAULT
        if not (COLLECTION_NAME_ENV or "").strip():
            try:
                _probe = embedding_functions.SentenceTransformerEmbeddingFunction(
                    model_name=model_name,
                    device=device,
                    normalize_embeddings=True,
                )(["collection probe"])
                _dim = len(_probe[0]) if _probe and isinstance(_probe[0], (list, tuple)) else None
                if isinstance(_dim, int) and _dim > 0:
                    _coll = f"{COLLECTION_NAME_DEFAULT}_{_dim}"
            except Exception:
                _coll = COLLECTION_NAME_DEFAULT
        print(
            f"[zotero-rag] starting (CHROMA_DIR={CHROMA_DIR}, COLLECTION={_coll}, EMB_MODEL={model_name}, EMB_DEVICE={device})",
            file=sys.stderr,
        )
        mcp.run()
        print("[zotero-rag] mcp.run() returned (unexpected). Exiting.", file=sys.stderr)
    except KeyboardInterrupt:
        raise
    except Exception as e:
        print(f"[zotero-rag] FATAL: {e}", file=sys.stderr)
        import traceback

        traceback.print_exc(file=sys.stderr)
        raise

if __name__ == "__main__":
    main()