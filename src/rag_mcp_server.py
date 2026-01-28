# MCP server for paragraph-level Zotero RAG (local Chroma).
from __future__ import annotations

import os
import sys
from typing import Any, Dict, Optional, List
from typing_extensions import TypedDict

import chromadb
from chromadb.utils import embedding_functions
from fastmcp import FastMCP


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


def _make_citation(md: dict) -> str:
    title = md.get("title") or ""
    year = md.get("year")
    page = md.get("page")
    if title and page and year:
        return f"{title} ({year}) p.{page}"
    if title and page:
        return f"{title} p.{page}"
    if title and year:
        return f"{title} ({year})"
    return title or ""


@mcp.tool()
def rag_search(
    query: str,
    k: int = 10,
    where: Optional[Dict[str, Any]] = None,
    context_window: int = 1,
    include_notes: bool = False,
) -> RagSearchResponse:
    """
    Paragraph-level semantic search over local Zotero PDFs/HTML snapshots (+ optionally Notes).
    Args:
        query:
            Natural-language query string.
        k:
            Number of results to return (after filtering short fragments). Default: 10.
        where:
            Optional metadata filter (Chroma `where` filter). Use this to restrict eligible chunks.

            Indexed metadata keys (this project):
                - title: str
                - year: int
                - creators: str (authors joined by '; ')
                - page: int (PDF only)
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
            Neighbor paragraphs to fetch around a hit. Default: 1.
            For PDF: neighbors are within the same page by para index.
            For HTML/Notes: neighbors are within the same doc by para index.
        include_notes:
            If True, include Zotero Notes chunks in the search space. Default: False.
            Notes are indexed but excluded by default.
    Returns:
        {"results": [ ... ]}
    """

    col = _col()
    if k <= 0:
        return {"results": []}

    def _where_requests_notes(w: Optional[Dict[str, Any]]) -> bool:
        """Detect whether a Chroma `where` filter explicitly includes Notes.

        We treat Notes as "requested" only when the filter positively selects
        source_type == "note" (or source_type $in includes "note"), accounting
        for nested `$and`/`$or` (and `$not`) compositions.

        This is intentionally conservative: negative constraints like
        {"source_type": {"$ne": "note"}} or {"$not": {"source_type": "note"}}
        do NOT count as requesting notes.
        """

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
            # Depth guard against pathological input
            if depth > 50:
                return False
            if node is None:
                return False

            if isinstance(node, dict):
                # Handle logical composition operators
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

                # Field-level check
                if "source_type" in node:
                    st = node.get("source_type")
                    if _positive_note_stype(st):
                        return not negated

                # Recurse through any nested dict/list values to catch embeddings of where clauses
                for v in node.values():
                    if isinstance(v, (dict, list)) and _walk(v, negated, depth + 1):
                        return True
                return False

            if isinstance(node, list):
                for sub in node:
                    if _walk(sub, negated, depth + 1):
                        return True
                return False

            return False

        if not w or not isinstance(w, dict):
            return False
        return _walk(w)

    effective_where = where
    if (not include_notes) and (not _where_requests_notes(where)):
        note_excl = {"source_type": {"$ne": "note"}}
        if effective_where is None:
            effective_where = note_excl
        else:
            effective_where = {"$and": [effective_where, note_excl]}

    res = col.query(
        query_texts=[query],
        n_results=max(k * 5, k),
        where=effective_where,
        include=["documents", "metadatas", "distances"],
    )

    ids0 = (res.get("ids") or [[]])[0]
    docs0 = (res.get("documents") or [[]])[0]
    metas0 = (res.get("metadatas") or [[]])[0]
    dists0 = (res.get("distances") or [[]])[0]

    def parse_id(chunk_id: str):
        """Parse chunk ids for pdf/html/note.

        pdf : {attachmentKey}:p{page}:para{para}:part{part}
        html: {attachmentKey}:html:para{para}:part{part}
        epub: {attachmentKey}:epub:para{para}:part{part}
        note: {noteKey}:note:para{para}:part{part}
        """
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
                "distance": dist,
                "citation": citation,
                "text": text,
                "context": ctx,
                "meta": {
                    "title": md.get("title"),
                    "year": md.get("year"),
                    "creators": md.get("creators"),
                    "page": md.get("page"),
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


if __name__ == "__main__":
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