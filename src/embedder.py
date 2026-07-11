# src/embedder.py
from __future__ import annotations

import json
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import chromadb
from chromadb.utils import embedding_functions

# Optional: resolve cached Hugging Face model snapshots (offline-friendly)
try:
    from huggingface_hub import snapshot_download  # type: ignore
except Exception:  # pragma: no cover
    snapshot_download = None


@dataclass
class EmbedderConfig:
    """Resolved embedder configuration."""
    provider: str       # "sentence_transformers" | "gemini"
    model_name: str     # model name or path
    device: str         # "cpu" | "mps" | "api"


# ---------------------------------------------------------------------------
# Process-level configuration
# ---------------------------------------------------------------------------

def configure_embedder_process() -> None:
    """Configure process-level environment variables for embedding.

    Call this once at process start to avoid thread contention
    from tokenizers / BLAS libraries. Uses ``setdefault`` so that
    explicit overrides set before this call are preserved.
    """
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("RAYON_RS_NUM_CPUS", "1")


# ---------------------------------------------------------------------------
# Configuration resolution
# ---------------------------------------------------------------------------

def resolve_embedder_settings(project_root: Path) -> EmbedderConfig:
    """
    Returns resolved embedder configuration.

    Profiles:
      - fast  (default): multilingual MiniLM (local, 384-dim)
      - bge  : BGE-M3 (local, 1024-dim)
      - gemini: Gemini text-embedding (API, 768-dim)
    """
    profile = (os.environ.get("EMB_PROFILE") or "fast").strip().lower()
    offline = (os.environ.get("HF_HUB_OFFLINE") == "1") or (os.environ.get("TRANSFORMERS_OFFLINE") == "1")

    def _pick_device(default: str = "cpu") -> str:
        return (os.environ.get("EMB_DEVICE") or default).strip()

    def _is_local_path(p: str) -> bool:
        try:
            return Path(p).expanduser().exists()
        except Exception:
            return False

    def _try_resolve_hf_cached_snapshot(model_id: str) -> Optional[str]:
        if snapshot_download is None:
            return None
        try:
            p = snapshot_download(repo_id=model_id, local_files_only=True)
            if p and Path(p).exists():
                return p
        except Exception:
            return None
        return None

    def _offline_resolve_or_exit(model: str) -> str:
        if not offline:
            return model
        if _is_local_path(model):
            return model
        cached = _try_resolve_hf_cached_snapshot(model)
        if cached:
            return cached
        raise RuntimeError(
            "ERROR: Offline mode is enabled (HF_HUB_OFFLINE=1 or TRANSFORMERS_OFFLINE=1), "
            f"but the requested embedding model is not available locally: {model}\n\n"
            "Fix options:\n"
            "  (1) Temporarily go online and cache it, then rerun offline:\n"
            "      HF_HUB_OFFLINE=0 TRANSFORMERS_OFFLINE=0 python -c "
            "\"from sentence_transformers import SentenceTransformer; "
            f"SentenceTransformer('{model}')\"\n"
            "  (2) Or set EMB_MODEL to a local directory path containing the model files.\n"
        )

    # --- Gemini profile (API-based) ---
    if profile == "gemini":
        if offline:
            raise RuntimeError(
                "Gemini embedding requires an active internet connection.\n"
                "Offline mode (HF_HUB_OFFLINE=1 or TRANSFORMERS_OFFLINE=1) is "
                "incompatible with EMB_PROFILE=gemini."
            )
        api_key = os.environ.get("GEMINI_API_KEY", "").strip()
        if not api_key:
            raise RuntimeError(
                "Gemini embedding selected (EMB_PROFILE=gemini) but GEMINI_API_KEY is not set.\n\n"
                "Gemini embedding is a paid Google Cloud API service.\n"
                "Pricing: $0.075/1M input tokens (batch) | $0.0001/1K tokens (online)\n"
                "Free tier available: https://ai.google.dev/pricing\n\n"
                "Get an API key: https://aistudio.google.com/apikey\n"
                "Then set it in your .env file:\n"
                "  GEMINI_API_KEY=your_key_here\n\n"
                "Or export it in your environment."
            )
        model = os.environ.get("EMB_MODEL", "gemini-embedding-001").strip()
        return EmbedderConfig(
            provider="gemini",
            model_name=model,
            device="api",
        )

    # --- Explicit EMB_MODEL override (assumes sentence_transformers) ---
    if "EMB_MODEL" in os.environ and (os.environ.get("EMB_MODEL") or "").strip():
        model = os.environ["EMB_MODEL"].strip()
        model = _offline_resolve_or_exit(model)
        return EmbedderConfig(
            provider="sentence_transformers",
            model_name=model,
            device=_pick_device("cpu"),
        )

    # --- bge profile ---
    if profile == "bge":
        model = str(project_root / "data" / "models" / "bge-m3")
        device_default = "mps" if sys.platform == "darwin" else "cpu"
        model = _offline_resolve_or_exit(model)
        return EmbedderConfig(
            provider="sentence_transformers",
            model_name=model,
            device=_pick_device(device_default),
        )

    # --- fast (default): multilingual MiniLM ---
    remote_model = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    local_model = str(project_root / "data" / "models" / "paraphrase-multilingual-MiniLM-L12-v2")

    if offline:
        if _is_local_path(local_model):
            return EmbedderConfig(
                provider="sentence_transformers",
                model_name=local_model,
                device=_pick_device("cpu"),
            )
        cached = _try_resolve_hf_cached_snapshot(remote_model)
        if cached:
            return EmbedderConfig(
                provider="sentence_transformers",
                model_name=cached,
                device=_pick_device("cpu"),
            )
        raise RuntimeError(
            "ERROR: Offline mode is enabled (HF_HUB_OFFLINE=1 or TRANSFORMERS_OFFLINE=1) but the default fast model "
            "is not available locally.\n"
            f"Expected local model dir (project-local): {local_model}\n"
            "Also checked Hugging Face cache for: sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2\n\n"
            "Fix options:\n"
            "  A) Temporarily download/cache it (online):\n"
            "     HF_HUB_OFFLINE=0 TRANSFORMERS_OFFLINE=0 python -c "
            "\"from sentence_transformers import SentenceTransformer; "
            "SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')\"\n"
            "  B) Or download into the project-local directory and keep using offline mode afterwards "
            "(set EMB_MODEL to the local dir).\n"
        )

    return EmbedderConfig(
        provider="sentence_transformers",
        model_name=remote_model,
        device=_pick_device("cpu"),
    )


# ---------------------------------------------------------------------------
# Embedding function factory
# ---------------------------------------------------------------------------

def create_embedding_function(cfg: EmbedderConfig, task_type: str = "RETRIEVAL_DOCUMENT"):
    """
    Create the appropriate embedding function callable for ONLINE use
    (MCP server queries, citation mapper).

    Returns a callable with signature List[str] -> List[np.ndarray].

    For batch indexing (Gemini), use gemini_batch.py instead.

    Args:
        cfg: Resolved embedder configuration.
        task_type: Gemini task type ("RETRIEVAL_DOCUMENT" for indexing,
                   "RETRIEVAL_QUERY" for search). Ignored for local models.
    """
    if cfg.provider == "gemini":
        from chromadb.utils.embedding_functions import GoogleGeminiEmbeddingFunction

        return GoogleGeminiEmbeddingFunction(
            model_name=cfg.model_name,
            task_type=task_type,
            api_key_env_var="GEMINI_API_KEY",
        )
    else:
        return embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=cfg.model_name,
            device=cfg.device,
            normalize_embeddings=True,
        )


# ---------------------------------------------------------------------------
# Dimension probe
# ---------------------------------------------------------------------------

def probe_embedding_dim(ef) -> Optional[int]:
    """Probe the output dimensionality of an embedding function.

    Returns the vector length, or None if the probe fails.
    """
    try:
        probe_vecs = ef(["collection probe"])
        if isinstance(probe_vecs, list) and probe_vecs and isinstance(probe_vecs[0], (list, tuple)):
            return len(probe_vecs[0])
    except Exception:
        return None
    return None


# ---------------------------------------------------------------------------
# Collection name resolution
# ---------------------------------------------------------------------------

def resolve_collection_name(
    ef,
    *,
    env_value: str | None = None,
    default: str = "zotero_paragraphs",
) -> str:
    """Resolve a ChromaDB collection name.

    If *env_value* is provided and non-empty, it is used verbatim.
    Otherwise the embedding dimension is probed and the *default* name
    is suffixed with ``_<dim>`` (e.g. ``zotero_paragraphs_384``) to prevent
    dimension mismatch when switching embedding models.

    Args:
        ef: Embedding function callable (list[str] -> list[list[float]]).
        env_value: Explicit collection name from environment (e.g. CHROMA_COLLECTION).
        default: Base collection name when no explicit override is set.

    Returns:
        Resolved collection name.
    """
    explicit = (env_value or "").strip()
    if explicit:
        return explicit
    dim = probe_embedding_dim(ef)
    if isinstance(dim, int) and dim > 0:
        return f"{default}_{dim}"
    return default


# ---------------------------------------------------------------------------
# ChromaDB collection helpers
# ---------------------------------------------------------------------------

def open_chroma_collection(
    chroma_dir: str | Path,
    collection_name: str,
    ef,
    *,
    metadata: dict | None = None,
):
    """Open (or create) a ChromaDB collection and attach an embedding function.

    Returns a ChromaDB Collection with ``_embedding_function`` set.
    The embedding function is attached manually to avoid ChromaDB Rust FFI
    deadlocks on reads.
    """
    chroma_dir = Path(chroma_dir)
    chroma_dir.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(chroma_dir))

    if metadata is None:
        metadata = {"hnsw:space": "cosine", "hnsw:sync_threshold": 100}

    col = client.get_or_create_collection(
        name=collection_name,
        metadata=metadata,
    )
    col._embedding_function = ef
    # Keep the Client wrapper reachable from the collection so callers can
    # close() it.  Without close(), chromadb keeps the System (and its
    # in-memory HNSW segment) cached per-path in SharedSystemClient, and
    # re-creating a PersistentClient silently reuses the stale one.
    col._chroma_client = client

    # For existing collections, get_or_create_collection does NOT update metadata.
    # Explicitly apply sync_threshold=100 so the indexer flushes more frequently.
    try:
        col.modify(configuration={"hnsw": {"sync_threshold": 100}})
    except Exception:
        pass  # Non-fatal: older ChromaDB versions may not support this

    return col


def save_embedder_config(
    chroma_dir: str | Path,
    cfg: EmbedderConfig,
    dim: int | None,
    collection_name: str,
) -> None:
    """Persist resolved embedder configuration for reproducibility/debugging.

    This is best-effort — failures are silently ignored (non-essential artifact).
    """
    try:
        cfg_path = Path(chroma_dir) / "embedder_config.json"
        tmp_path = Path(chroma_dir) / "embedder_config.json.tmp"
        cfg_out = {
            "python": sys.executable,
            "provider": cfg.provider,
            "emb_model": cfg.model_name,
            "emb_device": cfg.device,
            "embedding_dim": dim,
            "collection": collection_name,
        }
        tmp_path.write_text(json.dumps(cfg_out, ensure_ascii=False, indent=2), encoding="utf-8")
        tmp_path.replace(cfg_path)
    except Exception:
        pass


# ---------------------------------------------------------------------------
# High-level collection factory (used by the indexer)
# ---------------------------------------------------------------------------

def get_collection(
    *,
    chroma_dir: Path,
    project_root: Path,
    chroma_collection_env: Optional[str],
    chroma_collection_default: str,
):
    """
    Create / open Chroma collection with an embedding function.

    - If CHROMA_COLLECTION is not explicitly set, suffix the default collection name
      by embedding dimension to prevent mismatch when switching models.
    - Write embedder_config.json for reproducibility/debug.
    """
    cfg = resolve_embedder_settings(project_root)
    ef = create_embedding_function(cfg)
    dim = probe_embedding_dim(ef)

    collection_name = resolve_collection_name(
        ef,
        env_value=chroma_collection_env,
        default=chroma_collection_default,
    )

    save_embedder_config(chroma_dir, cfg, dim, collection_name)

    if os.environ.get("DEBUG_EMBEDDER") == "1":
        try:
            probe_text = "sanity check: embedder probe"
            t0 = time.perf_counter()
            _ = ef([probe_text])
            t1 = time.perf_counter()
            print(
                "[DEBUG] Embedder probe: "
                f"python={sys.executable} "
                f"provider={cfg.provider} "
                f"EMB_MODEL={cfg.model_name} "
                f"EMB_DEVICE={cfg.device} "
                f"dim={dim} "
                f"encode_time_s={(t1 - t0):.3f}",
                file=sys.__stderr__,
            )
        except Exception as e:
            print(f"[DEBUG] Embedder probe failed: {e}", file=sys.__stderr__)

    col = open_chroma_collection(chroma_dir, collection_name, ef)
    return col


# Apply process-level configuration at import time (legacy behaviour).
configure_embedder_process()
