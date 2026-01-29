# src/embedder.py
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path
from typing import Optional, Tuple

import chromadb
from chromadb.utils import embedding_functions

# Optional: resolve cached Hugging Face model snapshots (offline-friendly)
try:
    from huggingface_hub import snapshot_download  # type: ignore
except Exception:  # pragma: no cover
    snapshot_download = None


def _resolve_embedder_settings(project_root: Path) -> Tuple[str, str]:
    """
    Returns (model_name_or_path, device).
    Logic is ported from the previous index_from_zotero.py implementation.
    """
    profile = (os.environ.get("EMB_PROFILE") or "fast").strip().lower()

    # Treat either flag as "offline".
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
        raise SystemExit(
            "ERROR: Offline mode is enabled (HF_HUB_OFFLINE=1 or TRANSFORMERS_OFFLINE=1), "
            f"but the requested embedding model is not available locally: {model}\n\n"
            "Fix options:\n"
            "  (1) Temporarily go online and cache it, then rerun offline:\n"
            "      HF_HUB_OFFLINE=0 TRANSFORMERS_OFFLINE=0 python -c "
            "\"from sentence_transformers import SentenceTransformer; "
            f"SentenceTransformer('{model}')\"\n"
            "  (2) Or set EMB_MODEL to a local directory path containing the model files.\n"
        )

    # Explicit override wins.
    if "EMB_MODEL" in os.environ and (os.environ.get("EMB_MODEL") or "").strip():
        model = os.environ["EMB_MODEL"].strip()
        model = _offline_resolve_or_exit(model)
        return model, _pick_device("cpu")

    # Profile-based defaults.
    if profile == "bge":
        model = str(project_root / "data" / "models" / "bge-m3")
        device_default = "mps" if sys.platform == "darwin" else "cpu"
        model = _offline_resolve_or_exit(model)
        return model, _pick_device(device_default)

    # fast (default): multilingual MiniLM
    remote_model = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    local_model = str(project_root / "data" / "models" / "paraphrase-multilingual-MiniLM-L12-v2")

    if offline:
        if _is_local_path(local_model):
            return local_model, _pick_device("cpu")
        cached = _try_resolve_hf_cached_snapshot(remote_model)
        if cached:
            return cached, _pick_device("cpu")
        raise SystemExit(
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

    return remote_model, _pick_device("cpu")


def _probe_embedding_dim(ef) -> Optional[int]:
    try:
        probe_vecs = ef(["collection probe"])
        if isinstance(probe_vecs, list) and probe_vecs and isinstance(probe_vecs[0], (list, tuple)):
            return len(probe_vecs[0])
    except Exception:
        return None
    return None


def get_collection(
    *,
    chroma_dir: Path,
    project_root: Path,
    chroma_collection_env: Optional[str],
    chroma_collection_default: str,
):
    """
    Create / open Chroma collection with a SentenceTransformer embedding function.

    - If CHROMA_COLLECTION is not explicitly set, suffix the default collection name
      by embedding dimension to prevent mismatch when switching models.
    - Write embedder_config.json for reproducibility/debug.
    """
    chroma_dir.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(chroma_dir))

    model_name, device = _resolve_embedder_settings(project_root)
    ef = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name=model_name,
        device=device,
        normalize_embeddings=True,
    )

    probe_dim = _probe_embedding_dim(ef)

    collection_name = (chroma_collection_env or "").strip() or chroma_collection_default
    if not (chroma_collection_env or "").strip():
        if isinstance(probe_dim, int) and probe_dim > 0:
            collection_name = f"{chroma_collection_default}_{probe_dim}"
        else:
            collection_name = chroma_collection_default

    # Persist resolved embedder configuration for debugging / reproducibility.
    try:
        cfg_path = chroma_dir / "embedder_config.json"
        tmp_path = chroma_dir / "embedder_config.json.tmp"
        cfg = {
            "python": sys.executable,
            "emb_model": model_name,
            "emb_device": device,
            "embedding_dim": probe_dim,
            "collection": collection_name,
        }
        tmp_path.write_text(json.dumps(cfg, ensure_ascii=False, indent=2), encoding="utf-8")
        tmp_path.replace(cfg_path)
    except Exception:
        pass

    if os.environ.get("DEBUG_EMBEDDER") == "1":
        try:
            probe_text = "sanity check: embedder probe"
            t0 = time.perf_counter()
            _ = ef([probe_text])
            t1 = time.perf_counter()
            print(
                "[DEBUG] Embedder probe: "
                f"python={sys.executable} "
                f"EMB_MODEL={model_name} "
                f"EMB_DEVICE={device} "
                f"dim={probe_dim} "
                f"encode_time_s={(t1 - t0):.3f}",
                file=sys.__stderr__,
            )
        except Exception as e:
            print(f"[DEBUG] Embedder probe failed: {e}", file=sys.__stderr__)

    col = client.get_or_create_collection(
        name=collection_name,
        embedding_function=ef,
        metadata={"hnsw:space": "cosine"},
    )
    return col