#!/usr/bin/env bash
set -euo pipefail

# Project root (directory containing this script)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Zotero data directory (must contain: storage/ and zotero.sqlite)
export ZOTERO_DATA_DIR="${ZOTERO_DATA_DIR:-$HOME/Zotero}"

# Chroma persistence
export CHROMA_DIR="${CHROMA_DIR:-${SCRIPT_DIR}/data/chroma}"

# Optional: Pin a specific Chroma collection name.
# If you leave CHROMA_COLLECTION unset, the code will auto-suffix the default
# name by embedding dimension (e.g., zotero_paragraphs_384 / zotero_paragraphs_1024)
# to avoid dimension-mismatch when switching embedding models.
if [[ -n "${CHROMA_COLLECTION:-}" ]]; then
  export CHROMA_COLLECTION="${CHROMA_COLLECTION}"
fi

# Local cache for PDFs downloaded via Local API fallback
export PDF_CACHE_DIR="${PDF_CACHE_DIR:-${SCRIPT_DIR}/data/pdf_cache}"

# Manifest path (indexer state)
export MANIFEST_PATH="${MANIFEST_PATH:-${SCRIPT_DIR}/data/manifest.json}"

# Optional: show progress logs by default when running via Make
# export PROGRESS="${PROGRESS:-1}"

# Offline-only by default (recommended): prevent any Hugging Face network access.
# You can temporarily override for first-time downloads by exporting HF_HUB_OFFLINE=0 / TRANSFORMERS_OFFLINE=0
# *before* sourcing this file.
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"

# Embedding profile (choose once per machine/install)
# - fast: multilingual + relatively light
# - bge : higher quality (bge-m3) but heavier
export EMB_PROFILE="${EMB_PROFILE:-fast}"

# Device for embedding model: cpu | mps (Apple Silicon) | cuda (NVIDIA)
# If EMB_DEVICE is already set, respect it. Otherwise choose a sensible default.
if [[ -z "${EMB_DEVICE+x}" ]]; then
  case "${EMB_PROFILE}" in
    bge)
      export EMB_DEVICE="mps"
      ;;
    fast|*)
      export EMB_DEVICE="cpu"
      ;;
  esac
fi

# Embedding model (must be cached if running offline)
# If you want to override, set EMB_MODEL explicitly before sourcing this file.
if [[ -z "${EMB_MODEL:-}" || "${EMB_MODEL}" == "/path/to/the/snapshot_dir" ]]; then
  case "${EMB_PROFILE}" in
    bge)
      # Local path recommended for bge-m3 (avoid remote lookups during offline runs)
      export EMB_MODEL="${SCRIPT_DIR}/data/models/bge-m3"
      ;;
    fast|*)
      # Offline-first: use a project-local model directory (download/copy it here once).
      export EMB_MODEL="${SCRIPT_DIR}/data/models/paraphrase-multilingual-MiniLM-L12-v2"
      ;;
  esac
fi

# If running offline, fail fast when the local model directory is missing.
if [[ "${HF_HUB_OFFLINE:-0}" == "1" || "${TRANSFORMERS_OFFLINE:-0}" == "1" ]]; then
  if [[ ! -d "${EMB_MODEL}" ]]; then
    echo "ERROR: Offline mode is enabled (HF_HUB_OFFLINE=1 or TRANSFORMERS_OFFLINE=1), but EMB_MODEL is not a local directory: ${EMB_MODEL}" 1>&2
    echo "Fix: set EMB_MODEL to an existing local snapshot directory under ./data/models/, or unset EMB_MODEL to use the default for EMB_PROFILE." 1>&2
    exit 1
  fi
fi