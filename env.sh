#!/usr/bin/env bash
set -euo pipefail

# Project root (directory containing this script)
# Works in bash and zsh when the script is sourced.
if [[ -n "${BASH_VERSION:-}" ]]; then
  _ENV_SH_PATH="${BASH_SOURCE[0]}"
elif [[ -n "${ZSH_VERSION:-}" ]]; then
  _ENV_SH_PATH="${(%):-%N}"
else
  # Fallback: may be less accurate when sourced in other shells
  _ENV_SH_PATH="$0"
fi
SCRIPT_DIR="$(cd "$(dirname "${_ENV_SH_PATH}")" && pwd)"
unset _ENV_SH_PATH

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

# Chunking / overlap tuning (optional)
# MAX_CHARS controls the maximum chunk size after splitting.
# MIN_CHUNK_CHARS is the target minimum chunk size for space-delimited languages.
# MIN_CHUNK_CHARS_NO_SPACE is the target minimum chunk size for CJK/no-space docs.
# HARD_MIN_CHARS drops very short noise chunks even before merging.
#
# Overlap controls (characters):
# - If OVERLAP_CHARS is set to a positive value, it overrides both language-specific settings.
# - Otherwise, OVERLAP_CHARS_LATIN / OVERLAP_CHARS_CJK are used depending on document heuristic.
#
# Uncomment to pin values globally.
# export MAX_CHARS="${MAX_CHARS:-1200}"
# export MIN_CHUNK_CHARS="${MIN_CHUNK_CHARS:-200}"
# export MIN_CHUNK_CHARS_NO_SPACE="${MIN_CHUNK_CHARS_NO_SPACE:-120}"
# export HARD_MIN_CHARS="${HARD_MIN_CHARS:-40}"
# export BATCH_SIZE="${BATCH_SIZE:-128}"              # Chroma upsert batch size
#
# export OVERLAP_CHARS="${OVERLAP_CHARS:-0}"          # global override (0 = disabled)
# export OVERLAP_CHARS_LATIN="${OVERLAP_CHARS_LATIN:-80}"
# export OVERLAP_CHARS_CJK="${OVERLAP_CHARS_CJK:-60}"

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