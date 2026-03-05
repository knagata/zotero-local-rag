.PHONY: help setup check cache-model sync serve rebuild dump
SHELL := /bin/bash

# Python interpreter: defaults to the project venv created by `make setup`.
# Override only when needed: make PY=/other/python sync
PY ?= .venv/bin/python

# Extra args passed to the indexer (index_from_zotero.py)
INDEX_ARGS ?= --progress

# Extra args passed to the MCP server runner
SERVE_ARGS ?= -u

help:
	@echo "Targets:"
	@echo "  make setup        - Create venv and install dependencies (run once)"
	@echo "  make check        - Quick environment sanity check (fast)"
	@echo "  make cache-model  - Download embedding model (run once, requires internet)"
	@echo "  make sync         - Incremental index update"
	@echo "  make rebuild      - Full rebuild (slow)"
	@echo "  make serve        - Run MCP server"
	@echo "  make dump         - Dump resolved attachments (debug)"

setup:
	uv sync

cache-model:
	@. ./env.sh; \
		HF_HUB_OFFLINE=0 TRANSFORMERS_OFFLINE=0 \
		$(PY) -c "\
from huggingface_hub import snapshot_download; \
import os; \
profile = os.environ.get('EMB_PROFILE', 'fast'); \
repo = 'BAAI/bge-m3' if profile == 'bge' else 'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2'; \
dest = os.environ['EMB_MODEL']; \
print(f'Downloading {repo} -> {dest}'); \
snapshot_download(repo_id=repo, local_dir=dest, local_dir_use_symlinks=False); \
print('Done'); \
"

check:
	@if [ ! -f "$(PY)" ]; then \
		echo "ERROR: $(PY) not found. Run 'make setup' first."; \
		exit 1; \
	fi
	@set -euo pipefail; \
		. ./env.sh; \
		echo "[CHECK] python=$$($(PY) -c 'import sys; print(sys.executable)')"; \
		echo "[CHECK] HF_HUB_OFFLINE=$${HF_HUB_OFFLINE:-} TRANSFORMERS_OFFLINE=$${TRANSFORMERS_OFFLINE:-}"; \
		if [ -z "$${ZOTERO_DATA_DIR:-}" ]; then \
			echo "ERROR: ZOTERO_DATA_DIR is not set."; \
			echo "Set it to your Zotero data directory (must contain 'storage/' and 'zotero.sqlite')."; \
			exit 1; \
		fi; \
		if [ ! -d "$${ZOTERO_DATA_DIR}/storage" ] || [ ! -f "$${ZOTERO_DATA_DIR}/zotero.sqlite" ]; then \
			echo "ERROR: ZOTERO_DATA_DIR=$${ZOTERO_DATA_DIR} must contain 'storage/' and 'zotero.sqlite'."; \
			exit 1; \
		fi; \
		if [ -z "$${CHROMA_DIR:-}" ]; then \
			echo "ERROR: CHROMA_DIR is not set."; \
			exit 1; \
		fi; \
		mkdir -p "$${CHROMA_DIR}"; \
		$(PY) -c 'import chromadb, sentence_transformers, huggingface_hub, pymupdf, fastmcp, httpx, typing_extensions, trafilatura, ebooklib, pydantic; print("[CHECK] python imports OK")'; \
		echo "[CHECK] OK"

sync: check
	@. ./env.sh && $(PY) src/index_from_zotero.py $(INDEX_ARGS)

serve: check
	@. ./env.sh && $(PY) $(SERVE_ARGS) src/rag_mcp_server.py

rebuild: check
	@. ./env.sh && $(PY) src/index_from_zotero.py --rebuild $(INDEX_ARGS)

dump: check
	@. ./env.sh && $(PY) src/index_from_zotero.py --dump-attachments $(INDEX_ARGS)
