.PHONY: help check sync serve rebuild dump
SHELL := /bin/bash

# Use the same Python that has the dependencies installed.
# Example: make PY=/Users/user/.pyenv/versions/3.10.17/bin/python3 sync
PY ?= python3

# Extra args passed to the indexer (index_from_zotero.py)
INDEX_ARGS ?= --progress

# Extra args passed to the MCP server runner
SERVE_ARGS ?= -u

help:
	@echo "Targets:"
	@echo "  make check    - Quick environment sanity check (fast)"
	@echo "  make sync     - Incremental index update"
	@echo "  make rebuild  - Full rebuild (slow)"
	@echo "  make serve    - Run MCP server"
	@echo "  make dump     - Dump resolved attachments (debug)"

check:
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