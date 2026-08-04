# Post-refactor follow-up register

Date: 2026-08-04  
Baseline: `049b5bf` (`main`)  
Status: Deferred; no immediate production incident is known.

This note records findings from the comprehensive review after the indexing,
repository, MCP, and FastAPI refactoring. The current implementation passes
1,126 tests. These items should be handled as small, separately verified
changes rather than as another large refactoring batch.

## Recommended order

### 1. Share the indexing lock with re-OCR adoption

- Locations: `scripts/run_reocr_queue.py`, `src/reocr_adoption.py`,
  `src/index_from_zotero.py`
- Risk: `--adopt` mutates Chroma, the lexical index, the manifest, and the
  structure database without acquiring the lock used by the normal indexer.
  Concurrent indexing can overwrite either generation, while MCP queries can
  observe a partial update.
- Proposal: move lock acquisition/release into a reusable module and hold the
  same lock for the complete re-OCR adoption unit of work.
- Raise priority if re-OCR adoption is automated or run while the MCP server or
  normal indexer remains active.

### 2. Make re-OCR compensation resilient to rollback failures

- Location: `src/reocr_adoption.py`
- Risk: a failure during Chroma restoration stops lexical and manifest
  restoration and can also prevent the failed status from being recorded.
- Proposal: attempt every compensation independently, collect rollback errors,
  and raise one error that preserves both the original failure and all rollback
  failures. Reuse the attachment unit-of-work primitives where practical.
- Test: inject a failure into each forward and rollback phase and assert the
  final state of all stores.

### 3. Bound `search_items` resource use and validate inputs

- Location: `src/rag_mcp_server.py`
- Risk: `k_internal = max(k * 10, 100)` has no upper bound, so a very large `k`
  can request an excessive Chroma result set. Invalid `MIN_RETURN_CHARS` values
  also fail outside the query error boundary.
- Proposal: use `bounded_env_int`, cap by collection count and an absolute
  maximum, validate query/list parameters, and return a stable client error.

### 4. Serialize manifest writers

- Location: `src/manifest.py`
- Risk: every writer uses the same `.json.tmp` path. Concurrent writers can
  collide, and independent read-modify-write operations can lose updates.
- Proposal: use a unique temporary inode plus a shared writer lock. Define
  clearly which maintenance commands are authorized manifest writers.
- Note: a unique temporary filename prevents temp-file collisions but does not
  by itself prevent lost updates.

### 5. Add translation request limits

- Location: `citation_graph/server.py`
- Risk: the Azure Translator route does not limit text count, per-text length,
  or total characters, allowing excessive memory use, latency, or API cost.
- Proposal: enforce Pydantic limits, reject oversized payloads before the
  upstream request, and return sanitized upstream errors.

### 6. Move identifier override migration out of request handling

- Location: `citation_graph/server.py`
- Risk: each identifier update retries `ALTER TABLE` and suppresses all
  exceptions, including lock, read-only, and I/O failures.
- Proposal: migrate once during database initialization, ignoring only SQLite's
  duplicate-column error, and keep the HTTP handler limited to validation and
  the update transaction.

## Longer-term maintainability

### Per-attachment chunk generations

- Current constraint: `chunk_scheme` is bound to the whole Chroma collection by
  the pipeline fingerprint. A chunk-boundary change therefore requires a new
  homogeneous collection and a full rebuild, even though only attachment-level
  old/new generation mixing must strictly be prevented.
- Proposal: store `chunk_generation`, `chunk_scheme_version`, and a content
  fingerprint per attachment; build a candidate generation, verify its Chroma,
  lexical, structure, and summary artifacts, then atomically switch the active
  generation. Keep the prior generation available for rollback.
- Migration rule: embedding-model or vector-dimension changes still require a
  separate collection and full re-embedding. Chunking changes should permit
  attachment-at-a-time migration followed by an eventual background convergence.
- Do not relax the current collection fingerprint until generation-aware search,
  deletion, crash recovery, and audits are implemented; merely allowing mixed
  rows would reintroduce duplicate and orphaned chunks.

- Continue splitting `index_from_zotero.main_async` by attachment processing
  phase.
- Split `db_relations._init_db` into versioned migrations.
- Extract the ordinary 442-line `rag_search` into request normalization,
  candidate retrieval, fusion/filtering, and response formatting services.
- Split `pdf_extract.extract_chunks_from_pdf` and
  `citation_graph.build_graph_data` along their existing decision phases.
- Add static type checking and a coverage gate after establishing a baseline;
  CI currently checks compilation, fatal Ruff rules, imports, tests, and
  `RuntimeWarning` only.

## Reassessment triggers

Revisit the deferred P1 items immediately if any of the following occurs:

- re-OCR adoption and normal indexing may run concurrently;
- partial or mismatched Chroma/lexical/manifest generations are observed;
- MCP search returns data while a maintenance command is writing;
- `search_items` causes high memory use or long server stalls;
- the citation graph API is exposed beyond its current local-only workflow.
