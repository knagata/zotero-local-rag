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

### 7. Decide what to do with context-less incoming citations

- Location: `src/citation_mapper.py` (`map_item_global_citations`)
- Observation (2026-08-06): the citations loop does `if not contexts: continue`,
  so every citing paper S2 reports without an extracted context snippet is
  discarded. The references loop, a few hundred lines below, keeps the
  equivalent rows as `s2_status='no_context'`. The asymmetry looks unintended.
- Impact: the graph holds roughly 37% of the citations S2 knows about
  (`s2_citation_count` totals 54,132 against 20,188 stored citing papers).
  Per item the gap can be far larger — V2ESRAMA shows "Citations: 4,188" in its
  tooltip while contributing 191 citing nodes. Knowing *who* cited a work is
  useful even with no quotable context, so the discard also costs real signal.
- Resolved 2026-08-07: the citations loop now stores them as
  `chunk_status='no_context'` with an empty-string snippet, matching the
  references loop.
- Two objections recorded here on 2026-08-06 turned out not to hold:
  - *"A NULL snippet would never conflict, so every re-run would duplicate."*
    Wrong. Rows collide on `uq_global_citations_identity`, a partial expression
    index that COALESCEs the snippet, not on the table-level UNIQUE. Inserting
    the same context-less citation three times leaves one row with either NULL
    or `''`.
  - *"The references table already has this defect: 1,622 `no_context` rows
    cover only 1,095 distinct (paper, item) pairs."* That count used (paper,
    item) alone. The index identity also spans `raw_reference_text` and falls
    back to title+year+authors when there is no paper id; measured against the
    real expression the 1,622 rows are 1,622 distinct entries. No repair
    migration was needed.
  - What remains true is the graph-size question. It is bounded by the existing
    `max_citations` cap (5,000/item, which only 6 of 574 items reach), so the
    growth lands on a handful of heavily cited works rather than the library.
- Still open: the tooltip labels the S2 total as "Citations" while the graph
  draws a subset, so the number should say which it is.

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

- Flat-PDF structure recovery: a book whose printed contents page lost its own
  "Contents"/"目次" heading during extraction has nothing for the contents guard
  to key on, so its contents lines are read as part openers. One item is in this
  state (N6RU3QQG, *Japan-ness in Architecture*): four of its five recovered
  boundaries are contents lines. Every such line ends in a folio while a real
  opener does not, which is a possible discriminator if more cases appear.
- Flat-PDF structure recovery: a book that yields only part openers, with no
  chapter headings the extractor kept, fails the `len(events) >= 5` promotion
  gate and stays flat. 4GPDN33D (*The Spectre of Comparisons*) has four real
  parts at pages 19/43/137/171 and nothing else, so it recovers nothing. Worth
  revisiting the gate once the part vocabulary has fewer false positives to
  guard against; it was written when a bare Roman numeral could match prose.
- Flat-PDF structure recovery: `_numbering_is_contiguous` requires a run of
  1..n, so a volume whose chapters continue the previous volume's numbering is
  rejected and stays flat. Deliberate — nothing in the chunks tells it apart
  from an extractor that lost the opening chapters — but it is a real shape for
  multi-volume works and 全集.
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
