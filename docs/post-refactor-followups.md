# Post-refactor follow-up register

Opened: 2026-08-04, from the review after the indexing, repository, MCP and
FastAPI refactoring.
Last triaged: 2026-08-09. No production incident is known for anything here.

Every item below was re-read against the code on the triage date and names the
file and symbol it lives in, so a fix that lands without updating this file
leaves a reference that can be checked (`tests/test_documentation_references.py`
checks that the files exist; the symbols are the reader's job). Items that were
finished or overtaken are listed under [Closed](#closed) in one line each rather
than deleted silently, because the same review would otherwise raise them again.

Priority means: **P1** can lose or corrupt data, or has no bound; **P2** is a
robustness or cost defect within the local-only workflow; **P3** costs accuracy
or clarity in what the user is shown. Structural work is not ranked here — its
order is decided by the size ratchet (`tests/test_function_size_ratchet.py`).

## P1

### 1. Share the indexing lock with re-OCR adoption

- Locations: `scripts/run_reocr_queue.py`, `src/reocr_adoption.py`,
  `src/index_from_zotero.py`
- Still true 2026-08-09: `_acquire_indexing_lock` lives in
  `index_from_zotero.py` and nothing else can reach it; neither
  `reocr_adoption.py` nor `run_reocr_queue.py` mentions a lock.
- Risk: `--adopt` mutates Chroma, the lexical index, the manifest and the
  structure database without the lock the normal indexer takes. Concurrent
  indexing can overwrite either generation, and MCP queries can observe a
  partial update.
- Proposal: move acquisition/release into a module both callers can import, and
  hold it for the whole adoption unit of work.
- Raise priority if adoption is automated, or run while the MCP server or the
  indexer is active.

### 2. Make re-OCR compensation resilient to rollback failures

**Done 2026-08-09.** `src/reocr_adoption.py`

- Each compensation now runs independently and collects its own failure, so one
  store refusing no longer abandons the others or the `failed` status write that
  tells the next run to retry. The raised error names the original failure and
  every store left inconsistent.
- The wider problem found while testing it: the status writes ran inside the
  `try`, so a bookkeeping failure rolled back a complete, consistent adoption.
  They now run after the canonical stores are all new, and their failures are
  reported in `status_write_errors` rather than raised -- raising told the
  caller to redo the expensive adoption that had in fact succeeded.
- `tests/test_reocr_adoption_fault_injection.py` fails every phase in turn and
  asserts the same invariant each time, with the stores each case is still
  answerable for stated in the table. Four regressions were injected and caught.

### 3. Bound `search_items` resource use and validate inputs

- Location: `src/rag_mcp_server.py` (`search_items`)
- Still true 2026-08-09: `k_internal = max(k * 10, 100)` has no upper bound, and
  `MIN_RETURN_CHARS` is read with a bare `int(os.environ.get(...))` that raises
  outside the query error boundary. `rag_search` already does both correctly
  with `bounded_env_int` -- this is the one caller left behind, as it was for
  the query-embedding precompute (F3, 2026-07-30).
- Proposal: use `bounded_env_int`, cap by collection count and an absolute
  maximum, validate the query and list parameters, and return a stable client
  error.

### 4. The indexing lock is only released when the process exits

- Location: `src/index_from_zotero.py` (`_acquire_indexing_lock`, `main_async`)
- Found 2026-08-09 by the ingest fault-injection tests: after a run raises, the
  lock file is still held. `main_async` has no failure path that releases it --
  the release is registered with `atexit` and repeated once at the end of the
  happy path, read out of `locals().get("lock_data")`.
- Why it is not P1: the CLI is one run per process, so `atexit` covers a crash
  today. It bites any caller that runs an ingest and then keeps going -- an
  embedded run, a maintenance process doing two things, a test -- where the
  second attempt is refused with "another indexer is running" naming its own
  live PID, and the message tells the user to delete the file by hand.
- Proposal: release in a `finally` around the run, keeping `atexit` as the
  backstop, and stop reading the lock data out of `locals()`.

### 5. Serialize manifest writers

- Location: `src/manifest.py`
- Still true 2026-08-09: every writer builds the same `.json.tmp` path next to
  the manifest and `replace()`s it into place.
- Risk: concurrent writers collide on the temporary path, and independent
  read-modify-write cycles lose updates.
- Proposal: a unique temporary inode plus a shared writer lock. Decide which
  maintenance commands are authorized writers. A unique temporary name alone
  fixes the collision and not the lost update.

## P2

### 6. Add translation request limits

- Location: `citation_graph/server.py` (`_TranslateBatchRequest`)
- Still true 2026-08-09: the model is `texts: list[str]` with no constraint, so
  the Azure Translator route bounds neither text count, per-text length nor
  total characters. Cost, latency and memory are all unbounded by the request.
- Proposal: enforce the limits in the Pydantic model, reject oversized payloads
  before the upstream call, and return sanitized upstream errors.

### 7. Move identifier override migration out of request handling

- Location: `citation_graph/server.py` (`_ensure_override_table`)
- Still true 2026-08-09: the handler calls it per update, and it retries five
  `ALTER TABLE` statements under `except Exception: pass`, which swallows lock,
  read-only and I/O failures alongside the duplicate-column error it means.
- Proposal: migrate once at database initialization, ignore only SQLite's
  duplicate-column error, and leave the handler with validation and the update
  transaction.

## P3

### 8. Say which citation count the tooltip is showing

- Locations: `citation_graph/server.py` (~L1769), `citation_graph/static/app.js`
  (~L2186)
- The discard that caused the gap is fixed (see [Closed](#closed)), but the
  label is not: the tooltip prefers `s2_citation_count` when S2 has it and
  writes it under `被引用数`, while the graph draws the subset that is actually
  in the database. Two different numbers under one label.
- Proposal: name the S2 total and the drawn count separately, or label the
  total as S2's.

### 9. Give S2 identity a discriminator for author-less records

- Location: `src/citation_mapper.py` (`_record_names_a_creator`)
- The function passes a record that lists no author, on the principle that
  missing evidence cannot convict. On the DOI/ISBN path the identifier carries
  the identity and that is right. On the title-search path it leaves title
  similarity >= 0.5 as the only test, so an author-less record that is simply
  the wrong work passes.
- Why not a blanket rule: at the 2026-08-06 count, 21 of 274 mapped items
  resolved to a record with no authors, including correct ones (*Asia as
  Method*, *Ego and the Id*) alongside the doubtful ("Chapter Three.
  EARTHQUAKES"). Refusing them wholesale drops the correct with the wrong.
- Re-measure before acting; the count is from the mapping as it stood then.

### 10. Flat-PDF recovery: three shapes that stay flat

Verified against `tests/baselines/structure_recovery.json` on 2026-08-09, where
6 of 84 attachments recover a tree. Both named books are still flat.

- **A contents page that lost its own heading.** With no "Contents"/"目次" for
  the guard to key on, the contents lines read as part openers. N6RU3QQG
  (*Japan-ness in Architecture*) is in this state. It no longer produces a wrong
  tree -- the promotion gate refuses divisions holding almost none of the
  document -- but the structure is really there in the book. Every contents line
  ends in a folio and a real opener does not, which is a possible discriminator
  if more cases appear.
- **Part openers and nothing else.** `source_structure_refresh` requires
  `len(self.events) >= 5` to promote, so a book whose only kept headings are its
  parts recovers nothing. 4GPDN33D (*The Spectre of Comparisons*) has four real
  parts at pages 19/43/137/171. Worth revisiting the gate once the part
  vocabulary has fewer false positives to guard against -- it was written when a
  bare Roman numeral could match prose.
- **Numbering continued from a previous volume.** `_numbering_is_contiguous`
  requires a run of 1..n, so volume two of a work whose chapters carry on from
  volume one is rejected. Deliberate -- nothing in the chunks tells it apart
  from an extractor that lost the opening chapters -- but it is a real shape for
  multi-volume works and 全集.

## Longer-term maintainability

Not ranked with the above. The sizes below are held by the function-size ratchet
(`tests/function_size_budget.json`, checked by
`tests/test_function_size_ratchet.py`), which stops them growing and records
each split as it lands; this section keeps only what a splitter needs to know
that the size number does not say.

### Per-attachment chunk generations

- Current constraint: `chunk_scheme` is bound to the whole Chroma collection by
  the pipeline fingerprint, so a chunk-boundary change needs a new homogeneous
  collection and a full rebuild -- even though only attachment-level old/new
  mixing must strictly be prevented.
- Proposal: store `chunk_generation`, `chunk_scheme_version` and a content
  fingerprint per attachment; build a candidate generation, verify its Chroma,
  lexical, structure and summary artifacts, then switch the active generation
  atomically, keeping the prior one for rollback.
- Migration rule: embedding-model or vector-dimension changes still require a
  separate collection and full re-embedding. Chunking changes should permit
  attachment-at-a-time migration with eventual background convergence.
- Do not relax the collection fingerprint until generation-aware search,
  deletion, crash recovery and audits exist. Merely allowing mixed rows
  reintroduces duplicate and orphaned chunks.

### What the splitters need to know

- `index_from_zotero.main_async` (1,113 lines) is the largest remaining
  function. The PDF route came out of it in 2026-08-09 as `_extract_pdf_chunks`
  (705 lines), which is now the second largest: the seam moved rather than
  disappeared, and the same measured-interface approach applies to it.
- The ingestion net (`tests/test_ingestion_baseline.py`, `slow`-marked) reaches
  28% of the loop and about 15% of the PDF route, so verification is the
  expensive half of any split here. **Measure a block's reach before lifting
  it** -- an unreached block is not verified by the net, only unopposed by it.
- Widening the net needs attachments that take the OCR routes, and those are not
  deterministic: the library's one corrupted PDF returned 109 chunks twice and
  108 once, and the smallest image-page EPUB extracts through Mistral in the
  cloud. The way past this is to separate the deterministic decisions inside the
  PDF route from the extractor calls they choose between, so the decisions can
  be netted and the calls mocked.
- `db_relations._init_db` (698 lines) wants versioned migrations, not a split
  into helpers -- see item 7, which is the same problem leaking into a handler.
- `rag_search` (442 lines) divides into request normalization, candidate
  retrieval, fusion/filtering and response formatting.
- `pdf_extract.extract_chunks_from_pdf` (614) and
  `citation_graph.build_graph_data` (470) divide along their existing decision
  phases. `citation_mapper.map_item_global_citations` (444) is a fourth over 400
  and holds the citations/references asymmetry that produced item 8.

### Checking layers CI does not have

- Static type checking and a coverage gate, after establishing a baseline. CI
  checks compilation, fatal Ruff rules, the public import, tests, and
  `RuntimeWarning`.

## Closed

- **A source classification that failed read as "not a scan"** (found and fixed
  2026-08-09). `classify_pdf_source` failing was swallowed with a bare `pass` on
  the reused-OCR path and only printed on the extraction path, so
  `source_class` was absent -- which `reocr_quality` cannot tell apart from a
  born-digital document, since it tests `source_class == "scanned_no_text"`. A
  scan whose classification failed would never be offered for re-OCR again. Both
  paths now record `source_class_error`, and the assessment carries it as its
  own reason, scored below a measured scan because it says "unknown", not
  "known bad". Found by scanning every broad `except` inside a function that
  produces a verdict and reading the eight hits: five were already fail-closed
  and said so in a comment.


- **Context-less incoming citations were discarded** (raised 2026-08-06, fixed
  2026-08-07). `map_item_global_citations` stored nothing for a citing paper
  with no context snippet while the references loop kept the equivalent rows;
  the graph held roughly 37% of the citations S2 reported. Both loops now record
  them (`chunk_status` / `s2_status` = `'no_context'`). The fact worth keeping:
  those rows collide on `uq_global_citations_identity`, a partial expression
  index that COALESCEs the snippet and spans the raw reference text, so
  re-running does not duplicate them and no repair migration was needed. What
  remains is the label, now item 8.
- **Lift the PDF route out of `main_async`** (done 2026-08-09, `41c1484`). The
  665-line `else` branch named here as "the next unit to lift out" is now
  `_extract_pdf_chunks`.
- **`source_structure_refresh._refresh_pdf_rows_from_numbered_body_headings`
  is split** (done before 2026-08-04).

## Reassessment triggers

Revisit the P1 items immediately if any of the following occurs:

- re-OCR adoption and normal indexing may run concurrently;
- partial or mismatched Chroma/lexical/manifest generations are observed;
- MCP search returns data while a maintenance command is writing;
- `search_items` causes high memory use or long server stalls;
- the citation graph API is exposed beyond its current local-only workflow.
