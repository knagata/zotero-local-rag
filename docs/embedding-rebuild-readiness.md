# Clean rebuild readiness — 2026-08-10

Updated: 2026-08-11

This is the go/no-go record for the next V3 clean rebuild. A rebuild attempted
before M4.5 was fixed was stopped, and its incomplete active generation is not a
quality baseline. The previously verified pre-rebuild snapshot is no longer
present; the incomplete active generation was snapshotted for diagnosis before M5.

## Go/no-go

| Gate | Result | Evidence |
|---|---|---|
| Rebuild code generation | GO | Indexer code fingerprint `sha256:f3f1ebbac3e383b0c94d22c696caaad45c62d7eabc8b169d0b96a9ae01f5eec5` is pinned. The scale pilot rejects any nonempty pre-existing data plane before embedder creation, Chroma open or measurement, preventing a zero-write rerun from being reported as full throughput. Same-invocation interruption/resume remains supported. A clean rebuild now checks failed/deferred/missing attachments before publishing `hnsw_validated=true`; MCP refuses the durable false state after the process lock is released. |
| Setup configuration handoff | GO | `environment_with_saved_dotenv` overlays saved `.env`/`.env.policy` values on the inherited process environment, and DB lifecycle passes that explicit environment to the indexer, structure rebuild and audit children. A regression starts with stale parent long=Granite/queue=0 and proves saved long=Mistral/queue=1 reaches both rebuild children. |
| PDF extraction/chunk scheme | GO | Current heuristic, OCR fallback, language thresholds, chunk ID format and merge/rescue behavior are frozen. A rebuild must re-extract every attachment and must not reuse old chunks. |
| Zotero inventory | GO | `--rebuild --dry-run` resolved 590 items / 616 attachments: PDF 364, EPUB 208, HTML 44. It reported `canonical_data_modified=false` and zero legacy OCR reuse candidates. |
| Pre-M5 snapshot | GO WITH LOSS OF OLD ROLLBACK | The formerly recorded `data/backups/pre-clean-rebuild-20260810-4b58d6b` is absent and could not be found under the home directory, Trash or mounted volumes. The current generation was already incomplete and MCP-ineligible, so M5 does not destroy a working search generation. Before M5, `data/backups/pre-m5-incomplete-20260811` captured the incomplete generation as one verified unit: Chroma/FTS 272,007 rows, manifest 352 attachments, both SQLite quick checks `ok`, 10,832,960,633 bytes. It is a diagnostic rollback to the unusable pre-M5 state, not a quality baseline. |
| Active Chroma read health | GO | Read-only `check_chroma_health.py --no-repair-fts` found no integrity issue and no orphaned segment directory; no repair was attempted. |
| Capacity | GO | After keeping the snapshot, 70.6 GB remained free. The isolated 300-chunk pilot projected about 7.49 GB of Chroma growth for 513,683 chunks; one complete active-generation equivalent is 8.67 GB. The measured run used a fresh temporary plane, so the new reuse rejection does not invalidate it. The remaining space covers either estimate with a wide margin. |
| Indexing lock | GO | No `data/indexing.lock` remains. Snapshot creation held the production indexing lock and released it after verification. The rebuild entry point and its following non-dry-run document-structure rebuild both acquire the same lock used by re-OCR adoption and observed by MCP. |
| Embedder | GO | Local `bge-m3`, MPS, 1,024 dimensions, normalized output. M0 measured deterministic output; M3 measured batch 128 / HNSW sync 10,000 and successful recovery/reopen/query behavior. The next generation will write the current model-state fingerprint; the old saved fingerprint is not edited or adopted. |
| Paid LLM/OCR | CONDITIONAL | The user explicitly enabled AI TOC and the Mistral queue. Queue generation does not submit paid OCR by itself; submission and adoption remain separate steps requiring explicit approval. OCR-layer audit, query expansion and LLM summaries/reference extraction remain `0`. The slow baseline forces all hosted ingestion features off. `PDF_AI_TOC_DOCLING_REFERENCES_ENABLE=1` is local Docling enrichment, not a hosted call. |
| Incomplete attempted generation | GO TO RESTART | The pre-fix rebuild stopped at 33/618 and its incremental continuation stopped at 353/619. The active manifest has 352 files, one inflight attachment and durable `hnsw_validated=false`; no indexing lock remains and MCP refuses the generation. Because part of it used the stale OCR route, M5 must restart a clean rebuild rather than resume this generation. No Mistral batch submission/adoption was selected. |

The first approved M5 attempt was stopped at 8/619 when monitoring observed an
OCR-layer audit despite this record saying it was disabled. The saved `.env`
had OCR-layer audit, query expansion, LLM summaries and LLM reference extraction
set to `1`; all four were restored to `0`. That partial generation must also be
discarded by restarting clean. One Mistral queue deferral was recorded, but no
Batch submission or adoption was performed.

The Setup handoff is fixed and verified. M5 still requires explicit approval,
and it must replace the incomplete pre-fix generation from the beginning.
Mistral queue submission/adoption and paid AI-TOC calls retain their separate
approval boundaries.

One review item is deliberately outside this embedding gate: custom audit-report
paths are not yet protected from pointing at active data-plane files. The current
resolved gate, Zotero report and source report all live under `data/quality/`, so
the recorded post-rebuild audit command is safe with the present configuration.
Do not redirect those three paths until the follow-up guard is implemented.

## Start point (do not run without M5 approval)

The supported interactive entry point is:

```bash
./Setup.command
```

Keep the current BGE profile and feature settings: AI TOC and Mistral queue
generation are explicitly enabled, while OCR-layer audit, query expansion and
LLM summaries/reference extraction remain disabled. Entering the literal
`REBUILD` starts deletion of the incomplete generation, full extraction,
eligible paid AI-TOC calls, chunk generation and embedding, and therefore still
requires a new explicit approval. The
non-interactive equivalent shown by Setup is:

```bash
uv run src/index_from_zotero.py --rebuild --progress
```

After a successful rebuild, run the full source/database reconciliation:

```bash
uv run python scripts/run_db_audit.py
```

## Diagnostic rollback point

The old complete-generation rollback recorded on 2026-08-10 is no longer
present. If M5 fails, stop the indexer and preserve the failed generation for
diagnosis. The pre-M5 incomplete generation can be restored only as one unit;
it remains MCP-ineligible and is not a working fallback. Do not mix individual
stores from different generations. Re-verify it before any restore with:

```bash
uv run python scripts/backup_v3_generation.py --verify-only \
  data/backups/pre-m5-incomplete-20260811
```

Restoration is destructive and is intentionally not automated or executed by
this readiness milestone.
