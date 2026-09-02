# Clean rebuild readiness — 2026-08-10

Updated: 2026-08-13

## M5 result

M5 is complete; this file is retained as the pre-run decision record. The active
generation has 618 manifest attachments, no inflight/failed/deferred attachment,
and `hnsw_validated=true`. Zotero reconciliation resolved 619 eligible attachments
to 618 required attachments because one PDF has a committed EPUB sibling under
`rag:prefer-epub`. Source verification reports zero lost pages, orphan chunks,
dangling node references, unretrievable items, and unreadable source documents.
The new-only cutover audit passed all 593 selected items.

The 48 rebuild deferrals were satisfied from the verified 2026-08-02 Mistral
content-addressed cache; no new Batch was submitted. Post-run warning investigation
and remaining quality work are recorded in `TASKS.md` under
“2026-08-13 clean rebuild完了後の警告検証”.

The sections below are the historical go/no-go record used before the completed
M5 rebuild. Statements about an incomplete "active" generation describe that
pre-M5 point in time; they are not instructions or a description of the current
active V3 generation.

## Go/no-go

| Gate | Result | Evidence |
|---|---|---|
| Rebuild code generation | GO | Indexer code fingerprint `sha256:f3f1ebbac3e383b0c94d22c696caaad45c62d7eabc8b169d0b96a9ae01f5eec5` is pinned. The scale pilot rejects any nonempty pre-existing data plane before embedder creation, Chroma open or measurement, preventing a zero-write rerun from being reported as full throughput. Same-invocation interruption/resume remains supported. A clean rebuild now checks failed/deferred/missing attachments before publishing `hnsw_validated=true`; MCP refuses the durable false state after the process lock is released. |
| Setup configuration handoff | GO | `environment_with_saved_dotenv` overlays saved `.env`/`.env.policy` values on the inherited process environment, and DB lifecycle passes that explicit environment to the indexer, structure rebuild and audit children. A regression starts with stale parent long=Granite/queue=0 and proves saved long=Mistral/queue=1 reaches both rebuild children. |
| PDF extraction/chunk scheme | GO | Current heuristic, OCR fallback, language thresholds, chunk ID format and merge/rescue behavior are frozen. A rebuild must re-extract every attachment and must not reuse old chunks. |
| Zotero inventory | GO | `--rebuild --dry-run` resolved 590 items / 616 attachments: PDF 364, EPUB 208, HTML 44. It reported `canonical_data_modified=false` and zero legacy OCR reuse candidates. |
| Pre-M5 snapshot | GO WITH LOSS OF OLD ROLLBACK | The formerly recorded `data/backups/pre-clean-rebuild-20260810-4b58d6b` was absent. Before M5, a diagnostic snapshot captured the incomplete, MCP-ineligible generation, but it was not a quality baseline and was removed during the 2026-09-01 storage cleanup after the audited active generation and its later rollback had remained healthy. |
| Active Chroma read health | GO | Read-only `check_chroma_health.py --no-repair-fts` found no integrity issue and no orphaned segment directory; no repair was attempted. |
| Capacity | GO | After keeping the snapshot, 70.6 GB remained free. The isolated 300-chunk pilot projected about 7.49 GB of Chroma growth for 513,683 chunks; one complete active-generation equivalent is 8.67 GB. The measured run used a fresh temporary plane, so the new reuse rejection does not invalidate it. The remaining space covers either estimate with a wide margin. |
| Indexing lock | GO | No `data/indexing.lock` remains. Snapshot creation held the production indexing lock and released it after verification. The rebuild entry point and its following non-dry-run document-structure rebuild both acquire the same lock used by re-OCR adoption and observed by MCP. |
| Embedder | GO | Local `bge-m3`, MPS, 1,024 dimensions, normalized output. M0 measured deterministic output; M3 measured batch 128 / HNSW sync 10,000 and successful recovery/reopen/query behavior. The next generation will write the current model-state fingerprint; the old saved fingerprint is not edited or adopted. |
| Paid LLM/OCR | CONDITIONAL | The user explicitly enabled AI TOC, the Mistral queue, query expansion and LLM summaries/reference extraction. Queue generation does not submit paid OCR by itself; submission and adoption remain separate steps requiring explicit approval. OCR-layer audit remains `0`. The slow baseline forces all hosted ingestion features off. |
| Incomplete attempted generation | GO TO RESTART | The pre-fix rebuild stopped at 33/618 and its incremental continuation stopped at 353/619. The active manifest has 352 files, one inflight attachment and durable `hnsw_validated=false`; no indexing lock remains and MCP refuses the generation. Because part of it used the stale OCR route, M5 must restart a clean rebuild rather than resume this generation. No Mistral batch submission/adoption was selected. |

The first approved M5 attempt was stopped at 8/619 when monitoring observed an
OCR-layer audit despite this record saying it was disabled. OCR-layer audit was
restored to `0`, and that partial generation was discarded by restarting clean.
The user confirmed that query expansion, LLM summaries, LLM reference extraction
and Mistral OCR may remain enabled; the running rebuild child still has the first
three disabled because it received a fixed environment at process start. One
Mistral queue deferral was recorded, but no Batch submission or adoption was
performed.

The Setup handoff was fixed and verified, and M5 subsequently completed as
recorded at the top of this file. Mistral queue submission/adoption and paid
AI-TOC calls retain their separate approval boundaries for future runs.

One review item is deliberately outside this embedding gate: custom audit-report
paths are not yet protected from pointing at active data-plane files. The current
resolved gate, Zotero report and source report all live under `data/quality/`, so
the recorded post-rebuild audit command is safe with the present configuration.
Do not redirect those three paths until the follow-up guard is implemented.

## Historical M5 start point (completed; do not rerun as an instruction)

The interactive entry point recorded for that run was:

```bash
./Setup.command
```

At the recorded pre-M5 point, entering `REBUILD` started deletion of the
incomplete generation, full extraction, eligible paid AI-TOC calls, chunk
generation and embedding. That operation is complete. For any future rebuild,
use the current Setup guidance and obtain fresh approval instead of replaying
this historical command block. The non-interactive equivalent recorded then was:

```bash
uv run src/index_from_zotero.py --rebuild --progress
```

The completed rebuild was followed by this full source/database reconciliation:

```bash
uv run python scripts/run_db_audit.py
```

## Historical diagnostic rollback point

The old complete-generation rollback recorded on 2026-08-10 was no longer
present. The pre-M5 incomplete generation was retained only for diagnosis; it
is MCP-ineligible and is not a working fallback. Do not restore it over the
current audited generation or mix individual stores from different generations.
Its historical verification command was the following. The snapshot no longer
exists, so this command is retained only as an audit record:

```bash
uv run python scripts/backup_v3_generation.py --verify-only \
  data/backups/pre-m5-incomplete-20260811
```

Restoration is destructive and was intentionally not automated or executed by
this readiness milestone.
