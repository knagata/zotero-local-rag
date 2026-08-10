# Clean rebuild readiness — 2026-08-10

Updated: 2026-08-11

This is the go/no-go record for the next V3 clean rebuild. It stops before any
destructive rebuild or embedding write. The existing V3 generation remains a
rollback source, not a quality baseline.

## Go/no-go

| Gate | Result | Evidence |
|---|---|---|
| Rebuild code generation | GO | Indexer code fingerprint `sha256:f3f1ebbac3e383b0c94d22c696caaad45c62d7eabc8b169d0b96a9ae01f5eec5` is pinned. The scale pilot rejects any nonempty pre-existing data plane before embedder creation, Chroma open or measurement, preventing a zero-write rerun from being reported as full throughput. Same-invocation interruption/resume remains supported. A clean rebuild now checks failed/deferred/missing attachments before publishing `hnsw_validated=true`; MCP refuses the durable false state after the process lock is released. |
| Setup configuration handoff | **NO-GO** | The wizard can load the pre-edit `.env` while checking Granite availability, then launch its rebuild child with that stale inherited environment after saving the new file. An observed 811-page PDF consequently reached the former long-document Granite route although the saved configuration selected Mistral. Do not start M5 from the same Setup invocation until the child environment is built explicitly from the saved configuration and covered by a regression test. |
| PDF extraction/chunk scheme | GO | Current heuristic, OCR fallback, language thresholds, chunk ID format and merge/rescue behavior are frozen. A rebuild must re-extract every attachment and must not reuse old chunks. |
| Zotero inventory | GO | `--rebuild --dry-run` resolved 590 items / 616 attachments: PDF 364, EPUB 208, HTML 44. It reported `canonical_data_modified=false` and zero legacy OCR reuse candidates. |
| Rollback snapshot | GO | `data/backups/pre-clean-rebuild-20260810-4b58d6b` is 8,672,747,653 bytes and was read back successfully: Chroma 513,683 rows, lexical 513,684 rows, manifest 614 attachments, both SQLite quick checks `ok`. The known one-row lexical orphan is preserved, not repaired. New snapshot creation rejects any destination resolving to source Chroma or its descendants before copying. |
| Active Chroma read health | GO | Read-only `check_chroma_health.py --no-repair-fts` found no integrity issue and no orphaned segment directory; no repair was attempted. |
| Capacity | GO | After keeping the snapshot, 70.6 GB remained free. The isolated 300-chunk pilot projected about 7.49 GB of Chroma growth for 513,683 chunks; one complete active-generation equivalent is 8.67 GB. The measured run used a fresh temporary plane, so the new reuse rejection does not invalidate it. The remaining space covers either estimate with a wide margin. |
| Indexing lock | GO | No `data/indexing.lock` remains. Snapshot creation held the production indexing lock and released it after verification. The rebuild entry point and its following non-dry-run document-structure rebuild both acquire the same lock used by re-OCR adoption and observed by MCP. |
| Embedder | GO | Local `bge-m3`, MPS, 1,024 dimensions, normalized output. M0 measured deterministic output; M3 measured batch 128 / HNSW sync 10,000 and successful recovery/reopen/query behavior. The next generation will write the current model-state fingerprint; the old saved fingerprint is not edited or adopted. |
| Paid LLM/OCR | CONDITIONAL | The user explicitly enabled AI TOC and the Mistral queue. Queue generation does not submit paid OCR by itself; submission and adoption remain separate steps requiring explicit approval. OCR-layer audit, query expansion and LLM summaries/reference extraction remain `0`. The slow baseline forces all hosted ingestion features off. `PDF_AI_TOC_DOCLING_REFERENCES_ENABLE=1` is local Docling enrichment, not a hosted call. |
| Prohibited shortcuts | GO | No active-V3 incremental ingest, fingerprint edit, orphan-row repair, saved-chunk-only re-embedding, paid LLM/OCR call or rebuild was performed. The baseline child-environment guard keeps validation independent of local feature approval; pilot reports cannot target active Chroma, manifest, lexical or relations SQLite artifacts. |

The data plane remains protected, but M5 is **not ready to start through Setup**
until the stale child-environment handoff is fixed and verified. Mistral queue
submission/adoption and paid AI-TOC calls also retain their separate approval
boundaries.

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
LLM summaries/reference extraction remain disabled. Until the Setup environment
handoff is fixed, do not enter the literal `REBUILD`; the just-saved settings
may not be the settings inherited by its child indexer. Entering that word also
starts deletion, full extraction, eligible paid AI-TOC calls, chunk generation
and embedding, and therefore still requires a new explicit approval. The
non-interactive equivalent shown by Setup is:

```bash
uv run src/index_from_zotero.py --rebuild --progress
```

After a successful rebuild, run the full source/database reconciliation:

```bash
uv run python scripts/run_db_audit.py
```

## Rollback point

The verified snapshot above is the rollback point. If M5 fails, stop the
indexer and preserve the failed generation for diagnosis before restoring the
snapshot's `chroma`, `manifest_v3.json`, `lexical_v3.sqlite3`, and
`relations.db` as one unit. Do not mix individual stores from different
generations. Re-verify the snapshot before any restore with:

```bash
uv run python scripts/backup_v3_generation.py --verify-only \
  data/backups/pre-clean-rebuild-20260810-4b58d6b
```

Restoration is destructive and is intentionally not automated or executed by
this readiness milestone.
