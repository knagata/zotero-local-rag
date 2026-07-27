# zotero-local-rag: current implementation specification

Status: active implementation contract as of 2026-07-27.  This document
supersedes historical plans where they conflict.  Design evidence and rollout
records remain in `dev-notes/current/64_ingestion_redesign_audit_and_plan.md`,
`75_v3_cutover_20260723.md`, and `79_embedding_gates.md`.

## Active data plane

The only active search and ingestion plane is V3:

| component | active value |
|---|---|
| Chroma chunks | `zotero_paragraphs_v3` |
| manifest | `data/manifest_v3.json` |
| lexical index | `data/lexical_v3.sqlite3` |
| structured ingestion | `INGEST_STRUCTURED_V3_ENABLE=1` |
| hierarchical retrieval | `HIERARCHICAL_SEARCH_V2_ENABLE=1` |

An attachment is processed transactionally.  Canonical chunks retain source
locator, zone, extraction provenance, policy, and structure-node metadata.
The manifest, Chroma IDs, and FTS IDs must agree; pending/inflight state or an
unvalidated HNSW index fails the cutover gate.  Notes are searchable annotations
but are not canonical document structure.

## Extraction and quality routing

EPUB and HTML use DOM-aware block extraction.  Structural boundaries, zones,
footnotes, and cross-file note links are retained.  Fixed-layout image EPUBs
are rendered to pages and return to their source EPUB spine locators after OCR.

PDF routing is evidence- and policy-based:

1. A reliable embedded text layer and compatible outline use PyMuPDF;
   long unstructured PDFs can use the AI outline fast path when explicitly
   enabled.
2. A scanned PDF without an OCR layer goes straight to Docling below 30 pages.
   At 30 pages or more it is deferred to the Mistral OCR Batch queue when the
   cloud policy permits; no-cloud material and failed tag checks use Docling
   regardless of length. A scanned PDF with an OCR layer is adopted only when
   the stage-2 quality gate accepts it; every other verdict follows the same
   30-page rule.
3. RapidOCR is not an ordinary PDF route (it remains available for fixed-layout
   EPUB and explicit re-OCR work). Mistral Batch deferral is non-canonical; a
   staged result is adopted only after deterministic quality checks and an
   explicit per-item commit.

The U0–U4 gates are implemented for new/reingested material: deterministic
defects are re-extracted, page-level OCR corruption can be repaired, known
corruption is opt-in searchable, and uncertain text remains searchable with
quality provenance.  U5 is an operational reingest decision, not an implicit
rewrite of existing sources.

## Summaries and LLM use

Document summaries are V3 node summaries built bottom-up (leaf, parent, item
root).  Inputs observe retrieval policy, source fingerprints, prompt versions,
and a bounded input scope.  Current summaries skip without an API call when
their source/prompt state is current.  Only accepted/candidate LLM summaries
are searchable in `zotero_paragraphs_v3__sum_node`; extractive summaries are
displayable fallbacks but never summary-index candidates.

DeepSeek is the standard hosted summarization provider: `LLM_CHEAP` uses
`deepseek-v4-flash`; `LLM_STANDARD` and `LLM_REVIEW` use `deepseek-v4-pro`
unless an explicitly configured compatible provider overrides them.  Summary
workers may run concurrently by item because each item has an independent DB
transaction; rate limits and failed work are recorded in the artifact ledger.
Cloud-excluded material is never sent.

## Retrieval

Normal retrieval is hybrid vector plus FTS search, filtered by retrieval policy.
Hierarchical retrieval routes through eligible V3 node summaries and fuses
leaf-restricted, same-item, and global direct searches with RRF.  Direct global
search remains mandatory as the availability fallback if routing or the summary
collection is unavailable.  Returned results preserve source/provenance fields
for verification.

`search_mode="case"` remains a direct full-text/chunk retrieval strategy with
context; it does not depend on a generated case database.

## State, references, and retired features

`artifact_processing_status` and its event ledger are the authoritative process
state.  Valid artifact types are extraction, structure, summary, summary index,
references, and embeddings.  `pending`, `running`, `success`, `empty`,
`degraded`, `blocked`, `failed`, `stale`, and `excluded` remain distinct.

Citation/reference extraction, work identity, review queues, and GROBID
enrichment remain separate from ingestion commits.  GROBID is optional
review-only enrichment for qualifying English scholarly PDFs and never blocks
canonical indexing.

The structured-case generation/database/MCP/UI pipeline has been retired.  Its
historical evaluation material (including Gold Case QA) is retained only as
evaluation evidence; it is not a production quality gate.  The legacy item and
section summary stores are rollback artifacts.  A few transitional read/write
compatibility paths remain while the V3 summary/index/search final audit is in
progress; they are not an authorized normal V3 generation route and must be
removed before physical retirement.  Physical removal requires the Phase 6
write-zero audit, backup verification, and final V3 summary/index/search pass.

## Phase 6 rollback guard

Legacy deletion is deliberately separate from cutover.  Before deletion,
`scripts/audit_legacy_retirement.py` must compare a baseline and a post-
maintenance snapshot (no legacy table/collection/FTS/manifest mutations) and
verify the existing Chroma rollback snapshot.  `data/backups/legacy-retirement-
20260727/RESTORE.md` records the restore procedure.  Physical deletion requires
an explicit operator decision after those checks; it is never performed by a
normal ingestion or maintenance run.
