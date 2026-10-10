# Search snapshots during indexing

`SPEC.md` defines the publication contract. The canonical indexing paths stay
unchanged. Query entry points apply `src/search_snapshot.py` before importing
database readers, so Chroma, manifest, FTS and relations resolve to one saved
copy. Graph and MCP use that copy's read lock. `environment_with_saved_dotenv`
restores canonical paths when a query process launches maintenance. Local Zotero
import indexing also supplies this cleaned environment explicitly.

Create the first copy only when the existing indexing manifest is query-ready:

```bash
.venv/bin/python scripts/publish_search_snapshot.py --initialize --restart-services
```

This holds the writer lock, closes canonical query services, copies the stores,
checks SQLite storage and native collection counts/search/close/reopen, then
publishes the pointer and restores services. Failed unpublished partial copies
are removed in a finally block; process kills or power loss can still leave them.
A capacity check runs before copying. macOS uses APFS copy-on-write cloning for the HNSW files; SQLite
online backup includes WAL contents. The initial snapshot records whether the
full audit passed; a corpus with explicitly deferred PDFs remains an audit
failure even when the existing indexed corpus is physically query-ready.

Run normal differential ingestion while queries use the saved generation:

```bash
.venv/bin/python scripts/run_searchable_index.py --progress
.venv/bin/python scripts/run_db_audit.py
```

The admin library steps use the first command. Explicit caller feature overrides
remain authoritative; this wrapper must not re-enable hosted processing. The
audit runner refreshes an existing snapshot after all original checks pass and
after releasing its writer lock. Publication rejects an unvalidated manifest,
inflight/post-index work, overlapping paths and a failed full audit. Old copies
are pruned before and after publication, retaining the current generation, its
immediate predecessor, and generations with open files reported by lsof. Without
lsof no pruning occurs; inspection errors abort cleanup. Only completed copies
with matching generation certificates are eligible. Canonical databases, source
files, unknown directories and recovery backups are never pruned by this policy.

Python SQLite 3.53.1 reports the Chroma FTS5 trigram index as malformed when its
content includes NUL. The same result was reproduced in a brand-new native
Chroma control with one explicit vector and NUL-containing document. Snapshot
copy verification distinguishes only this exact error: all other SQLite checks
must pass, every stored internal FTS document must equal its embedding metadata,
and a fresh native control must reproduce the same error. The compatibility
finding is saved in the snapshot certificate; no source data or audit criterion
is changed. Separate lexical SQLite verification remains strict.

`tests/test_search_snapshot.py` verifies real isolated read/write clients,
failed publication, canonical child paths, caller hosted-feature overrides,
invalid certificates, NUL compatibility/content mismatch, and publication only
after a successful audit releases its lock.
