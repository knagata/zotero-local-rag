#!/usr/bin/env python3
"""Purge index and bookkeeping left behind by items deleted from Zotero.

Why this exists as its own command rather than only inside ingestion:
``purge_removed_items`` runs only on a *full-scope* ingest, and recent
operation has been almost entirely partial-scope (``--item``,
``--reocr-candidates``, ``--source-type``), so nothing had purged anything for
weeks. A deleted item was found still holding 1,897 chunks in both Chroma and
the FTS index -- deleted material was still being returned by search
(2026-07-27).

Safety, in order of importance:

1. **Dry-run by default.** ``--apply`` is required to delete anything.
2. **Every candidate is confirmed against Zotero individually.** Being absent
   from the library listing is treated as a hint, not proof; the item is only
   purged when a direct lookup says it is gone. A lookup that fails for any
   other reason (network, server down) leaves the item alone -- fail-closed,
   because a transient error must never be read as "deleted".
3. **The live set includes notes.** An item whose only content is notes never
   appears among attachments, so deriving "live" from attachments alone would
   classify it as deleted. One such item exists in this library.
4. **The manifest is purged too.** It was the one store left untouched, and a
   stale row there fails the cutover audit's global gate for every item.
5. **Re-parented attachments are not purged.** A top-level PDF is tracked under
   its attachment key; once filed under a parent it is tracked under the
   parent. Its old identity looks like a dead item key but its content is alive,
   so only the stranded bookkeeping rows are retired.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import sqlite3
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

#: Guards the ledger purge below, mirroring index_from_zotero's stale-deletion
#: guard: a removal this large indicates a short Zotero enumeration, not a
#: shrunken library.
LEDGER_PURGE_MAX_RATIO = 0.05
LEDGER_PURGE_MIN_KEYS = 10
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from src.env_utils import load_dotenv_native  # noqa: E402

load_dotenv_native(ROOT)

from src import db_relations  # noqa: E402
from src.chunk_store import active_collection_name  # noqa: E402
from src.chunk_store import list_attachment_keys  # noqa: E402
from src.manifest import load_manifest, save_manifest  # noqa: E402
from src.orphan_cleanup import (  # noqa: E402
    attachment_parents, classify_ledger_keys, live_item_keys, stale_manifest_keys,
)
from src.v3_data_plane import chroma_dir, lexical_path  # noqa: E402
from src.zotero_source_localapi import ZoteroLocalAPI  # noqa: E402

CHROMA_DIR = chroma_dir(ROOT)


def _ledger_keys() -> set[str]:
    """Every item key that bookkeeping or canonical structure refers to."""
    conn = sqlite3.connect(f"file:{db_relations.DB_PATH}?mode=ro", uri=True)
    try:
        keys: set[str] = set()
        for table in ("artifact_processing_status", "document_structures"):
            try:
                keys |= {row[0] for row in conn.execute(f"SELECT DISTINCT item_key FROM {table}")}
            except sqlite3.Error:
                pass
        return {str(k).strip() for k in keys if str(k or "").strip()}
    finally:
        conn.close()


async def _confirm_deleted(api: ZoteroLocalAPI, keys: list[str]) -> tuple[list[str], list[str]]:
    """Split candidates into (confirmed gone, unconfirmed).

    A direct lookup is the only evidence accepted for deletion. Anything that
    still resolves, or that fails to resolve for an unclear reason, is left
    alone: an ambiguous answer must never authorise a delete.
    """
    gone: list[str] = []
    unconfirmed: list[str] = []
    for key in keys:
        try:
            await api.get_item(key)
            unconfirmed.append(key)  # still present after all
        except Exception as exc:
            if "404" in str(exc) or "Not Found" in str(exc):
                gone.append(key)
            else:
                unconfirmed.append(key)
    return gone, unconfirmed


def _count_content(item_keys: list[str]) -> dict[str, dict[str, int]]:
    """Chunks each candidate still holds in Chroma and the FTS index."""
    import chromadb

    counts: dict[str, dict[str, int]] = {}
    collection = None
    try:
        name = active_collection_name(chroma_dir=CHROMA_DIR)
        collection = chromadb.PersistentClient(path=str(CHROMA_DIR)).get_collection(name)
    except Exception as exc:
        print(f"[WARN] Chroma unavailable: {exc}", file=sys.stderr)
    lexical_db = lexical_path(ROOT)
    for key in item_keys:
        chroma_n = 0
        if collection is not None:
            try:
                chroma_n = len(collection.get(where={"itemKey": key}, include=[]).get("ids") or [])
            except Exception:
                chroma_n = -1
        fts_n = 0
        try:
            conn = sqlite3.connect(f"file:{lexical_db}?mode=ro", uri=True)
            fts_n = conn.execute(
                "SELECT COUNT(*) FROM chunks_fts WHERE item_key = ?", (key,)
            ).fetchone()[0]
            conn.close()
        except Exception:
            fts_n = -1
        counts[key] = {"chroma": chroma_n, "fts": fts_n}
    return counts


def _purge_content(item_keys: list[str]) -> dict[str, int]:
    """Delete a deleted item's chunks from Chroma and the FTS index."""
    import chromadb

    from src.lexical_index import delete_by_chunk_ids

    removed = {"chroma": 0, "fts": 0}
    name = active_collection_name(chroma_dir=CHROMA_DIR)
    collection = chromadb.PersistentClient(path=str(CHROMA_DIR)).get_collection(name)
    lexical_db = lexical_path(ROOT)
    for key in item_keys:
        got = collection.get(where={"itemKey": key}, include=[])
        ids = got.get("ids") or []
        if ids:
            for start in range(0, len(ids), 500):
                collection.delete(ids=ids[start:start + 500])
            removed["chroma"] += len(ids)
            delete_by_chunk_ids(ids, path=lexical_db)
        # Note chunks carry the parent item key but no attachment key, so the
        # id-based delete above covers them too.
        conn = sqlite3.connect(lexical_db)
        try:
            removed["fts"] += conn.execute(
                "DELETE FROM chunks_fts WHERE item_key = ?", (key,)
            ).rowcount or 0
            conn.commit()
        finally:
            conn.close()
    client = getattr(collection, "_chroma_client", None)
    if client is not None:
        try:
            client.close()
        except Exception:
            pass
    return removed


def _purge_attachments(attachment_keys: list[str]) -> dict[str, int]:
    """Delete a deleted attachment's chunks from Chroma and the FTS index.

    Keyed on the attachment rather than the item because that is the identity
    an attachment deleted from Zotero still carries in the index. The FTS rows
    are removed by chunk id, in the same pass: a Chroma-only delete leaves the
    text lexically searchable, which is how deleted material keeps answering
    keyword queries after it has visibly gone from vector search.
    """
    import chromadb

    from src.lexical_index import delete_by_chunk_ids

    removed = {"chroma": 0, "fts": 0}
    name = active_collection_name(chroma_dir=CHROMA_DIR)
    collection = chromadb.PersistentClient(path=str(CHROMA_DIR)).get_collection(name)
    lexical_db = lexical_path(ROOT)
    for key in attachment_keys:
        got = collection.get(where={"attachmentKey": key}, include=[])
        ids = got.get("ids") or []
        if not ids:
            continue
        for start in range(0, len(ids), 500):
            collection.delete(ids=ids[start:start + 500])
        removed["chroma"] += len(ids)
        removed["fts"] += delete_by_chunk_ids(ids, path=lexical_db) or 0
    return removed


async def main_async(args: argparse.Namespace) -> int:
    api = ZoteroLocalAPI()
    attachments = await api.list_normalized_attachments(
        zotero_data_dir=os.environ.get("ZOTERO_DATA_DIR"),
        pdf_cache_dir=str(ROOT / "data" / "pdf_cache"), collection_key=None,
    )
    try:
        notes = await api.list_notes(collection_key=None)
    except Exception as exc:
        print(f"[ERROR] Could not enumerate notes: {exc}\n"
              "Refusing to purge on a partial view of the library.", file=sys.stderr)
        return 1

    manifest_path = ROOT / "data" / "manifest_v3.json"
    live = live_item_keys(attachments, notes)
    parents = attachment_parents(attachments)
    report = classify_ledger_keys(_ledger_keys(), live_keys=live,
                                  attachment_parents_map=parents)

    # Candidates must come from what is *indexed*, not only from what the
    # ledger happens to know. An attachment deleted from Zotero that still
    # holds chunks has no ledger rows to be found by, and is not a stale
    # manifest row either (stale requires the chunks to be gone already), so
    # nothing proposed it for removal: two such attachments were serving 1,286
    # chunks of deleted material (2026-07-28). Every candidate is still
    # confirmed against Zotero individually before anything is deleted.
    live_attachments = {
        str(getattr(attachment, "attachmentKey", "")) for attachment in attachments
    }
    live_attachments.discard("")
    indexed_attachments = set(list_attachment_keys(collection_name=active_collection_name()))
    indexed_attachments |= set(load_manifest(manifest_path).get("files") or {})
    orphan_attachments = sorted(indexed_attachments - live_attachments - live)

    gone, unconfirmed = await _confirm_deleted(api, report.deleted)
    gone_attachments, unconfirmed_attachments = await _confirm_deleted(api, orphan_attachments)
    counts = _count_content(gone + unconfirmed + report.reparented)

    manifest = load_manifest(manifest_path)
    stale_rows = stale_manifest_keys(
        manifest.get("files") or {},
        list_attachment_keys(collection_name=active_collection_name()),
    )

    payload = {
        "dry_run": not args.apply,
        "live_item_keys": len(live),
        "confirmed_deleted": [{"item_key": k, **counts.get(k, {})} for k in gone],
        "unconfirmed_skipped": [{"item_key": k, **counts.get(k, {})} for k in unconfirmed],
        "deleted_attachments_still_indexed": gone_attachments,
        "attachments_unconfirmed_skipped": unconfirmed_attachments,
        "stale_manifest_rows": [
            {"attachment_key": k,
             "title": ((manifest.get("files") or {}).get(k) or {}).get("title", "")}
            for k in stale_rows
        ],
        "reparented_ledger_only": [
            {"item_key": k, "now_under": parents.get(k, ""), **counts.get(k, {})}
            for k in report.reparented
        ],
    }

    if args.apply:
        if gone:
            payload["content_removed"] = _purge_content(gone)
            # purge_removed_items(live) deletes every ledger key absent from
            # `live` -- computed from the same Zotero enumeration that P1-11
            # showed can come back short -- with no per-item confirmation
            # against Zotero at all (2026-07-28, found in code review). This
            # script's whole design is to confirm every deletion individually,
            # so a wholesale removal here is refused for the same reason
            # index_from_zotero refuses one: it is evidence the listing came
            # back short, not that the library shrank.
            candidate_removal = db_relations.ledger_keys_pending_removal(live)
            removal_limit = max(
                LEDGER_PURGE_MIN_KEYS, int(len(candidate_removal | live) * LEDGER_PURGE_MAX_RATIO),
            )
            if len(candidate_removal) > removal_limit:
                print(
                    f"[ERROR] Refusing to purge ledger rows for {len(candidate_removal)} items: "
                    f"that is more than {removal_limit} of {len(live)} live items, which indicates "
                    "the Zotero listing came back short, not that the library shrank. Nothing was "
                    "purged. Re-run once Zotero responds fully.",
                    file=sys.stderr,
                )
                payload["ledger_removed"] = {}
                payload["ledger_purge_refused"] = len(candidate_removal)
            else:
                payload["ledger_removed"] = db_relations.purge_removed_items(live)
        for key in report.reparented:
            db_relations.drop_stale_identity_rows(key, parents.get(key, ""))
        payload["reparented_rows_retired"] = len(report.reparented)
        if gone_attachments:
            payload["attachment_content_removed"] = _purge_attachments(gone_attachments)
            files = manifest.get("files") or {}
            for key in gone_attachments:
                files.pop(key, None)
            save_manifest(manifest_path, manifest)
        if stale_rows:
            files = manifest.get("files") or {}
            for key in stale_rows:
                files.pop(key, None)
            save_manifest(manifest_path, manifest)
        payload["manifest_rows_removed"] = len(stale_rows) if args.apply else 0

    print(json.dumps(payload, ensure_ascii=False, indent=2))
    if not args.apply and (gone or report.reparented or stale_rows or gone_attachments):
        print("\n[DRY-RUN] Nothing was deleted. Re-run with --apply to purge.",
              file=sys.stderr)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--apply", action="store_true",
        help="Actually delete. Without it the command only reports what it would do.",
    )
    return asyncio.run(main_async(parser.parse_args()))


if __name__ == "__main__":
    raise SystemExit(main())
