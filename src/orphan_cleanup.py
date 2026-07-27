# src/orphan_cleanup.py
"""Decide which ledger identities no longer correspond to anything in Zotero.

The status ledger (``artifact_processing_status``) and the canonical structure
tables are keyed by an *item key* that the ingestion pipeline derives as
``parentItemKey or attachmentKey``. Two things follow from that fallback, and
both were observed in the live library (2026-07-27):

**Deleted items leave their content behind.** ``purge_removed_items`` is only
called on a full-scope run, and recent operation has been almost entirely
partial-scope (``--item`` / ``--reocr-candidates`` / ``--source-type``), so
nothing has purged anything for weeks. One deleted item still had 1,897 chunks
in both Chroma and the FTS index, i.e. deleted material was still being
returned by search.

**Re-parenting silently changes an identity.** A PDF filed at the top level of
Zotero is tracked under its *attachment* key. When the user later files it
under a parent item -- an ordinary thing to do -- subsequent runs write status
under the real parent key and the old rows are stranded forever. These are not
deleted items: the attachment is alive and its content is correctly tracked
elsewhere, so purging by "not a live item key" would be the wrong verb. Only
the stale ledger rows should go.

Distinguishing the two is the whole job of this module, and it is kept pure so
that the decision can be tested without touching Zotero, Chroma or the DB.

Deriving the live key set correctly matters as much as the classification. It
must be built the same way the pipeline builds the keys it *writes*:

- every attachment contributes ``parentItemKey or attachmentKey``
- every note contributes its ``parentItemKey``

Omitting either side is dangerous rather than merely incomplete. The call site
in ``index_from_zotero.py`` previously used ``{a.parentItemKey for a in
attachments if a.parentItemKey}``, which drops parentless attachments *and*
every note-only item. Once ``purge_removed_items`` was extended to delete
document structures and status rows, that narrow set meant a full-scope run
would have purged live note-only items -- e.g. ``FSIXT5VE``, an item with 42
note chunks and no file attachment, which exists in Zotero and is not deleted.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


@dataclass
class OrphanReport:
    """Classification of ledger keys against what Zotero currently holds."""

    #: Keys with nothing in Zotero: purge content and bookkeeping alike.
    deleted: list[str] = field(default_factory=list)
    #: Live attachments tracked under an old identity: drop bookkeeping only.
    reparented: list[str] = field(default_factory=list)
    #: Keys that correspond to something Zotero still has.
    live: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, list[str]]:
        return {"deleted": list(self.deleted), "reparented": list(self.reparented),
                "live": list(self.live)}


def _key_of(value: Any) -> str:
    return str(value or "").strip()


def live_item_keys(
    attachments: Iterable[Any], notes: Iterable[Mapping[str, Any]] = (),
) -> set[str]:
    """Every item key the pipeline would currently write status under.

    Mirrors ``scope_item_key`` for attachments and ``index_notes``' ``itemKey``
    for notes. Anything this misses becomes a purge candidate, so it must not
    be narrowed for convenience.
    """
    keys: set[str] = set()
    for attachment in attachments:
        parent = _key_of(getattr(attachment, "parentItemKey", None))
        own = _key_of(getattr(attachment, "attachmentKey", None))
        resolved = parent or own
        if resolved:
            keys.add(resolved)
    for note in notes:
        if not isinstance(note, Mapping):
            continue
        parent = _key_of(note.get("parentItemKey"))
        if parent:
            keys.add(parent)
    return keys


def attachment_parents(attachments: Iterable[Any]) -> dict[str, str]:
    """Map each live attachment key to its current parent item key ('' if none)."""
    mapping: dict[str, str] = {}
    for attachment in attachments:
        own = _key_of(getattr(attachment, "attachmentKey", None))
        if own:
            mapping[own] = _key_of(getattr(attachment, "parentItemKey", None))
    return mapping


def classify_ledger_keys(
    ledger_keys: Iterable[str],
    *,
    live_keys: set[str],
    attachment_parents_map: Mapping[str, str],
) -> OrphanReport:
    """Split ledger keys into deleted, re-parented, and live.

    A key that is itself a live attachment key but is not a live *item* key
    means that attachment has since acquired a parent: its content is tracked
    under the parent now, so only the stranded rows are stale. Zotero item and
    attachment keys share one namespace and are unique, so this test cannot
    confuse an attachment with some other item.
    """
    report = OrphanReport()
    for raw in ledger_keys:
        key = _key_of(raw)
        if not key:
            continue
        if key in live_keys:
            report.live.append(key)
        elif attachment_parents_map.get(key):
            report.reparented.append(key)
        else:
            report.deleted.append(key)
    report.deleted.sort()
    report.reparented.sort()
    report.live.sort()
    return report


def stale_identity_keys(attachment_key: str, parent_item_key: str) -> list[str]:
    """Ledger identities superseded once this attachment gained a parent.

    Called during ingestion, where both keys are in hand, so the stranded row
    can be cleared at the moment it becomes stale instead of accumulating until
    someone runs a cleanup. Returns nothing for a genuinely top-level
    attachment, whose attachment key *is* its legitimate ledger identity.
    """
    attachment_key = _key_of(attachment_key)
    parent_item_key = _key_of(parent_item_key)
    if not attachment_key or not parent_item_key:
        return []
    if attachment_key == parent_item_key:
        return []
    return [attachment_key]


def note_only_item(chunks: Sequence[Mapping[str, Any]]) -> bool:
    """Whether an item's indexed chunks are all Zotero notes.

    Such an item has no canonical document structure by design (notes are
    searchable annotations but are excluded from the document tree), so the
    structure builder finding "no chunks" is the expected outcome rather than a
    blockage. Recording it as ``blocked`` leaves a permanent entry in the
    unresolved list, which the maintenance summary then reports forever.
    """
    if not chunks:
        return False
    for chunk in chunks:
        metadata = chunk.get("metadata") if isinstance(chunk, Mapping) else None
        source_type = str((metadata or {}).get("source_type") or "")
        if source_type != "note":
            return False
    return True


def stale_manifest_keys(
    manifest_files: Mapping[str, Any], indexed_attachment_keys: Iterable[str],
) -> list[str]:
    """Manifest rows whose file is gone *and* whose chunks are already purged.

    The manifest was the one bookkeeping store the orphan purge never touched,
    so an attachment deleted from Zotero left a row behind permanently. That is
    not cosmetic: the cutover audit compares the manifest's attachment set with
    Chroma's, so a single stale row failed ``manifest_chroma_attachment_mismatch``
    on *every* item audited from then on -- a global gate stuck open on a
    document that no longer exists (C66HF59V, 2026-07-27).

    Both conditions are required. A missing file alone may mean a detached
    external drive, and a row is retired only once its content is demonstrably
    absent from the index too.
    """
    indexed = set(indexed_attachment_keys)
    stale = []
    for key, entry in (manifest_files or {}).items():
        if key in indexed or not isinstance(entry, Mapping):
            continue
        path = str(entry.get("pdf_path") or entry.get("path") or "")
        if path and not Path(path).exists():
            stale.append(str(key))
    return sorted(stale)


__all__ = [
    "OrphanReport", "attachment_parents", "classify_ledger_keys", "live_item_keys",
    "note_only_item", "stale_identity_keys", "stale_manifest_keys",
]
