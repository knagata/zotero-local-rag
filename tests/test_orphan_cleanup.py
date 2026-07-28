"""Tests for orphan classification (dev-notes 79 follow-up, 2026-07-27).

These encode the two live cases that motivated the module, so a regression that
confuses them fails here:

- ``RTVZVXT8`` -- deleted from Zotero, 1,897 chunks still in Chroma and FTS.
- ``AJSX4LFZ`` -- alive, but tracked under its attachment key until the user
  filed it under parent ``Q56RQ6H6``; only the stranded rows are stale.

plus the safety property that made the purge dangerous in the first place:
note-only items must survive, because the live key set is what purging trusts.
"""
from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from orphan_cleanup import (  # noqa: E402
    attachment_parents, classify_ledger_keys, live_item_keys, note_only_item,
    stale_identity_keys, stale_manifest_keys,
)


class _Att:
    def __init__(self, attachment_key, parent_item_key=None):
        self.attachmentKey = attachment_key
        self.parentItemKey = parent_item_key


class LiveItemKeysTests(unittest.TestCase):
    def test_attachment_with_parent_contributes_the_parent(self):
        self.assertEqual(live_item_keys([_Att("ATT1", "ITEM1")]), {"ITEM1"})

    def test_parentless_attachment_contributes_its_own_key(self):
        # This is what the pipeline writes as scope_item_key, so purging must
        # see it as live or it would delete a top-level PDF's bookkeeping.
        self.assertEqual(live_item_keys([_Att("ATT1", None)]), {"ATT1"})

    def test_note_only_item_is_live(self):
        # FSIXT5VE: 42 note chunks, no file attachment, present in Zotero. The
        # previous call site derived live keys from attachments alone and would
        # have purged its structure and status rows.
        keys = live_item_keys([_Att("ATT1", "ITEM1")], [{"parentItemKey": "NOTEONLY"}])
        self.assertEqual(keys, {"ITEM1", "NOTEONLY"})

    def test_malformed_entries_are_ignored(self):
        keys = live_item_keys([_Att("", ""), _Att("ATT1", "ITEM1")], [{}, None, "x"])
        self.assertEqual(keys, {"ITEM1"})


class ClassifyLedgerKeysTests(unittest.TestCase):
    def setUp(self):
        self.attachments = [_Att("ATT_LIVE", "ITEM_LIVE"), _Att("AJSX4LFZ", "Q56RQ6H6")]
        self.live = live_item_keys(self.attachments)
        self.parents = attachment_parents(self.attachments)

    def _classify(self, keys):
        return classify_ledger_keys(
            keys, live_keys=self.live, attachment_parents_map=self.parents,
        )

    def test_deleted_item_is_marked_for_purge(self):
        report = self._classify(["RTVZVXT8"])
        self.assertEqual(report.deleted, ["RTVZVXT8"])
        self.assertEqual(report.reparented, [])

    def test_reparented_attachment_is_not_treated_as_deleted(self):
        # Purging by "not a live item key" alone would delete the content of a
        # file that is alive and correctly indexed under its parent.
        report = self._classify(["AJSX4LFZ"])
        self.assertEqual(report.reparented, ["AJSX4LFZ"])
        self.assertEqual(report.deleted, [])

    def test_live_key_is_left_alone(self):
        report = self._classify(["ITEM_LIVE"])
        self.assertEqual(report.live, ["ITEM_LIVE"])
        self.assertEqual(report.deleted, [])

    def test_mixed_input_is_split(self):
        report = self._classify(["ITEM_LIVE", "AJSX4LFZ", "RTVZVXT8", ""])
        self.assertEqual(report.as_dict(), {
            "deleted": ["RTVZVXT8"], "reparented": ["AJSX4LFZ"], "live": ["ITEM_LIVE"],
        })

    def test_top_level_attachment_stays_live_not_reparented(self):
        attachments = [_Att("TOPLEVEL", None)]
        report = classify_ledger_keys(
            ["TOPLEVEL"], live_keys=live_item_keys(attachments),
            attachment_parents_map=attachment_parents(attachments),
        )
        self.assertEqual(report.live, ["TOPLEVEL"])


class StaleIdentityKeysTests(unittest.TestCase):
    def test_reparented_attachment_yields_its_old_identity(self):
        self.assertEqual(stale_identity_keys("AJSX4LFZ", "Q56RQ6H6"), ["AJSX4LFZ"])

    def test_top_level_attachment_yields_nothing(self):
        # Its attachment key is the legitimate ledger identity.
        self.assertEqual(stale_identity_keys("AJSX4LFZ", ""), [])

    def test_self_parented_yields_nothing(self):
        self.assertEqual(stale_identity_keys("SAME", "SAME"), [])


class NoteOnlyItemTests(unittest.TestCase):
    def _chunk(self, source_type):
        return {"id": "x", "metadata": {"source_type": source_type}}

    def test_all_note_chunks(self):
        self.assertTrue(note_only_item([self._chunk("note"), self._chunk("note")]))

    def test_mixed_with_a_pdf_chunk_is_not_note_only(self):
        self.assertFalse(note_only_item([self._chunk("note"), self._chunk("pdf")]))

    def test_no_chunks_is_not_note_only(self):
        # A genuinely empty item is a different condition and must stay blocked.
        self.assertFalse(note_only_item([]))


class DropStaleIdentityRowsTests(unittest.TestCase):
    """The destructive half of the re-parenting fix (db_relations)."""

    def setUp(self) -> None:
        import tempfile
        from unittest.mock import patch
        import db_relations
        self.db_relations = db_relations
        self.tempdir = tempfile.TemporaryDirectory()
        self.patch = patch.object(
            db_relations, "DB_PATH", str(Path(self.tempdir.name) / "relations.db"))
        self.patch.start()
        db_relations._db_initialized = False
        db_relations.mark_artifact_status("AJSX4LFZ", "extraction", "blocked",
                                          reason_code="awaiting_mistral_ocr_batch")
        db_relations.mark_artifact_status("Q56RQ6H6", "extraction", "success")

    def tearDown(self) -> None:
        self.patch.stop()
        self.db_relations._db_initialized = False
        self.tempdir.cleanup()

    def test_superseded_identity_rows_are_removed(self):
        removed = self.db_relations.drop_stale_identity_rows("AJSX4LFZ", "Q56RQ6H6")
        self.assertEqual(removed, 1)
        self.assertEqual(self.db_relations.get_item_processing_status("AJSX4LFZ"), [])

    def test_current_identity_is_untouched(self):
        self.db_relations.drop_stale_identity_rows("AJSX4LFZ", "Q56RQ6H6")
        rows = self.db_relations.get_item_processing_status("Q56RQ6H6")
        self.assertEqual([r["status"] for r in rows], ["success"])

    def test_no_op_when_keys_match_or_are_blank(self):
        # A genuinely top-level attachment must keep its rows: its attachment
        # key is its legitimate ledger identity, not a superseded one.
        self.assertEqual(self.db_relations.drop_stale_identity_rows("AJSX4LFZ", ""), 0)
        self.assertEqual(self.db_relations.drop_stale_identity_rows("SAME", "SAME"), 0)
        self.assertEqual(len(self.db_relations.get_item_processing_status("AJSX4LFZ")), 1)

    def test_orphaned_nodes_and_summaries_are_retired_too(self):
        """document_nodes has no foreign key to document_structures.

        Only document_structures was retired here, so a re-parented attachment
        left its nodes -- and their summaries -- alive under the dead key,
        still feeding the summary index and leaf routing (2026-07-28, found in
        code review).
        """
        conn = self.db_relations.get_db_connection()
        conn.execute(
            "INSERT INTO document_nodes (node_id, item_key, node_type, depth, ordinal, "
            "source_kind, zone, summary_policy, retrieval_policy, citation_policy) "
            "VALUES ('dn:x', 'AJSX4LFZ', 'semantic_segment', 0, 0, "
            "'metadata_heading', 'body', 'include', 'normal', 'none')"
        )
        conn.execute(
            "INSERT INTO document_node_chunks (node_id, chunk_id, ordinal) VALUES ('dn:x', 'c1', 0)"
        )
        conn.commit()
        conn.close()

        self.db_relations.drop_stale_identity_rows("AJSX4LFZ", "Q56RQ6H6")

        conn = self.db_relations.get_db_connection()
        try:
            self.assertEqual(
                conn.execute("SELECT count(*) FROM document_nodes WHERE item_key='AJSX4LFZ'")
                .fetchone()[0], 0)
            self.assertEqual(
                conn.execute("SELECT count(*) FROM document_node_chunks WHERE node_id='dn:x'")
                .fetchone()[0], 0)
        finally:
            conn.close()



class PurgeRemovedItemsCandidateTests(unittest.TestCase):
    """Candidates must come from every table purge_removed_items cleans.

    Reading only item_citation_status meant an item that never went through
    citation mapping was invisible: five items deleted from Zotero kept their
    V3 structure and status rows while the purge reported success (2026-07-27).
    """

    def setUp(self) -> None:
        import tempfile
        from unittest.mock import patch
        import db_relations
        self.db_relations = db_relations
        self.tempdir = tempfile.TemporaryDirectory()
        self.patch = patch.object(
            db_relations, "DB_PATH", str(Path(self.tempdir.name) / "relations.db"))
        self.patch.start()
        db_relations._db_initialized = False

    def tearDown(self) -> None:
        self.patch.stop()
        self.db_relations._db_initialized = False
        self.tempdir.cleanup()

    def test_item_known_only_to_the_status_ledger_is_purged(self):
        # No citation row exists for it at all -- the case that was missed.
        self.db_relations.mark_artifact_status("GONE", "extraction", "success")
        self.db_relations.mark_artifact_status("ALIVE", "extraction", "success")
        counts = self.db_relations.purge_removed_items({"ALIVE"})
        self.assertEqual(counts["artifact_processing_status"], 1)
        self.assertEqual(self.db_relations.get_item_processing_status("GONE"), [])
        self.assertEqual(len(self.db_relations.get_item_processing_status("ALIVE")), 1)

    def test_live_items_are_never_purged(self):
        self.db_relations.mark_artifact_status("ALIVE", "structure", "success")
        counts = self.db_relations.purge_removed_items({"ALIVE"})
        self.assertEqual(counts["artifact_processing_status"], 0)
        self.assertEqual(len(self.db_relations.get_item_processing_status("ALIVE")), 1)

    def test_empty_live_set_purges_everything_known(self):
        # Guards the shape of the diff: an empty live set is a legitimate (if
        # drastic) request, not a no-op.
        self.db_relations.mark_artifact_status("A", "extraction", "success")
        counts = self.db_relations.purge_removed_items(set())
        self.assertEqual(counts["artifact_processing_status"], 1)

class StaleManifestKeyTests(unittest.TestCase):
    """The manifest is bookkeeping too, and it was never purged.

    A row for a deleted attachment made the cutover audit's global gate fail
    with ``manifest_chroma_attachment_mismatch`` for every item audited
    afterwards, because that gate compares the manifest's attachment set with
    Chroma's (C66HF59V, a partial PDF replaced by the full scan).
    """

    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.present = Path(self.tmp.name) / "here.pdf"
        self.present.write_bytes(b"%PDF-1.4")
        self.missing = str(Path(self.tmp.name) / "gone.pdf")

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def test_a_row_with_no_file_and_no_chunks_is_stale(self):
        files = {"GONE": {"pdf_path": self.missing, "title": "Deleted"}}
        self.assertEqual(stale_manifest_keys(files, []), ["GONE"])

    def test_a_row_still_holding_chunks_is_left_alone(self):
        # Content in the index is the stronger signal: purge that first.
        files = {"GONE": {"pdf_path": self.missing}}
        self.assertEqual(stale_manifest_keys(files, ["GONE"]), [])

    def test_a_row_whose_file_exists_is_left_alone(self):
        files = {"HERE": {"pdf_path": str(self.present)}}
        self.assertEqual(stale_manifest_keys(files, []), [])

    def test_a_row_with_no_path_is_left_alone(self):
        # Without a path there is no evidence either way, so do nothing --
        # an unreadable external drive must not look like a deletion.
        self.assertEqual(stale_manifest_keys({"X": {"title": "t"}}, []), [])

    def test_only_the_stale_rows_are_returned(self):
        files = {
            "HERE": {"pdf_path": str(self.present)},
            "GONE": {"pdf_path": self.missing},
            "INDEXED": {"pdf_path": self.missing},
        }
        self.assertEqual(stale_manifest_keys(files, ["INDEXED"]), ["GONE"])




class LedgerKeysPendingRemovalTests(unittest.TestCase):
    """Exposes what purge_removed_items would delete, without deleting it.

    purge_removed_items itself takes no confirmation and deletes everything in
    the diff, trusting the caller's enumeration to be complete -- the same risk
    P1-11 found in the stale-attachment deletion. A caller sizing the removal
    before committing to it needs this (2026-07-28, found in code review).
    """

    def setUp(self) -> None:
        import tempfile
        from unittest.mock import patch
        import db_relations
        self.db_relations = db_relations
        self.tempdir = tempfile.TemporaryDirectory()
        self.patch = patch.object(
            db_relations, "DB_PATH", str(Path(self.tempdir.name) / "relations.db"))
        self.patch.start()
        db_relations._db_initialized = False

    def tearDown(self) -> None:
        self.patch.stop()
        self.db_relations._db_initialized = False
        self.tempdir.cleanup()

    def test_matches_what_purge_removed_items_would_remove(self):
        self.db_relations.mark_artifact_status("GONE", "extraction", "success")
        self.db_relations.mark_artifact_status("ALIVE", "extraction", "success")
        self.assertEqual(
            self.db_relations.ledger_keys_pending_removal({"ALIVE"}), {"GONE"})

    def test_nothing_is_actually_deleted(self):
        self.db_relations.mark_artifact_status("GONE", "extraction", "success")
        self.db_relations.ledger_keys_pending_removal(set())
        self.assertEqual(len(self.db_relations.get_item_processing_status("GONE")), 1)


if __name__ == "__main__":
    unittest.main()
