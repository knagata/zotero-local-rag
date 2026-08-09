from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest.mock import MagicMock, call, patch


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import index_from_zotero as module  # noqa: E402
import index_batch  # noqa: E402


class AttachmentBatchAtomicityTests(unittest.TestCase):
    def _snapshot(self):
        return {
            "ids": ["old-1"],
            "documents": ["old document"],
            "metadatas": [{"attachmentKey": "A1", "itemKey": "I1"}],
            "embeddings": [[0.1, 0.2]],
        }

    def test_second_subbatch_failure_restores_old_vector_and_lexical_rows(self):
        col = MagicMock()
        col.get.return_value = self._snapshot()
        col.upsert.side_effect = [None, RuntimeError("second batch failed"), None]

        # ``index_from_zotero`` imported the lexical functions under other
        # names (``upsert_lexical_chunks``), so patching them on ``index_batch``
        # left the forward path writing into the real data/lexical_v3.sqlite3
        # on every run of this test. It was invisible until the data-directory
        # guard refused the connection (2026-08-09). Both bindings are patched
        # now: the one the rollback uses and the one the forward path uses.
        with patch.object(index_batch, "delete_by_attachment_keys") as lexical_delete, \
                patch.object(index_batch, "upsert_chunks") as lexical_upsert, \
                patch.object(module, "upsert_lexical_chunks"), \
                patch.object(module, "delete_lexical_attachments") as initial_lexical_delete, \
                patch.object(module, "relieve_memory_pressure"):
            with self.assertRaisesRegex(RuntimeError, "second batch failed"):
                module._replace_attachment_batch(
                    col,
                    attachment_keys={"A1"},
                    ids=["new-1", "new-2"],
                    documents=["new one", "new two"],
                    metadatas=[
                        {"attachmentKey": "A1", "itemKey": "I1"},
                        {"attachmentKey": "A1", "itemKey": "I1"},
                    ],
                    expected_ids={"A1": {"new-1", "new-2"}},
                    attachment_item_keys={"A1": "I1"},
                    subbatch_size=1,
                    show_progress=False,
                    label="test",
                    context_label="test flush",
                    strict_lexical=True,
                )

        self.assertEqual(col.delete.call_count, 2)
        restore = col.upsert.call_args_list[-1].kwargs
        self.assertEqual(restore["ids"], ["old-1"])
        self.assertEqual(restore["documents"], ["old document"])
        self.assertEqual(restore["embeddings"], [[0.1, 0.2]])
        initial_lexical_delete.assert_called_once_with(["A1"])
        lexical_delete.assert_called_once_with(["A1"])
        self.assertEqual(
            lexical_upsert.call_args_list[-1],
            call(
                ["old-1"], ["old document"],
                [{"attachmentKey": "A1", "itemKey": "I1"}],
            ),
        )

    def test_snapshot_failure_happens_before_any_delete(self):
        col = MagicMock()
        col.get.side_effect = RuntimeError("snapshot unavailable")
        with patch.object(index_batch, "delete_by_attachment_keys") as lexical_delete:
            with self.assertRaisesRegex(RuntimeError, "snapshot unavailable"):
                module._replace_attachment_batch(
                    col,
                    attachment_keys={"A1"}, ids=[], documents=[], metadatas=[],
                    expected_ids={"A1": set()}, attachment_item_keys={"A1": "I1"},
                    subbatch_size=1, show_progress=False, label="test",
                    context_label="test", strict_lexical=True,
                )
        col.delete.assert_not_called()
        lexical_delete.assert_not_called()


class PendingBatchCommitTests(unittest.TestCase):
    def test_flush_commits_once_and_clears_all_pending_state(self):
        pending = index_batch.PendingIndexBatch.empty()
        pending.ids.extend(["new-1", "new-1"])
        pending.documents.extend(["older duplicate", "winning document"])
        pending.metadatas.extend([
            {"attachmentKey": "A1", "itemKey": "I1"},
            {"attachmentKey": "A1", "itemKey": "I1"},
        ])
        pending.manifest_updates["A1"] = {"mtime": 1}
        pending.extraction_statuses["A1"] = ("I1", "success", {})
        pending.delete_attachment_keys.add("A1")
        pending.source_types["A1"] = "pdf"
        pending.item_keys["A1"] = "I1"
        manifest = {"files": {}, "inflight_attachments": []}
        files_manifest = manifest["files"]

        with patch.object(module, "_replace_attachment_batch") as replace, \
                patch.object(module, "assert_code_unchanged"), \
                patch.object(module, "save_manifest"), \
                patch.object(module, "mark_artifact_status"), \
                patch.object(module, "_finalize_v3_pending"):
            outcome = module._flush_pending_index_batch(
                MagicMock(), pending,
                manifest=manifest,
                files_manifest=files_manifest,
                run_code_fingerprint="code",
                show_progress=False,
                label="test",
                context_label="test flush",
            )

        self.assertEqual(replace.call_args.kwargs["ids"], ["new-1"])
        self.assertEqual(replace.call_args.kwargs["documents"], ["winning document"])
        self.assertEqual(files_manifest["A1"], {"mtime": 1})
        self.assertEqual(outcome.updated_pdf, 1)
        self.assertEqual(outcome.last_written_id, "new-1")
        self.assertEqual(outcome.committed_item_keys, frozenset({"I1"}))
        self.assertEqual(manifest["inflight_attachments"], [])
        self.assertEqual(pending.ids, [])
        self.assertEqual(pending.manifest_updates, {})
    def test_inconsistent_snapshot_refuses_the_destructive_boundary(self):
        col = MagicMock()
        col.get.return_value = {
            "ids": ["old-1"], "documents": [], "metadatas": [], "embeddings": [],
        }
        with self.assertRaisesRegex(RuntimeError, "snapshot is inconsistent"):
            module._replace_attachment_batch(
                col,
                attachment_keys={"A1"}, ids=[], documents=[], metadatas=[],
                expected_ids={"A1": set()}, attachment_item_keys={"A1": "I1"},
                subbatch_size=1, show_progress=False, label="test",
                context_label="test", strict_lexical=True,
            )
        col.delete.assert_not_called()


if __name__ == "__main__":
    unittest.main()
