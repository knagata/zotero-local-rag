from __future__ import annotations

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import index_from_zotero as module  # noqa: E402
import index_batch  # noqa: E402


class FailFlushOnUnhealthyCollectionTests(unittest.TestCase):
    """Covers the post-flush Chroma health check (`_fail_flush_on_unhealthy_collection`).

    This is the general safety net wired in after both `_upsert_in_subbatches`
    call sites (periodic flush and final flush) in `main_async`: if
    `col.count()` fails right after a flush's writes land, the run must mark
    every attachment in that flush as blocked/retryable and stop immediately,
    rather than letting `save_manifest` record success against a collection
    that is actually broken.
    """

    def test_healthy_collection_is_a_noop(self):
        col = SimpleNamespace(count=lambda: 42)
        with patch.object(module, "mark_artifact_status") as mock_mark:
            module._fail_flush_on_unhealthy_collection(
                col, {"ATT1": "ITEM1"}, context_label="periodic flush",
            )
        mock_mark.assert_not_called()

    def test_unhealthy_collection_marks_attachments_blocked_and_aborts(self):
        def _boom():
            raise RuntimeError(
                "Error sending backfill request to compactor: "
                "Failed to apply logs to the hnsw segment writer"
            )

        col = SimpleNamespace(count=_boom)
        item_keys = {"ATT1": "ITEM1", "ATT2": "ITEM2"}

        with patch.object(module, "mark_artifact_status") as mock_mark:
            with self.assertRaises(SystemExit) as ctx:
                module._fail_flush_on_unhealthy_collection(
                    col, item_keys, context_label="final flush",
                )
            self.assertEqual(ctx.exception.code, 1)

        # Every attachment in the just-flushed batch must be marked blocked,
        # not just the first one.
        self.assertEqual(mock_mark.call_count, 2)
        calls_by_attachment = {
            call.kwargs.get("attachment_key"): call for call in mock_mark.call_args_list
        }
        self.assertEqual(set(calls_by_attachment), {"ATT1", "ATT2"})
        for attachment_key, item_key in item_keys.items():
            call = calls_by_attachment[attachment_key]
            self.assertEqual(call.args[0], item_key)
            self.assertEqual(call.args[1], "extraction")
            self.assertEqual(call.args[2], "blocked")
            self.assertEqual(call.kwargs.get("reason_code"), "chroma_collection_unhealthy")
            self.assertTrue(call.kwargs.get("retryable"))
            self.assertIn(
                "Error sending backfill request to compactor",
                call.kwargs.get("message", ""),
            )

    def test_source_around_both_upsert_call_sites_checks_health_before_save(self):
        """Static-shape guard: both flush call sites must run the health check
        before their corresponding save_manifest/manifest-commit step, so a
        broken collection can never be masked by a later manifest write.
        """
        source = Path(module.__file__).read_text()
        periodic_marker = "label=\"upsert batch\","
        final_marker = "label=\"final upsert\","

        # Writes and health verification now live inside one compensating unit
        # of work. Verify its internal ordering, then verify that both callers
        # only commit manifest state after that unit returns.
        batch_source = Path(index_batch.__file__).read_text()
        helper_start = batch_source.index("def replace_attachment_batch(")
        helper_source = batch_source[helper_start:]
        self.assertLess(
            helper_source.index("upsert_batch("),
            helper_source.index("health_check("),
        )

        flush_start = source.index("def _flush_pending_index_batch(")
        flush_end = source.index("\ndef _source_document_chunks", flush_start)
        flush_source = source[flush_start:flush_end]
        self.assertLess(
            flush_source.index("_replace_attachment_batch("),
            flush_source.index("files_manifest.update("),
            "replacement and health checks must finish before success state is committed",
        )

        # Both periodic and final call sites must route through the same unit.
        self.assertIn(periodic_marker, source)
        self.assertIn(final_marker, source)

    def test_attachment_commit_requires_exact_chroma_and_lexical_ids(self):
        class Collection:
            def get(self, **_kwargs):
                return {"ids": ["a", "b"]}

        with patch.object(
            module, "lexical_chunk_ids_by_attachment_keys",
            return_value={"ATT": {"a", "b"}},
        ):
            module._verify_written_attachments(Collection(), {"ATT": {"a", "b"}})
        with patch.object(
            module, "lexical_chunk_ids_by_attachment_keys",
            return_value={"ATT": {"a"}},
        ):
            with self.assertRaisesRegex(RuntimeError, "Lexical attachment ID mismatch"):
                module._verify_written_attachments(Collection(), {"ATT": {"a", "b"}})

    def test_hnsw_flush_runs_real_query_and_restores_threshold(self):
        class Collection:
            def __init__(self):
                self.configurations = []
                self.queried = False
                self.configuration = {"hnsw": {"sync_threshold": 37}}

            def modify(self, *, configuration):
                self.configurations.append(configuration)

            def upsert(self, **_kwargs):
                return None

            def delete(self, **_kwargs):
                return None

            def get(self, **_kwargs):
                return {"embeddings": [[0.1, 0.2]]}

            def query(self, **_kwargs):
                self.queried = True
                return {"ids": [["sample"]]}

        collection = Collection()
        module._flush_and_verify_hnsw(collection, "sample")
        self.assertTrue(collection.queried)
        self.assertEqual(collection.configurations[-1], {"hnsw": {"sync_threshold": 37}})

    def test_stale_hnsw_sentinels_are_removed_from_both_indexes(self):
        class Collection:
            def __init__(self):
                self.deleted = []

            def delete(self, *, ids):
                self.deleted.extend(ids)

        collection = Collection()
        sentinel = "__hnsw_flush_sentinel_abcd_0__"
        with patch.object(module, "list_chunk_ids", side_effect=[["real", sentinel], ["real"]]), patch.object(
            module, "delete_lexical_chunk_ids",
        ) as lexical_delete:
            removed = module._remove_stale_hnsw_sentinels(
                collection, collection_name="zotero_paragraphs_v3",
            )
        self.assertEqual(removed, [sentinel])
        self.assertEqual(collection.deleted, [sentinel])
        lexical_delete.assert_called_once_with([sentinel])


if __name__ == "__main__":
    unittest.main()
