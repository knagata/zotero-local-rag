from __future__ import annotations

import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import index_from_zotero as module  # noqa: E402


class StructuredV3IndexConfigTests(unittest.TestCase):
    def test_v3_defaults_are_isolated(self):
        env = os.environ.copy()
        env["INGEST_STRUCTURED_V3_ENABLE"] = "1"
        for name in ("MANIFEST_PATH", "CHROMA_COLLECTION", "LEXICAL_DB_PATH"):
            env.pop(name, None)
        result = subprocess.run(
            [sys.executable, "-c", (
                "import sys; sys.path.insert(0, 'src'); import index_from_zotero as m; "
                "import os; print(m.MANIFEST_PATH.name, m.CHROMA_COLLECTION_ENV, "
                "os.environ['LEXICAL_DB_PATH'].split('/')[-1])"
            )], cwd=ROOT, env=env, text=True, capture_output=True, check=True,
        )
        self.assertEqual(result.stdout.strip(), "manifest_v3.json zotero_paragraphs_v3 lexical_v3.sqlite3")

    def test_item_limit_keeps_all_attachments_of_selected_item(self):
        from index_from_zotero import _select_item_scope

        attachments = [
            SimpleNamespace(parentItemKey="A", attachmentKey="A1"),
            SimpleNamespace(parentItemKey="B", attachmentKey="B1"),
            SimpleNamespace(parentItemKey="A", attachmentKey="A2"),
        ]
        selected = _select_item_scope(attachments, None, 1)
        self.assertEqual([row.attachmentKey for row in selected], ["A1", "A2"])

    def test_exact_attachment_scope_does_not_expand_to_pdf_siblings(self):
        from index_from_zotero import _select_attachment_scope, _select_item_scope

        # The U5 queue holds one attachment per row. --item=PARENT alone
        # intentionally selects both rows; applying --attachment first must
        # keep the runner's repair to the one exact PDF.
        attachments = [
            SimpleNamespace(parentItemKey="PARENT", attachmentKey="PDF1"),
            SimpleNamespace(parentItemKey="PARENT", attachmentKey="PDF2"),
            SimpleNamespace(parentItemKey="OTHER", attachmentKey="PDF3"),
        ]
        scoped = _select_attachment_scope(attachments, ["PDF2"])
        selected = _select_item_scope(scoped, ["PARENT"], 0)
        self.assertEqual([row.attachmentKey for row in selected], ["PDF2"])

    def test_retryable_failed_requires_both_fields(self):
        with patch.object(module, "get_item_processing_status", return_value=[
            {"status": "failed", "retryable": 0},
            {"status": "success", "retryable": 1},
        ]):
            self.assertFalse(module._retryable_failed("ITEM"))
        with patch.object(module, "get_item_processing_status", return_value=[
            {"status": "failed", "retryable": 1},
        ]):
            self.assertTrue(module._retryable_failed("ITEM"))

    def test_current_mistral_candidate_requires_same_attachment_and_source_stat(self):
        status = [{
            "artifact_type": "extraction", "attachment_key": "A1",
            "status": "blocked", "reason_code": "awaiting_mistral_ocr_batch",
            "processor_version": module.MISTRAL_TOC_QUEUE_PROCESSOR_VERSION,
            "counts": {"source_mtime": 12.5, "source_size": 99},
        }]
        with patch.object(module, "get_item_processing_status", return_value=status):
            self.assertTrue(module._is_current_mistral_toc_candidate(
                "ITEM", "A1", mtime=12.5, size=99,
            ))
            self.assertFalse(module._is_current_mistral_toc_candidate(
                "ITEM", "A1", mtime=13.0, size=99,
            ))
            self.assertFalse(module._is_current_mistral_toc_candidate(
                "ITEM", "A2", mtime=12.5, size=99,
            ))

    def test_old_mistral_candidate_is_re_evaluated_after_gate_change(self):
        status = [{
            "artifact_type": "extraction", "attachment_key": "A1",
            "status": "blocked", "reason_code": "awaiting_mistral_ocr_batch",
            "processor_version": "mistral-ocr-queue-v1",
            "counts": {"source_mtime": 12.5, "source_size": 99},
        }]
        with patch.object(module, "get_item_processing_status", return_value=status):
            self.assertFalse(module._is_current_mistral_toc_candidate(
                "ITEM", "A1", mtime=12.5, size=99,
            ))

    def test_mistral_candidate_is_not_retryable_failed(self):
        with patch.object(module, "get_item_processing_status", return_value=[{
            "status": "blocked", "reason_code": "awaiting_mistral_ocr_batch",
            "retryable": 0,
        }]):
            self.assertFalse(module._retryable_failed("ITEM"))
        with patch.object(module, "get_item_processing_status", return_value=[
            {"status": "degraded", "retryable": 1},
        ]):
            self.assertTrue(module._retryable_failed("ITEM"))

    def test_source_document_chunks_excludes_notes(self):
        rows = [
            {"id": "a", "metadata": {"source_type": "pdf"}},
            {"id": "n", "metadata": {"source_type": "note"}},
        ]
        self.assertEqual([row["id"] for row in module._source_document_chunks(rows)], ["a"])

    def test_finalize_pending_is_a_durable_recovery_checkpoint(self):
        with tempfile.TemporaryDirectory() as directory:
            manifest_path = Path(directory) / "manifest.json"
            manifest = {"files": {}, "post_index_pending": ["OLD"]}
            with (
                patch.object(module, "MANIFEST_PATH", manifest_path),
                patch.object(module, "_finalize_v3_item") as finalize,
            ):
                module._finalize_v3_pending(
                    manifest, ["NEW"], collection_name="zotero_paragraphs_v3",
                )
            self.assertEqual([call.args[0] for call in finalize.call_args_list], ["NEW", "OLD"])
            self.assertEqual(manifest["post_index_pending"], [])
            self.assertEqual(__import__("json").loads(manifest_path.read_text())["post_index_pending"], [])

    def test_v3_rebuild_does_not_remove_chroma_directory(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            chroma = root / "chroma"
            chroma.mkdir()
            legacy_marker = chroma / "legacy-kept"
            legacy_marker.write_text("yes")
            manifest = root / "manifest_v3.json"
            manifest.write_text("{}")
            pipeline_config = chroma / "embedder_config_v3.json"
            pipeline_config.write_text("{}")
            lexical = root / "lexical_v3.sqlite3"
            lexical.write_text("x")
            deleted = []
            fake_client = SimpleNamespace(
                list_collections=lambda: [
                    SimpleNamespace(name="zotero_paragraphs_v3"),
                    SimpleNamespace(name="zotero_paragraphs_v3__sum_node"),
                    SimpleNamespace(name="zotero_paragraphs"),
                ],
                delete_collection=deleted.append,
            )
            with (
                patch.object(module, "STRUCTURED_V3_ENABLE", True),
                patch.object(module, "CHROMA_DIR", chroma),
                patch.object(module, "MANIFEST_PATH", manifest),
                patch.object(module, "V3_PIPELINE_CONFIG_PATH", pipeline_config),
                patch.dict(os.environ, {"LEXICAL_DB_PATH": str(lexical)}),
                patch("chromadb.PersistentClient", return_value=fake_client),
                patch.object(module, "reset_ingestion_derived_state") as reset_derived,
            ):
                module._reset_rebuild_target()
            self.assertEqual(deleted, [
                "zotero_paragraphs_v3",
                "zotero_paragraphs_v3__sum_node",
            ])
            reset_derived.assert_called_once_with()
            self.assertTrue(legacy_marker.exists())
            self.assertFalse(manifest.exists())
            self.assertFalse(pipeline_config.exists())
            self.assertFalse(lexical.exists())

    def test_v3_rebuild_delete_failure_keeps_sidecar_state(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            chroma = root / "chroma"
            chroma.mkdir()
            manifest = root / "manifest_v3.json"
            manifest.write_text("{}")
            pipeline_config = chroma / "embedder_config_v3.json"
            pipeline_config.write_text("{}")
            lexical = root / "lexical_v3.sqlite3"
            lexical.write_text("x")

            def fail_delete(_name):
                raise RuntimeError("delete failed")

            fake_client = SimpleNamespace(
                list_collections=lambda: [SimpleNamespace(name="zotero_paragraphs_v3")],
                delete_collection=fail_delete,
            )
            with (
                patch.object(module, "STRUCTURED_V3_ENABLE", True),
                patch.object(module, "CHROMA_DIR", chroma),
                patch.object(module, "MANIFEST_PATH", manifest),
                patch.object(module, "V3_PIPELINE_CONFIG_PATH", pipeline_config),
                patch.dict(os.environ, {"LEXICAL_DB_PATH": str(lexical)}),
                patch("chromadb.PersistentClient", return_value=fake_client),
                patch.object(module, "reset_ingestion_derived_state"),
                self.assertRaisesRegex(RuntimeError, "delete failed"),
            ):
                module._reset_rebuild_target()
            self.assertTrue(manifest.exists())
            self.assertTrue(pipeline_config.exists())
            self.assertTrue(lexical.exists())

    def test_rebuild_rejects_scoped_options(self):
        for scoped in (
            ["--rebuild", "--item", "ITEM"],
            ["--rebuild", "--attachment", "ATTACH"],
            ["--rebuild", "--limit", "1"],
            ["--rebuild", "--source-type", "pdf"],
            ["--rebuild", "--retry-failed"],
        ):
            with patch.object(sys, "argv", ["index_from_zotero.py", *scoped]):
                with self.assertRaises(SystemExit):
                    module.parse_args()


if __name__ == "__main__":
    unittest.main()
