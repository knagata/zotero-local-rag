from __future__ import annotations

import argparse
import asyncio
import io
from dataclasses import replace
from unittest.mock import AsyncMock, patch
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import index_from_zotero as module  # noqa: E402
from zotero_source_localapi import ZoteroAttachment  # noqa: E402


def attachment(path: Path, *, source_type: str = "pdf", content_type: str | None = None):
    return ZoteroAttachment(
        attachmentKey="A1", parentItemKey="I1", title="Title", year=None,
        creators=None, pdf_path=str(path), source_type=source_type,
        contentType=content_type,
    )


class AttachmentSourceResolverTests(unittest.TestCase):
    def test_snapshot_directory_prefers_index_html(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "z-other.html").write_text("other", encoding="utf-8")
            (root / "index.html").write_text("index", encoding="utf-8")
            resolved = module._resolve_attachment_source(attachment(root, source_type="html"))
        self.assertEqual(resolved.path.name, "index.html")
        self.assertEqual(resolved.source_type, "html")

    def test_snapshot_directory_uses_stable_shallow_fallback(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "b.html").write_text("b", encoding="utf-8")
            (root / "a.htm").write_text("a", encoding="utf-8")
            resolved = module._resolve_attachment_source(attachment(root, source_type="html"))
        self.assertEqual(resolved.path.name, "a.htm")

    def test_missing_or_html_free_source_is_not_processable(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "image.png").write_bytes(b"png")
            self.assertIsNone(module._resolve_attachment_source(attachment(root)))
            self.assertIsNone(module._resolve_attachment_source(attachment(root / "missing.pdf")))


class AwaitingSourceSyncTests(unittest.TestCase):
    def test_ready_sources_continue_and_inventory_is_preserved(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "ready.pdf"
            path.write_bytes(b"%PDF-ready")
            ready = attachment(path)
            missing = replace(attachment(path), attachmentKey="MISSING", pdf_path="")
            empty_path = Path(directory) / "empty.pdf"
            empty_path.touch()
            empty = replace(attachment(empty_path), attachmentKey="EMPTY")
            rows = [missing, ready, empty]
            with patch.object(sys, "argv", ["index_from_zotero"]):
                args = module.parse_args()
            api = AsyncMock()
            api.list_normalized_attachments.return_value = rows
            with patch("sys.stderr", new_callable=io.StringIO) as log:
                result = asyncio.run(module._discover_attachments(
                    args, api=api, manifest={}, files_manifest={},
                    reocr_routes={}, show_progress=False,
                ))
            self.assertEqual(result.attachments, [ready])
            self.assertEqual(result.inventory_attachments, rows)
            self.assertIn("MISSING", log.getvalue())
            self.assertIn("EMPTY", log.getvalue())
            self.assertIn("再実行", log.getvalue())
            self.assertTrue(api.list_normalized_attachments.call_args.kwargs["allow_missing_files"])
            empty_path.write_bytes(b"%PDF-now-synced")
            self.assertEqual(module._ready_source_attachments([empty], rebuild=False), [empty])

    def test_unreadable_source_is_deferred_without_aborting_ready_sources(self):
        rows = [attachment(Path("unreadable.pdf"))]
        with patch.object(module, "_resolve_attachment_source", side_effect=PermissionError()), \
                patch("sys.stderr", new_callable=io.StringIO) as log:
            self.assertEqual(module._ready_source_attachments(rows, rebuild=False), [])
        self.assertIn("同期待ち", log.getvalue())

    def test_rebuild_rejects_empty_sources_before_index_writes(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "empty.pdf"
            path.touch()
            with self.assertRaisesRegex(RuntimeError, "A1"):
                module._ready_source_attachments([attachment(path)], rebuild=True)

    def test_missing_path_does_not_resolve_to_working_directory(self):
        row = replace(attachment(Path("unused")), pdf_path="")
        self.assertIsNone(module._resolve_attachment_source(row))


class ReparseDecisionTests(unittest.TestCase):
    def _args(self, **values):
        return argparse.Namespace(
            use_docling=values.get("use_docling", False),
            reparse_corrupted=values.get("reparse_corrupted", False),
        )

    def test_explicit_routes_take_priority(self):
        mistral = module._reparse_decision(
            self._args(), source_type="pdf", previous=None,
            reocr_route={"target_engine": "mistral_ocr", "lang": "ja"},
        )
        self.assertTrue(mistral.force_mistral)
        self.assertFalse(mistral.force_ndlocr)
        japanese = module._reparse_decision(
            self._args(), source_type="pdf", previous=None,
            reocr_route={"target_engine": "local", "language": "ja"},
        )
        self.assertTrue(japanese.force_ndlocr)

    def test_docling_is_not_repeated_for_an_existing_docling_result(self):
        decision = module._reparse_decision(
            self._args(use_docling=True), source_type="pdf",
            previous={"quality": {"parser": "docling"}}, reocr_route=None,
        )
        self.assertFalse(decision.force_docling)


if __name__ == "__main__":
    unittest.main()
