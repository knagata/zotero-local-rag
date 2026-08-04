from __future__ import annotations

import argparse
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
            reocr_route={"target_engine": "local", "lang": "ja"},
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
