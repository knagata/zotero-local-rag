"""Tests for the Zotero-as-truth reconciliation check.

Every existing check compares stores derived from Zotero against each other
(manifest vs Chroma, Chroma vs FTS); none compares against Zotero itself.
75QYJJYK -- a linked_url attachment whose resolution failure was silently
swallowed -- was the single Zotero-eligible attachment missing from the
manifest, and nothing but a manual audit noticed (2026-07-28).
"""
from __future__ import annotations

import asyncio
import importlib.util
import sys
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

spec = importlib.util.spec_from_file_location(
    "verify_zotero_reconciliation", ROOT / "scripts" / "verify_zotero_reconciliation.py")
MODULE = importlib.util.module_from_spec(spec)
spec.loader.exec_module(MODULE)

from zotero_source_localapi import ZoteroLocalAPI  # noqa: E402


def _attachment(key, *, item_type="attachment", content_type="application/pdf",
                filename="doc.pdf", link_mode="imported_file", parent="ITEM", tags=None):
    return {
        "key": key,
        "data": {
            "key": key, "itemType": item_type, "contentType": content_type,
            "filename": filename, "linkMode": link_mode, "parentItem": parent,
            "tags": tags or [],
        },
    }


class EligibleZoteroAttachmentsTests(unittest.TestCase):
    def _eligible(self, raw_attachments):
        api = ZoteroLocalAPI()
        with patch.object(api, "list_pdf_attachments", return_value=raw_attachments):
            return asyncio.run(MODULE.eligible_zotero_attachments(api))

    def test_a_pdf_attachment_is_eligible(self):
        eligible = self._eligible([_attachment("A1")])
        self.assertIn("A1", eligible)
        self.assertEqual(eligible["A1"]["source_type"], "pdf")

    def test_a_linked_url_attachment_is_still_eligible(self):
        # Eligibility is about the *type* Zotero says it is, independent of
        # whether the ingestion path can actually resolve a local file for it
        # -- that distinction is exactly what this check exists to surface.
        eligible = self._eligible([
            _attachment("A2", content_type="text/html", link_mode="linked_url"),
        ])
        self.assertIn("A2", eligible)
        self.assertEqual(eligible["A2"]["linkMode"], "linked_url")

    def test_reconciliation_requires_a_complete_zotero_inventory(self):
        api = ZoteroLocalAPI()
        with patch.object(api, "list_pdf_attachments", return_value=[]) as listed:
            asyncio.run(MODULE.eligible_zotero_attachments(api))
        listed.assert_called_once_with(require_complete=True)

    def test_a_non_attachment_item_is_excluded(self):
        raw = [_attachment("A3", item_type="journalArticle")]
        self.assertEqual(self._eligible(raw), {})

    def test_an_unrecognised_content_type_is_excluded(self):
        raw = [_attachment("A4", content_type="application/zip", filename="data.zip")]
        self.assertEqual(self._eligible(raw), {})


class ReconciliationReportTests(unittest.TestCase):
    def test_an_eligible_attachment_absent_from_the_manifest_is_reported(self):
        args = type("Args", (), {"output": None})()
        api = ZoteroLocalAPI()
        raw = [_attachment("MISSING"), _attachment("PRESENT")]
        with patch.object(api, "list_pdf_attachments", return_value=raw), \
             patch.object(api, "get_item", AsyncMock(side_effect=lambda key: next(
                 row for row in raw if row["key"] == key))), \
             patch.object(MODULE, "load_manifest", return_value={"files": {"PRESENT": {}}}):
            exit_code = asyncio.run(MODULE.main_async(args, api=api))
        self.assertEqual(exit_code, 2)

    def test_full_reconciliation_passes(self):
        args = type("Args", (), {"output": None})()
        api = ZoteroLocalAPI()
        raw = [_attachment("A1")]
        with patch.object(api, "list_pdf_attachments", return_value=raw), \
             patch.object(MODULE, "load_manifest", return_value={"files": {"A1": {}}}):
            exit_code = asyncio.run(MODULE.main_async(args, api=api))
        self.assertEqual(exit_code, 0)

    def test_linked_url_is_reported_but_does_not_block_a_local_rebuild(self):
        args = type("Args", (), {"output": None})()
        api = ZoteroLocalAPI()
        raw = [_attachment("LINK", link_mode="linked_url")]
        with patch.object(api, "list_pdf_attachments", return_value=raw), \
             patch.object(MODULE, "load_manifest", return_value={"files": {}}):
            exit_code = asyncio.run(MODULE.main_async(args, api=api))
        self.assertEqual(exit_code, 0)

    def test_attachment_local_rag_exclusion_does_not_block_reconciliation(self):
        args = type("Args", (), {"output": None})()
        api = ZoteroLocalAPI()
        raw = [_attachment("EXCLUDED")]
        precise = _attachment("EXCLUDED", tags=[{"tag": "RAG:Exclude"}])
        with patch.object(api, "list_pdf_attachments", return_value=raw), \
             patch.object(api, "get_item", AsyncMock(return_value=precise)) as fetched, \
             patch.object(MODULE, "load_manifest", return_value={"files": {}}):
            exit_code = asyncio.run(MODULE.main_async(args, api=api))
        self.assertEqual(exit_code, 0)
        fetched.assert_awaited_once_with("EXCLUDED")

    def test_parent_rag_exclusion_does_not_exclude_attachment(self):
        # Reconciliation reads attachment-local tags only, matching ingestion.
        args = type("Args", (), {"output": None})()
        api = ZoteroLocalAPI()
        raw = [_attachment("REQUIRED")]
        with patch.object(api, "list_pdf_attachments", return_value=raw), \
             patch.object(api, "get_item", AsyncMock(return_value=raw[0])), \
             patch.object(MODULE, "load_manifest", return_value={"files": {}}):
            exit_code = asyncio.run(MODULE.main_async(args, api=api))
        self.assertEqual(exit_code, 2)

    def test_preferred_pdf_with_committed_epub_sibling_does_not_block(self):
        args = type("Args", (), {"output": None})()
        api = ZoteroLocalAPI()
        raw = [
            _attachment("PDF", parent="PARENT"),
            _attachment("EPUB", content_type="application/epub+zip", filename="doc.epub",
                        parent="PARENT"),
        ]
        parent = {"key": "PARENT", "data": {"key": "PARENT", "tags": [
            {"tag": "rag:prefer-epub"},
        ]}}
        with patch.object(api, "list_pdf_attachments", return_value=raw), \
             patch.object(api, "get_item", AsyncMock(side_effect=[raw[0], parent])), \
             patch.object(MODULE, "load_manifest", return_value={"files": {"EPUB": {}}}):
            exit_code = asyncio.run(MODULE.main_async(args, api=api))
        self.assertEqual(exit_code, 0)


if __name__ == "__main__":
    unittest.main()
