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
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

spec = importlib.util.spec_from_file_location(
    "verify_zotero_reconciliation", ROOT / "scripts" / "verify_zotero_reconciliation.py")
MODULE = importlib.util.module_from_spec(spec)
spec.loader.exec_module(MODULE)

from zotero_source_localapi import ZoteroLocalAPI  # noqa: E402


def _attachment(key, *, item_type="attachment", content_type="application/pdf",
                filename="doc.pdf", link_mode="imported_file", parent="ITEM"):
    return {
        "key": key,
        "data": {
            "key": key, "itemType": item_type, "contentType": content_type,
            "filename": filename, "linkMode": link_mode, "parentItem": parent,
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


if __name__ == "__main__":
    unittest.main()
