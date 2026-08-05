"""A linked_url attachment must be skipped explicitly, not silently.

A linked_url attachment is a pointer to an external URL -- Zotero never stores
a local file for it, so local path resolution and the Local API file download
fallback were guaranteed to fail every time, and the failure was swallowed by
a bare `except Exception: continue` printed only behind DEBUG_ZOTERO_LOCALAPI.
One such attachment (75QYJJYK) was the single Zotero-eligible attachment
absent from the manifest with no record anywhere of why (2026-07-28).
"""
from __future__ import annotations

import asyncio
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from zotero_source_localapi import ZoteroLocalAPI  # noqa: E402


def _attachment(key, *, link_mode="imported_file", content_type="application/pdf",
                 filename="doc.pdf", parent_item="ITEM", tags=None):
    return {
        "key": key,
        "data": {
            "key": key, "itemType": "attachment", "linkMode": link_mode,
            "contentType": content_type, "filename": filename, "parentItem": parent_item,
            "tags": tags or [],
        },
    }


class LinkedUrlSkipTests(unittest.TestCase):
    def _run(self, raw_attachments, *, resolve_returns=None, download_side_effect=None):
        api = ZoteroLocalAPI()
        yielded = []
        warnings = []

        async def go():
            with patch.object(api, "list_pdf_attachments", return_value=raw_attachments), \
                 patch.object(api, "get_item", return_value={"key": "ITEM", "data": {"key": "ITEM", "title": "Parent"}}), \
                 patch.object(api, "resolve_pdf_path_from_attachment",
                               return_value=resolve_returns), \
                 patch.object(api, "fetch_attachment_file_to_cache",
                               side_effect=download_side_effect or Exception("no such file")), \
                 patch("sys.stderr") as stderr:
                async for attachment in api.iter_normalized_attachments(
                    zotero_data_dir="/tmp/zotero", pdf_cache_dir="/tmp/cache",
                ):
                    yielded.append(attachment)
                warnings.extend(
                    call.args[0] for call in stderr.write.call_args_list if call.args
                )

        asyncio.run(go())
        return yielded, "".join(warnings)

    def test_a_linked_url_attachment_is_skipped_without_attempting_resolution(self):
        raw = [_attachment("A1", link_mode="linked_url", content_type="text/html")]
        with patch.object(
            ZoteroLocalAPI, "resolve_pdf_path_from_attachment",
            side_effect=AssertionError("must not attempt to resolve a linked_url attachment"),
        ):
            yielded, warning_text = self._run(raw)
        self.assertEqual(yielded, [])
        self.assertIn("linked_url", warning_text)
        self.assertIn("A1", warning_text)

    def test_an_ordinary_attachment_is_unaffected(self):
        raw = [_attachment("A2", link_mode="imported_file")]
        yielded, _ = self._run(raw, resolve_returns="/tmp/zotero/storage/A2/doc.pdf")
        self.assertEqual([attachment.attachmentKey for attachment in yielded], ["A2"])

    def test_attachment_and_parent_tags_are_normalized(self):
        raw = [_attachment("A2", tags=[{"tag": " RAG:Exclude "}])]
        api = ZoteroLocalAPI()
        yielded = []

        async def go():
            with patch.object(api, "list_pdf_attachments", return_value=raw), \
                 patch.object(api, "get_item", return_value={
                     "key": "ITEM", "data": {
                         "key": "ITEM", "title": "Parent",
                         "tags": [{"tag": "RAG:Prefer-EPUB"}],
                     },
                 }), patch.object(
                     api, "resolve_pdf_path_from_attachment",
                     return_value="/tmp/zotero/storage/A2/doc.pdf",
                 ):
                async for attachment in api.iter_normalized_attachments(
                    zotero_data_dir="/tmp/zotero", pdf_cache_dir="/tmp/cache",
                ):
                    yielded.append(attachment)

        asyncio.run(go())
        self.assertEqual(yielded[0].tags, ("rag:exclude",))
        self.assertEqual(yielded[0].parentTags, ("rag:prefer-epub",))

    def test_a_genuine_resolution_failure_stops_incomplete_enumeration(self):
        # The other half of the same defect: a failed download for a non-
        # linked_url attachment used to be silent unless a debug flag was set.
        raw = [_attachment("A3", link_mode="imported_file")]
        with self.assertRaisesRegex(RuntimeError, "A3"):
            self._run(raw, resolve_returns=None)


class InventoryCompletenessTests(unittest.TestCase):
    def test_rebuild_inventory_requires_total_results(self):
        api = ZoteroLocalAPI()

        async def fake_get(_path, params=None, timeout=None):
            api._last_total_results = None
            return [_attachment("A1")]

        with patch.object(api, "_get_json", side_effect=fake_get):
            with self.assertRaisesRegex(RuntimeError, "Total-Results"):
                asyncio.run(api.list_pdf_attachments(limit=2, require_complete=True))

    def test_rebuild_inventory_rejects_short_partial_listing(self):
        api = ZoteroLocalAPI()

        async def fake_get(_path, params=None, timeout=None):
            api._last_total_results = 3
            # A transiently short first page used to be accepted as EOF.
            return [_attachment("A1")]

        with patch.object(api, "_get_json", side_effect=fake_get):
            with self.assertRaisesRegex(RuntimeError, "expected=3"):
                asyncio.run(api.list_pdf_attachments(limit=2, require_complete=True))

    def test_rebuild_inventory_accepts_exact_unique_total(self):
        api = ZoteroLocalAPI()

        async def fake_get(_path, params=None, timeout=None):
            api._last_total_results = 3
            start = int((params or {}).get("start") or 0)
            return (
                [_attachment("A1"), _attachment("A2")]
                if start == 0 else [_attachment("A3")]
            )

        with patch.object(api, "_get_json", side_effect=fake_get):
            rows = asyncio.run(
                api.list_pdf_attachments(limit=2, require_complete=True)
            )
        self.assertEqual([row["key"] for row in rows], ["A1", "A2", "A3"])


if __name__ == "__main__":
    unittest.main()
