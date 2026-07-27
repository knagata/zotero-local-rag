from __future__ import annotations

from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from src.manifest import load_manifest, save_manifest
from src.reocr_adoption import adopt_prepared_reocr, canonicalize_prepared_blocks


class FakeCollection:
    def __init__(self):
        self.rows = {}

    def delete(self, *, where):
        attachment = where["attachmentKey"]
        self.rows = {
            key: row for key, row in self.rows.items()
            if row[1].get("attachmentKey") != attachment
        }

    def upsert(self, *, ids, documents, metadatas):
        for chunk_id, document, metadata in zip(ids, documents, metadatas):
            self.rows[chunk_id] = (document, dict(metadata))


def old_chunks():
    return [{
        "id": "ATT:p1:old", "text": "Old readable source text. " * 8,
        "metadata": {
            "itemKey": "ITEM", "attachmentKey": "ATT", "source_type": "pdf",
            "page": 1, "locator": "p1", "lang": "en", "title": "Test",
        },
    }]


def prepared():
    return {
        "engine": "docling", "version": "2",
        "quality": {"total_pages": 1, "page_coverage": 1},
        "blocks": [{
            "text": "New readable OCR source text. " * 8,
            "metadata": {"page": 1, "locator": "p1", "structure_path": ["Chapter 1"]},
        }],
    }


class ReocrAdoptionTests(unittest.TestCase):
    def test_canonical_ids_are_new_and_content_versioned(self):
        first = canonicalize_prepared_blocks("ITEM", "ATT", prepared())
        second_payload = prepared()
        second_payload["blocks"][0]["text"] += "changed"
        second = canonicalize_prepared_blocks("ITEM", "ATT", second_payload)
        self.assertTrue(first[0]["id"].startswith("ATT:v3ocr:"))
        self.assertNotEqual(first[0]["id"], second[0]["id"])
        self.assertEqual(first[0]["metadata"]["extraction_engine"], "docling")

    def test_adoption_updates_v3_indexes_manifest_and_stales_summary(self):
        collection = FakeCollection()
        rows = old_chunks()
        collection.upsert(
            ids=[rows[0]["id"]], documents=[rows[0]["text"]],
            metadatas=[rows[0]["metadata"]],
        )
        statuses = []
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest_path = root / "manifest_v3.json"
            save_manifest(manifest_path, {"version": 1, "files": {"ATT": {"title": "Test"}}, "notes": {}})
            with (
                patch("src.reocr_adoption.replace_document_structure") as replace_structure,
                patch("src.reocr_adoption.delete_document_node_summaries"),
                patch("src.reocr_adoption.invalidate_item_summaries"),
            ):
                result = adopt_prepared_reocr(
                    item_key="ITEM", attachment_key="ATT", prepared=prepared(),
                    collection=collection, old_item_chunks=rows,
                    manifest_path=manifest_path, lexical_path=root / "lexical.sqlite3",
                    status_writer=lambda *args, **kwargs: statuses.append((args, kwargs)),
                )
            self.assertTrue(result["canonical_data_modified"])
            self.assertNotIn("ATT:p1:old", collection.rows)
            self.assertTrue(any(key.startswith("ATT:v3ocr:") for key in collection.rows))
            self.assertTrue(load_manifest(manifest_path)["files"]["ATT"]["quality"]["reocr_adopted"])
            replace_structure.assert_called_once()
            self.assertTrue(any(args[1:3] == ("summary", "stale") for args, _ in statuses))

    def test_failure_restores_old_search_rows_and_manifest(self):
        collection = FakeCollection()
        rows = old_chunks()
        collection.upsert(ids=[rows[0]["id"]], documents=[rows[0]["text"]], metadatas=[rows[0]["metadata"]])
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest_path = root / "manifest_v3.json"
            original = {"version": 1, "files": {"ATT": {"title": "Before"}}, "notes": {}}
            save_manifest(manifest_path, original)
            with patch("src.reocr_adoption.replace_document_structure", side_effect=RuntimeError("db failed")):
                with self.assertRaises(RuntimeError):
                    adopt_prepared_reocr(
                        item_key="ITEM", attachment_key="ATT", prepared=prepared(),
                        collection=collection, old_item_chunks=rows,
                        manifest_path=manifest_path, lexical_path=root / "lexical.sqlite3",
                        status_writer=lambda *args, **kwargs: None,
                    )
            self.assertEqual(set(collection.rows), {"ATT:p1:old"})
            self.assertEqual(load_manifest(manifest_path), original)


if __name__ == "__main__":
    unittest.main()
