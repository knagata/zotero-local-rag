from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from scripts.export_ocr_cache import export, verify
from src.ocr_cache import MISTRAL_REQUEST_CONTRACT, cache_root, entry_path, store_result


class ExportOcrCacheTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.data_dir = Path(self._tmp.name) / "data"
        self.dest = Path(self._tmp.name) / "export"
        self.digest = "a" * 64
        store_result(
            self.data_dir, engine="mistral_ocr", model="mistral-ocr-latest",
            contract_version=MISTRAL_REQUEST_CONTRACT, digest=self.digest,
            result={"pages": [{"index": 0, "markdown": "x"}]},
            attachment_key="ATT", title="A Title",
        )

    def tearDown(self):
        self._tmp.cleanup()

    def _entry(self, digest: str | None = None) -> Path:
        return entry_path(
            self.data_dir, engine="mistral_ocr", model="mistral-ocr-latest",
            contract_version=MISTRAL_REQUEST_CONTRACT, digest=digest or self.digest,
        )

    def test_export_produces_a_droppable_ocr_cache_tree_and_index(self):
        self.assertEqual(export(cache_root(self.data_dir), self.dest, dry_run=False), 0)
        copied = self.dest / "ocr_cache" / "mistral_ocr" / "mistral-ocr-latest__req1" / f"{self.digest}.json"
        self.assertTrue(copied.exists())
        self.assertTrue((self.dest / "README.md").exists())
        index = json.loads((self.dest / "INDEX.json").read_text(encoding="utf-8"))
        self.assertEqual(index["entry_count"], 1)
        self.assertEqual(index["entries"][0]["attachment_key"], "ATT")
        self.assertEqual(index["entries"][0]["pages"], 1)

    def test_dry_run_writes_nothing(self):
        self.assertEqual(export(cache_root(self.data_dir), self.dest, dry_run=True), 0)
        self.assertFalse(self.dest.exists())

    def test_incomplete_and_stray_files_are_excluded(self):
        # A live archive can hold a half-written entry or an OS stray file; the
        # copied set must contain only entries checked to be complete.
        root = cache_root(self.data_dir)
        broken = root / "mistral_ocr" / "mistral-ocr-latest__req1" / f"{'b' * 64}.json"
        broken.write_text("{ not json", encoding="utf-8")
        (root / "mistral_ocr" / "mistral-ocr-latest__req1" / ".DS_Store").write_text("x")
        partial = root / "mistral_ocr" / "mistral-ocr-latest__req1" / f"{'c' * 64}.json"
        partial.write_text(json.dumps({"schema_version": "ocr-cache-1"}), encoding="utf-8")

        export(cache_root(self.data_dir), self.dest, dry_run=False)
        copied = sorted(p.name for p in (self.dest / "ocr_cache").rglob("*.json"))
        self.assertEqual(copied, [f"{self.digest}.json"])

    def test_entry_filed_under_a_mismatched_digest_is_excluded(self):
        # The filename is the lookup key, so an entry whose recorded identity
        # disagrees would be found under an identity it does not claim.
        path = self._entry()
        payload = json.loads(path.read_text(encoding="utf-8"))
        payload["source_sha256"] = "d" * 64
        path.write_text(json.dumps(payload), encoding="utf-8")
        export(cache_root(self.data_dir), self.dest, dry_run=False)
        self.assertEqual(list((self.dest / "ocr_cache").rglob("*.json")), [])

    def test_verify_passes_on_a_fresh_export(self):
        export(cache_root(self.data_dir), self.dest, dry_run=False)
        self.assertEqual(verify(self.dest), 0)

    def test_verify_fails_when_a_copy_was_damaged_in_transit(self):
        export(cache_root(self.data_dir), self.dest, dry_run=False)
        copied = next((self.dest / "ocr_cache").rglob("*.json"))
        copied.write_text("{ truncated", encoding="utf-8")
        self.assertEqual(verify(self.dest), 1)

    def test_verify_fails_when_entries_are_missing_against_the_index(self):
        export(cache_root(self.data_dir), self.dest, dry_run=False)
        next((self.dest / "ocr_cache").rglob("*.json")).unlink()
        self.assertEqual(verify(self.dest), 1)

    def test_verify_reports_a_missing_directory(self):
        self.assertEqual(verify(self.dest), 1)


if __name__ == "__main__":
    unittest.main()
