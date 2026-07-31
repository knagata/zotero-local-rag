from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from src.ocr_cache import (
    MISTRAL_REQUEST_CONTRACT, contract_slug, entry_path, load_entry, load_result,
    load_result_any_model, source_digest, store_result,
)


class OcrCacheTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.data_dir = Path(self._tmp.name)
        self.pdf = self.data_dir / "sample.pdf"
        self.pdf.write_bytes(b"%PDF-1.4 fake bytes for hashing")
        self.digest = source_digest(self.pdf)
        self.result = {"pages": [{"index": 0, "markdown": "hello"}], "model": "m"}

    def tearDown(self):
        self._tmp.cleanup()

    def _store(self, **overrides):
        kwargs = dict(
            engine="mistral_ocr", model="mistral-ocr-latest",
            contract_version=MISTRAL_REQUEST_CONTRACT, digest=self.digest,
            result=self.result, attachment_key="ATT", title="A Title",
        )
        kwargs.update(overrides)
        return store_result(self.data_dir, **kwargs)

    def _load(self, **overrides):
        kwargs = dict(
            engine="mistral_ocr", model="mistral-ocr-latest",
            contract_version=MISTRAL_REQUEST_CONTRACT, digest=self.digest,
        )
        kwargs.update(overrides)
        return load_result(self.data_dir, **kwargs)

    def test_roundtrip_returns_the_raw_response_unmodified(self):
        self._store()
        self.assertEqual(self._load(), self.result)

    def test_identity_is_the_bytes_not_the_path_or_mtime(self):
        # Portability: the same document copied to another machine (new path,
        # new mtime from Zotero sync) must still hit. Only the bytes are keyed.
        self._store()
        moved = self.data_dir / "elsewhere" / "renamed.pdf"
        moved.parent.mkdir()
        moved.write_bytes(self.pdf.read_bytes())
        import os
        os.utime(moved, (0, 0))  # deliberately different mtime
        self.assertEqual(source_digest(moved), self.digest)
        self.assertEqual(self._load(), self.result)

    def test_changing_the_request_contract_misses_instead_of_serving_old_shape(self):
        # The safety property that makes "defer include_blocks" reversible: an
        # entry fetched under the old contract must never be read as if it
        # carried fields the new request would have added.
        self._store()
        self.assertIsNone(self._load(contract_version="req2"))

    def test_changing_the_model_misses(self):
        # Mistral's older OCR models return an empty blocks array rather than
        # an error, so the model is part of the response shape, not metadata.
        self._store()
        self.assertIsNone(self._load(model="mistral-ocr-4-0"))

    def test_different_bytes_miss(self):
        self._store()
        other = self.data_dir / "other.pdf"
        other.write_bytes(b"%PDF-1.4 completely different")
        self.assertIsNone(self._load(digest=source_digest(other)))

    def test_entry_is_self_describing_for_a_copied_directory(self):
        # A copied ocr_cache/ must be auditable without the manifest or the
        # batch job's work directory.
        self._store(batch_job_id="job-1", source_size=31, source_path=str(self.pdf))
        entry = load_entry(
            self.data_dir, engine="mistral_ocr", model="mistral-ocr-latest",
            contract_version=MISTRAL_REQUEST_CONTRACT, digest=self.digest,
        )
        for field in (
            "schema_version", "engine", "model", "request_contract_version",
            "source_sha256", "attachment_key", "title", "fetched_at", "batch_job_id",
        ):
            self.assertIn(field, entry)
        self.assertEqual(entry["attachment_key"], "ATT")
        self.assertEqual(entry["source_sha256"], self.digest)

    def test_entry_whose_recorded_digest_disagrees_is_not_served(self):
        # A hand-edited or half-migrated file must not be served as a match
        # just because it sits at the right filename.
        path = self._store()
        payload = json.loads(path.read_text(encoding="utf-8"))
        payload["source_sha256"] = "0" * 64
        path.write_text(json.dumps(payload), encoding="utf-8")
        self.assertIsNone(self._load())

    def test_damaged_entry_reads_as_a_miss_not_an_error(self):
        # Falling back to a refetch is correct and costs what it would have
        # cost anyway; raising would fail an ingest over a cache problem.
        path = self._store()
        path.write_text("{ this is not json", encoding="utf-8")
        self.assertIsNone(self._load())

    def test_store_is_atomic_leaving_no_partial_file(self):
        self._store()
        leftovers = list(entry_path(
            self.data_dir, engine="mistral_ocr", model="mistral-ocr-latest",
            contract_version=MISTRAL_REQUEST_CONTRACT, digest=self.digest,
        ).parent.glob("*.tmp"))
        self.assertEqual(leftovers, [])

    def test_contract_slug_is_filesystem_safe(self):
        self.assertEqual(contract_slug("mistral-ocr-4-0", "req1"), "mistral-ocr-4-0__req1")
        slug = contract_slug("weird/model name", "v1")
        self.assertNotIn("/", slug)
        self.assertNotIn(" ", slug)

    def test_cache_root_honours_the_env_override(self):
        import os
        from unittest import mock
        from src.ocr_cache import cache_root

        with mock.patch.dict(os.environ, {"OCR_CACHE_DIR": "/tmp/elsewhere"}):
            self.assertEqual(cache_root(self.data_dir), Path("/tmp/elsewhere"))

    def test_load_result_any_model_finds_an_entry_stored_under_a_different_model(self):
        # run_mistral_ocr_batch.py --model can archive under a model that
        # differs from this machine's MISTRAL_OCR_MODEL; a lookup for the
        # configured model alone must not be the only way to find the entry.
        self._store(model="mistral-ocr-4-0")
        self.assertIsNone(self._load(model="mistral-ocr-latest"))
        found = load_result_any_model(
            self.data_dir, engine="mistral_ocr",
            contract_version=MISTRAL_REQUEST_CONTRACT, digest=self.digest,
        )
        self.assertIsNotNone(found)
        model, result = found
        self.assertEqual(model, "mistral-ocr-4-0")
        self.assertEqual(result, self.result)

    def test_load_result_any_model_still_respects_the_request_contract(self):
        # The contract (not the model) governs response shape, so a match
        # under a different contract must still miss.
        self._store(model="mistral-ocr-4-0")
        found = load_result_any_model(
            self.data_dir, engine="mistral_ocr",
            contract_version="req2", digest=self.digest,
        )
        self.assertIsNone(found)

    def test_load_result_any_model_misses_cleanly_when_nothing_is_archived(self):
        found = load_result_any_model(
            self.data_dir, engine="mistral_ocr",
            contract_version=MISTRAL_REQUEST_CONTRACT, digest="0" * 64,
        )
        self.assertIsNone(found)


if __name__ == "__main__":
    unittest.main()
