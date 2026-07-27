from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from src.v3_runtime import (
    assert_code_unchanged, bind_manifest_pipeline, code_fingerprint,
    ensure_pipeline_config, pipeline_payload,
)


class V3RuntimeTests(unittest.TestCase):
    def _runtime(self, code: str = "sha256:code") -> dict:
        return pipeline_payload({
            "embedding_fingerprint": "sha256:model", "embedding_dim": 1024,
            "normalize_embeddings": True, "model_state_fingerprint": "sha256:state",
        }, collection="zotero_paragraphs_v3", run_code_fingerprint=code)

    def test_existing_collection_is_adopted_once_then_validated(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.json"
            created, was_created = ensure_pipeline_config(path, self._runtime(), existing_chunk_count=12)
            self.assertTrue(was_created)
            self.assertEqual(created["adopted_existing_chunks"], 12)
            loaded, was_created = ensure_pipeline_config(path, self._runtime("sha256:other-code"), existing_chunk_count=99)
            self.assertFalse(was_created)
            self.assertEqual(loaded["pipeline_fingerprint"], created["pipeline_fingerprint"])

    def test_incompatible_embedder_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.json"
            ensure_pipeline_config(path, self._runtime(), existing_chunk_count=0)
            other = self._runtime()
            other["pipeline_fingerprint"] = "sha256:different"
            with self.assertRaisesRegex(RuntimeError, "refusing to mix"):
                ensure_pipeline_config(path, other, existing_chunk_count=0)

    def test_manifest_adoption_stamps_old_entries_but_future_missing_stamp_fails(self):
        manifest = {"files": {"A": {"mtime": 1}}}
        self.assertTrue(bind_manifest_pipeline(manifest, "sha256:p", adopt_existing=True))
        self.assertEqual(manifest["files"]["A"]["pipeline_fingerprint"], "sha256:p")
        with self.assertRaisesRegex(RuntimeError, "missing"):
            bind_manifest_pipeline({"files": {"A": {}}}, "sha256:p", adopt_existing=False)

    def test_code_change_is_detected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "module.py"
            path.write_text("one", encoding="utf-8")
            expected = code_fingerprint([path])
            assert_code_unchanged([path], expected)
            path.write_text("two", encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "changed"):
                assert_code_unchanged([path], expected)


if __name__ == "__main__":
    unittest.main()
