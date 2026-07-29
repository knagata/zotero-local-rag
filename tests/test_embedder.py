from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from src.embedder import (
    EmbedderConfig,
    embedder_config_payload,
    ensure_embedding_model,
    probe_embedding_dim,
    resolve_collection_name,
    resolve_embedder_settings,
)


class EmbedderSettingsTests(unittest.TestCase):
    def test_ensure_embedding_model_downloads_selected_profile(self):
        with patch("src.embedder.snapshot_download") as download:
            def materialize(*, repo_id, local_dir):
                target = Path(local_dir)
                target.mkdir(parents=True)
                (target / "config.json").write_text("{}", encoding="utf-8")
                self.assertEqual(repo_id, "BAAI/bge-m3")

            download.side_effect = materialize
            with tempfile.TemporaryDirectory() as directory:
                config = {"EMB_PROFILE": "bge"}
                model = ensure_embedding_model(config, Path(directory))
                self.assertTrue(Path(model).joinpath("config.json").is_file())
                download.assert_called_once()

    def test_ensure_embedding_model_reuses_existing_local_model(self):
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "data" / "models" / "bge-m3"
            target.mkdir(parents=True)
            (target / "config.json").write_text("{}", encoding="utf-8")
            with patch("src.embedder.snapshot_download") as download:
                model = ensure_embedding_model({"EMB_PROFILE": "bge"}, Path(directory))
            self.assertEqual(model, str(target))
            download.assert_not_called()

    def test_default_collection_name_is_the_canonical_v3_name_without_probing(self):
        def should_not_run(_texts):
            raise AssertionError("collection-name resolution must not probe embeddings")

        self.assertEqual(
            resolve_collection_name(should_not_run),
            "zotero_paragraphs_v3",
        )

    def test_removed_gemini_profile_fails_with_migration_guidance(self):
        with patch.dict(os.environ, {"EMB_PROFILE": "gemini"}, clear=True):
            with self.assertRaisesRegex(RuntimeError, "no longer supported"):
                resolve_embedder_settings(Path("."))

    def test_probe_accepts_numpy_like_embedding_rows(self):
        class Vector:
            def __len__(self):
                return 3

        self.assertEqual(probe_embedding_dim(lambda _texts: [Vector()]), 3)

    def test_config_records_normalization_and_embedding_fingerprint(self):
        cfg = EmbedderConfig("sentence_transformers", "remote/model", "cpu")
        payload = embedder_config_payload(cfg, 1024, "v3")
        self.assertTrue(payload["normalize_embeddings"])
        self.assertEqual(payload["embedding_dim"], 1024)
        self.assertTrue(payload["embedding_fingerprint"].startswith("sha256:"))


if __name__ == "__main__":
    unittest.main()
