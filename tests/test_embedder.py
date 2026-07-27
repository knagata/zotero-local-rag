from __future__ import annotations

import os
import unittest
from pathlib import Path
from unittest.mock import patch

from src.embedder import EmbedderConfig, embedder_config_payload, probe_embedding_dim, resolve_embedder_settings


class EmbedderSettingsTests(unittest.TestCase):
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
