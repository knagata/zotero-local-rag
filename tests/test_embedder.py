from __future__ import annotations

import os
import unittest
from pathlib import Path
from unittest.mock import patch

from src.embedder import resolve_embedder_settings


class EmbedderSettingsTests(unittest.TestCase):
    def test_removed_gemini_profile_fails_with_migration_guidance(self):
        with patch.dict(os.environ, {"EMB_PROFILE": "gemini"}, clear=True):
            with self.assertRaisesRegex(RuntimeError, "no longer supported"):
                resolve_embedder_settings(Path("."))


if __name__ == "__main__":
    unittest.main()
