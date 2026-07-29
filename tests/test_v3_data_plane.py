from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from src.v3_data_plane import (
    V3_COLLECTION, collection_name, enforce_environment, lexical_path,
    manifest_path,
)


class V3DataPlaneTests(unittest.TestCase):
    def test_empty_environment_resolves_only_v3_targets(self):
        with tempfile.TemporaryDirectory() as directory, patch.dict(os.environ, {}, clear=True):
            root = Path(directory)
            self.assertEqual(collection_name(), V3_COLLECTION)
            self.assertEqual(manifest_path(root).name, "manifest_v3.json")
            self.assertEqual(lexical_path(root).name, "lexical_v3.sqlite3")

    def test_explicit_legacy_flag_is_rejected(self):
        with patch.dict(os.environ, {"INGEST_STRUCTURED_V3_ENABLE": "0"}, clear=True):
            with self.assertRaisesRegex(RuntimeError, "legacy data plane is retired"):
                collection_name()

    def test_explicit_legacy_collection_is_rejected(self):
        with patch.dict(os.environ, {"CHROMA_COLLECTION": "zotero_paragraphs"}, clear=True):
            with self.assertRaisesRegex(RuntimeError, "only production collection"):
                collection_name()

    def test_legacy_sidecar_names_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with patch.dict(os.environ, {"MANIFEST_PATH": "data/manifest.json"}, clear=True):
                with self.assertRaisesRegex(RuntimeError, "legacy manifest is retired"):
                    manifest_path(root)
            with patch.dict(os.environ, {"LEXICAL_DB_PATH": "data/lexical.sqlite3"}, clear=True):
                with self.assertRaisesRegex(RuntimeError, "legacy FTS is retired"):
                    lexical_path(root)

    def test_environment_enforcement_overrides_retired_search_switch(self):
        with tempfile.TemporaryDirectory() as directory, patch.dict(
            os.environ, {"HIERARCHICAL_SEARCH_V2_ENABLE": "0"}, clear=True,
        ):
            enforce_environment(Path(directory))
            self.assertEqual(os.environ["HIERARCHICAL_SEARCH_V2_ENABLE"], "1")
            self.assertEqual(os.environ["CHROMA_COLLECTION"], V3_COLLECTION)

    def test_relative_sidecar_paths_are_resolved_from_project_root(self):
        with tempfile.TemporaryDirectory() as directory, patch.dict(
            os.environ,
            {
                "MANIFEST_PATH": "data/manifest_v3.json",
                "LEXICAL_DB_PATH": "data/lexical_v3.sqlite3",
            },
            clear=True,
        ):
            root = Path(directory)
            self.assertEqual(manifest_path(root), root / "data" / "manifest_v3.json")
            self.assertEqual(lexical_path(root), root / "data" / "lexical_v3.sqlite3")


if __name__ == "__main__":
    unittest.main()
