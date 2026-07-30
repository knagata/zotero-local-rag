from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from src.v3_data_plane import (
    V3_COLLECTION, chroma_collection_populated, chroma_dir, collection_name,
    enforce_environment, lexical_path, manifest_path, pipeline_config_path,
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

    def test_pipeline_config_defaults_inside_the_chroma_directory(self):
        with tempfile.TemporaryDirectory() as directory, patch.dict(os.environ, {}, clear=True):
            root = Path(directory)
            self.assertEqual(
                pipeline_config_path(root),
                root / "data" / "chroma" / "embedder_config_v3.json",
            )

    def test_pipeline_config_follows_a_relocated_chroma_directory(self):
        with tempfile.TemporaryDirectory() as directory, patch.dict(
            os.environ, {"CHROMA_DIR": "elsewhere/chroma"}, clear=True,
        ):
            root = Path(directory)
            self.assertEqual(chroma_dir(root), root / "elsewhere" / "chroma")
            self.assertEqual(
                pipeline_config_path(root),
                root / "elsewhere" / "chroma" / "embedder_config_v3.json",
            )

    def test_pipeline_config_outside_the_chroma_directory_is_rejected(self):
        # The config describes the collection beside it; one stored elsewhere
        # can outlive or contradict that collection.
        with tempfile.TemporaryDirectory() as directory, patch.dict(
            os.environ, {"PIPELINE_CONFIG_PATH": "data/embedder_config_v3.json"}, clear=True,
        ):
            with self.assertRaisesRegex(RuntimeError, "pinned to"):
                pipeline_config_path(Path(directory))

    def test_an_explicit_pipeline_config_at_the_pinned_location_is_accepted(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            pinned = root / "data" / "chroma" / "embedder_config_v3.json"
            with patch.dict(os.environ, {"PIPELINE_CONFIG_PATH": str(pinned)}, clear=True):
                self.assertEqual(pipeline_config_path(root), pinned)

    def test_environment_enforcement_publishes_and_guards_the_pipeline_config(self):
        with tempfile.TemporaryDirectory() as directory, patch.dict(
            os.environ, {}, clear=True,
        ):
            root = Path(directory)
            enforce_environment(root)
            self.assertEqual(os.environ["CHROMA_DIR"], str(root / "data" / "chroma"))
            self.assertEqual(
                os.environ["PIPELINE_CONFIG_PATH"],
                str(root / "data" / "chroma" / "embedder_config_v3.json"),
            )
        with tempfile.TemporaryDirectory() as directory, patch.dict(
            os.environ, {"PIPELINE_CONFIG_PATH": "/tmp/somewhere_else.json"}, clear=True,
        ):
            with self.assertRaisesRegex(RuntimeError, "pinned to"):
                enforce_environment(Path(directory))

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


class ChromaCollectionPopulatedTests(unittest.TestCase):
    # Only chromadb.errors.NotFoundError -- confirmed to be exactly what
    # get_collection raises for a genuinely absent collection -- may resolve
    # to False. Any other failure (a locked file, corruption, permissions)
    # must fall through to None so callers don't mistake "cannot tell" for
    # "empty" (2026-07-30).
    def test_no_chroma_directory_at_all_is_not_populated(self):
        with tempfile.TemporaryDirectory() as directory:
            chroma = Path(directory) / "chroma"
            self.assertIs(chroma_collection_populated(chroma), False)

    def test_a_bare_sqlite_file_with_no_valid_store_is_unknown(self):
        # chroma.sqlite3 is created the moment anything opens the persistent
        # client (a failed build, or just configuring the MCP server), and a
        # rebuild deletes collections without deleting it. A file that exists
        # but isn't a real chromadb store must not read as "empty".
        with tempfile.TemporaryDirectory() as directory:
            chroma = Path(directory) / "chroma"
            chroma.mkdir(parents=True)
            (chroma / "chroma.sqlite3").write_bytes(b"SQLite format 3\x00")
            self.assertIs(chroma_collection_populated(chroma), None)

    def test_a_missing_collection_in_a_real_store_is_not_populated(self):
        with tempfile.TemporaryDirectory() as directory:
            chroma = Path(directory) / "chroma"
            import chromadb

            client = chromadb.PersistentClient(path=str(chroma))
            client.create_collection("some_other_collection")
            close = getattr(client, "close", None)
            if callable(close):
                close()
            self.assertIs(chroma_collection_populated(chroma), False)

    def test_an_empty_v3_collection_is_not_populated(self):
        with tempfile.TemporaryDirectory() as directory:
            chroma = Path(directory) / "chroma"
            import chromadb

            client = chromadb.PersistentClient(path=str(chroma))
            client.create_collection(V3_COLLECTION)
            close = getattr(client, "close", None)
            if callable(close):
                close()
            self.assertIs(chroma_collection_populated(chroma), False)

    def test_a_v3_collection_with_rows_is_populated(self):
        with tempfile.TemporaryDirectory() as directory:
            chroma = Path(directory) / "chroma"
            import chromadb

            client = chromadb.PersistentClient(path=str(chroma))
            collection = client.create_collection(V3_COLLECTION)
            collection.add(ids=["1"], documents=["doc"])
            close = getattr(client, "close", None)
            if callable(close):
                close()
            self.assertIs(chroma_collection_populated(chroma), True)

    def test_a_non_notfound_error_is_unknown_not_empty(self):
        with tempfile.TemporaryDirectory() as directory:
            chroma = Path(directory) / "chroma"
            chroma.mkdir(parents=True)

            class FakeClient:
                def get_collection(self, name):
                    raise PermissionError("locked")

                def close(self):
                    pass

            with patch("chromadb.PersistentClient", return_value=FakeClient()):
                self.assertIs(chroma_collection_populated(chroma), None)


if __name__ == "__main__":
    unittest.main()
