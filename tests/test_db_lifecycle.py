from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from src import db_lifecycle


class ExistingDatabaseStateTests(unittest.TestCase):
    # Only a positively-proven-empty state may skip the destructive
    # confirmation. Every unreadable state must classify as "there is
    # something to lose" (2026-07-30). Paths are passed explicitly rather
    # than read from os.environ, since Setup.command never loads .env back
    # into its own process -- reading the environment here would silently
    # check the default location instead of whatever was just configured.
    def _paths(self, root: Path) -> tuple[Path, Path]:
        return root / "data" / "chroma", root / "data" / "manifest_v3.json"

    def _write_manifest(self, root: Path, body: str) -> Path:
        _chroma, manifest = self._paths(root)
        manifest.parent.mkdir(parents=True, exist_ok=True)
        manifest.write_text(body, encoding="utf-8")
        return manifest

    def test_absent_manifest_and_absent_chroma_is_empty(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            chroma, manifest = self._paths(root)
            self.assertEqual(
                db_lifecycle.existing_database_state(chroma, manifest),
                db_lifecycle.DB_STATE_EMPTY,
            )

    def test_empty_files_map_is_empty(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._write_manifest(root, '{"version": 1, "files": {}, "notes": {}}')
            chroma, manifest = self._paths(root)
            self.assertEqual(
                db_lifecycle.existing_database_state(chroma, manifest),
                db_lifecycle.DB_STATE_EMPTY,
            )

    def test_a_file_entry_is_populated(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._write_manifest(
                root, '{"version": 1, "files": {"ABCD1234": {"mtime": 1.0}}, "notes": {}}',
            )
            chroma, manifest = self._paths(root)
            self.assertEqual(
                db_lifecycle.existing_database_state(chroma, manifest),
                db_lifecycle.DB_STATE_POPULATED,
            )

    def test_a_corrupt_manifest_is_unknown_not_empty(self):
        # A truncated manifest is not evidence that the collection is empty;
        # treating it as such would drop the rebuild confirmation entirely.
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._write_manifest(root, '{"version": 1, "files": {"ABC')
            chroma, manifest = self._paths(root)
            self.assertEqual(
                db_lifecycle.existing_database_state(chroma, manifest),
                db_lifecycle.DB_STATE_UNKNOWN,
            )

    def test_a_populated_collection_outranks_a_missing_manifest(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            chroma, manifest = self._paths(root)
            with patch.object(db_lifecycle, "chroma_collection_populated", return_value=True):
                self.assertEqual(
                    db_lifecycle.existing_database_state(chroma, manifest),
                    db_lifecycle.DB_STATE_POPULATED,
                )

    def test_unresolvable_chroma_directory_is_unknown(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            chroma, manifest = self._paths(root)
            with patch.object(db_lifecycle, "chroma_collection_populated", return_value=None):
                self.assertEqual(
                    db_lifecycle.existing_database_state(chroma, manifest),
                    db_lifecycle.DB_STATE_UNKNOWN,
                )


class RunRebuildTests(unittest.TestCase):
    def test_stops_at_the_first_failing_step(self):
        results = iter([type("R", (), {"returncode": 7})()])
        with patch("subprocess.run", side_effect=lambda *a, **k: next(results)) as mock_run:
            code = db_lifecycle.run_rebuild(Path("/unused"))
        self.assertEqual(code, 7)
        mock_run.assert_called_once()

    def test_runs_both_steps_and_returns_zero_on_success(self):
        ok = type("R", (), {"returncode": 0})()
        with patch("subprocess.run", return_value=ok) as mock_run:
            code = db_lifecycle.run_rebuild(Path("/unused"))
        self.assertEqual(code, 0)
        self.assertEqual(mock_run.call_count, 2)
        first_args = mock_run.call_args_list[0].args[0]
        second_args = mock_run.call_args_list[1].args[0]
        self.assertIn("--rebuild", first_args)
        self.assertTrue(any("rebuild_document_structure.py" in part for part in second_args))

    def test_saved_config_overrides_stale_parent_environment_for_every_child(self):
        ok = type("R", (), {"returncode": 0})()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / ".env").write_text(
                "PDF_STRUCTURE_ENGINE_LONG=mistral\n"
                "PDF_MISTRAL_TOC_QUEUE_ENABLE=1\n",
                encoding="utf-8",
            )
            with patch.dict(
                "os.environ",
                {
                    "PDF_STRUCTURE_ENGINE_LONG": "granite",
                    "PDF_MISTRAL_TOC_QUEUE_ENABLE": "0",
                },
            ), patch("subprocess.run", return_value=ok) as mock_run:
                code = db_lifecycle.run_rebuild(root)

        self.assertEqual(code, 0)
        self.assertEqual(mock_run.call_count, 2)
        for call in mock_run.call_args_list:
            child_env = call.kwargs["env"]
            self.assertEqual(child_env["PDF_STRUCTURE_ENGINE_LONG"], "mistral")
            self.assertEqual(child_env["PDF_MISTRAL_TOC_QUEUE_ENABLE"], "1")


class RunAuditTests(unittest.TestCase):
    def test_returns_the_subprocess_exit_code(self):
        result = type("R", (), {"returncode": 3})()
        with patch("subprocess.run", return_value=result) as mock_run:
            code = db_lifecycle.run_audit(Path("/unused"))
        self.assertEqual(code, 3)
        args = mock_run.call_args.args[0]
        self.assertTrue(any("run_db_audit.py" in part for part in args))


if __name__ == "__main__":
    unittest.main()
