from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import index_from_zotero as module  # noqa: E402


class FinalizeStoragePhaseTests(unittest.TestCase):
    def test_success_commits_manifests_only_after_hnsw_validation(self):
        col = MagicMock()
        manifest = {}
        files = {"A1": {"mtime": 1}}
        notes = {"N1": {"hash": "x"}}
        events = []

        def validate(_col, sample_id):
            events.append(("validate", sample_id))

        def save(_path, saved_manifest):
            events.append(("save", bool(saved_manifest.get("hnsw_validated"))))

        with patch.object(module, "_flush_and_verify_hnsw", side_effect=validate), \
                patch.object(module, "save_manifest", side_effect=save):
            result = module._finalize_index_storage(
                col, manifest=manifest, files_manifest=files,
                notes_manifest=notes, last_written_id="chunk-1",
            )

        self.assertEqual(result, "chunk-1")
        self.assertEqual(events, [("validate", "chunk-1"), ("save", True)])
        self.assertIs(manifest["files"], files)
        self.assertIs(manifest["notes"], notes)

    def test_validation_failure_records_failed_gate_and_aborts(self):
        manifest = {}
        with patch.object(
            module, "_flush_and_verify_hnsw", side_effect=RuntimeError("broken index"),
        ), patch.object(module, "save_manifest") as save:
            with self.assertRaisesRegex(RuntimeError, "HNSW final validation failed"):
                module._finalize_index_storage(
                    MagicMock(), manifest=manifest, files_manifest={},
                    notes_manifest={}, last_written_id="chunk-1",
                )
        self.assertFalse(manifest["hnsw_validated"])
        save.assert_called_once()


class QualityWarningPhaseTests(unittest.TestCase):
    def test_only_unresolved_quality_damage_becomes_a_warning(self):
        warnings = module._quality_warnings({
            "GOOD": {"title": "Good", "quality": {"is_scanned": False}},
            "BAD": {
                "title": "Damaged",
                "quality": {"corrupted_pages": [2, 3]},
            },
            "PARTIAL": {
                "title": "Partial",
                "quality": {
                    "source_coverage_adopted": True,
                    "source_coverage_shortfall": {
                        "accounted_units": 9, "expected_units": 10, "unit_kind": "page",
                    },
                },
            },
        })
        self.assertEqual([warning.attachment_key for warning in warnings], ["BAD", "PARTIAL"])
        self.assertIn("2 unresolved corrupted page", warnings[0].reasons[0])
        self.assertIn("9/10 pages", warnings[1].reasons[0])


if __name__ == "__main__":
    unittest.main()
