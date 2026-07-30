from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from scripts import run_db_audit


def _env(root: Path, **overrides: str) -> dict[str, str]:
    base = {
        "INGEST_STRUCTURED_V3_ENABLE": "1",
        "CHROMA_COLLECTION": "zotero_paragraphs_v3",
        "MANIFEST_PATH": str(root / "data" / "manifest_v3.json"),
        "LEXICAL_DB_PATH": str(root / "data" / "lexical_v3.sqlite3"),
        "CHROMA_DIR": str(root / "data" / "chroma"),
        "SERVER_DB_GATE_PATH": str(root / "data" / "quality" / "gate.json"),
        "SERVER_ZOTERO_AUDIT_PATH": str(root / "data" / "quality" / "zotero.json"),
        "SERVER_SOURCE_AUDIT_PATH": str(root / "data" / "quality" / "source.json"),
    }
    base.update(overrides)
    return base


class RunDbAuditTests(unittest.TestCase):
    def test_happy_path_runs_the_three_checks_in_order(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            calls = []

            def fake_run(cmd, cwd=None):
                calls.append(cmd)
                return type("R", (), {"returncode": 0})()

            with patch.dict("os.environ", _env(root), clear=True), \
                 patch("subprocess.run", side_effect=fake_run):
                run_db_audit.main()

        scripts_called = [
            next(part for part in call if part.startswith("scripts/"))
            for call in calls
        ]
        self.assertEqual(scripts_called, [
            "scripts/verify_zotero_reconciliation.py",
            "scripts/verify_against_source.py",
            "scripts/audit_v3_cutover.py",
        ])
        self.assertIn("--new-only", calls[2])

    def test_a_failing_step_stops_before_the_next_one(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            calls = []

            def fake_run(cmd, cwd=None):
                calls.append(cmd)
                returncode = 9 if any("verify_against_source.py" in part for part in cmd) else 0
                return type("R", (), {"returncode": returncode})()

            with patch.dict("os.environ", _env(root), clear=True), \
                 patch("subprocess.run", side_effect=fake_run):
                with self.assertRaises(SystemExit) as ctx:
                    run_db_audit.main()
            self.assertEqual(ctx.exception.code, 9)
        self.assertEqual(len(calls), 2)
        self.assertFalse(any(
            "audit_v3_cutover.py" in part for call in calls for part in call
        ))

    def test_a_stale_gate_is_removed_before_any_check_runs(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            gate = root / "data" / "quality" / "gate.json"
            gate.parent.mkdir(parents=True)
            gate.write_text('{"gate": {"passed": true}}', encoding="utf-8")

            def fake_run(cmd, cwd=None):
                # The gate must already be gone by the time the first check runs.
                self.assertFalse(gate.exists())
                return type("R", (), {"returncode": 0})()

            with patch.dict("os.environ", _env(root), clear=True), \
                 patch("subprocess.run", side_effect=fake_run):
                run_db_audit.main()

    def test_legacy_collection_is_rejected_with_a_plain_stop_message(self):
        # A RuntimeError traceback is not an explanation: the caller only sees
        # a nonzero exit, so the reason has to be stated in the message.
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with patch.dict(
                "os.environ", _env(root, CHROMA_COLLECTION="zotero_paragraphs"), clear=True,
            ), patch("subprocess.run") as mock_run:
                with self.assertRaises(SystemExit) as ctx:
                    run_db_audit.main()
            self.assertIn("[停止]", str(ctx.exception.code))
            self.assertIn("zotero_paragraphs", str(ctx.exception.code))
            mock_run.assert_not_called()

    def test_a_legacy_manifest_name_is_rejected_with_a_plain_stop_message(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with patch.dict(
                "os.environ",
                _env(root, MANIFEST_PATH=str(root / "data" / "manifest.json")),
                clear=True,
            ), patch("subprocess.run") as mock_run:
                with self.assertRaises(SystemExit) as ctx:
                    run_db_audit.main()
            self.assertIn("[停止]", str(ctx.exception.code))
            mock_run.assert_not_called()

    def test_a_custom_pipeline_config_path_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with patch.dict(
                "os.environ",
                _env(root, PIPELINE_CONFIG_PATH=str(root / "elsewhere.json")),
                clear=True,
            ), patch("subprocess.run") as mock_run:
                with self.assertRaises(SystemExit) as ctx:
                    run_db_audit.main()
            self.assertIn("[停止]", str(ctx.exception.code))
            mock_run.assert_not_called()


if __name__ == "__main__":
    unittest.main()
