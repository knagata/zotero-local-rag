from __future__ import annotations

import os
import subprocess
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
COMMAND = ROOT / "Maintenance-Widget.command"


class MaintenanceCommandTests(unittest.TestCase):
    def run_command(self, answers: str, *, fail_first: bool = False, extra_env: dict[str, str] | None = None):
        with tempfile.TemporaryDirectory() as directory:
            home = Path(directory)
            bin_dir = home / ".local" / "bin"
            bin_dir.mkdir(parents=True)
            log_path = home / "calls.log"
            fake_uv = bin_dir / "uv"
            fake_uv.write_text(
                "#!/bin/bash\n"
                "printf '%s\\n' \"$*\" >> \"$MAINTENANCE_TEST_LOG\"\n"
                "if [[ \"${MAINTENANCE_FAIL_FIRST:-0}\" == \"1\" ]]; then exit 7; fi\n",
                encoding="utf-8",
            )
            fake_uv.chmod(0o755)
            env = os.environ.copy()
            env.update({
                "HOME": str(home),
                "MAINTENANCE_TEST_LOG": str(log_path),
                "MAINTENANCE_FAIL_FIRST": "1" if fail_first else "0",
                "MAINTENANCE_AUTO_APPROVE": "0",
            })
            env.update(extra_env or {})
            result = subprocess.run(
                ["/bin/bash", str(COMMAND)], input=answers, text=True,
                encoding="utf-8", errors="replace", capture_output=True,
                env=env, timeout=10,
            )
            calls = log_path.read_text(encoding="utf-8").splitlines() if log_path.exists() else []
            return result, calls

    def test_enter_defaults_skip_paid_summary_step(self):
        result, calls = self.run_command("\n\n\n\n")
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertEqual(calls, [
            "run src/index_from_zotero.py --progress",
            "run python scripts/rebuild_document_structure.py --all",
            "run src/update_citations.py --all",
            "run python scripts/list_artifact_status.py --unresolved-only",
        ])

    def test_explicit_mistral_permission_submits_new_batch_and_explains_followup(self):
        # The fourth response explicitly permits the cloud Batch submission;
        # the fifth confirms the planned work.
        result, calls = self.run_command(
            "n\nn\nn\ny\n\n",
            extra_env={"MISTRAL_BATCH_STATE_PATH": "tmp/test_widget_mistral_state.json"},
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertEqual(calls, [
            "run python scripts/run_mistral_ocr_batch.py --submit --state tmp/test_widget_mistral_state.json",
            "run python scripts/list_artifact_status.py --unresolved-only",
        ])
        self.assertIn("処理完了後にMaintenance-Widget.commandを再度起動", result.stdout)

    def test_summary_requires_database_gate(self):
        result, calls = self.run_command(
            "\ny\nn\nn\n\n",
            extra_env={"SUMMARY_DATABASE_GATE": "tmp/nonexistent-database-gate.json"},
        )
        self.assertEqual(result.returncode, 2, result.stdout + result.stderr)
        self.assertFalse(any("build_structure_summaries.py" in call for call in calls))
        self.assertIn("DB監査gateがありません", result.stdout)

    def test_auto_approval_never_runs_paid_cloud_steps(self):
        result, calls = self.run_command(
            "",
            extra_env={
                "MAINTENANCE_AUTO_APPROVE": "1",
                "MISTRAL_BATCH_STATE_PATH": "tmp/test_widget_auto_state.json",
            },
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn("ローカル更新を自動許可", result.stdout)
        self.assertFalse(any("build_structure_summaries.py" in call for call in calls))
        self.assertFalse(any("run_mistral_ocr_batch.py" in call for call in calls))
        self.assertFalse(any("triage_quality_reports.py" in call for call in calls))

    def test_user_can_skip_one_update(self):
        result, calls = self.run_command("\nn\n\n\n")
        self.assertEqual(result.returncode, 0)
        self.assertEqual(calls, [
            "run src/index_from_zotero.py --progress",
            "run python scripts/rebuild_document_structure.py --all",
            "run src/update_citations.py --all",
            "run python scripts/list_artifact_status.py --unresolved-only",
        ])

    def test_failure_stops_later_updates(self):
        result, calls = self.run_command("\n\n\n\n", fail_first=True)
        self.assertEqual(result.returncode, 7)
        self.assertEqual(calls, ["run src/index_from_zotero.py --progress"])
        self.assertIn("後続処理を実行せず終了", result.stdout)

if __name__ == "__main__":
    unittest.main()
