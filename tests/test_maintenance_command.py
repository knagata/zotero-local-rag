from __future__ import annotations

import os
import subprocess
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
COMMAND = ROOT / "Maintenance-Widget.command"


class MaintenanceCommandTests(unittest.TestCase):
    def run_command(self, answers: str, *, fail_first: bool = False):
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
            })
            result = subprocess.run(
                ["/bin/bash", str(COMMAND)], input=answers, text=True,
                encoding="utf-8", errors="replace", capture_output=True,
                env=env, timeout=10,
            )
            calls = log_path.read_text(encoding="utf-8").splitlines() if log_path.exists() else []
            return result, calls

    def test_enter_defaults_run_all_maintenance_steps(self):
        result, calls = self.run_command("\n\n\n\n\n")
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertEqual(calls, [
            "run src/index_from_zotero.py --progress",
            "run python scripts/rebuild_document_structure.py --all",
            "run python -m src.build_summaries",
            "run python scripts/build_deepseek_summaries.py --output data/quality/maintenance-summary-report.json",
            "run python scripts/build_deepseek_cases.py --output data/quality/maintenance-case-report.json",
            "run src/update_citations.py --all",
            "run python scripts/triage_quality_reports.py",
            "run python scripts/review_relation_reports.py",
            "run python scripts/review_summary_quality_reports.py",
            "run python scripts/review_case_quality_reports.py",
        ])

    def test_user_can_skip_one_update(self):
        result, calls = self.run_command("\nn\n\n\n\n")
        self.assertEqual(result.returncode, 0)
        self.assertEqual(calls, [
            "run src/index_from_zotero.py --progress",
            "run python scripts/rebuild_document_structure.py --all",
            "run src/update_citations.py --all",
            "run python scripts/triage_quality_reports.py",
            "run python scripts/review_relation_reports.py",
            "run python scripts/review_summary_quality_reports.py",
            "run python scripts/review_case_quality_reports.py",
        ])

    def test_failure_stops_later_updates(self):
        result, calls = self.run_command("\n\n\n\n\n", fail_first=True)
        self.assertEqual(result.returncode, 7)
        self.assertEqual(calls, ["run src/index_from_zotero.py --progress"])
        self.assertIn("後続処理を実行せず終了", result.stdout)


if __name__ == "__main__":
    unittest.main()
