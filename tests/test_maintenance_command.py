from __future__ import annotations

import os
import subprocess
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
COMMAND = ROOT / "Maintenance-Widget.command"


class MaintenanceCommandTests(unittest.TestCase):
    def run_command(
        self, answers: str, *, fail_first: bool = False, extra_env: dict[str, str] | None = None,
        gate_exists: bool = False, gate_passed: bool = True, create_gate_on_audit: bool = False,
        audit_fails: bool = False, log_answer: str = "n",
    ):
        with tempfile.TemporaryDirectory() as directory:
            home = Path(directory)
            bin_dir = home / ".local" / "bin"
            bin_dir.mkdir(parents=True)
            log_path = home / "calls.log"
            gate_path = home / "data" / "quality" / "gate.json"
            gate_path.parent.mkdir(parents=True)
            if gate_exists:
                # scripts/audit_v3_cutover.py writes this file unconditionally,
                # including on a failing audit -- gate_passed=False models that
                # stale-but-present failing gate.
                gate_path.write_text(
                    '{"gate": {"passed": %s}}' % ("true" if gate_passed else "false"),
                    encoding="utf-8",
                )
            summary_audit_path = home / "data" / "quality" / "summary_audit.json"
            fake_uv = bin_dir / "uv"
            fake_uv.write_text(
                "#!/bin/bash\n"
                "printf '%s\\n' \"$*\" >> \"$MAINTENANCE_TEST_LOG\"\n"
                "if [[ \"${MAINTENANCE_FAIL_FIRST:-0}\" == \"1\" ]]; then exit 7; fi\n"
                "if [[ \"$*\" == *run_db_audit.py* ]]; then\n"
                "    if [[ \"${MAINTENANCE_TEST_CREATE_GATE:-0}\" == \"1\" ]]; then\n"
                "        printf '{\"gate\": {\"passed\": true}}' > \"$SERVER_DB_GATE_PATH\"\n"
                "    fi\n"
                # A real early failure (bad env, uv launch error) leaves any
                # previous gate untouched -- that is the case the widget must
                # not mistake for \"summaries are safe to run\".
                "    if [[ \"${MAINTENANCE_TEST_AUDIT_FAILS:-0}\" == \"1\" ]]; then exit 2; fi\n"
                "fi\n",
                encoding="utf-8",
            )
            fake_uv.chmod(0o755)
            env = os.environ.copy()
            env.update({
                "HOME": str(home),
                "MAINTENANCE_TEST_LOG": str(log_path),
                "MAINTENANCE_FAIL_FIRST": "1" if fail_first else "0",
                "MAINTENANCE_TEST_CREATE_GATE": "1" if create_gate_on_audit else "0",
                "MAINTENANCE_TEST_AUDIT_FAILS": "1" if audit_fails else "0",
                "MAINTENANCE_AUTO_APPROVE": "0",
                "SERVER_DB_GATE_PATH": str(gate_path),
                "SERVER_SUMMARY_AUDIT_PATH": str(summary_audit_path),
            })
            log_dir = home / "logs"
            env.setdefault("MAINTENANCE_LOG_DIR", str(log_dir))
            env.update(extra_env or {})
            # The "save a log?" prompt added 2026-07-30 comes before any of
            # the numbered-item prompts below; most tests decline it.
            result = subprocess.run(
                ["/bin/bash", str(COMMAND)], input=f"{log_answer}\n" + answers, text=True,
                encoding="utf-8", errors="replace", capture_output=True,
                env=env, timeout=10,
            )
            calls = log_path.read_text(encoding="utf-8").splitlines() if log_path.exists() else []
            # The temp HOME is removed when this `with` block exits, so any
            # assertion about the saved transcript file must read it now.
            self.last_log_files = list(log_dir.glob("*.log")) if log_dir.exists() else []
            self.last_log_contents = [
                path.read_text(encoding="utf-8") for path in self.last_log_files
            ]
            return result, calls, gate_path, summary_audit_path

    # Reads consumed, in order, when every item is answered interactively:
    # 1 library, 2 audit, 3 differential summary, 4 bulk summary,
    # 5 citations, (6 is a static notice, no read), 7 Mistral batch, final
    # confirmation. Bulk summary additionally reads a typed SUMMARIZE
    # confirmation during execution, not during this up-front sequence.

    def test_log_prompt_defaults_off_and_writes_no_file_on_bare_enter(self):
        result, _calls, _gate, _summary_audit = self.run_command("\n" * 7, log_answer="")
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertEqual(self.last_log_files, [])

    def test_log_prompt_saves_a_transcript_when_accepted(self):
        result, _calls, _gate, _summary_audit = self.run_command("\n" * 7, log_answer="y")
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertEqual(len(self.last_log_files), 1)
        self.assertIn("ライブラリ差分更新", self.last_log_contents[0])

    def test_enter_defaults_run_library_audit_and_citations_when_gate_is_stale(self):
        # U6/U7 (2026-07-30): a first run (no passing gate yet) defaults the
        # audit item to yes even on a bare Enter -- the paid summary items
        # stay opt-in regardless.
        result, calls, _gate, _summary_audit = self.run_command("\n" * 7)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertEqual(calls, [
            "run src/index_from_zotero.py --progress",
            "run python scripts/rebuild_document_structure.py --all",
            "run python scripts/run_db_audit.py",
            "run src/update_citations.py --all",
            "run python scripts/list_artifact_status.py --unresolved-only",
        ])

    def test_enter_defaults_skip_the_audit_once_the_gate_already_passes(self):
        result, calls, _gate, _summary_audit = self.run_command("\n" * 7, gate_exists=True)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertEqual(calls, [
            "run src/index_from_zotero.py --progress",
            "run python scripts/rebuild_document_structure.py --all",
            "run src/update_citations.py --all",
            "run python scripts/list_artifact_status.py --unresolved-only",
        ])

    def test_enter_defaults_still_run_the_audit_when_the_gate_exists_but_failed(self):
        # audit_v3_cutover.py writes the gate file unconditionally, even on
        # failure -- its mere presence must not be mistaken for a pass.
        result, calls, _gate, _summary_audit = self.run_command(
            "\n" * 7, gate_exists=True, gate_passed=False,
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertEqual(calls, [
            "run src/index_from_zotero.py --progress",
            "run python scripts/rebuild_document_structure.py --all",
            "run python scripts/run_db_audit.py",
            "run src/update_citations.py --all",
            "run python scripts/list_artifact_status.py --unresolved-only",
        ])

    def test_a_stale_failing_gate_still_blocks_a_requested_summary(self):
        # Declining the audit this run (item 2 = n) while a stale failing
        # gate sits on disk from an earlier run must not let the summary
        # step mistake that file's existence for a pass.
        result, calls, _gate, _summary_audit = self.run_command(
            "n\nn\ny\nn\nn\nn\n\n", gate_exists=True, gate_passed=False,
        )
        self.assertEqual(result.returncode, 2, result.stdout + result.stderr)
        self.assertFalse(any("build_structure_summaries.py" in call for call in calls))
        self.assertIn("DB監査の合格証明がありません", result.stdout)

    def test_user_can_skip_the_audit_specifically(self):
        result, calls, _gate, _summary_audit = self.run_command("\nn\n\n\n\n\n\n")
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertEqual(calls, [
            "run src/index_from_zotero.py --progress",
            "run python scripts/rebuild_document_structure.py --all",
            "run src/update_citations.py --all",
            "run python scripts/list_artifact_status.py --unresolved-only",
        ])

    def test_audit_then_differential_summary_run_in_the_same_invocation(self):
        # The freshly created gate from this run's own audit step must be
        # visible to the differential-summary step later in the same run.
        answers = "n\ny\ny\nn\nn\nn\ny\n"
        result, calls, gate, _summary_audit = self.run_command(answers, create_gate_on_audit=True)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertEqual(calls, [
            # The structure refresh writes document_structures, an input to
            # the gate's fingerprint, so it must precede the audit -- not
            # follow it inside the summary block.
            "run python scripts/rebuild_document_structure.py --all",
            "run python scripts/run_db_audit.py",
            f"run python scripts/build_structure_summaries.py --all --mode llm --limit 10 "
            f"--workers 10 --embed --database-gate {gate}",
            "run python scripts/list_artifact_status.py --unresolved-only",
        ])

    def test_differential_summary_requires_database_gate(self):
        result, calls, _gate, _summary_audit = self.run_command("n\nn\ny\nn\nn\nn\n\n")
        self.assertEqual(result.returncode, 2, result.stdout + result.stderr)
        self.assertFalse(any("build_structure_summaries.py" in call for call in calls))
        self.assertIn("DB監査の合格証明がありません", result.stdout)

    def test_bulk_summary_requires_database_gate(self):
        result, calls, _gate, _summary_audit = self.run_command("n\nn\nn\ny\nn\nn\n\n")
        self.assertEqual(result.returncode, 2, result.stdout + result.stderr)
        self.assertFalse(any("build_structure_summaries.py" in call for call in calls))
        self.assertIn("DB監査の合格証明がありません", result.stdout)

    def test_the_audit_runs_after_the_mistral_adoption_it_would_invalidate(self):
        # Mistral adoption rewrites canonical chunks, the manifest and the
        # document structure -- all inputs to the gate fingerprint. Auditing
        # before it would hand the summary step a gate that is already stale.
        answers = "n\ny\nn\nn\nn\ny\n\n"
        result, calls, _gate, _summary_audit = self.run_command(
            answers, extra_env={"MISTRAL_BATCH_STATE_PATH": "tmp/test_widget_order_state.json"},
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        mistral_index = next(
            i for i, call in enumerate(calls) if "run_mistral_ocr_batch.py" in call
        )
        audit_index = next(i for i, call in enumerate(calls) if "run_db_audit.py" in call)
        self.assertLess(mistral_index, audit_index)

    def test_a_failed_audit_blocks_summaries_even_when_a_stale_gate_survives(self):
        # run_db_audit.py can die before it invalidates the previous gate, so
        # the gate file alone must not authorise a paid summary run the
        # operator was just told would not happen.
        answers = "n\ny\ny\nn\nn\nn\n\n"
        result, calls, _gate, _summary_audit = self.run_command(
            answers, gate_exists=True, audit_fails=True,
        )
        self.assertEqual(result.returncode, 2, result.stdout + result.stderr)
        self.assertFalse(any("build_structure_summaries.py" in call for call in calls))
        self.assertIn("要約の生成はこの実行では行われません", result.stdout)

    def test_a_failed_audit_also_blocks_the_bulk_summary(self):
        answers = "n\ny\nn\ny\nn\nn\n\n"
        result, calls, _gate, _summary_audit = self.run_command(
            answers, gate_exists=True, audit_fails=True,
        )
        self.assertEqual(result.returncode, 2, result.stdout + result.stderr)
        self.assertFalse(any("build_structure_summaries.py" in call for call in calls))
        self.assertFalse(any("audit_structure_summaries.py" in call for call in calls))

    def test_a_missing_gate_still_lets_independent_steps_finish(self):
        # The summary step depends on the audit; the citation update and the
        # closing unresolved-status listing do not. A stale gate must not cost
        # the operator the work they explicitly selected alongside it.
        result, calls, _gate, _summary_audit = self.run_command("n\nn\ny\nn\ny\nn\n\n")
        self.assertEqual(result.returncode, 2, result.stdout + result.stderr)
        self.assertFalse(any("build_structure_summaries.py" in call for call in calls))
        self.assertEqual(calls, [
            # The structure refresh precedes the gate check inside the summary
            # block and is idempotent, so it still runs.
            "run python scripts/rebuild_document_structure.py --all",
            "run src/update_citations.py --all",
            "run python scripts/list_artifact_status.py --unresolved-only",
        ])
        self.assertIn("他の更新は最後まで実行済みです", result.stdout)

    def test_bulk_summary_requires_a_typed_summarize_confirmation(self):
        answers = "n\nn\nn\ny\nn\nn\ny\n\n"  # trailing blank cancels the typed prompt
        result, calls, _gate, _summary_audit = self.run_command(answers, gate_exists=True)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertFalse(any("build_structure_summaries.py" in call for call in calls))
        self.assertIn("全件要約の一括生成をキャンセルしました", result.stdout)

    def test_bulk_summary_runs_once_typed_confirmation_matches(self):
        answers = "n\nn\nn\ny\nn\nn\ny\nSUMMARIZE\n"
        result, calls, gate, summary_audit = self.run_command(answers, gate_exists=True)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertEqual(calls, [
            # run_library == 0 with the bulk summary selected still needs a
            # fresh structure, even though the audit itself was declined.
            "run python scripts/rebuild_document_structure.py --all",
            f"run python scripts/build_structure_summaries.py --all --mode llm "
            f"--workers 20 --embed --database-gate {gate}",
            "run python scripts/audit_structure_summaries.py "
            f"--output {summary_audit}",
            "run python scripts/list_artifact_status.py --unresolved-only",
        ])

    def test_explicit_mistral_permission_submits_new_batch_and_explains_followup(self):
        result, calls, _gate, _summary_audit = self.run_command(
            "n\nn\nn\nn\nn\ny\n\n",
            extra_env={"MISTRAL_BATCH_STATE_PATH": "tmp/test_widget_mistral_state.json"},
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertEqual(calls, [
            "run python scripts/run_mistral_ocr_batch.py --submit --state tmp/test_widget_mistral_state.json",
            "run python scripts/list_artifact_status.py --unresolved-only",
        ])
        self.assertIn("処理完了後にMaintenance-Widget.commandを再度起動", result.stdout)

    def test_auto_approval_runs_the_audit_when_the_gate_is_stale_but_never_pays(self):
        result, calls, _gate, _summary_audit = self.run_command(
            "",
            extra_env={
                "MAINTENANCE_AUTO_APPROVE": "1",
                "MISTRAL_BATCH_STATE_PATH": "tmp/test_widget_auto_state.json",
            },
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn("ローカル更新を自動許可", result.stdout)
        self.assertTrue(any("run_db_audit.py" in call for call in calls))
        self.assertFalse(any("build_structure_summaries.py" in call for call in calls))
        self.assertFalse(any("run_mistral_ocr_batch.py" in call for call in calls))
        self.assertFalse(any("audit_structure_summaries.py" in call for call in calls))

    def test_auto_approval_skips_the_audit_once_the_gate_already_passes(self):
        result, calls, _gate, _summary_audit = self.run_command(
            "", extra_env={"MAINTENANCE_AUTO_APPROVE": "1"}, gate_exists=True,
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertFalse(any("run_db_audit.py" in call for call in calls))

    def test_failure_stops_later_updates(self):
        result, calls, _gate, _summary_audit = self.run_command("\n" * 7, fail_first=True)
        self.assertEqual(result.returncode, 7)
        self.assertEqual(calls, ["run src/index_from_zotero.py --progress"])
        self.assertIn("後続処理を実行せず終了", result.stdout)


if __name__ == "__main__":
    unittest.main()
