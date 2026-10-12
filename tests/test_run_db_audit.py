from __future__ import annotations

import sys
import tempfile
import unittest
import pytest
from pathlib import Path
from unittest.mock import patch

from scripts import run_db_audit


@pytest.fixture(autouse=True)
def isolated_audit_root(tmp_path, monkeypatch):
    monkeypatch.setattr(run_db_audit, "ROOT", tmp_path)


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
    def test_structure_failure_explains_target_and_non_ocr_action(self):
        import json
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "gate.json"
            output.write_text(json.dumps({
                "gate": {"failed_items": 1},
                "items": [{"item_key": "A", "title": "Book",
                           "failures": ["stale_structure_fingerprint"]}],
            }))
            with patch("builtins.print") as printed:
                run_db_audit._explain_failure(output, None, 2)
            log = "\n".join(str(call.args[0]) for call in printed.call_args_list)
            self.assertIn("Book", log)
            self.assertIn("更新状態", log)
            self.assertIn("再OCRは必要ありません", log)

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
        self.assertTrue(all(call[0] == sys.executable for call in calls))
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


@pytest.mark.parametrize("output_name", [
    "SERVER_DB_GATE_PATH", "SERVER_ZOTERO_AUDIT_PATH", "SERVER_SOURCE_AUDIT_PATH",
    "SERVER_DB_AUDIT_STATUS_PATH",
])
@pytest.mark.parametrize("target", [
    "data/manifest_v3.json", "data/lexical_v3.sqlite3", "data/relations.db",
    "data/lexical_v3.sqlite3-wal", "data/lexical_v3.sqlite3-shm",
    "data/lexical_v3.sqlite3-journal", "data/relations.db-wal",
    "data/relations.db-shm", "data/relations.db-journal",
    "data/chroma/embedder_config_v3.json", "data/chroma/segment/index.bin",
])
def test_protected_outputs_leave_all_files_intact(tmp_path, monkeypatch, output_name, target):
    victim = tmp_path / target
    victim.parent.mkdir(parents=True)
    victim.write_bytes(b"original data")
    gate = tmp_path / "data/quality/gate.json"
    gate.parent.mkdir(parents=True)
    gate.write_bytes(b"old gate")
    monkeypatch.setattr(run_db_audit, "ROOT", tmp_path)
    environment = _env(tmp_path, RELATIONS_DB_PATH=str(tmp_path / "data/relations.db"))
    environment[output_name] = target
    with patch.dict("os.environ", environment, clear=True), patch.object(run_db_audit, "_run") as run:
        with pytest.raises(SystemExit, match=r"\[停止\]"):
            run_db_audit.main()
        run.assert_not_called()
    assert victim.read_bytes() == b"original data"
    assert gate.read_bytes() == b"old gate"


@pytest.mark.parametrize("alias_kind", ["symlink", "hardlink", "parent_symlink"])
def test_alias_of_protected_data_is_rejected(tmp_path, monkeypatch, alias_kind):
    victim = tmp_path / "data/manifest_v3.json"
    victim.parent.mkdir()
    victim.write_bytes(b"original")
    alias = tmp_path / "report.json"
    if alias_kind == "symlink":
        alias.symlink_to(victim)
    elif alias_kind == "hardlink":
        alias.hardlink_to(victim)
    else:
        directory = tmp_path / "alias"
        directory.symlink_to(victim.parent, target_is_directory=True)
        alias = directory / victim.name
    monkeypatch.setattr(run_db_audit, "ROOT", tmp_path)
    with patch.dict("os.environ", _env(tmp_path, SERVER_SOURCE_AUDIT_PATH=str(alias)), clear=True), \
         patch.object(run_db_audit, "_run") as run:
        with pytest.raises(SystemExit, match="protected data"):
            run_db_audit.main()
        run.assert_not_called()
    assert victim.read_bytes() == b"original"
    assert not (tmp_path / "data/quality").exists()


@pytest.mark.parametrize("target", ["data/quality/gate.json", "data/quality/gate.json/child.json"])
def test_colliding_outputs_are_rejected_before_gate_deletion(tmp_path, monkeypatch, target):
    monkeypatch.setattr(run_db_audit, "ROOT", tmp_path)
    with patch.dict("os.environ", _env(tmp_path, SERVER_SOURCE_AUDIT_PATH=target), clear=True), \
         patch.object(run_db_audit, "_run") as run:
        with pytest.raises(SystemExit, match="collide"):
            run_db_audit.main()
        run.assert_not_called()
    assert not (tmp_path / "data").exists()


def test_relative_reports_are_resolved_against_root(tmp_path, monkeypatch):
    monkeypatch.setattr(run_db_audit, "ROOT", tmp_path)
    with patch.dict("os.environ", _env(tmp_path, SERVER_SOURCE_AUDIT_PATH="reports/source.json"), clear=True), \
         patch.object(run_db_audit, "_run") as run:
        run_db_audit.main()
    assert str(tmp_path / "reports/source.json") in run.call_args_list[1].args
    assert str(tmp_path / "reports/source.json") in run.call_args_list[2].args


def test_safe_gate_is_invalidated_even_for_invalid_environment(tmp_path, monkeypatch):
    monkeypatch.setattr(run_db_audit, "ROOT", tmp_path)
    gate = tmp_path / "gate.json"
    gate.write_text("old gate")
    with patch.dict("os.environ", _env(tmp_path, SERVER_DB_GATE_PATH=str(gate), CHROMA_COLLECTION="legacy"), clear=True), \
         patch.object(run_db_audit, "_run") as run:
        with pytest.raises(SystemExit, match="Unsupported Chroma"):
            run_db_audit.main()
        run.assert_not_called()
    assert not gate.exists()


def test_failed_source_check_reports_title_pages_and_next_action(tmp_path, capsys):
    import json
    report = tmp_path / "source.json"

    def fail(command, cwd):
        report.write_text(json.dumps({"documents": [{
            "title": "Fixture document", "attachment_key": "TESTKEY1", "lost_pages": [1, 3, 4],
        }], "summary": {"orphan_chunks": 2}}))
        return type("R", (), {"returncode": 2})()

    with patch("subprocess.run", side_effect=fail), pytest.raises(SystemExit) as exc:
        run_db_audit._run("python", "check.py", "--output", str(report))
    assert exc.value.code == 2
    output = capsys.readouterr().out
    assert "[監査失敗]" in output
    assert "Fixture document" in output and "TESTKEY1" in output
    assert "3頁" in output and "1, 3, 4" in output
    assert "Mistral OCR" in output and "有料" in output
    assert "原本と対応しない本文：2件" in output


def test_failed_check_does_not_describe_previous_report(tmp_path, capsys):
    report = tmp_path / "old.json"
    report.write_text('{"gate":{"global_failures":["incomplete_source_coverage"]}}')
    with patch("subprocess.run", return_value=type("R", (), {"returncode": 9})()), pytest.raises(SystemExit):
        run_db_audit._run("python", "check.py", "--output", str(report))
    output = capsys.readouterr().out
    assert "終了コード9" in output
    assert "原本の抽出範囲" not in output


@pytest.mark.parametrize("report_body", ["not json", "[]", '{"gate":{"global_failures":["chroma_lexical_id_mismatch","new_reason"],"failed_items":1}}', '{"missing_required_from_manifest":["TESTKEY1"]}'])
def test_failure_explanation_handles_gate_missing_and_invalid_reports(tmp_path, capsys, report_body):
    report = tmp_path / "report.json"
    report.write_text(report_body)
    run_db_audit._explain_failure(report, None, 2)
    output = capsys.readouterr().out
    assert "[監査失敗]" in output and "[必要な作業]" in output
    if "chroma_lexical" in report_body:
        assert "全文索引とベクトル索引" in output and "new_reason" in output
    if "TESTKEY1" in report_body:
        assert "TESTKEY1" in output


if __name__ == "__main__":
    unittest.main()


def test_auto_repair_reaudits_all_checks_exactly_once(tmp_path, capsys):
    import json
    calls = []
    report = {"items": [{"item_key": "FIXTURE", "failures": ["stale_structure_fingerprint"]}]}
    def run(command, cwd):
        script = command[1]
        calls.append(script)
        if script.endswith("audit_v3_cutover.py") and len(calls) == 3:
            Path(command[command.index("--output") + 1]).write_text(json.dumps(report))
            return type("R", (), {"returncode": 2})()
        return type("R", (), {"returncode": 0})()
    with patch.dict("os.environ", _env(tmp_path), clear=True), \
         patch("subprocess.run", side_effect=run), \
         patch("src.audit_repair.repair_derived_data", return_value={"repaired": True}) as repair:
        run_db_audit.main()
    assert len(calls) == 6 and calls[:3] == calls[3:]
    repair.assert_called_once_with(report, tmp_path)
    status = json.loads((tmp_path / "data/quality/server_database_audit_status.json").read_text())
    assert status["status"] == "completed"
    assert not (tmp_path / "data/indexing.lock").exists()


def test_persistent_failure_stops_after_one_repair(tmp_path):
    import json
    attempts = []
    def audit():
        attempts.append(True)
        raise run_db_audit.AuditFailure(2, {"gate": {"global_failures": ["hnsw_not_validated"]}})
    with patch.dict("os.environ", _env(tmp_path), clear=True), \
         patch.object(run_db_audit, "_audit_once", side_effect=audit), \
         patch("src.audit_repair.repair_derived_data", return_value={"repaired": True}) as repair:
        with pytest.raises(SystemExit):
            run_db_audit.main()
    assert len(attempts) == 2
    repair.assert_called_once()
    status = json.loads((tmp_path / "data/quality/server_database_audit_status.json").read_text())
    assert status["status"] == "failed" and status["findings"][0]["reason"] == "hnsw_not_validated"


def test_audit_holds_writer_lock_for_checks_and_repair(tmp_path):
    from src import indexing_lock
    def audit():
        with pytest.raises(SystemExit):
            indexing_lock.acquire(indexing_lock.default_path(tmp_path))
    with patch.dict("os.environ", _env(tmp_path), clear=True), patch.object(run_db_audit, "_audit_once", side_effect=audit):
        run_db_audit.main()


def test_lock_conflict_does_not_invalidate_existing_gate(tmp_path):
    from src import indexing_lock
    gate = tmp_path / "data/quality/gate.json"
    gate.parent.mkdir(parents=True)
    gate.write_text("existing gate")
    with patch.dict("os.environ", _env(tmp_path), clear=True), indexing_lock.held(indexing_lock.default_path(tmp_path)):
        with pytest.raises(SystemExit):
            run_db_audit.main()
    assert gate.read_text() == "existing gate"


def test_child_crash_after_publishing_success_does_not_leave_passing_gate(tmp_path):
    import json
    gate = tmp_path / "data/quality/gate.json"
    def fail():
        gate.parent.mkdir(parents=True, exist_ok=True)
        gate.write_text(json.dumps({"gate": {"passed": True}}))
        raise run_db_audit.AuditFailure(9, {"gate": {"passed": True}})
    with patch.dict("os.environ", _env(tmp_path), clear=True), \
         patch.object(run_db_audit, "_audit_once", side_effect=fail):
        with pytest.raises(SystemExit):
            run_db_audit.main()
    assert not gate.exists()
