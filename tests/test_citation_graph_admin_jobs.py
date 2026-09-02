from __future__ import annotations

import json
import os
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from citation_graph import admin_jobs


def project(tmp_path: Path) -> Path:
    root = tmp_path / "repo"
    (root / ".venv" / "bin").mkdir(parents=True)
    (root / ".venv" / "bin" / "python").write_text("")
    (root / "scripts").mkdir(exist_ok=True)
    return root


def test_job_catalog_contains_only_fixed_argv_and_no_destructive_rebuild(tmp_path):
    definitions = admin_jobs.job_definitions(project(tmp_path))
    assert set(definitions) == {
        "update_check", "quick_update", "library_update", "database_audit", "structure_update",
        "summary_batch", "citation_update",
    }
    for definition in definitions.values():
        for _label, command in definition.steps:
            assert isinstance(command, tuple)
            assert command[0].endswith("/.venv/bin/python")
            assert "--rebuild" not in command
            assert "--force" not in command
    assert definitions["summary_batch"].confirmation == "SUMMARIZE"
    assert definitions["summary_batch"].paid is True
    assert definitions["database_audit"].confirmation == "AUDIT"


def test_quick_update_runs_the_bounded_daily_sequence(tmp_path):
    definition = admin_jobs.job_definitions(project(tmp_path))["quick_update"]
    assert definition.confirmation == "QUICK"
    assert [label for label, _command in definition.steps] == [
        "ライブラリ差分更新",
        "文書構造・目次の差分更新",
        "DB監査",
        "Citation Network更新",
    ]
    assert [Path(command[1]).name for _label, command in definition.steps] == [
        "index_from_zotero.py",
        "rebuild_document_structure.py",
        "run_db_audit.py",
        "update_citations.py",
    ]


def test_start_requires_exact_confirmation_and_redacts_runner_token(tmp_path, monkeypatch):
    root = project(tmp_path)
    monkeypatch.setattr(admin_jobs, "active_job", lambda root: None)
    fake = SimpleNamespace(pid=4321)
    with pytest.raises(ValueError, match="UPDATE"):
        admin_jobs.start_job("library_update", "yes", "user@example.com", root)
    with patch.object(admin_jobs.subprocess, "Popen", return_value=fake):
        record = admin_jobs.start_job("library_update", "UPDATE", "user@example.com", root)
    assert record["pid"] == 4321
    assert record["actor"] == "user@example.com"
    assert "token" not in admin_jobs.public_record(record)
    saved = admin_jobs.read_record(record["id"], root)
    assert saved["token"]


def test_active_job_rejects_pid_reuse(tmp_path, monkeypatch):
    root = project(tmp_path)
    record = {
        "id": "20260902-120000-deadbeef", "type": "database_audit",
        "label": "DB監査", "status": "running", "pid": os.getpid(), "token": "token",
    }
    admin_jobs.write_record(record, root)
    monkeypatch.setattr(admin_jobs, "_runner_command", lambda pid: "python unrelated.py")
    assert admin_jobs.active_job(root) is None


def test_stale_running_record_is_marked_interrupted(tmp_path, monkeypatch):
    root = project(tmp_path)
    record = {
        "id": "20260902-120000-deadbeef", "type": "database_audit",
        "label": "DB監査", "status": "running", "pid": 99999, "token": "token",
    }
    admin_jobs.write_record(record, root)
    monkeypatch.setattr(admin_jobs, "_runner_command", lambda pid: "")
    admin_jobs.reconcile_stale_jobs(root)
    assert admin_jobs.read_record(record["id"], root)["status"] == "interrupted"


def test_runner_executes_steps_and_persists_completion(tmp_path, monkeypatch):
    root = project(tmp_path)
    job_id = "20260902-120000-1234abcd"
    token = "token"
    record = {
        "id": job_id, "type": "test", "label": "test", "status": "queued",
        "created_at": "now", "started_at": None, "finished_at": None,
        "current_step": None, "step_index": 0, "step_total": 2,
        "pid": os.getpid(), "exit_code": None, "token": token,
    }
    admin_jobs.write_record(record, root)
    definition = admin_jobs.JobDefinition(
        "test", "test", "test",
        (
            ("one", (sys.executable, "-c", "print('first')")),
            ("two", (sys.executable, "-c", "print('second')")),
        ),
    )
    monkeypatch.setattr(admin_jobs, "job_definitions", lambda root: {"test": definition})
    assert admin_jobs.run_job(job_id, "test", token, root) == 0
    finished = admin_jobs.read_record(job_id, root)
    assert finished["status"] == "completed"
    assert finished["step_index"] == 2
    assert "first" in admin_jobs.log_tail(job_id, root)
    assert "second" in admin_jobs.log_tail(job_id, root)


def test_failed_step_stops_the_sequence(tmp_path, monkeypatch):
    root = project(tmp_path)
    job_id = "20260902-120001-1234abcd"
    token = "token"
    admin_jobs.write_record({
        "id": job_id, "type": "test", "label": "test", "status": "queued",
        "created_at": "now", "started_at": None, "finished_at": None,
        "current_step": None, "step_index": 0, "step_total": 2,
        "pid": os.getpid(), "exit_code": None, "token": token,
    }, root)
    definition = admin_jobs.JobDefinition(
        "test", "test", "test",
        (
            ("fail", (sys.executable, "-c", "raise SystemExit(7)")),
            ("never", (sys.executable, "-c", "print('must not run')")),
        ),
    )
    monkeypatch.setattr(admin_jobs, "job_definitions", lambda root: {"test": definition})
    assert admin_jobs.run_job(job_id, "test", token, root) == 7
    finished = admin_jobs.read_record(job_id, root)
    assert finished["status"] == "failed"
    assert finished["step_index"] == 1
    assert "must not run" not in admin_jobs.log_tail(job_id, root)


def test_system_status_reports_manifest_gate_and_artifacts(tmp_path):
    root = project(tmp_path)
    data = root / "data"
    (data / "quality").mkdir(parents=True)
    (data / "manifest_v3.json").write_text(json.dumps({
        "files": {"A": {}}, "notes": {}, "hnsw_validated": True,
        "inflight_attachments": [], "post_index_pending": [],
    }))
    (data / "quality" / "server_database_gate.json").write_text(
        json.dumps({"gate": {"passed": True}})
    )
    (data / "admin_update_status.json").write_text(json.dumps({
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "index": {"pending": 2},
    }))
    import sqlite3
    with sqlite3.connect(data / "relations.db") as connection:
        connection.execute("CREATE TABLE artifact_processing_status (status TEXT)")
        connection.executemany(
            "INSERT INTO artifact_processing_status VALUES (?)", [("failed",), ("blocked",)],
        )
    result = admin_jobs.system_status(root)
    assert result["manifest"]["attachments"] == 1
    assert result["database_gate"]["passed"] is True
    assert result["artifacts"]["unresolved"] == 2
    assert result["update_status"]["index"]["pending"] == 2
    assert result["update_status"]["stale"] is False


def test_system_status_marks_old_or_invalidated_freshness_report(tmp_path):
    root = project(tmp_path)
    data = root / "data"
    data.mkdir(exist_ok=True)
    report_path = data / "admin_update_status.json"
    report_path.write_text(json.dumps({
        "generated_at": (datetime.now(timezone.utc) - timedelta(hours=2)).isoformat(),
    }))
    result = admin_jobs.system_status(root)
    assert result["update_status"]["stale"] is True
    assert result["update_status"]["recheck_pending"] is False

    admin_jobs.mark_update_status_recheck_pending("library_update", root)
    result = admin_jobs.system_status(root)
    assert result["update_status"]["recheck_pending"] is True


def test_completed_write_job_schedules_a_read_only_followup(tmp_path, monkeypatch):
    root = project(tmp_path)
    calls = []
    monkeypatch.setattr(
        admin_jobs, "start_job", lambda *args: calls.append(args) or {"id": "followup"},
    )
    result = admin_jobs.schedule_followup_update_check("citation_update", root)
    assert result == {"id": "followup"}
    assert calls == [("update_check", "", "post-job:citation_update", root)]
    report = json.loads((root / "data" / "admin_update_status.json").read_text())
    assert report["recheck_reason"] == "maintenance_completed"


def test_followup_suppresses_only_a_real_job_conflict(tmp_path, monkeypatch):
    root = project(tmp_path)
    monkeypatch.setattr(
        admin_jobs, "start_job",
        lambda *_args: (_ for _ in ()).throw(admin_jobs.JobAlreadyRunningError("busy")),
    )
    assert admin_jobs.schedule_followup_update_check("citation_update", root) is None

    monkeypatch.setattr(
        admin_jobs, "start_job",
        lambda *_args: (_ for _ in ()).throw(RuntimeError("python unavailable")),
    )
    with pytest.raises(RuntimeError, match="python unavailable"):
        admin_jobs.schedule_followup_update_check("citation_update", root)


def test_read_only_jobs_do_not_invalidate_or_schedule_freshness(tmp_path, monkeypatch):
    root = project(tmp_path)
    monkeypatch.setattr(
        admin_jobs, "start_job", lambda *_args: (_ for _ in ()).throw(AssertionError()),
    )
    admin_jobs.mark_update_status_recheck_pending("database_audit", root)
    assert not (root / "data" / "admin_update_status.json").exists()
    assert admin_jobs.schedule_followup_update_check("update_check", root) is None
