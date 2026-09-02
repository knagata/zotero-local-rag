from __future__ import annotations

from scripts import schedule_admin_update_check as scheduler


def test_scheduler_starts_read_only_catalog_job(monkeypatch, capsys):
    calls = []
    monkeypatch.setattr(scheduler, "start_job", lambda *args: calls.append(args) or {"id": "job"})
    assert scheduler.main() == 0
    assert calls == [("update_check", "", "scheduled-launch-agent", scheduler.ROOT)]
    assert "job_id=job" in capsys.readouterr().out


def test_scheduler_defers_when_an_admin_job_is_active(monkeypatch, capsys):
    def busy(*_args):
        raise scheduler.JobAlreadyRunningError("job already running: active")

    monkeypatch.setattr(scheduler, "start_job", busy)
    assert scheduler.main() == 0
    assert "deferred" in capsys.readouterr().out


def test_scheduler_does_not_hide_other_runtime_failures(monkeypatch):
    monkeypatch.setattr(
        scheduler, "start_job",
        lambda *_args: (_ for _ in ()).throw(RuntimeError("python unavailable")),
    )
    try:
        scheduler.main()
    except RuntimeError as exc:
        assert str(exc) == "python unavailable"
    else:
        raise AssertionError("unexpected runtime failures must propagate")
