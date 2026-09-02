from __future__ import annotations

import plistlib
from types import SimpleNamespace

from scripts import manage_citation_graph_service as service


def test_launch_agents_use_project_python_without_copying_secrets(tmp_path):
    root = tmp_path / "repo"
    (root / ".venv" / "bin").mkdir(parents=True)
    (root / ".venv" / "bin" / "python").write_text("")
    local, remote, update_check = service.service_definitions(root)
    local_payload = service.build_launch_agent(local, root)
    remote_payload = service.build_launch_agent(remote, root)
    update_payload = service.build_launch_agent(update_check, root)

    assert local_payload["ProgramArguments"][-3:] == ["--no-open", "--port", "7234"]
    assert remote_payload["ProgramArguments"][-1].endswith("citation_graph/remote_proxy.py")
    assert local_payload["KeepAlive"] == {"SuccessfulExit": False}
    assert remote_payload["ThrottleInterval"] == 10
    assert update_payload["ProgramArguments"][-1].endswith("schedule_admin_update_check.py")
    assert update_payload["StartInterval"] == 1800
    assert "KeepAlive" not in update_payload
    serialized = plistlib.dumps(remote_payload)
    assert b"GOOGLE_CLIENT_SECRET" not in serialized
    assert b"SESSION_SECRET" not in serialized


def test_launch_agent_file_is_private_and_atomic(tmp_path):
    destination = tmp_path / "LaunchAgents" / "graph.plist"
    service.write_launch_agent(destination, {"Label": "test"})
    assert plistlib.loads(destination.read_bytes()) == {"Label": "test"}
    assert destination.stat().st_mode & 0o777 == 0o600


def test_bootstrap_retries_launchd_transient_error(monkeypatch, tmp_path):
    outcomes = iter((5, 5, 0))
    calls = []

    def launchctl(*arguments, check=True):
        calls.append((arguments, check))
        return SimpleNamespace(returncode=next(outcomes), args=arguments, stdout="", stderr="")

    monkeypatch.setattr(service, "_launchctl", launchctl)
    monkeypatch.setattr(service.time, "sleep", lambda _seconds: None)
    service._bootstrap_launch_agent("gui/502", tmp_path / "agent.plist")
    assert len(calls) == 3
    assert all(check is False for _arguments, check in calls)


def test_service_snapshot_rollback_restores_old_plist_and_loaded_state(monkeypatch, tmp_path):
    destination = tmp_path / "agent.plist"
    destination.write_bytes(b"old-plist")
    snapshot = service.ServiceSnapshot(destination, b"old-plist", True)
    destination.write_bytes(b"new-plist")
    loaded = {"label": True}
    calls = []

    monkeypatch.setattr(service, "_loaded", lambda label: loaded[label])
    monkeypatch.setattr(
        service, "_launchctl",
        lambda *args, **kwargs: calls.append(args) or loaded.update(label=False)
        or SimpleNamespace(returncode=0),
    )
    monkeypatch.setattr(
        service, "_bootstrap_launch_agent",
        lambda domain, path: calls.append(("restore", domain, str(path))),
    )
    failures = service._restore_service_snapshots({"label": snapshot}, "gui/502")
    assert failures == []
    assert destination.read_bytes() == b"old-plist"
    assert any(call[0] == "restore" for call in calls)


def test_status_requires_both_agents_and_all_health_checks(monkeypatch, capsys):
    settings = type("Settings", (), {
        "host": "127.0.0.1", "port": 7244,
        "public_url": "https://graph.example.test:8443",
    })()
    monkeypatch.setattr(service, "_settings", lambda: settings)
    monkeypatch.setattr(service, "_loaded", lambda label: True)
    monkeypatch.setattr(service, "_http_status", lambda url: 200)
    assert service.status() == 0
    assert "public_oauth_proxy=ok" in capsys.readouterr().out

    monkeypatch.setattr(
        service, "_http_status", lambda url: None if url.startswith("https://") else 200,
    )
    assert service.status() == 1


def test_install_refuses_to_adopt_unknown_listener(monkeypatch):
    settings = type("Settings", (), {"host": "127.0.0.1", "port": 7244})()
    monkeypatch.setattr(service, "_settings", lambda: settings)
    monkeypatch.setattr(service, "service_definitions", lambda: ())
    monkeypatch.setattr(service, "_loaded", lambda label: False)
    monkeypatch.setattr(service, "_port_in_use", lambda host, port: port == 7234)
    try:
        service.install()
    except RuntimeError as error:
        assert "Unmanaged" in str(error)
    else:
        raise AssertionError("install adopted an unknown process")
