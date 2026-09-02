import plistlib
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import manage_remote_mcp_service as service


def _project(tmp_path):
    root = tmp_path / "project"
    (root / ".venv" / "bin").mkdir(parents=True)
    (root / ".venv" / "bin" / "python").write_text("", encoding="utf-8")
    return root


def test_launch_agent_restarts_failures_without_copying_secrets(tmp_path):
    root = _project(tmp_path)
    payload = service.build_launch_agent(root)

    assert payload["RunAtLoad"] is True
    assert payload["KeepAlive"] == {"SuccessfulExit": False}
    assert payload["ThrottleInterval"] == 10
    assert payload["ProcessType"] == "Interactive"
    assert payload["ProgramArguments"][-1] == str(root / "src" / "rag_mcp_http_server.py")
    assert all("SECRET" not in key for key in payload["EnvironmentVariables"])


def test_launch_agent_is_written_private_and_valid(tmp_path):
    destination = tmp_path / "Library" / "LaunchAgents" / "agent.plist"
    payload = {"Label": "test", "RunAtLoad": True}

    service.write_launch_agent(destination, payload)

    assert plistlib.loads(destination.read_bytes()) == payload
    assert destination.stat().st_mode & 0o777 == 0o600


def test_failed_reinstall_restores_previous_loaded_agent(monkeypatch, tmp_path):
    destination = tmp_path / "agent.plist"
    destination.write_bytes(b"old")
    loaded = {"value": True}
    calls = []
    monkeypatch.setattr(service, "_loaded", lambda: loaded["value"])
    monkeypatch.setattr(
        service, "_launchctl",
        lambda *args, **kwargs: calls.append(args) or loaded.update(value=False)
        or SimpleNamespace(returncode=0),
    )
    monkeypatch.setattr(
        service, "_bootstrap", lambda path: calls.append(("bootstrap", str(path))),
    )
    service._restore_previous(destination, b"old", True)
    assert destination.read_bytes() == b"old"
    assert any(call[0] == "bootstrap" for call in calls)


def test_missing_project_python_fails_before_install(tmp_path):
    with pytest.raises(RuntimeError, match="uv sync"):
        service.build_launch_agent(tmp_path)


def test_status_requires_launchd_and_both_http_routes(monkeypatch, capsys):
    settings = SimpleNamespace(
        host="127.0.0.1", port=8765, public_url="https://example.ts.net",
    )
    monkeypatch.setattr(service, "load_dotenv_native", lambda root: None)
    monkeypatch.setattr(service.RemoteMCPSettings, "from_environment", lambda: settings)
    monkeypatch.setattr(service, "_loaded", lambda: True)
    monkeypatch.setattr(service, "_http_status", lambda url: 200)

    assert service.status() == 0
    output = capsys.readouterr().out
    assert "launch_agent=loaded" in output
    assert "local_oauth=ok" in output
    assert "public_oauth=ok" in output


def test_status_fails_when_public_route_is_down(monkeypatch):
    settings = SimpleNamespace(
        host="127.0.0.1", port=8765, public_url="https://example.ts.net",
    )
    monkeypatch.setattr(service, "load_dotenv_native", lambda root: None)
    monkeypatch.setattr(service.RemoteMCPSettings, "from_environment", lambda: settings)
    monkeypatch.setattr(service, "_loaded", lambda: True)
    monkeypatch.setattr(
        service, "_http_status", lambda url: 200 if url.startswith("http://") else None,
    )

    assert service.status() == 1


def test_local_health_wait_stops_after_success(monkeypatch):
    settings = SimpleNamespace(host="127.0.0.1", port=8765)
    responses = iter((None, 200))
    monkeypatch.setattr(service, "_http_status", lambda url: next(responses))
    monkeypatch.setattr(service.time, "sleep", lambda seconds: None)

    assert service.wait_for_local_health(settings, timeout=1) is True


def test_local_health_wait_times_out(monkeypatch):
    settings = SimpleNamespace(host="127.0.0.1", port=8765)
    clock = iter((0.0, 0.1, 1.0))
    monkeypatch.setattr(service, "_http_status", lambda url: None)
    monkeypatch.setattr(service.time, "monotonic", lambda: next(clock))
    monkeypatch.setattr(service.time, "sleep", lambda seconds: None)

    assert service.wait_for_local_health(settings, timeout=0.5) is False
