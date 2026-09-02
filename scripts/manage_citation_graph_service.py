#!/usr/bin/env python3
"""Install and operate Citation Graph and periodic-status macOS LaunchAgents."""
from __future__ import annotations

import argparse
import os
import plistlib
import socket
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from citation_graph.remote_proxy import CitationGraphRemoteSettings
from src.env_utils import environment_with_saved_dotenv

LOCAL_LABEL = "dev.zotero-local-rag.citation-graph"
REMOTE_LABEL = "dev.zotero-local-rag.citation-graph-remote"
UPDATE_CHECK_LABEL = "dev.zotero-local-rag.citation-graph-update-check"
UPDATE_CHECK_INTERVAL_SECONDS = 30 * 60


@dataclass(frozen=True)
class ServiceDefinition:
    label: str
    program_arguments: tuple[str, ...]
    stdout_name: str
    stderr_name: str
    keep_alive: bool = True
    start_interval: int | None = None


@dataclass(frozen=True)
class ServiceSnapshot:
    destination: Path
    content: bytes | None
    loaded: bool


def launch_domain(uid: int | None = None) -> str:
    return f"gui/{os.getuid() if uid is None else uid}"


def launch_agent_path(label: str, home: Path | None = None) -> Path:
    base = Path.home() if home is None else home
    return base / "Library" / "LaunchAgents" / f"{label}.plist"


def service_definitions(root: Path = ROOT) -> tuple[ServiceDefinition, ...]:
    python = root / ".venv" / "bin" / "python"
    if not python.is_file():
        raise RuntimeError(f"Project Python not found: {python}. Run uv sync first.")
    return (
        ServiceDefinition(
            LOCAL_LABEL,
            (
                str(python), "-u", str(root / "citation_graph" / "server.py"),
                "--no-open", "--port", "7234",
            ),
            "citation-graph.stdout.log",
            "citation-graph.stderr.log",
        ),
        ServiceDefinition(
            REMOTE_LABEL,
            (str(python), "-u", str(root / "citation_graph" / "remote_proxy.py")),
            "citation-graph-remote.stdout.log",
            "citation-graph-remote.stderr.log",
        ),
        ServiceDefinition(
            UPDATE_CHECK_LABEL,
            (str(python), str(root / "scripts" / "schedule_admin_update_check.py")),
            "citation-graph-update-check.stdout.log",
            "citation-graph-update-check.stderr.log",
            keep_alive=False,
            start_interval=UPDATE_CHECK_INTERVAL_SECONDS,
        ),
    )


def build_launch_agent(service: ServiceDefinition, root: Path = ROOT) -> dict[str, Any]:
    data = root / "data"
    payload: dict[str, Any] = {
        "Label": service.label,
        "ProgramArguments": list(service.program_arguments),
        "WorkingDirectory": str(root),
        "RunAtLoad": True,
        "ThrottleInterval": 10,
        "ProcessType": "Interactive",
        "StandardOutPath": str(data / service.stdout_name),
        "StandardErrorPath": str(data / service.stderr_name),
        "EnvironmentVariables": {"PYTHONUNBUFFERED": "1"},
    }
    if service.keep_alive:
        payload["KeepAlive"] = {"SuccessfulExit": False}
    if service.start_interval is not None:
        payload["StartInterval"] = service.start_interval
    return payload


def write_launch_agent(destination: Path, payload: dict[str, Any]) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.parent.chmod(0o700)
    encoded = plistlib.dumps(payload, fmt=plistlib.FMT_XML, sort_keys=True)
    with tempfile.NamedTemporaryFile(dir=destination.parent, delete=False) as temporary:
        temporary.write(encoded)
        temporary_path = Path(temporary.name)
    temporary_path.chmod(0o600)
    os.replace(temporary_path, destination)


def _write_private_bytes(destination: Path, content: bytes) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=destination.parent, delete=False) as temporary:
        temporary.write(content)
        candidate = Path(temporary.name)
    candidate.chmod(0o600)
    os.replace(candidate, destination)


def _launchctl(*arguments: str, check: bool = True) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["launchctl", *arguments], text=True, capture_output=True, check=check,
    )


def _loaded(label: str) -> bool:
    result = _launchctl("print", f"{launch_domain()}/{label}", check=False)
    return result.returncode == 0


def _bootstrap_launch_agent(domain: str, destination: Path, attempts: int = 20) -> None:
    """Retry launchd's transient EIO after bootout before giving up."""
    result: subprocess.CompletedProcess[str] | None = None
    for attempt in range(attempts):
        result = _launchctl("bootstrap", domain, str(destination), check=False)
        if result.returncode == 0:
            return
        if attempt + 1 < attempts:
            time.sleep(0.1)
    assert result is not None
    raise subprocess.CalledProcessError(
        result.returncode, result.args, output=result.stdout, stderr=result.stderr,
    )


def _port_in_use(host: str, port: int) -> bool:
    try:
        with socket.create_connection((host, port), timeout=0.25):
            return True
    except OSError:
        return False


def _http_status(url: str) -> int | None:
    try:
        with urllib.request.urlopen(url, timeout=5) as response:
            return response.status
    except urllib.error.HTTPError as exc:
        return exc.code
    except urllib.error.URLError:
        return None


def _wait_for(url: str, expected: set[int], timeout: float = 45) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if _http_status(url) in expected:
            return True
        time.sleep(0.25)
    return False


def _settings() -> CitationGraphRemoteSettings:
    return CitationGraphRemoteSettings.from_environment(environment_with_saved_dotenv(ROOT))


def _service_snapshots(services: tuple[ServiceDefinition, ...]) -> dict[str, ServiceSnapshot]:
    snapshots = {}
    for service in services:
        destination = launch_agent_path(service.label)
        snapshots[service.label] = ServiceSnapshot(
            destination=destination,
            content=destination.read_bytes() if destination.exists() else None,
            loaded=_loaded(service.label),
        )
    return snapshots


def _restore_service_snapshots(
    snapshots: dict[str, ServiceSnapshot], domain: str,
) -> list[str]:
    failures: list[str] = []
    for label, snapshot in reversed(tuple(snapshots.items())):
        try:
            if _loaded(label):
                _launchctl("bootout", f"{domain}/{label}", check=False)
            if snapshot.content is None:
                if snapshot.destination.exists():
                    snapshot.destination.unlink()
            else:
                _write_private_bytes(snapshot.destination, snapshot.content)
            if snapshot.loaded and snapshot.content is not None:
                _bootstrap_launch_agent(domain, snapshot.destination)
                _launchctl("enable", f"{domain}/{label}")
        # Rollback is best-effort across every service; one failed restore must
        # not prevent the remaining snapshots from being restored.
        except Exception as exc:  # noqa: BLE001
            failures.append(f"{label}: {exc}")
    return failures


def install() -> None:
    settings = _settings()
    services = service_definitions()
    occupied = (
        (LOCAL_LABEL, "127.0.0.1", 7234),
        (REMOTE_LABEL, settings.host, settings.port),
    )
    conflicts = [
        f"{host}:{port}" for label, host, port in occupied
        if not _loaded(label) and _port_in_use(host, port)
    ]
    if conflicts:
        raise RuntimeError(
            "Unmanaged Citation Graph process is already using "
            f"{', '.join(conflicts)}. Stop it before installing the LaunchAgents."
        )
    (ROOT / "data").mkdir(parents=True, exist_ok=True)
    domain = launch_domain()
    snapshots = _service_snapshots(services)
    try:
        for service in services:
            destination = launch_agent_path(service.label)
            write_launch_agent(destination, build_launch_agent(service))
            if _loaded(service.label):
                _launchctl("bootout", f"{domain}/{service.label}")
            _bootstrap_launch_agent(domain, destination)
            _launchctl("enable", f"{domain}/{service.label}")
        if not _wait_for("http://127.0.0.1:7234/", {200}):
            raise RuntimeError("local Citation Graph did not become healthy on port 7234")
        if not _wait_for(f"http://{settings.host}:{settings.port}/healthz", {200}):
            raise RuntimeError("Citation Graph OAuth proxy did not become healthy")
        if not _wait_for(f"{settings.public_url}/healthz", {200}):
            raise RuntimeError("public Citation Graph Funnel endpoint did not become healthy")
    except Exception as exc:
        restore_failures = _restore_service_snapshots(snapshots, domain)
        if restore_failures:
            raise RuntimeError(
                f"Citation Graph install failed and rollback was incomplete: "
                f"{'; '.join(restore_failures)}"
            ) from exc
        raise
    print(f"local={settings.local_url}")
    print(f"public={settings.public_url}")


def restart() -> None:
    missing = [service.label for service in service_definitions() if not _loaded(service.label)]
    if missing:
        raise RuntimeError(f"Citation Graph LaunchAgent is not loaded: {', '.join(missing)}")
    for service in service_definitions():
        _launchctl("kickstart", "-k", f"{launch_domain()}/{service.label}")
    print("restart=requested")


def uninstall() -> None:
    domain = launch_domain()
    for service in service_definitions():
        if _loaded(service.label):
            _launchctl("bootout", f"{domain}/{service.label}")
        destination = launch_agent_path(service.label)
        if destination.exists():
            destination.unlink()
    print("uninstalled")
    print("funnel_8443=unchanged")


def status() -> int:
    settings = _settings()
    local_loaded = _loaded(LOCAL_LABEL)
    remote_loaded = _loaded(REMOTE_LABEL)
    update_check_loaded = _loaded(UPDATE_CHECK_LABEL)
    local_status = _http_status("http://127.0.0.1:7234/")
    proxy_status = _http_status(f"http://{settings.host}:{settings.port}/healthz")
    public_status = _http_status(f"{settings.public_url}/healthz")
    print(f"local_launch_agent={'loaded' if local_loaded else 'not_loaded'}")
    print(f"remote_launch_agent={'loaded' if remote_loaded else 'not_loaded'}")
    print(f"update_check_launch_agent={'loaded' if update_check_loaded else 'not_loaded'}")
    print(f"local_graph={'ok' if local_status == 200 else local_status or 'unreachable'}")
    print(f"local_oauth_proxy={'ok' if proxy_status == 200 else proxy_status or 'unreachable'}")
    print(f"public_oauth_proxy={'ok' if public_status == 200 else public_status or 'unreachable'}")
    print(f"local_stdout_log={ROOT / 'data' / 'citation-graph.stdout.log'}")
    print(f"local_stderr_log={ROOT / 'data' / 'citation-graph.stderr.log'}")
    print(f"remote_stdout_log={ROOT / 'data' / 'citation-graph-remote.stdout.log'}")
    print(f"remote_stderr_log={ROOT / 'data' / 'citation-graph-remote.stderr.log'}")
    print(f"update_check_stdout_log={ROOT / 'data' / 'citation-graph-update-check.stdout.log'}")
    print(f"update_check_stderr_log={ROOT / 'data' / 'citation-graph-update-check.stderr.log'}")
    healthy = local_loaded and remote_loaded and update_check_loaded and all(
        code == 200 for code in (local_status, proxy_status, public_status)
    )
    return 0 if healthy else 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("install", "status", "restart", "uninstall"))
    action = parser.parse_args().action
    if action == "install":
        install()
        return 0
    if action == "restart":
        restart()
        return 0
    if action == "uninstall":
        uninstall()
        return 0
    return status()


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
