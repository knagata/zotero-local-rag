#!/usr/bin/env python3
"""Install and operate the macOS LaunchAgent for the remote MCP server."""
from __future__ import annotations

import argparse
import os
import plistlib
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))

from env_utils import load_dotenv_native  # noqa: E402
from rag_mcp_http_server import RemoteMCPSettings  # noqa: E402


LABEL = "dev.zotero-local-rag.remote-mcp"


def launch_agent_path(home: Path | None = None) -> Path:
    base = Path.home() if home is None else home
    return base / "Library" / "LaunchAgents" / f"{LABEL}.plist"


def launch_domain(uid: int | None = None) -> str:
    return f"gui/{os.getuid() if uid is None else uid}"


def build_launch_agent(root: Path = ROOT) -> dict[str, Any]:
    python = root / ".venv" / "bin" / "python"
    if not python.is_file():
        raise RuntimeError(f"Project Python not found: {python}. Run uv sync first.")
    data = root / "data"
    return {
        "Label": LABEL,
        "ProgramArguments": [
            str(python), "-u", str(root / "src" / "rag_mcp_http_server.py"),
        ],
        "WorkingDirectory": str(root),
        "RunAtLoad": True,
        "KeepAlive": {"SuccessfulExit": False},
        "ThrottleInterval": 10,
        # Background launch jobs can be speculatively I/O-throttled while
        # Python imports the large local environment. This is a user-facing
        # query service, so launch it at normal interactive priority.
        "ProcessType": "Interactive",
        "StandardOutPath": str(data / "remote-mcp.stdout.log"),
        "StandardErrorPath": str(data / "remote-mcp.stderr.log"),
        "EnvironmentVariables": {"PYTHONUNBUFFERED": "1"},
    }


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
    with tempfile.NamedTemporaryFile(dir=destination.parent, delete=False) as temporary:
        temporary.write(content)
        candidate = Path(temporary.name)
    candidate.chmod(0o600)
    os.replace(candidate, destination)


def _launchctl(*arguments: str, check: bool = True) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["launchctl", *arguments], text=True, capture_output=True, check=check,
    )


def _loaded() -> bool:
    result = _launchctl("print", f"{launch_domain()}/{LABEL}", check=False)
    return result.returncode == 0


def _bootstrap(destination: Path, attempts: int = 20) -> None:
    result: subprocess.CompletedProcess[str] | None = None
    for attempt in range(attempts):
        result = _launchctl("bootstrap", launch_domain(), str(destination), check=False)
        if result.returncode == 0:
            return
        if attempt + 1 < attempts:
            time.sleep(0.1)
    assert result is not None
    raise subprocess.CalledProcessError(
        result.returncode, result.args, output=result.stdout, stderr=result.stderr,
    )


def _restore_previous(destination: Path, content: bytes | None, was_loaded: bool) -> None:
    domain = launch_domain()
    if _loaded():
        _launchctl("bootout", f"{domain}/{LABEL}", check=False)
    if content is None:
        if destination.exists():
            destination.unlink()
    else:
        _write_private_bytes(destination, content)
    if was_loaded and content is not None:
        _bootstrap(destination)
        _launchctl("enable", f"{domain}/{LABEL}")


def install() -> None:
    load_dotenv_native(ROOT)
    settings = RemoteMCPSettings.from_environment()
    payload = build_launch_agent()
    (ROOT / "data").mkdir(parents=True, exist_ok=True)
    destination = launch_agent_path()
    previous = destination.read_bytes() if destination.exists() else None
    was_loaded = _loaded()
    domain = launch_domain()
    try:
        write_launch_agent(destination, payload)
        if was_loaded:
            _launchctl("bootout", f"{domain}/{LABEL}")
        _bootstrap(destination)
        _launchctl("enable", f"{domain}/{LABEL}")
        if not wait_for_local_health(settings):
            raise RuntimeError(
                "LaunchAgent loaded but could not start the local OAuth endpoint. "
                f"If the project is under a protected macOS folder, grant Full Disk Access to "
                f"{payload['ProgramArguments'][0]} or move the repository outside Documents, "
                "then run install again."
            )
    except Exception:
        _restore_previous(destination, previous, was_loaded)
        raise
    print(f"installed={destination}")
    print(f"endpoint={settings.public_url}/mcp")


def restart() -> None:
    if not _loaded():
        raise RuntimeError("Remote MCP LaunchAgent is not loaded. Run install first.")
    _launchctl("kickstart", "-k", f"{launch_domain()}/{LABEL}")
    print("restart=requested")


def uninstall() -> None:
    if _loaded():
        _launchctl("bootout", f"{launch_domain()}/{LABEL}")
    destination = launch_agent_path()
    if destination.exists():
        destination.unlink()
    print("uninstalled")


def _http_status(url: str) -> int | None:
    try:
        with urllib.request.urlopen(url, timeout=5) as response:
            return response.status
    except urllib.error.HTTPError as exc:
        return exc.code
    except urllib.error.URLError:
        return None


def wait_for_local_health(
    settings: RemoteMCPSettings, *, timeout: float = 15, poll_interval: float = 0.25,
) -> bool:
    endpoint = (
        f"http://{settings.host}:{settings.port}/.well-known/oauth-authorization-server"
    )
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if _http_status(endpoint) == 200:
            return True
        time.sleep(poll_interval)
    return False


def status() -> int:
    load_dotenv_native(ROOT)
    settings = RemoteMCPSettings.from_environment()
    loaded = _loaded()
    local_status = _http_status(
        f"http://{settings.host}:{settings.port}/.well-known/oauth-authorization-server"
    )
    public_status = _http_status(
        f"{settings.public_url}/.well-known/oauth-authorization-server"
    )
    print(f"launch_agent={'loaded' if loaded else 'not_loaded'}")
    print(f"local_oauth={'ok' if local_status == 200 else local_status or 'unreachable'}")
    print(f"public_oauth={'ok' if public_status == 200 else public_status or 'unreachable'}")
    print(f"stdout_log={ROOT / 'data' / 'remote-mcp.stdout.log'}")
    print(f"stderr_log={ROOT / 'data' / 'remote-mcp.stderr.log'}")
    return 0 if loaded and local_status == 200 and public_status == 200 else 1


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
