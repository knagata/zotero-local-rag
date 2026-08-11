from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Mapping


_log = logging.getLogger(__name__)


def _dotenv_value(raw: str) -> str:
    """Parse the small dotenv value subset used by this project."""
    value = raw.strip()
    if value[:1] in {"'", '"'}:
        quote = value[0]
        escaped = False
        for index, char in enumerate(value[1:], start=1):
            if char == quote and not escaped:
                return value[1:index]
            escaped = char == "\\" and not escaped
            if char != "\\":
                escaped = False
        return value
    for index, char in enumerate(value):
        if char == "#" and (index == 0 or value[index - 1].isspace()):
            return value[:index].rstrip()
    return value


def dotenv_values_native(project_root: Path) -> dict[str, str]:
    """Read the project's dotenv files using the runtime precedence rules."""
    values: dict[str, str] = {}
    for env_file in (project_root / ".env", project_root / ".env.policy"):
        if not env_file.exists():
            continue
        try:
            with open(env_file, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith("#") or "=" not in line:
                        continue
                    k, v = line.split("=", 1)
                    k = k.strip()
                    v = _dotenv_value(v)
                    if k:
                        values.setdefault(k, v)
        except (OSError, UnicodeError) as exc:
            _log.warning("Could not read dotenv file %s: %s", env_file, exc)
    return values


def environment_with_saved_dotenv(
    project_root: Path, base: Mapping[str, str] | None = None,
) -> dict[str, str]:
    """Return a child environment where saved config beats stale parent values."""
    environment = dict(os.environ if base is None else base)
    environment.update(dotenv_values_native(project_root))
    return environment


def load_dotenv_native(project_root: Path | None = None) -> None:
    """Load .env and optional .env.policy without overriding existing vars.

    Detects project root automatically via the caller's file location when
    project_root is not provided.
    """
    if project_root is None:
        import inspect

        frame = inspect.currentframe()
        try:
            caller_file = frame.f_back.f_globals.get("__file__")
        finally:
            del frame
        if caller_file:
            project_root = Path(caller_file).resolve().parents[1]
        else:
            return

    for key, value in dotenv_values_native(project_root).items():
        os.environ.setdefault(key, value)
