from __future__ import annotations

import os
from pathlib import Path


def load_dotenv_native(project_root: Path | None = None) -> None:
    """Load .env file into os.environ (does not override existing vars).

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

    env_file = project_root / ".env"
    if not env_file.exists():
        return

    try:
        with open(env_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                k, v = line.split("=", 1)
                k = k.strip()
                v = v.strip()
                if len(v) >= 2 and (
                    (v.startswith('"') and v.endswith('"'))
                    or (v.startswith("'") and v.endswith("'"))
                ):
                    v = v[1:-1]
                if k and k not in os.environ:
                    os.environ[k] = v
    except Exception:
        pass
