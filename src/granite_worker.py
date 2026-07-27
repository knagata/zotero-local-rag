# src/granite_worker.py
"""Call Granite-Docling in its own virtualenv, from the main application.

``DoclingWorker`` isolates Docling in a subprocess to survive crashes, but it
can use ``multiprocessing`` because the child runs the *same* interpreter.
Granite cannot: ``mlx-vlm`` needs ``transformers>=5.14`` and the Docling
pipeline pins ``<5.9`` on macOS, so Granite lives in a separate virtualenv with
a separate interpreter. The boundary is therefore a plain subprocess speaking
JSON over pipes (``scripts/granite_runner.py``), not a forked worker.

One process per document, rather than the persistent worker Docling uses. Model
load costs a few seconds, which is small against Granite's own runtime -- it
measured ~2.3x slower than Docling per page -- and a fresh process means a
crash or a leaked MLX buffer cannot accumulate across a long batch.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

try:
    from .feature_gates import DEFAULT_GRANITE_PYTHON
except ImportError:  # direct execution with src/ on sys.path
    from feature_gates import DEFAULT_GRANITE_PYTHON

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RUNNER = PROJECT_ROOT / "scripts" / "granite_runner.py"

#: Generous: Granite is roughly 2.3x slower than Docling, and Docling's own
#: external backstop is an hour. This only catches a genuinely wedged process.
DEFAULT_TIMEOUT_SEC = 10800.0


class GraniteUnavailable(RuntimeError):
    """The Granite environment is missing or unusable."""


def granite_python() -> Path:
    configured = os.environ.get("GRANITE_VENV_PYTHON", "").strip() or DEFAULT_GRANITE_PYTHON
    path = Path(configured)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path


class GraniteWorker:
    """Extract PDFs with Granite-Docling through its isolated interpreter.

    ``extract()`` returns ``(chunks, quality_info)`` exactly like
    ``DoclingWorker.extract`` so the router can treat the two interchangeably.
    Failures raise ``RuntimeError``, which the caller already handles as an
    ordinary extraction failure for one document.
    """

    def __init__(self, timeout_sec: float = DEFAULT_TIMEOUT_SEC):
        self.timeout_sec = float(timeout_sec)

    def available(self) -> bool:
        return granite_python().exists() and RUNNER.exists()

    def extract(
        self, pdf_path: Path, attachment_key: str, meta_base: dict,
    ) -> Tuple[List[Any], Dict[str, Any]]:
        interpreter = granite_python()
        if not interpreter.exists():
            raise GraniteUnavailable(
                f"Granite interpreter not found at {interpreter}. Create the isolated "
                "environment (see dev-notes/current/68_granite_docling_mlx_bakeoff.md) "
                "or set GRANITE_VENV_PYTHON."
            )
        request = json.dumps({
            "pdf_path": str(pdf_path),
            "attachment_key": attachment_key,
            "meta_base": dict(meta_base),
        })
        try:
            completed = subprocess.run(
                [str(interpreter), str(RUNNER)],
                input=request, capture_output=True, text=True,
                timeout=self.timeout_sec, cwd=str(PROJECT_ROOT),
            )
        except subprocess.TimeoutExpired:
            raise RuntimeError(
                f"Granite worker timed out after {self.timeout_sec}s: {pdf_path}"
            )
        if completed.returncode != 0 and not completed.stdout.strip():
            # No JSON came back at all: the interpreter itself failed to start,
            # or the process was killed. stderr is the only diagnostic.
            detail = (completed.stderr or "").strip()[-500:]
            raise RuntimeError(f"Granite worker failed for {pdf_path}: {detail}")
        try:
            response = json.loads(completed.stdout or "{}")
        except ValueError as exc:
            raise RuntimeError(f"Granite worker returned unparseable output: {exc}")
        if response.get("status") != "ok":
            raise RuntimeError(
                f"Granite extraction failed for {pdf_path}: "
                f"{response.get('message') or 'unknown error'}"
            )
        chunks = [
            (str(row[0]), str(row[1]), dict(row[2]))
            for row in response.get("chunks") or []
            if isinstance(row, list) and len(row) == 3
        ]
        return chunks, dict(response.get("quality_info") or {})

    def shutdown(self) -> None:
        """No persistent process to stop; present so callers can treat the two
        workers alike."""
        return None


__all__ = ["DEFAULT_TIMEOUT_SEC", "GraniteUnavailable", "GraniteWorker", "granite_python"]
