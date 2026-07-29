from __future__ import annotations

import os
import subprocess
from pathlib import Path
from tempfile import TemporaryDirectory


ROOT = Path(__file__).resolve().parents[1]
COMMAND = ROOT / "Server-Database-Workflow.command"


def _run(phase: str, user_input: str = ""):
    with TemporaryDirectory() as directory:
        root = Path(directory)
        bin_dir = root / ".local" / "bin"
        bin_dir.mkdir(parents=True)
        log = root / "calls.log"
        uv = bin_dir / "uv"
        uv.write_text(
            "#!/bin/bash\n"
            "if [[ \"$*\" == *\"--rebuild\"* && "
            "( \"${INGEST_STRUCTURED_V3_ENABLE:-}\" != \"1\" || "
            "-z \"${CHROMA_COLLECTION:-}\" ) ]]; then exit 9; fi\n"
            "printf '%s\\n' \"$*\" >> \"$SERVER_TEST_LOG\"\n",
            encoding="utf-8",
        )
        uv.chmod(0o755)
        env = os.environ.copy()
        env.update({
            "HOME": str(root),
            "SERVER_TEST_LOG": str(log),
            "SERVER_WORKFLOW_PHASE": phase,
            "SERVER_DB_GATE_PATH": str(root / "gate.json"),
        })
        result = subprocess.run(
            ["/bin/bash", str(COMMAND)], input=user_input, text=True,
            capture_output=True, env=env, timeout=10,
        )
        calls = log.read_text(encoding="utf-8").splitlines() if log.exists() else []
        return result, calls


def test_rebuild_phase_never_invokes_summary_builder():
    result, calls = _run("1", "REBUILD\n")
    assert result.returncode == 0
    assert any("--rebuild" in call for call in calls)
    assert not any("build_structure_summaries.py" in call for call in calls)


def test_summary_phase_stops_without_database_gate():
    result, calls = _run("3")
    assert result.returncode == 2
    assert calls == []
    assert "先にフェーズ2" in result.stdout
