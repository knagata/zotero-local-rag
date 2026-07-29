from __future__ import annotations

import os
import subprocess
from pathlib import Path
from tempfile import TemporaryDirectory


ROOT = Path(__file__).resolve().parents[1]
COMMAND = ROOT / "Server-Database-Workflow.command"


def _run(
    phase: str, user_input: str = "", *, extra_env: dict[str, str] | None = None,
    existing_gate: bool = False, report_gate: bool = False,
):
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
            "printf '%s\\n' \"$*\" >> \"$SERVER_TEST_LOG\"\n"
            "if [[ -n \"${SERVER_TEST_FAIL_MATCH:-}\" && \"$*\" == *\"$SERVER_TEST_FAIL_MATCH\"* ]]; then exit 7; fi\n",
            encoding="utf-8",
        )
        uv.chmod(0o755)
        env = os.environ.copy()
        gate = root / "gate.json"
        if existing_gate:
            gate.write_text('{"gate": {"passed": true}}', encoding="utf-8")
        env.update({
            "HOME": str(root),
            "SERVER_TEST_LOG": str(log),
            "SERVER_WORKFLOW_PHASE": phase,
            "SERVER_DB_GATE_PATH": str(gate),
        })
        env.update(extra_env or {})
        result = subprocess.run(
            ["/bin/bash", str(COMMAND)], input=user_input, text=True,
            capture_output=True, env=env, timeout=10,
        )
        calls = log.read_text(encoding="utf-8").splitlines() if log.exists() else []
        if report_gate:
            return result, calls, gate.exists()
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


def test_audit_phase_reconciles_zotero_and_sources_before_creating_gate():
    result, calls = _run("2")
    assert result.returncode == 0, result.stdout + result.stderr
    assert [
        "scripts/verify_zotero_reconciliation.py",
        "scripts/verify_against_source.py",
        "scripts/audit_v3_cutover.py",
    ] == [next(part for part in call.split() if part.startswith("scripts/")) for call in calls]
    assert "--manifest " in calls[0] and "manifest_v3.json" in calls[0]
    assert "--collection zotero_paragraphs_v3" in calls[1]
    assert "--chroma-dir data/chroma" in calls[1]
    assert "--zotero-report data/quality/server_zotero_reconciliation.json" in calls[2]
    assert "--source-report data/quality/server_source_verification.json" in calls[2]


def test_failed_phase_two_invalidates_the_old_gate_and_never_runs_the_final_audit():
    result, calls, gate_exists = _run(
        "2", existing_gate=True, report_gate=True,
        extra_env={"SERVER_TEST_FAIL_MATCH": "verify_against_source.py"},
    )
    assert result.returncode == 7
    assert any("verify_zotero_reconciliation.py" in call for call in calls)
    assert any("verify_against_source.py" in call for call in calls)
    assert not any("audit_v3_cutover.py" in call for call in calls)
    assert gate_exists is False


def test_workflow_rejects_inherited_legacy_collection_before_any_command():
    result, calls = _run(
        "1", "REBUILD\n", extra_env={"CHROMA_COLLECTION": "zotero_paragraphs"},
    )
    assert result.returncode == 2
    assert calls == []
    assert "旧コレクション" in result.stdout
