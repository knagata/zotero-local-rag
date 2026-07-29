from __future__ import annotations

import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _run(script: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            sys.executable, str(ROOT / "scripts" / script), "--all", "--dry-run",
            "--collection", "zotero_paragraphs",
        ],
        cwd=ROOT, text=True, capture_output=True, timeout=15,
    )


def test_structure_rebuild_rejects_legacy_collection_argument():
    result = _run("rebuild_document_structure.py")
    assert result.returncode == 2
    assert "legacy data plane is retired" in result.stderr


def test_structure_summary_rejects_legacy_collection_argument():
    result = _run("build_structure_summaries.py")
    assert result.returncode == 2
    assert "legacy data plane is retired" in result.stderr


def test_new_only_audit_cannot_seal_gate_without_external_reports(tmp_path: Path):
    result = subprocess.run(
        [
            sys.executable, str(ROOT / "scripts" / "audit_v3_cutover.py"),
            "--new-only", "--output", str(tmp_path / "gate.json"),
        ],
        cwd=ROOT, text=True, capture_output=True, timeout=15,
    )
    assert result.returncode == 2
    assert "--new-only requires --zotero-report and --source-report" in result.stderr
    assert not (tmp_path / "gate.json").exists()
