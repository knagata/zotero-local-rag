#!/usr/bin/env python3
"""Non-destructive V3 DB audit: Zotero reconciliation + source verification +
completeness gate.

This is the one implementation of "phase 2" shared by Setup.command (right
after an initial build) and Maintenance-Widget.command (routine, opt-in).
There used to be a third .command file (Server-Database-Workflow.command)
whose only remaining reason to exist was hosting this sequence -- collapsing
it here removes that file without duplicating the sequence into two bash
scripts (2026-07-30 user decision).

Read-only against Zotero and the source files; the only thing it writes is
the audit reports and the gate file itself.
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.env_utils import load_dotenv_native  # noqa: E402

load_dotenv_native(ROOT)

from src.v3_data_plane import V3_COLLECTION, enforce_environment  # noqa: E402


def _run(*args: str) -> None:
    print(f"\n>> {' '.join(args)}")
    result = subprocess.run(["uv", "run", *args], cwd=ROOT)
    if result.returncode != 0:
        raise SystemExit(result.returncode)


def main() -> None:
    gate_path = Path(os.environ.get(
        "SERVER_DB_GATE_PATH", str(ROOT / "data" / "quality" / "server_database_gate.json"),
    ))
    # A failed re-audit must never leave a previous passing gate available for
    # the summary step, so the invalidation happens before anything that can
    # fail -- including the environment guard below. Its path does not depend
    # on that guard, and erring toward "no gate" only costs a re-audit.
    if gate_path.exists():
        print(f"[情報] 前回のDB監査合格証明を無効化します: {gate_path}")
        gate_path.unlink()
    gate_path.parent.mkdir(parents=True, exist_ok=True)

    # Fail-closed before any check runs: refuses a non-V3 collection, a
    # non-canonical manifest/lexical filename or a pipeline config outside the
    # Chroma directory, and publishes the resolved paths back into the
    # environment for the three subprocesses below. The guard states the
    # problem plainly rather than letting a RuntimeError traceback be the
    # only explanation the caller sees.
    try:
        enforce_environment(ROOT)
    except RuntimeError as exc:
        raise SystemExit(f"[停止] {exc}") from exc

    chroma_dir = Path(os.environ["CHROMA_DIR"])
    pipeline_config = Path(os.environ["PIPELINE_CONFIG_PATH"])
    manifest = Path(os.environ["MANIFEST_PATH"])
    lexical_db = Path(os.environ["LEXICAL_DB_PATH"])
    collection = os.environ.get("CHROMA_COLLECTION", V3_COLLECTION)

    zotero_audit_path = Path(os.environ.get(
        "SERVER_ZOTERO_AUDIT_PATH", str(ROOT / "data" / "quality" / "server_zotero_reconciliation.json"),
    ))
    source_audit_path = Path(os.environ.get(
        "SERVER_SOURCE_AUDIT_PATH", str(ROOT / "data" / "quality" / "server_source_verification.json"),
    ))

    _run(
        "python", "scripts/verify_zotero_reconciliation.py",
        "--manifest", str(manifest), "--output", str(zotero_audit_path),
    )
    _run(
        "python", "scripts/verify_against_source.py",
        "--collection", collection, "--manifest", str(manifest),
        "--chroma-dir", str(chroma_dir), "--output", str(source_audit_path),
    )
    _run(
        "python", "scripts/audit_v3_cutover.py",
        "--new-only", "--new-collection", collection, "--manifest", str(manifest),
        "--chroma-dir", str(chroma_dir),
        "--lexical-db", str(lexical_db), "--pipeline-config", str(pipeline_config),
        "--zotero-report", str(zotero_audit_path), "--source-report", str(source_audit_path),
        "--output", str(gate_path),
    )

    print(f"\n[合格] 現在のDB世代に結び付いた要約実行の合格証明を作成しました: {gate_path}")


if __name__ == "__main__":
    main()
