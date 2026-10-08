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

from src.env_utils import environment_with_saved_dotenv, load_dotenv_native  # noqa: E402

load_dotenv_native(ROOT)

from src.v3_data_plane import V3_COLLECTION, enforce_environment  # noqa: E402
from src.v3_data_plane import (  # noqa: E402
    chroma_dir as v3_chroma_dir, lexical_path, manifest_path, resolve_configured_path,
)


def _run(*args: str) -> None:
    print(f"\n>> {' '.join(args)}")
    command = list(args)
    if command and command[0] == "python":
        # Browser-admin jobs run under the project's virtualenv, while
        # launchd's deliberately small PATH may not include the user's uv.
        # Reuse the interpreter which is already running this audit.
        command[0] = sys.executable
    result = subprocess.run(command, cwd=ROOT)
    if result.returncode != 0:
        raise SystemExit(result.returncode)


def _validated_report_paths() -> tuple[Path, Path, Path]:
    """Validate all outputs before invalidating a gate or starting a writer.

    Read raw settings here so even an invalid V3 environment cannot turn gate
    invalidation into deletion of a configured input. Resolve aliases once and
    pass these same absolute paths to the children.
    """
    def configured(name: str, default: Path) -> Path:
        return resolve_configured_path(ROOT, os.environ.get(name, default)).resolve()

    chroma = configured("CHROMA_DIR", ROOT / "data" / "chroma")
    protected = [
        configured("MANIFEST_PATH", ROOT / "data" / "manifest_v3.json"),
        configured("PIPELINE_CONFIG_PATH", chroma / "embedder_config_v3.json"),
    ]
    for name, default in (
        ("LEXICAL_DB_PATH", ROOT / "data" / "lexical_v3.sqlite3"),
        ("RELATIONS_DB_PATH", ROOT / "data" / "relations.db"),
    ):
        database = configured(name, default)
        protected.extend(Path(str(database) + suffix).resolve()
                         for suffix in ("", "-wal", "-shm", "-journal"))

    outputs = tuple(configured(name, ROOT / "data" / "quality" / default) for name, default in (
        ("SERVER_DB_GATE_PATH", "server_database_gate.json"),
        ("SERVER_ZOTERO_AUDIT_PATH", "server_zotero_reconciliation.json"),
        ("SERVER_SOURCE_AUDIT_PATH", "server_source_verification.json"),
    ))

    def aliases(left: Path, right: Path) -> bool:
        return left == right or (left.exists() and right.exists() and left.samefile(right))

    for index, output in enumerate(outputs):
        if output == chroma or chroma in output.parents or output.is_dir():
            raise RuntimeError(f"Unsafe DB-audit output path: {output}")
        if any(aliases(output, source) for source in protected):
            raise RuntimeError(f"DB-audit output overlaps protected data: {output}")
        if any(aliases(output, other) or output in other.parents or other in output.parents
               for other in outputs[:index]):
            raise RuntimeError(f"DB-audit output paths collide: {output}")
    return outputs


def _refresh_search_snapshot() -> None:
    from src.search_snapshot import pointer_path
    if pointer_path(ROOT).exists():
        print("[検索DB更新] 全監査に合格した資料を検索側へ反映します。", flush=True)
        subprocess.run([
            sys.executable, str(ROOT / "scripts" / "publish_search_snapshot.py"),
            "--restart-services",
        ], cwd=ROOT, env=environment_with_saved_dotenv(ROOT), check=True)


def main() -> None:
    try:
        gate_path, zotero_audit_path, source_audit_path = _validated_report_paths()
    except (RuntimeError, OSError, ValueError) as exc:
        raise SystemExit(f"[停止] {exc}") from exc
    # A failed re-audit must never leave a previous passing gate available for
    # the summary step, so the invalidation happens before anything that can
    # fail -- including the environment guard below -- after output safety has
    # been established. An unsafe output configuration must change no files.
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

    # Asked of v3_data_plane rather than read from the environment here: it
    # owns both the default location and the rule that expands ~ and resolves
    # a relative value against the project root. Copies of that rule have
    # drifted twice, each time leaving one configuration naming two databases.
    chroma_directory = v3_chroma_dir(ROOT)
    pipeline_config = Path(os.environ["PIPELINE_CONFIG_PATH"])
    manifest = manifest_path(ROOT)
    lexical_db = lexical_path(ROOT)
    collection = os.environ.get("CHROMA_COLLECTION", V3_COLLECTION)

    _run(
        "python", "scripts/verify_zotero_reconciliation.py",
        "--manifest", str(manifest), "--output", str(zotero_audit_path),
    )
    _run(
        "python", "scripts/verify_against_source.py",
        "--collection", collection, "--manifest", str(manifest),
        "--chroma-dir", str(chroma_directory), "--output", str(source_audit_path),
    )
    _run(
        "python", "scripts/audit_v3_cutover.py",
        "--new-only", "--new-collection", collection, "--manifest", str(manifest),
        "--chroma-dir", str(chroma_directory),
        "--lexical-db", str(lexical_db), "--pipeline-config", str(pipeline_config),
        "--zotero-report", str(zotero_audit_path), "--source-report", str(source_audit_path),
        "--output", str(gate_path),
    )

    print(f"\n[合格] 現在のDB世代に結び付いた要約実行の合格証明を作成しました: {gate_path}")
    _refresh_search_snapshot()


if __name__ == "__main__":
    main()
