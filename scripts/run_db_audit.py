#!/usr/bin/env python3
"""V3 DB audit with backed-up, bounded derived-data repair: Zotero reconciliation + source verification +
completeness gate.

This is the one implementation of "phase 2" shared by Setup.command (right
after an initial build) and Maintenance-Widget.command (routine, opt-in).
There used to be a third .command file (Server-Database-Workflow.command)
whose only remaining reason to exist was hosting this sequence -- collapsing
it here removes that file without duplicating the sequence into two bash
scripts (2026-07-30 user decision).

Read-only against Zotero and source files. Known derived-data inconsistencies
may be repaired after backup, followed by all three audits again.
"""
from __future__ import annotations

import os
import json
import subprocess
import sys
import tempfile
import sqlite3
from datetime import datetime, timezone
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
from src.indexing_lock import held, default_path  # noqa: E402


class AuditFailure(SystemExit):
    def __init__(self, code: int, report: dict | None = None):
        super().__init__(code)
        self.report = report


def _run(*args: str) -> None:
    print(f"\n>> {' '.join(args)}")
    command = list(args)
    if command and command[0] == "python":
        # Browser-admin jobs run under the project's virtualenv, while
        # launchd's deliberately small PATH may not include the user's uv.
        # Reuse the interpreter which is already running this audit.
        command[0] = sys.executable
    output = Path(command[command.index("--output") + 1]) if "--output" in command else None
    previous = output.stat().st_mtime_ns if output and output.exists() else None
    result = subprocess.run(command, cwd=ROOT)
    if result.returncode != 0:
        _explain_failure(output, previous, result.returncode)
        report = None
        if output and output.exists() and output.stat().st_mtime_ns != previous:
            try:
                value = json.loads(output.read_text(encoding="utf-8"))
                if isinstance(value, dict):
                    report = value
            except (OSError, ValueError):
                report = None
        raise AuditFailure(result.returncode, report)


def _explain_failure(output: Path | None, previous: int | None, exit_code: int) -> None:
    """Explain only the report produced by this invocation, never an old one."""
    report = {}
    try:
        if output and output.exists() and output.stat().st_mtime_ns != previous:
            report = json.loads(output.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        report = {}
    if not isinstance(report, dict):
        report = {}
    messages = []
    for document in report.get("documents", []):
        if document.get("lost_pages"):
            pages = document["lost_pages"]
            messages.append(f"『{document.get('title') or '名称不明'}』"
                            f"（添付 {document.get('attachment_key')}）：本文が索引にないページ {len(pages)}頁。"
                            f"対象ページ：{', '.join(map(str, pages[:20]))}"
                            + (" ほか" if len(pages) > 20 else ""))
    missing = report.get("missing_required_from_manifest", [])
    if missing:
        messages.append(f"未索引の添付が{len(missing)}件あります。対象：{missing[:10]}")
    summary = report.get("summary", {})
    for key, label in (("orphan_chunks", "原本と対応しない本文"),
                       ("dangling_node_chunks", "文書構造との不整合"),
                       ("unretrievable_items", "検索不能な資料"),
                       ("unreadable_source_documents", "読み取れない原本")):
        if summary.get(key):
            messages.append(f"{label}：{summary[key]}件")
    labels = {
        "manifest_chroma_attachment_mismatch": "索引管理記録と本文索引の添付が一致していません",
        "incomplete_source_coverage": "原本の抽出範囲が不足、または確認できません",
        "chroma_lexical_id_mismatch": "全文索引とベクトル索引が一致していません",
        "hnsw_not_validated": "ベクトル索引の検証が完了していません",
        "inflight_attachments": "取込が途中の添付があります",
        "post_index_pending": "索引後の処理が未完了です",
    }
    gate = report.get("gate", {})
    from src.audit_diagnostics import item_failure_details
    details = item_failure_details(report)
    for detail in details:
        messages.append(f"『{detail['title']}』（{detail['item_key']}）：{detail['message']}")
    messages.extend(labels.get(reason, f"整合性検査：{reason}")
                    for reason in gate.get("global_failures", []))
    if gate.get("failed_items"):
        messages.append(f"資料単位の整合性検査で{gate['failed_items']}件が不合格です")
    if not messages:
        messages.append(f"監査処理が終了コード{exit_code}で停止しました。実行ログの直前のエラーを確認してください")
    for message in messages:
        print(f"[監査失敗] {message}", flush=True)
    action = ("対象資料の取込状態を確認してください。OCR待ちの場合は「Mistral OCR」を実行し、"
              "回収・採用後にクイック実行とDB監査を行ってください。有料APIの確認が必要です。"
              if any(document.get("lost_pages") for document in report.get("documents", [])) or missing
              else "原本・取込状態と実行ログを確認してください。索引不整合は対象を特定して修復後、DB監査を再実行してください。")
    print(f"[必要な作業] {action}", flush=True)
    for action in dict.fromkeys(detail["action"] for detail in details):
        print(f"[必要な作業] {action}", flush=True)


def _validated_report_paths() -> tuple[Path, Path, Path, Path]:
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
        ("SERVER_DB_AUDIT_STATUS_PATH", "server_database_audit_status.json"),
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


def _audit_once() -> None:
    try:
        gate_path, zotero_audit_path, source_audit_path, _status_path = _validated_report_paths()
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


def _write_status(path: Path, status: str, message: str, **details) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    value = {**details, "status": status, "message": message,
             "pid": os.getpid(),
             "updated_at": datetime.now(timezone.utc).isoformat()}
    with tempfile.NamedTemporaryFile(mode="w", dir=path.parent, delete=False,
                                     encoding="utf-8") as stream:
        temporary = Path(stream.name)
        json.dump(value, stream, ensure_ascii=False)
    try:
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _discard_passing_gate(path: Path) -> None:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return  # unreadable or invalid reports cannot authorize summaries
    if isinstance(value, dict) and isinstance(value.get("gate"), dict) and value["gate"].get("passed") is True:
        path.unlink()


def _refresh_search_snapshot() -> None:
    from src.search_snapshot import pointer_path
    if pointer_path(ROOT).exists():
        print("[検索DB更新] 全監査に合格した資料を検索側へ反映します。", flush=True)
        subprocess.run([
            sys.executable, str(ROOT / "scripts" / "publish_search_snapshot.py"),
            "--restart-services",
        ], cwd=ROOT, env=environment_with_saved_dotenv(ROOT), check=True)


def main() -> None:
    from src.audit_repair import repair_derived_data
    try:
        paths = _validated_report_paths()
    except (RuntimeError, OSError, ValueError) as exc:
        raise SystemExit(f"[停止] {exc}") from exc
    status_path = paths[3]
    # Keep the same generation stable across checks, backup, repair and re-audit.
    with held(default_path(ROOT)):
        _write_status(status_path, "running", "DB監査を実行しています。")
        repair = {}
        findings = []
        try:
            try:
                _audit_once()
            except AuditFailure as failure:
                from src.audit_diagnostics import report_failure_details
                findings = report_failure_details(failure.report or {})
                if not failure.report:
                    raise
                _write_status(status_path, "repairing", "監査不合格の原因を確認し、安全な自動修復を検査しています。")
                repair = repair_derived_data(failure.report, ROOT)
                if not repair["repaired"]:
                    raise
                _write_status(status_path, "running", "自動修復後、同じ基準で全DB監査を再実行しています。", repair=repair)
                _audit_once()  # exactly one retry, all three original audits
        except (SystemExit, RuntimeError, OSError, ValueError, sqlite3.Error, KeyboardInterrupt) as exc:
            if isinstance(exc, AuditFailure):
                from src.audit_diagnostics import report_failure_details
                findings = report_failure_details(exc.report or {})
            # A child may publish a passing report and then crash. Its nonzero
            # exit must not leave a usable success certificate behind.
            _discard_passing_gate(paths[0])
            message = ("DB監査は合格していません。原因と必要な対応は下記の監査結果・実行ログを確認してください。"
                       if isinstance(exc, SystemExit) else f"自動修復を停止しました：{exc}。DB監査は未合格です。")
            _write_status(status_path, "failed", message, repair=repair, findings=findings)
            print(f"[必要な作業] {message}", flush=True)
            raise
        _write_status(status_path, "completed", "DB監査に合格しました。", repair=repair)
    _refresh_search_snapshot()


if __name__ == "__main__":
    main()
