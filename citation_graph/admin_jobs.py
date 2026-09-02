"""Bounded maintenance jobs and read-only status for the browser admin page."""
from __future__ import annotations

import fcntl
import json
import os
import signal
import sqlite3
import subprocess
import tempfile
import time
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.env_utils import environment_with_saved_dotenv

ROOT = Path(__file__).resolve().parents[1]
JOBS_DIR = ROOT / "data" / "admin_jobs"
ACTIVE_LOCK = JOBS_DIR / ".active.lock"
START_LOCK = JOBS_DIR / ".start.lock"
RUNNER = ROOT / "scripts" / "run_admin_job.py"
UPDATE_STATUS_REPORT = ROOT / "data" / "admin_update_status.json"
UPDATE_STATUS_MAX_AGE_SECONDS = 45 * 60
FRESHNESS_AFFECTING_JOBS = frozenset({
    "quick_update", "library_update", "structure_update", "summary_batch", "citation_update",
})


@dataclass(frozen=True)
class JobDefinition:
    key: str
    label: str
    description: str
    steps: tuple[tuple[str, tuple[str, ...]], ...]
    confirmation: str | None = None
    paid: bool = False


class JobAlreadyRunningError(RuntimeError):
    """A maintenance runner already owns the single-job slot."""


def _python(root: Path = ROOT) -> str:
    value = root / ".venv" / "bin" / "python"
    if not value.is_file():
        raise RuntimeError(f"Project Python not found: {value}")
    return str(value)


def job_definitions(root: Path = ROOT) -> dict[str, JobDefinition]:
    python = _python(root)
    gate = str(root / "data" / "quality" / "server_database_gate.json")
    library_step = (
        "ライブラリ差分更新",
        (python, str(root / "src" / "index_from_zotero.py"), "--progress"),
    )
    structure_step = (
        "文書構造・目次の差分更新",
        (python, str(root / "scripts" / "rebuild_document_structure.py"), "--all"),
    )
    audit_step = (
        "DB監査",
        (python, str(root / "scripts" / "run_db_audit.py")),
    )
    citation_step = (
        "Citation Network更新",
        (python, str(root / "src" / "update_citations.py"), "--all"),
    )
    return {
        "update_check": JobDefinition(
            "update_check", "更新状況を確認",
            "Zotero・索引・文書構造・Citation台帳を読み取り専用で照合し、未更新件数を表示します。",
            (("更新状況を確認", (
                python, str(root / "scripts" / "check_admin_update_status.py"),
            )),),
        ),
        "quick_update": JobDefinition(
            "quick_update", "クイック実行",
            "索引・文書構造・目次の差分更新、DB監査、Citation Network更新を順番に実行します。",
            (library_step, structure_step, audit_step, citation_step),
            confirmation="QUICK",
        ),
        "library_update": JobDefinition(
            "library_update", "ライブラリ差分更新",
            "Zotero差分を索引へ反映し、文書構造・目次も差分更新します。",
            (library_step, structure_step),
            confirmation="UPDATE",
        ),
        "database_audit": JobDefinition(
            "database_audit", "DB監査",
            "Zotero・原本・索引を監査します。開始時に前回の合格証明を無効化します。",
            (audit_step,),
            confirmation="AUDIT",
        ),
        "structure_update": JobDefinition(
            "structure_update", "文書構造・目次の差分更新",
            "原本の目次と見出しを再確認し、変更資料だけを更新します。再埋め込みはしません。",
            (structure_step,),
            confirmation="STRUCTURE",
        ),
        "summary_batch": JobDefinition(
            "summary_batch", "階層要約の差分更新（10件）",
            "構造更新とDB監査後、未処理・変更資料を最大10件、10並列でDeepSeek要約します。",
            (
                ("文書構造・目次の差分確認", structure_step[1]),
                audit_step,
                ("階層要約10件バッチ", (
                    python, str(root / "scripts" / "build_structure_summaries.py"),
                    "--all", "--mode", "llm", "--limit", "10", "--workers", "10",
                    "--embed", "--database-gate", gate,
                )),
            ),
            confirmation="SUMMARIZE",
            paid=True,
        ),
        "citation_update": JobDefinition(
            "citation_update", "Citation Network更新",
            "未処理・エラー分の引用・参照関係を更新します。外部APIを利用します。",
            (citation_step,),
            confirmation="CITATIONS",
        ),
    }


def public_job_definitions(root: Path = ROOT) -> list[dict[str, Any]]:
    return [
        {
            "key": job.key,
            "label": job.label,
            "description": job.description,
            "confirmation": job.confirmation,
            "paid": job.paid,
        }
        for job in job_definitions(root).values()
    ]


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _record_path(job_id: str, root: Path = ROOT) -> Path:
    if not job_id or any(char not in "0123456789abcdefghijklmnopqrstuvwxyz-" for char in job_id):
        raise ValueError("invalid job id")
    return root / "data" / "admin_jobs" / f"{job_id}.json"


def _log_path(job_id: str, root: Path = ROOT) -> Path:
    return _record_path(job_id, root).with_suffix(".log")


def write_record(record: dict[str, Any], root: Path = ROOT) -> None:
    path = _record_path(str(record["id"]), root)
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = (json.dumps(record, ensure_ascii=False, indent=2) + "\n").encode()
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as temporary:
        temporary.write(encoded)
        temporary.flush()
        os.fsync(temporary.fileno())
        candidate = Path(temporary.name)
    candidate.chmod(0o600)
    os.replace(candidate, path)


def _write_json_private(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = (json.dumps(value, ensure_ascii=False, indent=2) + "\n").encode()
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as temporary:
        temporary.write(encoded)
        temporary.flush()
        os.fsync(temporary.fileno())
        candidate = Path(temporary.name)
    candidate.chmod(0o600)
    os.replace(candidate, path)


def mark_update_status_recheck_pending(
    job_type: str, root: Path = ROOT, *, reason: str = "maintenance_started",
) -> None:
    if job_type not in FRESHNESS_AFFECTING_JOBS:
        return
    path = root / "data" / "admin_update_status.json"
    try:
        report = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        report = {}
    report.update({
        "recheck_required_at": _now(),
        "recheck_reason": reason,
        "recheck_job_type": job_type,
    })
    _write_json_private(path, report)


def schedule_followup_update_check(job_type: str, root: Path = ROOT) -> dict[str, Any] | None:
    if job_type not in FRESHNESS_AFFECTING_JOBS:
        return None
    mark_update_status_recheck_pending(job_type, root, reason="maintenance_completed")
    try:
        return start_job("update_check", "", f"post-job:{job_type}", root)
    except JobAlreadyRunningError:
        # A periodic/manual check won the race. It will replace the pending report.
        return None


def read_record(job_id: str, root: Path = ROOT) -> dict[str, Any]:
    return json.loads(_record_path(job_id, root).read_text(encoding="utf-8"))


def list_records(root: Path = ROOT, limit: int = 30) -> list[dict[str, Any]]:
    directory = root / "data" / "admin_jobs"
    if not directory.exists():
        return []
    records: list[dict[str, Any]] = []
    for path in sorted(directory.glob("*.json"), key=lambda item: item.stat().st_mtime, reverse=True):
        try:
            records.append(json.loads(path.read_text(encoding="utf-8")))
        except (OSError, ValueError):
            continue
        if len(records) >= limit:
            break
    return records


def public_record(record: dict[str, Any] | None) -> dict[str, Any] | None:
    if record is None:
        return None
    return {key: value for key, value in record.items() if key != "token"}


def active_job(root: Path = ROOT) -> dict[str, Any] | None:
    for record in list_records(root):
        if record.get("status") in {"queued", "running", "stopping"}:
            pid = record.get("pid")
            if isinstance(pid, int):
                try:
                    os.kill(pid, 0)
                    command = _runner_command(pid)
                    if (
                        str(root / "scripts" / "run_admin_job.py") in command
                        and str(record.get("id") or "") in command
                        and str(record.get("token") or "") in command
                    ):
                        return record
                except OSError:
                    pass
    return None


def reconcile_stale_jobs(root: Path = ROOT) -> None:
    for record in list_records(root):
        if record.get("status") not in {"queued", "running", "stopping"}:
            continue
        pid = record.get("pid")
        command = _runner_command(pid) if isinstance(pid, int) else ""
        if (
            command
            and str(root / "scripts" / "run_admin_job.py") in command
            and str(record.get("id") or "") in command
            and str(record.get("token") or "") in command
        ):
            continue
        record.update({
            "status": "interrupted", "finished_at": _now(), "current_step": None,
        })
        write_record(record, root)


def start_job(
    job_type: str, confirmation: str, actor: str, root: Path = ROOT,
) -> dict[str, Any]:
    definitions = job_definitions(root)
    if job_type not in definitions:
        raise ValueError("unknown maintenance job")
    definition = definitions[job_type]
    if definition.confirmation and confirmation != definition.confirmation:
        raise ValueError(f"confirmation must be {definition.confirmation}")
    jobs_dir = root / "data" / "admin_jobs"
    jobs_dir.mkdir(parents=True, exist_ok=True)
    lock_fd = os.open(jobs_dir / ".start.lock", os.O_RDWR | os.O_CREAT, 0o600)
    try:
        fcntl.flock(lock_fd, fcntl.LOCK_EX)
        running = active_job(root)
        if running:
            raise JobAlreadyRunningError(f"job already running: {running['id']}")
        job_id = f"{datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:8]}"
        token = uuid.uuid4().hex
        record = {
            "id": job_id,
            "type": definition.key,
            "label": definition.label,
            "status": "queued",
            "created_at": _now(),
            "started_at": None,
            "finished_at": None,
            "current_step": None,
            "step_index": 0,
            "step_total": len(definition.steps),
            "pid": None,
            "exit_code": None,
            "actor": actor,
            "token": token,
        }
        write_record(record, root)
        process = subprocess.Popen(
            [
                _python(root), str(root / "scripts" / "run_admin_job.py"),
                "--job-id", job_id, "--job-type", job_type, "--token", token,
            ],
            cwd=root,
            env=environment_with_saved_dotenv(root),
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
        )
        record["pid"] = process.pid
        write_record(record, root)
        mark_update_status_recheck_pending(job_type, root)
        return record
    finally:
        fcntl.flock(lock_fd, fcntl.LOCK_UN)
        os.close(lock_fd)


def _runner_command(pid: int) -> str:
    result = subprocess.run(
        ["ps", "-p", str(pid), "-o", "command="], capture_output=True, text=True, check=False,
    )
    return result.stdout.strip() if result.returncode == 0 else ""


def stop_job(
    job_id: str, confirmation: str, actor: str, root: Path = ROOT,
) -> dict[str, Any]:
    if confirmation != "STOP":
        raise ValueError("confirmation must be STOP")
    record = read_record(job_id, root)
    if record.get("status") not in {"queued", "running", "stopping"}:
        raise ValueError("job is not running")
    pid = record.get("pid")
    token = str(record.get("token") or "")
    command = _runner_command(pid) if isinstance(pid, int) else ""
    if not command or str(RUNNER) not in command or job_id not in command or token not in command:
        raise RuntimeError("refusing to signal an unverified process")
    record["status"] = "stopping"
    record["stopped_by"] = actor
    write_record(record, root)
    os.killpg(os.getpgid(pid), signal.SIGTERM)
    return record


def log_tail(job_id: str, root: Path = ROOT, max_bytes: int = 64 * 1024) -> str:
    path = _log_path(job_id, root)
    if not path.exists():
        return ""
    with path.open("rb") as stream:
        stream.seek(0, os.SEEK_END)
        size = stream.tell()
        stream.seek(max(0, size - max_bytes))
        value = stream.read().decode("utf-8", errors="replace")
    return value[-max_bytes:]


def _artifact_counts(root: Path) -> dict[str, int | str]:
    database = root / "data" / "relations.db"
    try:
        connection = sqlite3.connect(f"{database.resolve().as_uri()}?mode=ro", uri=True, timeout=5)
        rows = connection.execute(
            "SELECT status, COUNT(*) FROM artifact_processing_status GROUP BY status"
        ).fetchall()
        connection.close()
        counts = {str(status): int(count) for status, count in rows}
        counts["unresolved"] = counts.get("failed", 0) + counts.get("blocked", 0)
        return counts
    except sqlite3.Error as exc:
        return {"error": str(exc)[:200]}


def system_status(root: Path = ROOT) -> dict[str, Any]:
    reconcile_stale_jobs(root)
    manifest_path = root / "data" / "manifest_v3.json"
    gate_path = root / "data" / "quality" / "server_database_gate.json"
    lock_path = root / "data" / "indexing.lock"
    manifest: dict[str, Any] = {}
    gate: dict[str, Any] = {}
    lock: dict[str, Any] | None = None
    update_status: dict[str, Any] | None = None
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        pass
    try:
        gate = json.loads(gate_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        pass
    if lock_path.exists():
        try:
            lock = json.loads(lock_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            lock = {"error": "indexing lock is unreadable"}
    try:
        update_status = json.loads(
            (root / "data" / "admin_update_status.json").read_text(encoding="utf-8")
        )
    except (OSError, ValueError):
        pass
    if update_status is not None:
        generated_at = update_status.get("generated_at")
        age_seconds: float | None = None
        try:
            generated = datetime.fromisoformat(str(generated_at))
            age_seconds = max(0.0, (datetime.now(timezone.utc) - generated).total_seconds())
        except (TypeError, ValueError):
            pass
        update_status = {
            **update_status,
            "age_seconds": age_seconds,
            "stale": age_seconds is None or age_seconds > UPDATE_STATUS_MAX_AGE_SECONDS,
            "recheck_pending": bool(update_status.get("recheck_required_at")),
        }
    return {
        "generated_at": _now(),
        "manifest": {
            "attachments": len(manifest.get("files") or {}),
            "notes": len(manifest.get("notes") or {}),
            "hnsw_validated": manifest.get("hnsw_validated"),
            "inflight": len(manifest.get("inflight_attachments") or []),
            "post_index_pending": len(manifest.get("post_index_pending") or []),
            "modified_at": datetime.fromtimestamp(
                manifest_path.stat().st_mtime, timezone.utc
            ).isoformat() if manifest_path.exists() else None,
        },
        "database_gate": {
            "passed": (gate.get("gate") or {}).get("passed") is True,
            "modified_at": datetime.fromtimestamp(
                gate_path.stat().st_mtime, timezone.utc
            ).isoformat() if gate_path.exists() else None,
        },
        "indexing_lock": lock,
        "artifacts": _artifact_counts(root),
        "update_status": update_status,
        "active_job": public_record(active_job(root)),
        "jobs": [public_record(record) for record in list_records(root, limit=20)],
        "definitions": public_job_definitions(root),
    }


def run_job(job_id: str, job_type: str, token: str, root: Path = ROOT) -> int:
    definition = job_definitions(root).get(job_type)
    if definition is None:
        return 2
    deadline = time.monotonic() + 5
    while True:
        record = read_record(job_id, root)
        if record.get("pid") == os.getpid():
            break
        if record.get("pid") is not None or time.monotonic() >= deadline:
            return 2
        time.sleep(0.01)
    if record.get("token") != token or record.get("type") != job_type:
        return 2
    jobs_dir = root / "data" / "admin_jobs"
    jobs_dir.mkdir(parents=True, exist_ok=True)
    lock_fd = os.open(jobs_dir / ".active.lock", os.O_RDWR | os.O_CREAT, 0o600)
    try:
        try:
            fcntl.flock(lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            record.update({"status": "rejected", "finished_at": _now(), "exit_code": 3})
            write_record(record, root)
            return 3
        record.update({"status": "running", "started_at": _now(), "pid": os.getpid()})
        write_record(record, root)
        environment = environment_with_saved_dotenv(root)
        cancelled = False
        child: subprocess.Popen[bytes] | None = None

        def terminate(_signum, _frame):
            nonlocal cancelled
            cancelled = True
            if child and child.poll() is None:
                child.terminate()

        signal.signal(signal.SIGTERM, terminate)
        signal.signal(signal.SIGINT, terminate)
        log_path = _log_path(job_id, root)
        with log_path.open("ab", buffering=0) as log:
            for index, (label, command) in enumerate(definition.steps, start=1):
                if cancelled:
                    break
                record.update({"current_step": label, "step_index": index})
                write_record(record, root)
                log.write(f"\n[{_now()}] STEP {index}/{len(definition.steps)}: {label}\n".encode())
                child = subprocess.Popen(
                    command, cwd=root, env=environment, stdin=subprocess.DEVNULL,
                    stdout=log, stderr=subprocess.STDOUT,
                )
                record["child_pid"] = child.pid
                write_record(record, root)
                exit_code = child.wait()
                record.pop("child_pid", None)
                if cancelled:
                    break
                if exit_code != 0:
                    record.update({
                        "status": "failed", "finished_at": _now(), "exit_code": exit_code,
                    })
                    write_record(record, root)
                    return exit_code
            final_status = "cancelled" if cancelled else "completed"
            exit_code = 130 if cancelled else 0
            record.update({
                "status": final_status, "finished_at": _now(), "exit_code": exit_code,
                "current_step": None,
            })
            write_record(record, root)
            log.write(f"\n[{_now()}] JOB {final_status} (exit={exit_code})\n".encode())
            return exit_code
    finally:
        try:
            fcntl.flock(lock_fd, fcntl.LOCK_UN)
        finally:
            os.close(lock_fd)
