"""Persistence boundary for artifact processing state and audit events."""
from __future__ import annotations

import json
from collections.abc import Callable
from contextlib import contextmanager
from typing import Any


@contextmanager
def _connection(factory: Callable[[], Any]):
    connection = factory()
    try:
        yield connection
    finally:
        connection.close()


class ArtifactRepository:
    def __init__(
        self,
        connection_factory: Callable[[], Any],
        *,
        artifact_types: set[str],
        artifact_statuses: set[str],
    ) -> None:
        self._connect = connection_factory
        self._artifact_types = artifact_types
        self._artifact_statuses = artifact_statuses

    def mark_status(
        self, item_key: str, artifact_type: str, status: str, *,
        attachment_key: str | None = None, reason_code: str | None = None,
        message: str | None = None, retryable: bool = False,
        source_fingerprint: str | None = None, processor_version: str | None = None,
        model: str | None = None, counts: dict[str, Any] | None = None,
        fallback_kind: str | None = None, run_id: str | None = None,
    ) -> dict[str, Any]:
        if not item_key:
            raise ValueError("item_key is required")
        if artifact_type not in self._artifact_types:
            raise ValueError(f"invalid artifact type: {artifact_type}")
        if status not in self._artifact_statuses:
            raise ValueError(f"invalid artifact status: {status}")
        attachment = str(attachment_key or "")
        with _connection(self._connect) as connection:
            try:
                connection.execute("BEGIN IMMEDIATE")
                previous = connection.execute('''
                    SELECT status, attempt_count FROM artifact_processing_status
                    WHERE item_key = ? AND attachment_key = ? AND artifact_type = ?
                ''', (item_key, attachment, artifact_type)).fetchone()
                previous_status = str(previous["status"]) if previous else None
                attempts = int(previous["attempt_count"]) if previous else 0
                if status == "running" and previous_status != "running":
                    attempts += 1
                started_at = "CURRENT_TIMESTAMP" if status == "running" else "NULL"
                finished_at = (
                    "NULL" if status in {"pending", "running", "stale"}
                    else "CURRENT_TIMESTAMP"
                )
                connection.execute(f'''
                    INSERT INTO artifact_processing_status (
                        item_key, attachment_key, artifact_type, status, reason_code, message,
                        retryable, attempt_count, source_fingerprint, processor_version, model,
                        counts_json, fallback_kind, started_at, finished_at, updated_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,
                              {started_at}, {finished_at}, CURRENT_TIMESTAMP)
                    ON CONFLICT(item_key, attachment_key, artifact_type) DO UPDATE SET
                        status=excluded.status, reason_code=excluded.reason_code,
                        message=excluded.message, retryable=excluded.retryable,
                        attempt_count=excluded.attempt_count,
                        source_fingerprint=COALESCE(excluded.source_fingerprint,
                            artifact_processing_status.source_fingerprint),
                        processor_version=COALESCE(excluded.processor_version,
                            artifact_processing_status.processor_version),
                        model=COALESCE(excluded.model, artifact_processing_status.model),
                        counts_json=COALESCE(excluded.counts_json,
                            artifact_processing_status.counts_json),
                        fallback_kind=excluded.fallback_kind,
                        started_at=CASE WHEN excluded.status = 'running' THEN CURRENT_TIMESTAMP
                            ELSE artifact_processing_status.started_at END,
                        finished_at=CASE WHEN excluded.status IN ('pending', 'running', 'stale')
                            THEN NULL ELSE CURRENT_TIMESTAMP END,
                        updated_at=CURRENT_TIMESTAMP
                ''', (
                    item_key, attachment, artifact_type, status, reason_code, message,
                    int(bool(retryable)), attempts, source_fingerprint, processor_version,
                    model, json.dumps(counts, ensure_ascii=False, sort_keys=True)
                    if counts is not None else None, fallback_kind,
                ))
                connection.execute('''
                    INSERT INTO artifact_processing_events
                        (item_key, attachment_key, artifact_type, from_status, to_status,
                         reason_code, message, run_id)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ''', (item_key, attachment, artifact_type, previous_status, status,
                      reason_code, message, run_id))
                row = connection.execute('''
                    SELECT * FROM artifact_processing_status
                    WHERE item_key = ? AND attachment_key = ? AND artifact_type = ?
                ''', (item_key, attachment, artifact_type)).fetchone()
                connection.commit()
                return dict(row)
            except Exception:
                connection.rollback()
                raise

    def list_for_item(self, item_key: str) -> list[dict[str, Any]]:
        return self.list_statuses(item_key=item_key)

    def list_statuses(
        self, *, item_key: str | None = None, artifact_type: str | None = None,
        reason_code: str | None = None,
    ) -> list[dict[str, Any]]:
        filters = {"item_key": item_key, "artifact_type": artifact_type,
                   "reason_code": reason_code}
        clauses = [f"{key} = ?" for key, value in filters.items() if value is not None]
        params = [value for value in filters.values() if value is not None]
        where = f" WHERE {' AND '.join(clauses)}" if clauses else ""
        order = "attachment_key, artifact_type" if item_key is not None else (
            "item_key, attachment_key, artifact_type"
        )
        with _connection(self._connect) as connection:
            rows = connection.execute(
                f"SELECT * FROM artifact_processing_status{where} ORDER BY {order}", params,
            ).fetchall()
        result = []
        for row in rows:
            record = dict(row)
            try:
                record["counts"] = json.loads(record.pop("counts_json") or "{}")
            except (TypeError, ValueError):
                record["counts"] = {}
            result.append(record)
        return result

    def status_summary(self) -> list[dict[str, Any]]:
        with _connection(self._connect) as connection:
            rows = connection.execute('''
                SELECT artifact_type, status, COUNT(*) AS count
                FROM artifact_processing_status
                GROUP BY artifact_type, status ORDER BY artifact_type, status
            ''').fetchall()
            return [dict(row) for row in rows]

    def recover_interrupted(self, *, older_than_seconds: int = 3600) -> int:
        with _connection(self._connect) as connection:
            rows = connection.execute('''
                SELECT item_key, attachment_key, artifact_type
                FROM artifact_processing_status
                WHERE status = 'running' AND started_at IS NOT NULL
                  AND CAST(strftime('%s', 'now') AS INTEGER)
                    - CAST(strftime('%s', started_at) AS INTEGER) >= ?
            ''', (max(0, int(older_than_seconds)),)).fetchall()
        for row in rows:
            self.mark_status(
                str(row["item_key"]), str(row["artifact_type"]), "failed",
                attachment_key=str(row["attachment_key"] or ""),
                reason_code="interrupted",
                message="Previous maintenance run did not finish.", retryable=True,
            )
        return len(rows)
