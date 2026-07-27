from __future__ import annotations

import sqlite3
import unittest

from scripts.list_artifact_status import _build_report


_SCHEMA = """
CREATE TABLE artifact_processing_status (
    item_key TEXT NOT NULL,
    attachment_key TEXT NOT NULL DEFAULT '',
    artifact_type TEXT NOT NULL,
    status TEXT NOT NULL,
    reason_code TEXT,
    message TEXT,
    retryable INTEGER NOT NULL DEFAULT 0,
    attempt_count INTEGER NOT NULL DEFAULT 0,
    source_fingerprint TEXT,
    processor_version TEXT,
    model TEXT,
    counts_json TEXT,
    fallback_kind TEXT,
    started_at TIMESTAMP,
    finished_at TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY(item_key, attachment_key, artifact_type)
)
"""


def _isolated_conn() -> sqlite3.Connection:
    """A private in-memory DB — never touches data/relations.db."""
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    conn.execute(_SCHEMA)
    return conn


def _insert(conn: sqlite3.Connection, **row) -> None:
    row.setdefault("attachment_key", "ATT1")
    row.setdefault("reason_code", None)
    row.setdefault("message", None)
    row.setdefault("retryable", 0)
    row.setdefault("counts_json", None)
    conn.execute(
        """INSERT INTO artifact_processing_status
           (item_key, attachment_key, artifact_type, status, reason_code, message, retryable, counts_json)
           VALUES (:item_key, :attachment_key, :artifact_type, :status, :reason_code, :message, :retryable, :counts_json)""",
        row,
    )
    conn.commit()


class ListArtifactStatusTests(unittest.TestCase):
    def test_message_body_is_preserved_not_collapsed_to_a_flag(self):
        # Regression: `message or ''` in SQL is logical OR, not COALESCE, and
        # used to collapse real message text down to '0'/'1'/None.
        conn = _isolated_conn()
        _insert(
            conn, item_key="I1", artifact_type="extraction", status="failed",
            message="extraction failed: MuPDF error on page 3", retryable=1,
        )
        report = _build_report(conn)
        self.assertEqual(
            report["unresolved"][0]["message"], "extraction failed: MuPDF error on page 3",
        )

    def test_null_message_becomes_empty_string_not_none_or_digit(self):
        conn = _isolated_conn()
        _insert(conn, item_key="I1", artifact_type="extraction", status="blocked", message=None)
        report = _build_report(conn)
        self.assertEqual(report["unresolved"][0]["message"], "")

    def test_empty_and_degraded_are_not_counted_as_unresolved(self):
        conn = _isolated_conn()
        _insert(conn, item_key="I1", artifact_type="references", status="empty")
        _insert(conn, item_key="I2", artifact_type="summary", status="degraded")
        report = _build_report(conn)
        self.assertEqual(report["unresolved_count"], 0)
        self.assertEqual(report["unresolved"], [])
        self.assertEqual(report["informational_count"], 2)
        statuses = {row["status"] for row in report["informational"]}
        self.assertEqual(statuses, {"empty", "degraded"})

    def test_failed_and_blocked_are_counted_as_unresolved(self):
        conn = _isolated_conn()
        _insert(conn, item_key="I1", artifact_type="extraction", status="failed", retryable=1)
        _insert(conn, item_key="I2", artifact_type="structure", status="blocked")
        report = _build_report(conn)
        self.assertEqual(report["unresolved_count"], 2)
        statuses = {row["status"] for row in report["unresolved"]}
        self.assertEqual(statuses, {"failed", "blocked"})

    def test_a_real_failure_is_not_buried_alongside_many_empty_results(self):
        conn = _isolated_conn()
        for i in range(20):
            _insert(conn, item_key=f"E{i}", artifact_type="references", status="empty")
        _insert(conn, item_key="REAL_FAILURE", artifact_type="extraction", status="failed", retryable=1)
        report = _build_report(conn)
        self.assertEqual(report["unresolved_count"], 1)
        self.assertEqual(report["unresolved"][0]["item_key"], "REAL_FAILURE")

    def test_counts_are_exposed_on_unresolved_rows(self):
        conn = _isolated_conn()
        _insert(
            conn, item_key="I1", artifact_type="extraction", status="failed",
            counts_json='{"chunks": 0, "pages": 12}',
        )
        report = _build_report(conn)
        self.assertEqual(report["unresolved"][0]["counts"], {"chunks": 0, "pages": 12})

    def test_item_key_filter_restricts_rows(self):
        conn = _isolated_conn()
        _insert(conn, item_key="I1", artifact_type="extraction", status="failed")
        _insert(conn, item_key="I2", artifact_type="extraction", status="failed")
        report = _build_report(conn, item_key="I1")
        self.assertEqual(report["unresolved_count"], 1)
        self.assertEqual(report["unresolved"][0]["item_key"], "I1")


if __name__ == "__main__":
    unittest.main()
