"""Schema-migration primitives shared by the relations repositories."""
from __future__ import annotations

import sqlite3


def add_column(cursor: sqlite3.Cursor, sql: str) -> None:
    """Run idempotent ADD COLUMN without hiding unrelated DB failures."""
    try:
        cursor.execute(sql)
    except sqlite3.OperationalError as exc:
        if "duplicate column name" not in str(exc).casefold():
            raise
