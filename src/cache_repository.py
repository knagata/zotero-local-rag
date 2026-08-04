"""Repositories for small reusable query, resolver, and summary records."""
from __future__ import annotations

from collections.abc import Callable
from contextlib import contextmanager
from typing import Any


@contextmanager
def _managed_connection(factory: Callable[[], Any]):
    connection = factory()
    try:
        with connection:
            yield connection
    finally:
        connection.close()


class CacheRepository:
    def __init__(self, connection_factory: Callable[[], Any]) -> None:
        self._connect = connection_factory

    def get_query_expansion(self, query_hash: str) -> str | None:
        with _managed_connection(self._connect) as connection:
            row = connection.execute(
                "SELECT expansions FROM query_expansion_cache WHERE query_hash = ?",
                (query_hash,),
            ).fetchone()
            return row[0] if row else None

    def save_query_expansion(self, query_hash: str, expansions: str) -> None:
        with _managed_connection(self._connect) as connection:
            connection.execute('''
                INSERT INTO query_expansion_cache (query_hash, expansions, created_at)
                VALUES (?, ?, CURRENT_TIMESTAMP)
                ON CONFLICT(query_hash) DO UPDATE SET
                    expansions = excluded.expansions,
                    created_at = CURRENT_TIMESTAMP
            ''', (query_hash, expansions))

    def get_resolver(self, query_hash: str, source: str) -> str | None:
        with _managed_connection(self._connect) as connection:
            row = connection.execute(
                "SELECT response_json FROM resolver_cache WHERE query_hash = ? AND source = ?",
                (query_hash, source),
            ).fetchone()
            return row[0] if row else None

    def save_resolver(self, query_hash: str, source: str, response_json: str) -> None:
        with _managed_connection(self._connect) as connection:
            connection.execute('''
                INSERT INTO resolver_cache (query_hash, source, response_json, created_at)
                VALUES (?, ?, ?, CURRENT_TIMESTAMP)
                ON CONFLICT(query_hash, source) DO UPDATE SET
                    response_json = excluded.response_json,
                    created_at = CURRENT_TIMESTAMP
            ''', (query_hash, source, response_json))

    def get_external_abstract(self, paper_id: str) -> dict[str, Any] | None:
        with _managed_connection(self._connect) as connection:
            row = connection.execute(
                "SELECT abstract, tldr, status, updated_at "
                "FROM external_abstracts WHERE paper_id = ?",
                (paper_id,),
            ).fetchone()
            if not row:
                return None
            return {
                "abstract": row[0], "tldr": row[1],
                "status": row[2], "updated_at": row[3],
            }

    def save_external_abstract(
        self, paper_id: str, abstract: str | None, tldr: str | None, status: str,
    ) -> None:
        with _managed_connection(self._connect) as connection:
            connection.execute('''
                INSERT INTO external_abstracts (paper_id, abstract, tldr, status, updated_at)
                VALUES (?, ?, ?, ?, CURRENT_TIMESTAMP)
                ON CONFLICT(paper_id) DO UPDATE SET
                    abstract = excluded.abstract,
                    tldr = excluded.tldr,
                    status = excluded.status,
                    updated_at = CURRENT_TIMESTAMP
            ''', (paper_id, abstract, tldr, status))


class SummaryRepository:
    def __init__(self, connection_factory: Callable[[], Any]) -> None:
        self._connect = connection_factory

    def get_item_abstract(self, item_key: str) -> str | None:
        with _managed_connection(self._connect) as connection:
            row = connection.execute(
                "SELECT abstract FROM item_citation_status WHERE item_key = ?",
                (item_key,),
            ).fetchone()
            return row[0] if row else None

    def get_item_summary(self, item_key: str) -> dict[str, Any] | None:
        with _managed_connection(self._connect) as connection:
            row = connection.execute('''
                SELECT summary, summary_en, keywords, model, chunk_count,
                       source_mtime, updated_at
                FROM item_summaries WHERE item_key = ?
            ''', (item_key,)).fetchone()
            return dict(row) if row else None

    def get_section_summaries(self, item_key: str) -> list[dict[str, Any]]:
        with _managed_connection(self._connect) as connection:
            rows = connection.execute(
                "SELECT * FROM section_summaries WHERE item_key = ? ORDER BY section_id",
                (item_key,),
            ).fetchall()
            return [dict(row) for row in rows]
