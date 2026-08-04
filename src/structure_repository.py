"""Persistence boundary for canonical document structures."""
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


class StructureRepository:
    def __init__(
        self, connection_factory: Callable[[], Any], *, statuses: set[str],
    ) -> None:
        self._connect = connection_factory
        self._statuses = statuses

    def replace(
        self, item_key: str, *, source_fingerprint: str, structure_version: str,
        status: str, nodes: list[dict[str, Any]],
        diagnostics: dict[str, Any] | None = None,
        confidence: float | None = None,
    ) -> None:
        if not item_key:
            raise ValueError("item_key is required")
        if status not in self._statuses:
            raise ValueError(f"invalid structure status: {status}")
        node_ids = [str(node.get("node_id") or "") for node in nodes]
        if not all(node_ids) or len(node_ids) != len(set(node_ids)):
            raise ValueError("document structure node_id values must be present and unique")
        seen_chunks: set[str] = set()
        for node in nodes:
            for chunk in node.get("chunks") or []:
                chunk_id = str(
                    chunk.get("chunk_id") if isinstance(chunk, dict) else chunk
                )
                if not chunk_id or chunk_id in seen_chunks:
                    raise ValueError(
                        "a source chunk must belong to exactly one document leaf"
                    )
                seen_chunks.add(chunk_id)

        with _connection(self._connect) as connection:
            try:
                connection.execute("BEGIN IMMEDIATE")
                self._cache_previous_summaries(connection, item_key)
                connection.execute(
                    "DELETE FROM document_nodes WHERE item_key = ?", (item_key,),
                )
                leaf_count = self._insert_nodes(connection, item_key, nodes)
                connection.execute('''
                    INSERT INTO document_structures
                        (item_key, source_fingerprint, structure_version, status, confidence,
                         node_count, leaf_count, diagnostics_json, updated_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
                    ON CONFLICT(item_key) DO UPDATE SET
                        source_fingerprint=excluded.source_fingerprint,
                        structure_version=excluded.structure_version,
                        status=excluded.status, confidence=excluded.confidence,
                        node_count=excluded.node_count, leaf_count=excluded.leaf_count,
                        diagnostics_json=excluded.diagnostics_json,
                        updated_at=CURRENT_TIMESTAMP
                ''', (
                    item_key, source_fingerprint, structure_version, status, confidence,
                    len(nodes), leaf_count,
                    json.dumps(diagnostics or {}, ensure_ascii=False, sort_keys=True),
                ))
                connection.commit()
            except Exception:
                connection.rollback()
                raise

    @staticmethod
    def _cache_previous_summaries(connection: Any, item_key: str) -> None:
        connection.execute('''
            WITH RECURSIVE titled_nodes AS (
                SELECT node_id, parent_node_id,
                       COALESCE(NULLIF(title, ''), '資料') AS prompt_title
                FROM document_nodes WHERE item_key = ? AND parent_node_id IS NULL
                UNION ALL
                SELECT n.node_id, n.parent_node_id,
                       COALESCE(NULLIF(n.title, ''), titled_nodes.prompt_title)
                FROM document_nodes n JOIN titled_nodes
                    ON n.parent_node_id = titled_nodes.node_id
            )
            INSERT INTO document_node_summary_reuse_cache (
                item_key, node_id, title, summary, summary_kind, model,
                prompt_version, source_fingerprint, source_chunk_count,
                source_chars, quality_status, input_scope_json, cached_at
            )
            SELECT s.item_key, s.node_id, titled_nodes.prompt_title, s.summary,
                   s.summary_kind, s.model, s.prompt_version, s.source_fingerprint,
                   s.source_chunk_count, s.source_chars, s.quality_status,
                   s.input_scope_json, CURRENT_TIMESTAMP
            FROM document_node_summaries s
            JOIN titled_nodes ON titled_nodes.node_id = s.node_id
            WHERE s.item_key = ?
            ON CONFLICT(item_key, node_id) DO UPDATE SET
                title=excluded.title, summary=excluded.summary,
                summary_kind=excluded.summary_kind, model=excluded.model,
                prompt_version=excluded.prompt_version,
                source_fingerprint=excluded.source_fingerprint,
                source_chunk_count=excluded.source_chunk_count,
                source_chars=excluded.source_chars,
                quality_status=excluded.quality_status,
                input_scope_json=excluded.input_scope_json,
                cached_at=CURRENT_TIMESTAMP
        ''', (item_key, item_key))

    @staticmethod
    def _insert_nodes(
        connection: Any, item_key: str, nodes: list[dict[str, Any]],
    ) -> int:
        leaf_count = 0
        for node in nodes:
            chunks = node.get("chunks") or []
            if chunks:
                leaf_count += 1
            connection.execute('''
                INSERT INTO document_nodes (
                    node_id, item_key, attachment_key, parent_node_id, node_type,
                    depth, ordinal, title, normalized_title, source_kind,
                    source_locator_json, confidence, content_chars, first_chunk_id,
                    last_chunk_id, zone, summary_policy, retrieval_policy,
                    citation_policy, extraction_engine, extraction_version
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                node["node_id"], item_key, node.get("attachment_key"),
                node.get("parent_node_id"), node.get("node_type") or "section",
                int(node.get("depth") or 0), int(node.get("ordinal") or 0),
                node.get("title"), node.get("normalized_title"),
                node.get("source_kind") or "semantic_fallback",
                json.dumps(
                    node.get("source_locator") or {}, ensure_ascii=False, sort_keys=True,
                ),
                node.get("confidence"), int(node.get("content_chars") or 0),
                node.get("first_chunk_id"), node.get("last_chunk_id"),
                node.get("zone") or "body", node.get("summary_policy") or "include",
                node.get("retrieval_policy") or "normal",
                node.get("citation_policy") or "none", node.get("extraction_engine"),
                node.get("extraction_version"),
            ))
            for ordinal, chunk in enumerate(chunks):
                chunk_id = str(
                    chunk.get("chunk_id") if isinstance(chunk, dict) else chunk
                )
                chunk_ordinal = int(
                    chunk.get("ordinal", ordinal) if isinstance(chunk, dict) else ordinal
                )
                connection.execute(
                    "INSERT INTO document_node_chunks (node_id, chunk_id, ordinal) "
                    "VALUES (?, ?, ?)",
                    (node["node_id"], chunk_id, chunk_ordinal),
                )
        return leaf_count

    def get_structure(self, item_key: str) -> dict[str, Any] | None:
        with _connection(self._connect) as connection:
            row = connection.execute(
                "SELECT * FROM document_structures WHERE item_key = ?", (item_key,),
            ).fetchone()
        if row is None:
            return None
        result = dict(row)
        try:
            result["diagnostics"] = json.loads(result.pop("diagnostics_json") or "{}")
        except (TypeError, ValueError):
            result["diagnostics"] = {}
        return result

    def get_nodes(
        self, item_key: str, *, include_chunks: bool = False,
    ) -> list[dict[str, Any]]:
        with _connection(self._connect) as connection:
            rows = connection.execute('''
                WITH RECURSIVE tree AS (
                    SELECT node_id, parent_node_id, 0 AS walk_depth,
                           printf('%08d', ordinal) AS walk_order
                    FROM document_nodes
                    WHERE item_key = ? AND parent_node_id IS NULL
                    UNION ALL
                    SELECT n.node_id, n.parent_node_id, tree.walk_depth + 1,
                           tree.walk_order || '.' || printf('%08d', n.ordinal)
                    FROM document_nodes n JOIN tree ON n.parent_node_id = tree.node_id
                )
                SELECT n.* FROM document_nodes n JOIN tree ON tree.node_id = n.node_id
                ORDER BY tree.walk_order
            ''', (item_key,)).fetchall()
            result = []
            for row in rows:
                node = dict(row)
                try:
                    node["source_locator"] = json.loads(
                        node.pop("source_locator_json") or "{}"
                    )
                except (TypeError, ValueError):
                    node["source_locator"] = {}
                if include_chunks:
                    chunks = connection.execute(
                        "SELECT chunk_id, ordinal FROM document_node_chunks "
                        "WHERE node_id = ? ORDER BY ordinal", (node["node_id"],),
                    ).fetchall()
                    node["chunks"] = [dict(chunk) for chunk in chunks]
                result.append(node)
            return result

    def descendant_chunks(self, node_ids: list[str]) -> list[str]:
        return self._descendants(node_ids, return_leaf_ids=False)

    def descendant_leaf_ids(self, node_ids: list[str]) -> list[str]:
        return self._descendants(node_ids, return_leaf_ids=True)

    def _descendants(self, node_ids: list[str], *, return_leaf_ids: bool) -> list[str]:
        ids = [str(value) for value in node_ids if value]
        if not ids:
            return []
        placeholders = ",".join("?" for _ in ids)
        if return_leaf_ids:
            query = f'''
                WITH RECURSIVE descendants(node_id) AS (
                    SELECT node_id FROM document_nodes WHERE node_id IN ({placeholders})
                    UNION
                    SELECT n.node_id FROM document_nodes n
                    JOIN descendants d ON n.parent_node_id = d.node_id
                )
                SELECT DISTINCT n.node_id, n.first_chunk_id
                FROM descendants d JOIN document_nodes n ON n.node_id = d.node_id
                WHERE EXISTS (
                    SELECT 1 FROM document_node_chunks c WHERE c.node_id = n.node_id
                )
                ORDER BY n.first_chunk_id, n.node_id
            '''
            key = "node_id"
        else:
            query = f'''
                WITH RECURSIVE descendants(node_id) AS (
                    SELECT node_id FROM document_nodes WHERE node_id IN ({placeholders})
                    UNION
                    SELECT n.node_id FROM document_nodes n
                    JOIN descendants d ON n.parent_node_id = d.node_id
                )
                SELECT DISTINCT c.chunk_id, c.ordinal, n.first_chunk_id
                FROM document_node_chunks c
                JOIN descendants d ON d.node_id = c.node_id
                JOIN document_nodes n ON n.node_id = c.node_id
                ORDER BY n.first_chunk_id, c.ordinal, c.chunk_id
            '''
            key = "chunk_id"
        with _connection(self._connect) as connection:
            rows = connection.execute(query, ids).fetchall()
            return [str(row[key]) for row in rows]
