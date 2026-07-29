"""Bind a successful database audit to the exact inputs used by summaries."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Mapping

from .v3_data_plane import V3_COLLECTION

def _file_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def external_audit_attestation(path: Path) -> dict[str, str]:
    """Require a passing external audit and bind its exact bytes to the DB gate."""
    try:
        report = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise RuntimeError(f"External audit report is unreadable: {path}") from exc
    if not isinstance(report, Mapping) or report.get("passed") is not True:
        raise RuntimeError(f"External audit did not pass: {path}")
    return {"path": str(path.resolve()), "sha256": _file_digest(path)}


def _validate_external_audits(value: Any) -> None:
    if not isinstance(value, Mapping):
        raise RuntimeError("Database audit lacks required Zotero/source attestations.")
    for name in ("zotero_reconciliation", "source_verification"):
        attestation = value.get(name)
        if not isinstance(attestation, Mapping):
            raise RuntimeError(f"Database audit lacks required {name} attestation.")
        path_value = str(attestation.get("path") or "")
        expected_digest = str(attestation.get("sha256") or "")
        if not path_value or not expected_digest:
            raise RuntimeError(f"Database audit has an incomplete {name} attestation.")
        path = Path(path_value)
        try:
            report = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            raise RuntimeError(f"Attested external audit is unreadable: {path}") from exc
        if not isinstance(report, Mapping) or report.get("passed") is not True:
            raise RuntimeError(f"Attested external audit no longer passes: {path}")
        if _file_digest(path) != expected_digest:
            raise RuntimeError(f"Attested external audit changed after gate creation: {path}")


def database_state_fingerprint(
    *,
    manifest_path: Path,
    pipeline_config_path: Path,
    chroma_rows: Iterable[Mapping[str, Any]],
    lexical_ids: Iterable[str],
    structure_rows: Iterable[tuple[Any, ...]],
) -> str:
    """Fingerprint every deterministic input that hierarchical summaries read."""
    digest = hashlib.sha256()
    for label, path in (
        ("manifest", manifest_path),
        ("pipeline_config", pipeline_config_path),
    ):
        digest.update(label.encode())
        digest.update(str(path.resolve()).encode())
        digest.update(_file_digest(path).encode() if path.exists() else b"<missing>")
    canonical_chunks = sorted((
        (
            str(row.get("id") or ""),
            str(row.get("text") or ""),
            dict(row.get("metadata") or {}) if isinstance(row.get("metadata"), Mapping) else {},
        )
        for row in chroma_rows
    ), key=lambda row: row[0])
    for label, values in (
        ("chroma_rows", canonical_chunks),
        ("lexical", sorted(str(value) for value in lexical_ids)),
        ("structure_rows", sorted(tuple(map(str, row)) for row in structure_rows)),
    ):
        digest.update(label.encode())
        for value in values:
            digest.update(json.dumps(
                value, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
            ).encode())
            digest.update(b"\n")
    return f"sha256:{digest.hexdigest()}"


def live_database_state_fingerprint(
    *, collection_name: str, manifest_path: Path,
    lexical_db_path: Path, pipeline_config_path: Path,
) -> str:
    from .chunk_store import get_collection_chunks
    from .db_relations import get_db_connection
    from .lexical_index import list_chunk_ids as list_lexical_chunk_ids

    connection = get_db_connection()
    try:
        structure_rows = [
            ("structure", *tuple(row)) for row in connection.execute(
                "SELECT item_key, source_fingerprint, structure_version, status, "
                "COALESCE(confidence, ''), node_count, leaf_count, "
                "COALESCE(diagnostics_json, '') "
                "FROM document_structures"
            ).fetchall()
        ]
        structure_rows.extend(
            ("node", *tuple(row)) for row in connection.execute(
                "SELECT item_key, node_id, COALESCE(parent_node_id, ''), node_type, "
                "depth, ordinal, COALESCE(title, ''), COALESCE(normalized_title, ''), "
                "COALESCE(attachment_key, ''), source_kind, "
                "COALESCE(source_locator_json, ''), COALESCE(confidence, ''), "
                "content_chars, zone, summary_policy, retrieval_policy, citation_policy, "
                "COALESCE(first_chunk_id, ''), COALESCE(last_chunk_id, '') "
                ", COALESCE(extraction_engine, ''), COALESCE(extraction_version, '') "
                "FROM document_nodes"
            ).fetchall()
        )
        structure_rows.extend(
            ("mapping", *tuple(row)) for row in connection.execute(
                "SELECT n.item_key, c.node_id, c.chunk_id, c.ordinal "
                "FROM document_node_chunks c "
                "JOIN document_nodes n ON n.node_id = c.node_id"
            ).fetchall()
        )
    finally:
        connection.close()
    return database_state_fingerprint(
        manifest_path=manifest_path,
        pipeline_config_path=pipeline_config_path,
        chroma_rows=get_collection_chunks(collection_name=collection_name),
        lexical_ids=list_lexical_chunk_ids(path=lexical_db_path),
        structure_rows=structure_rows,
    )


def validate_database_gate(report_path: Path, *, collection_name: str | None = None) -> dict[str, Any]:
    """Fail closed unless a passing audit describes the current DB generation."""
    try:
        report = json.loads(report_path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise RuntimeError(f"Database audit report is unreadable: {report_path}") from exc
    gate = report.get("gate") if isinstance(report.get("gate"), Mapping) else {}
    state = report.get("database_state") if isinstance(report.get("database_state"), Mapping) else {}
    if gate.get("passed") is not True:
        raise RuntimeError("Database audit did not pass; hierarchical summaries are blocked.")
    if report.get("new_only") is not True:
        raise RuntimeError("Database audit is not a server rebuild gate (--new-only required).")
    audited_collection = str(report.get("new_collection") or "")
    if audited_collection != V3_COLLECTION:
        raise RuntimeError(
            f"Database audit targets retired/unsupported collection {audited_collection!r}; "
            f"expected {V3_COLLECTION!r}."
        )
    if collection_name and audited_collection != collection_name:
        raise RuntimeError(
            f"Database audit collection mismatch: audited={audited_collection}, requested={collection_name}"
        )
    _validate_external_audits(report.get("external_audits"))
    required = ("fingerprint", "manifest_path", "lexical_db_path", "pipeline_config_path")
    if any(not state.get(key) for key in required):
        raise RuntimeError("Database audit lacks a complete state attestation.")
    current = live_database_state_fingerprint(
        collection_name=audited_collection,
        manifest_path=Path(str(state["manifest_path"])),
        lexical_db_path=Path(str(state["lexical_db_path"])),
        pipeline_config_path=Path(str(state["pipeline_config_path"])),
    )
    if current != state["fingerprint"]:
        raise RuntimeError(
            "Database state changed after its audit; run the database audit again before summaries."
        )
    return report


__all__ = [
    "database_state_fingerprint", "external_audit_attestation",
    "live_database_state_fingerprint",
    "validate_database_gate",
]
