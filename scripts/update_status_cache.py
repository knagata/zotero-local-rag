"""Reuse expensive read-only checks only while every recorded input is unchanged."""
from __future__ import annotations
import json
from pathlib import Path


def signature(paths):
    values = []
    for path in paths:
        try:
            stat = path.stat()
            values.append([str(path), stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns])
        except FileNotFoundError:
            values.append([str(path), None])
    return values


def cached_checks(report_path: Path, inputs, compute):
    before = signature(inputs)
    try:
        previous = json.loads(report_path.read_text())
    except (OSError, ValueError):
        previous = {}
    if isinstance(previous, dict) and previous.get('local_checks_signature') == before:
        if isinstance(previous.get('structure'), dict) and isinstance(previous.get('summaries'), dict):
            return previous['structure'], previous['summaries'], before, True
    structure, summaries = compute()
    # A concurrent write invalidates reuse; never save a signature for mixed state.
    after = signature(inputs)
    return structure, summaries, before if before == after else None, False


def lightweight_checks(database, expected_items):
    import sqlite3
    expected = set(expected_items)
    with sqlite3.connect(f'{database.resolve().as_uri()}?mode=ro', uri=True) as connection:
        structures = {str(k): (str(f or ''), str(status or '')) for k, f, status in connection.execute(
            'SELECT item_key, source_fingerprint, status FROM document_structures')}
        statuses = {(str(k), str(kind)): str(status or '') for k, kind, status in connection.execute(
            'SELECT item_key, artifact_type, status FROM artifact_processing_status')}
        stale = {str(k) for k, fingerprint in connection.execute(
            'SELECT DISTINCT item_key, source_fingerprint FROM document_node_summaries')
                 if str(k) in structures and str(fingerprint or '') != structures[str(k)][0]}
    bad = {'running', 'stale', 'failed', 'blocked', 'degraded'}
    structure_pending = {k for k in expected if k not in structures or structures[k][1] in bad
                         or statuses.get((k, 'structure')) in bad}
    summary_pending = {k for k in structures if statuses.get((k, 'summary'), '') not in {'success', 'completed', 'skipped', 'empty'}} | stale
    def status(keys, kind):
        failures = sum(statuses.get((key, kind)) in {"failed", "blocked", "degraded"} for key in keys)
        return dict(pending=len(keys), failed=failures, attention=failures, source_refresh_failed=0,
                    sample_keys=sorted(keys)[:20], lightweight=True)
    return status(structure_pending, "structure"), status(summary_pending, "summary")
