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
    structure_status = classify_structure_status(expected, structures, statuses)
    summary_pending = {k for k in structures if statuses.get((k, 'summary'), '') not in {'success', 'completed', 'skipped', 'empty'}} | stale
    def status(keys, kind):
        failures = sum(statuses.get((key, kind)) in {"failed", "blocked", "degraded"} for key in keys)
        return dict(pending=len(keys), failed=failures, attention=failures, source_refresh_failed=0,
                    sample_keys=sorted(keys)[:20], lightweight=True)
    return structure_status, status(summary_pending, "summary")


def classify_structure_status(expected, structures, statuses):
    """Keep usable flat structures separate from unfinished or failed work."""
    groups = {name: set() for name in ('unprocessed', 'outdated', 'failed', 'flat_fallback', 'degraded')}
    for key in expected:
        saved = structures.get(key, ('', ''))[1]
        artifact = statuses.get((key, 'structure'), '')
        if artifact in {'failed', 'blocked'} or saved == 'unavailable':
            group = 'failed'
        elif key not in structures:
            group = 'unprocessed'
        elif artifact in {'running', 'stale'} or saved in {'running', 'stale'}:
            group = 'outdated'
        elif saved == 'flat_fallback':
            group = 'flat_fallback'
        elif artifact == 'degraded' or saved == 'degraded':
            group = 'degraded'
        else:
            continue
        groups[group].add(key)
    pending = set().union(*(keys for name, keys in groups.items() if name != 'flat_fallback'))
    return {
        **{name: len(keys) for name, keys in groups.items()},
        'pending': len(pending), 'attention': len(groups['failed'] | groups['degraded']),
        'source_refresh_failed': 0, 'sample_keys': sorted(pending)[:20],
        'flat_fallback_sample_keys': sorted(groups['flat_fallback'])[:20], 'lightweight': True,
    }
