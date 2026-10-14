"""A separate, bounded attestation for indexed items while new PDFs await OCR."""
from __future__ import annotations

from src.source_coverage import coverage_gap_is_adoptable, validate_source_coverage


def indexed_gate(files: dict, indexed: set[str], statuses: list[dict],
                 comparisons: list[dict], global_failures: list[str]) -> dict:
    waiting = {row.get('attachment_key'): row.get('item_key') for row in statuses
               if row.get('artifact_type') == 'extraction'
               and row.get('status') == 'blocked'
               and row.get('reason_code') == 'awaiting_mistral_ocr_batch'}
    pending = (set(files) - indexed) & waiting.keys()
    uncovered = set()
    for key, entry in files.items():
        quality = entry.get('quality') or {}
        verdict = validate_source_coverage(quality.get('source_coverage'))
        if not verdict['passed'] and not (quality.get('source_coverage_adopted')
                                         and coverage_gap_is_adoptable(verdict)):
            uncovered.add(key)
    allowed = {'manifest_chroma_attachment_mismatch', 'incomplete_source_coverage'}
    safe = (not (set(global_failures) - allowed) and not (set(files) ^ indexed) - pending
            and not uncovered - pending)
    excluded_items = {waiting[key] for key in pending}
    eligible = sorted(row['item_key'] for row in comparisons
                      if row.get('passed') is True and row['item_key'] not in excluded_items)
    return {'passed': bool(safe and eligible), 'eligible_item_keys': eligible if safe else [],
            'pending_ocr_attachments': sorted(pending),
            'excluded_item_keys': sorted(row['item_key'] for row in comparisons if row['item_key'] not in eligible)}


def summary_scope(report: dict, keys: list[str]) -> list[str]:
    """Always restrict a partial audit to the exact items it approved."""
    if (report.get('gate') or {}).get('passed') is True:
        return keys
    gate = report.get('indexed_gate') or {}
    if gate.get('passed') is not True:
        raise RuntimeError('索引済み資料の監査が未合格です。DB監査を実行してください。')
    eligible = set(gate.get('eligible_item_keys') or [])
    return [key for key in keys if key in eligible]
