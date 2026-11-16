"""Recognize bounded, document-local source failures without accepting a full audit."""
from __future__ import annotations


def source_exclusions(report: dict) -> set[str]:
    """Return lost-page attachments only; unknown or global failures stop publication."""
    summary = report.get('summary') or {}
    documents = report.get('documents')
    if (report.get('passed') is not False or not isinstance(documents, list)
            or any(type(summary.get(key)) is not int or summary[key] != 0 for key in (
                'orphan_chunks', 'dangling_node_chunks', 'unretrievable_items',
                'unreadable_source_documents'))):
        raise RuntimeError('Source audit has global or unclassified failures')
    excluded = set()
    lost = 0
    for document in documents:
        if document.get('lost_pages'):
            key = document.get('attachment_key')
            if not isinstance(key, str) or not key or key in excluded:
                raise RuntimeError('Source audit has ambiguous attachment failures')
            excluded.add(key)
            lost += len(document['lost_pages'])
    if (not excluded or summary.get('documents_with_lost_pages') != len(excluded)
            or summary.get('lost_pages_total') != lost):
        raise RuntimeError('Source audit failure totals cannot be reconciled')
    return excluded


def exclude_source_items(comparisons: list[dict], rows: list[dict], statuses: list[dict],
                         attachments: set[str]) -> set[str]:
    """Resolve every excluded attachment to one item and mark its comparison failed."""
    owners = {key: set() for key in attachments}
    for row in rows:
        md = row.get('metadata') or {}
        if md.get('attachmentKey') in owners and md.get('itemKey'):
            owners[md['attachmentKey']].add(md['itemKey'])
    for row in statuses:
        if row.get('attachment_key') in owners and row.get('item_key'):
            owners[row['attachment_key']].add(row['item_key'])
    if any(len(items) != 1 for items in owners.values()):
        raise RuntimeError('Source audit failure has no unique item owner')
    excluded_items = set().union(*owners.values()) if owners else set()
    for row in comparisons:
        if row['item_key'] in excluded_items:
            row['failures'] = list(dict.fromkeys([*row.get('failures', []), 'source_verification_failed']))
            row['passed'] = False
    return excluded_items
