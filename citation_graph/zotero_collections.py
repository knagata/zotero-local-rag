"""Read collection membership through Zotero Local API, without SQLite locks."""
from __future__ import annotations

import httpx

from src.zotero_source_localapi import local_api_url, zotero_api_headers


def _pages(client: httpx.Client, endpoint: str) -> list[dict]:
    rows = []
    while True:
        response = client.get(local_api_url(endpoint), params={'start': len(rows), 'limit': 100})
        response.raise_for_status()
        page = response.json()
        if not isinstance(page, list):
            raise TypeError('Zotero Local APIの応答が一覧ではありません。')
        rows.extend(page)
        if len(page) < 100:
            return rows


def load_collections() -> list[dict]:
    with httpx.Client(timeout=15, headers=zotero_api_headers()) as client:
        collections = _pages(client, 'collections')
        items = _pages(client, 'items/top')
    by_key = {row['key']: row['data'] for row in collections}
    memberships: dict[str, list[str]] = {}
    for row in items:
        for key in row['data'].get('collections', []):
            memberships.setdefault(key, []).append(row['key'])
    result = []
    for key in by_key:
        parts, visited = [], set()
        current = key
        while current in by_key and current not in visited:
            visited.add(current)
            data = by_key[current]
            parts.append(data['name'])
            current = data.get('parentCollection')
        result.append({'id': key, 'path': ' / '.join(reversed(parts)),
                       'item_keys': memberships.get(key, [])})
    return sorted(result, key=lambda row: row['path'])
