import json

import httpx
import pytest

from citation_graph import server
from citation_graph import zotero_collections as module


def install_client(monkeypatch, handler):
    factory = httpx.Client
    monkeypatch.setattr(module.httpx, 'Client', lambda **kw: factory(transport=httpx.MockTransport(handler), **kw))


def test_collections_paths_membership_and_pagination(monkeypatch):
    calls = []
    def handler(request):
        start = int(request.url.params['start'])
        calls.append((request.url.path, start))
        if request.url.path.endswith('/collections'):
            rows = [{'key': 'C', 'data': {'name': 'Child', 'parentCollection': 'P'}},
                    {'key': 'P', 'data': {'name': 'Parent', 'parentCollection': False}},
                    {'key': 'E', 'data': {'name': 'Empty', 'parentCollection': False}}]
        elif not start:
            rows = [{'key': str(i), 'data': {'collections': ['C', 'P']}} for i in range(100)]
        else:
            rows = [{'key': 'last', 'data': {'collections': ['C']}}]
        return httpx.Response(200, json=rows, request=request)
    install_client(monkeypatch, handler)
    response = server._route_collections()
    assert response.status_code == 200
    rows = json.loads(response.body)['collections']
    assert [row['path'] for row in rows] == ['Empty', 'Parent', 'Parent / Child']
    assert rows[0]['item_keys'] == []
    assert len(rows[2]['item_keys']) == 101
    assert len(rows[1]['item_keys']) == 100
    assert calls[-1][1] == 100


def test_cycle_and_missing_parent_terminate(monkeypatch):
    def handler(request):
        rows = [{'key': 'C', 'data': {'name': 'Self', 'parentCollection': 'C'}},
                {'key': 'O', 'data': {'name': 'Orphan', 'parentCollection': 'missing'}}] if request.url.path.endswith('/collections') else []
        return httpx.Response(200, json=rows, request=request)
    install_client(monkeypatch, handler)
    assert [row['path'] for row in module.load_collections()] == ['Orphan', 'Self']


@pytest.mark.parametrize('kind', ['unavailable', 'invalid', 'missing_field'])
def test_actionable_error_without_sqlite_access(monkeypatch, kind):
    def handler(request):
        if kind == 'unavailable':
            raise httpx.ConnectError('connection refused', request=request)
        return httpx.Response(200, json={} if kind == 'invalid' else [{}], request=request)
    install_client(monkeypatch, handler)
    response = server._route_collections()
    assert response.status_code == 503
    assert 'Zotero Local API' in json.loads(response.body)['message']
    assert '通信を許可' in json.loads(response.body)['message']
