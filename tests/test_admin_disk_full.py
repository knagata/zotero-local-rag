import errno
from types import SimpleNamespace
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from citation_graph import admin_routes


@pytest.mark.parametrize('number,status', [(errno.ENOSPC, 507), (errno.EACCES, 500)])
def test_admin_status_reports_disk_full_without_hiding_other_errors(monkeypatch, number, status):
    app = FastAPI()
    settings = SimpleNamespace(allowed_google_emails={'allowed'}, public_url='https://example.test')
    admin_routes.register_admin_routes(app, settings, lambda request, allowed: 'allowed')
    def fail(root):
        raise OSError(number, 'write failed')
    monkeypatch.setattr(admin_routes, 'system_status', fail)
    response = TestClient(app, raise_server_exceptions=False).get('/admin/api/status')
    assert response.status_code == status
    if status == 507:
        assert '空き容量' in response.json()['error']
        assert response.headers['cache-control'] == 'no-store'
