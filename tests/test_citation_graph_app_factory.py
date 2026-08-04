from citation_graph import server


def _route_contract(application):
    return {
        (route.path, tuple(sorted(route.methods or ())))
        for route in application.routes
        if hasattr(route, "path")
    }


def test_create_app_preserves_route_contract_without_sharing_app_object():
    first = server.create_app()
    second = server.create_app()

    assert first is not second
    expected = _route_contract(server.app)
    assert _route_contract(first) == expected
    assert _route_contract(second) == expected


def test_create_app_does_not_start_close_watcher(monkeypatch):
    starts = []
    monkeypatch.setattr(server, "_start_close_watcher", lambda: starts.append(True))

    server.create_app()

    assert starts == []
