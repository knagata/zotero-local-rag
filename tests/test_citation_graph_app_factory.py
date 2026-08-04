from citation_graph import server


def test_create_app_preserves_route_contract_without_sharing_app_object():
    first = server.create_app()
    second = server.create_app()

    assert first is not second
    expected = {
        (route.path, tuple(sorted(route.methods or ())))
        for route in server.app.routes
    }
    assert {
        (route.path, tuple(sorted(route.methods or ())))
        for route in first.routes
    } == expected
    assert {
        (route.path, tuple(sorted(route.methods or ())))
        for route in second.routes
    } == expected


def test_create_app_does_not_start_close_watcher(monkeypatch):
    starts = []
    monkeypatch.setattr(server, "_start_close_watcher", lambda: starts.append(True))

    server.create_app()

    assert starts == []
