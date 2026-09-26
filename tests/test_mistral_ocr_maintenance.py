from __future__ import annotations

from scripts import run_mistral_ocr_maintenance as maintenance


def test_empty_state_submits_and_stops_until_a_later_run(monkeypatch):
    calls = []
    monkeypatch.setattr(maintenance, "_load", lambda: {})
    monkeypatch.setattr(maintenance, "_run", lambda *args: calls.append(args))

    assert maintenance.main() == 0

    assert calls == [(str(maintenance.BATCH), "--submit", "--state", str(maintenance.STATE))]


def test_running_state_refreshes_without_collecting_until_success(monkeypatch):
    states = iter([{"phase": "running"}, {"phase": "running"}])
    calls = []
    monkeypatch.setattr(maintenance, "_load", lambda: next(states))
    monkeypatch.setattr(maintenance, "_run", lambda *args: calls.append(args))

    assert maintenance.main() == 0

    assert calls == [(str(maintenance.BATCH), "--status", "--state", str(maintenance.STATE))]


def test_successful_batch_is_collected_then_adopted(monkeypatch):
    states = iter([{"phase": "success"}, {"phase": "collected", "adoptable_count": 2}])
    calls = []
    adopted = []
    monkeypatch.setattr(maintenance, "_load", lambda: next(states))
    monkeypatch.setattr(maintenance, "_run", lambda *args: calls.append(args))
    monkeypatch.setattr(maintenance, "_adopt", lambda state: adopted.append(state))

    assert maintenance.main() == 0

    assert calls == [(str(maintenance.BATCH), "--collect", "--state", str(maintenance.STATE))]
    assert adopted == [{"phase": "collected", "adoptable_count": 2}]


def test_collected_transient_failure_is_resubmitted_for_only_that_item(monkeypatch):
    states = iter([
        {
            "phase": "collected", "adoptable_count": 0,
            "reports": [{"item_key": "ITEM1", "retryable": True}],
        },
        {
            "phase": "collected", "adoptable_count": 0,
            "adoption_applied_at": "now",
            "reports": [{"item_key": "ITEM1", "retryable": True}],
        },
    ])
    calls = []
    monkeypatch.setattr(maintenance, "_load", lambda: next(states))
    monkeypatch.setattr(maintenance, "_adopt", lambda state: None)
    monkeypatch.setattr(maintenance, "_run", lambda *args: calls.append(args))

    assert maintenance.main() == 0
    assert calls == [(
        str(maintenance.BATCH), "--submit", "--state", str(maintenance.STATE),
        "--item", "ITEM1",
    )]


def test_adopted_batch_submits_newly_queued_candidates(monkeypatch):
    calls = []
    monkeypatch.setattr(
        maintenance, "_load",
        lambda: {"phase": "collected", "adoption_applied_at": "earlier"},
    )
    monkeypatch.setattr(maintenance, "_queued_candidate_count", lambda: 2)
    monkeypatch.setattr(maintenance, "_run", lambda *args: calls.append(args))

    assert maintenance.main() == 0

    assert calls == [(str(maintenance.BATCH), "--submit", "--state", str(maintenance.STATE))]


def test_adopted_batch_does_nothing_when_queue_is_empty(monkeypatch):
    calls = []
    monkeypatch.setattr(
        maintenance, "_load",
        lambda: {"phase": "collected", "adoption_applied_at": "earlier"},
    )
    monkeypatch.setattr(maintenance, "_queued_candidate_count", lambda: 0)
    monkeypatch.setattr(maintenance, "_run", lambda *args: calls.append(args))

    assert maintenance.main() == 0
    assert calls == []
