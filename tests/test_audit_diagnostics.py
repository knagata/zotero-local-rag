from src.audit_diagnostics import item_failure_details, report_failure_details


def test_item_failure_details_translate_structure_and_preserve_unknown_reasons():
    details = item_failure_details({"items": [
        {"item_key": "A", "title": "Book", "failures": ["stale_structure_fingerprint"]},
        {"item_key": "B", "failures": ["unknown_reason"]},
        {"item_key": "C", "failures": []},
    ]})
    assert len(details) == 2
    assert details[0]["title"] == "Book"
    assert "更新状態" in details[0]["message"]
    assert "再OCRは必要ありません" in details[0]["action"]
    assert details[1]["title"] == "B"
    assert "unknown_reason" in details[1]["message"]
    assert "実行ログ" in details[1]["action"]
    assert item_failure_details({}) == []


def test_report_explains_missing_source_and_manual_actions():
    details = report_failure_details({
        "gate": {"global_failures": ["inflight_attachments", "hnsw_not_validated"]},
        "documents": [{"title": "Missing", "attachment_key": "ATT", "error": "not found"},
                      {"title": "Lost", "lost_pages": [3, 5]}],
        "missing_required_from_manifest": ["NEW"],
        "orphan_chunks": ["ORPHAN"], "unretrievable_items": ["HIDDEN"],
    })
    assert len(details) == 7
    assert all(detail["message"] and detail["action"] for detail in details)
    assert "記録だけは消しません" in details[0]["action"]


def test_reconciliation_attachment_records_show_sync_recovery():
    details = report_failure_details({"missing_required_from_manifest": [
        {"attachment_key": "WAITING", "filename": "book.pdf"}, "OLD",
    ]})
    assert details[0]["attachment_key"] == "WAITING"
    assert details[0]["title"] == "book.pdf"
    assert details[1]["attachment_key"] == "OLD"
    assert all("未合格" in row["message"] for row in details)
    assert all("同期" in row["action"] and "再実行" in row["action"] for row in details)
