from scripts.list_mistral_toc_candidates import build_queue_rows


def test_build_queue_rows_only_exports_blocked_ai_toc_candidates():
    rows = build_queue_rows([
        {
            "item_key": "I1", "attachment_key": "A1", "artifact_type": "extraction",
            "status": "blocked", "reason_code": "awaiting_mistral_ocr_batch",
            "counts": {"total_pages": 210, "ai_toc_reason": "body_coverage_below_threshold"},
        },
        {
            "item_key": "I2", "attachment_key": "A2", "artifact_type": "extraction",
            "status": "failed", "reason_code": "no_chunks", "counts": {},
        },
    ])
    assert rows == [{
        "item_key": "I1", "attachment_key": "A1", "target_engine": "mistral_ocr",
        "recommendation": "mistral_ocr_batch_after_ai_toc_gate",
        "queue_reason": "awaiting_mistral_ocr_batch",
        "ai_toc_reason": "body_coverage_below_threshold",
        "ai_toc_diagnostics": {}, "total_pages": 210,
        "source_mtime": None, "source_size": None,
        "source_type": "pdf", "batch_document_path": None,
        "epub_mapping_path": None,
    }]


def test_build_queue_rows_carries_fixed_layout_epub_derivative():
    rows = build_queue_rows([{
        "item_key": "I1", "attachment_key": "E1", "artifact_type": "extraction",
        "status": "blocked", "reason_code": "awaiting_mistral_ocr_batch",
        "counts": {
            "source_type": "epub", "total_pages": 130,
            "batch_document_path": "/cache/E1.pdf",
            "epub_mapping_path": "/cache/E1.json",
        },
    }])
    assert rows[0]["source_type"] == "epub"
    assert rows[0]["batch_document_path"] == "/cache/E1.pdf"
    assert rows[0]["epub_mapping_path"] == "/cache/E1.json"
