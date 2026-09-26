from src.evidence_reference import build_evidence_reference


def test_reference_is_portable_and_locatable():
    ref = build_evidence_reference("A:p2:para1:part0", "quote", {
        "itemKey": "I", "attachmentKey": "A", "title": "Book", "page": 2,
        "chapter": "One", "pdf_path": "/private/book.pdf",
    })
    assert ref["provider"] == "zotero-local-rag"
    assert ref["item_key"] == "I"
    assert ref["locator"] == {"page": 2, "chapter": "One"}
    assert "pdf_path" not in ref


def test_reference_requires_item_and_chunk():
    assert build_evidence_reference("chunk", "quote", {}) is None
