"""Path-free evidence hand-off records for downstream research tools."""
from typing import Any, Mapping

SCHEMA_VERSION = "1.0"


def build_evidence_reference(chunk_id: str, quote: str, metadata: Mapping[str, Any] | None):
    metadata = metadata or {}
    item_key = str(metadata.get("itemKey") or "").strip()
    chunk_id = str(chunk_id or "").strip()
    if not item_key or not chunk_id:
        return None
    result = {
        "schema_version": SCHEMA_VERSION, "provider": "zotero-local-rag",
        "item_key": item_key, "chunk_id": chunk_id, "quote": str(quote or ""),
    }
    for target, source in (("attachment_key", "attachmentKey"), ("note_key", "noteKey"),
                           ("title", "title"), ("creators", "creators"),
                           ("year", "year"), ("source_type", "source_type")):
        if metadata.get(source) not in (None, ""):
            result[target] = metadata[source]
    locator = {key: metadata[key] for key in
               ("page", "page_label", "chapter", "section", "locator")
               if metadata.get(key) not in (None, "")}
    if locator:
        result["locator"] = locator
    return result
