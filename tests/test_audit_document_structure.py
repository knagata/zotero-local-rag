from __future__ import annotations

import unittest

from scripts.audit_document_structure import audit_metrics
from src.document_structure import build_document_structure


class AuditDocumentStructureTests(unittest.TestCase):
    def test_reports_zone_distribution_and_old_new_text_delta(self):
        baseline = [{"id": "A:old", "text": "x", "metadata": {"attachmentKey": "A"}}]
        candidate = [
            {"id": "A:p1", "text": "body", "metadata": {"attachmentKey": "A", "zone": "body"}},
            {"id": "A:p2", "text": "note", "metadata": {"attachmentKey": "A", "zone": "footnote"}},
        ]
        built = build_document_structure("ITEM", candidate)
        report = audit_metrics(built["nodes"], candidate, baseline=baseline)
        self.assertTrue(report["valid"])
        self.assertEqual(report["zone_chunk_counts"], {"body": 1, "footnote": 1})
        self.assertEqual(report["comparison"]["char_delta"], 7)


if __name__ == "__main__":
    unittest.main()


def test_node_coverage_requires_the_node_to_exist():
    """Coverage must mean the node exists, not that the string is non-empty.

    The gate counted a chunk as structurally covered whenever its metadata held
    a node_id, without asking whether anything answered to it. After a rebuild
    renamed every node, 53 items still scored a coverage of 1.0 while none of
    their chunks pointed at a live node -- the audit's most substantive check
    passing on nothing (2026-07-28).
    """
    import importlib.util
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location(
        "audit_v3_cutover", root / "scripts" / "audit_v3_cutover.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    rows = [
        {"id": "a", "text": "body", "metadata": {"node_id": "dn:gone", "source_type": "pdf"}},
        {"id": "b", "text": "body", "metadata": {"node_id": "dn:live", "source_type": "pdf"}},
    ]
    self_checked = module.item_metrics(rows, {"dn:live"})
    assert self_checked["node_assigned_chunks"] == 1
    assert self_checked["node_coverage"] == 0.5

    # Without a live set the old, weaker reading is preserved for callers that
    # genuinely have nothing to check against.
    assert module.item_metrics(rows)["node_coverage"] == 1.0


def test_a_document_no_query_can_reach_fails_the_gate():
    """zone was reported but never asserted; retrieval_policy was absent entirely.

    That is how a 65,000-character essay marked zone="index" throughout
    disappeared from search while the cutover gate returned passed
    (ND49KK4N, 2026-07-28).
    """
    import importlib.util
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location(
        "audit_v3_cutover", root / "scripts" / "audit_v3_cutover.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    old_rows = [{"id": "o", "text": "essay text", "metadata": {"source_type": "pdf"}}]
    hidden = [{"id": "a", "text": "essay text", "metadata": {
        "source_type": "pdf", "node_id": "dn:live", "retrieval_policy": "exclude"}}]
    verdict = module.compare_item("ITEM", old_rows, hidden, {"dn:live"})
    assert "no_retrievable_chunks" in verdict["failures"]

    visible = [{"id": "a", "text": "essay text", "metadata": {
        "source_type": "pdf", "node_id": "dn:live", "retrieval_policy": "normal"}}]
    assert "no_retrievable_chunks" not in module.compare_item(
        "ITEM", old_rows, visible, {"dn:live"})["failures"]


def test_a_document_with_some_excluded_zones_still_passes():
    # An index or bibliography section alongside body text is exactly what the
    # zone system is for; only a document with nothing reachable is a failure.
    import importlib.util
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location(
        "audit_v3_cutover", root / "scripts" / "audit_v3_cutover.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    rows = [
        {"id": "a", "text": "body", "metadata": {
            "source_type": "pdf", "node_id": "dn:live", "retrieval_policy": "normal"}},
        {"id": "b", "text": "index entry", "metadata": {
            "source_type": "pdf", "node_id": "dn:live", "retrieval_policy": "exclude"}},
    ]
    verdict = module.compare_item("ITEM", rows, rows, {"dn:live"})
    assert "no_retrievable_chunks" not in verdict["failures"]
