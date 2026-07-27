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
