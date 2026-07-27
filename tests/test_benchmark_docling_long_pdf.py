from __future__ import annotations

import importlib.util
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "benchmark_docling_long_pdf", ROOT / "scripts" / "benchmark_docling_long_pdf.py",
)
assert SPEC and SPEC.loader
module = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(module)


class LongDoclingBenchmarkTests(unittest.TestCase):
    def test_document_metrics_counts_content_pages_and_labels(self):
        class Label:
            value = "text"

        class Prov:
            page_no = 2

        class Item:
            text = "abc"
            label = Label()
            prov = [Prov()]

        class Document:
            def iterate_items(self):
                return iter([(Item(), 0), (Item(), 0)])

        self.assertEqual(module.document_metrics(Document()), {
            "items": 2, "characters": 6, "pages_with_content": 1,
            "labels": {"text": 2},
        })


if __name__ == "__main__":
    unittest.main()
