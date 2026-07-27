from __future__ import annotations

import importlib.util
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "run_granite_docling_bakeoff", ROOT / "scripts" / "run_granite_docling_bakeoff.py",
)
assert SPEC and SPEC.loader
module = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(module)


class GraniteDoclingBakeoffTests(unittest.TestCase):
    def test_document_adapter_preserves_order_structure_and_bbox(self):
        fake_document = object()
        item = {
            "text": "Section text", "reading_order": 3, "block_type": "text",
            "zone": "body", "structure_path": ["Chapter"],
            "provenance": [{"bbox": {"l": 1, "t": 2, "r": 3, "b": 4}}],
        }
        original = module._docling_items
        module._docling_items = lambda _document: {2: [item]}
        try:
            blocks = module.document_to_bakeoff_blocks(fake_document)
        finally:
            module._docling_items = original
        self.assertEqual(blocks[0]["ordinal"], 0)
        self.assertEqual(blocks[0]["metadata"]["structure_path"], ["Chapter"])
        self.assertEqual(blocks[0]["metadata"]["bbox"]["l"], 1)

    def test_frozen_english_fixtures_exist(self):
        for sample in module.ENGLISH_SAMPLES:
            self.assertTrue((module.FIXTURE_ROOT / "sources" / f"{sample}.pdf").is_file())
            self.assertTrue((module.EVALUATION_ROOT / "annotations" / f"{sample}.json").is_file())
            self.assertTrue((module.FIXTURE_ROOT / "raw" / sample / "docling.json").is_file())


if __name__ == "__main__":
    unittest.main()
