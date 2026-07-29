from __future__ import annotations

import importlib
import os
import unittest
from pathlib import Path
from unittest import mock

from src.extraction_engine import (
    ChunkAdapter, EngineCapabilities, EngineRegistry, MistralOCREngine, SourceRef,
    inspect_engine_availability, pymupdf_fast_path_passes,
    pymupdf_fast_path_rejection_reason,
)


class _FakeEngine(ChunkAdapter):
    name = "fake"
    version = "test"
    _capabilities = EngineCapabilities(headings=True, zones=True)

    def extract(self, source: SourceRef):  # type: ignore[no-untyped-def]
        return [("c1", "text", {"locator": "page:1"})], {"embedded_ok": True}


class ExtractionEngineTests(unittest.TestCase):
    def test_adapter_adds_provenance_without_replacing_source_metadata(self):
        result = _FakeEngine().parse(SourceRef(Path("sample.pdf"), "A", {"itemKey": "I"}))
        self.assertEqual(result.blocks[0]["metadata"]["locator"], "page:1")
        self.assertEqual(result.blocks[0]["metadata"]["extraction_engine"], "fake")
        self.assertEqual(result.blocks[0]["metadata"]["extraction_version"], "test")

    def test_explicit_engine_selection_is_deterministic(self):
        registry = EngineRegistry([_FakeEngine()])
        self.assertIsInstance(registry.select(requested="fake"), _FakeEngine)
        with self.assertRaises(ValueError):
            registry.select(requested="missing")

    def test_default_registry_exposes_all_local_bakeoff_candidates(self):
        self.assertEqual(
            EngineRegistry().names(),
            ("pymupdf", "docling", "ndlocr_lite", "yomitoku", "mistral_ocr"),
        )

    def test_default_selection_is_docling(self):
        self.assertEqual(EngineRegistry().select().name, "docling")

    def test_fast_path_requires_outline_and_healthy_document(self):
        clean = {
            "has_outline": True, "is_scanned": False, "is_corrupted": False,
            "scanned_ratio": 0.0, "corrupted_ratio": 0.0, "extraction_failure_ratio": 0.0,
        }
        self.assertTrue(pymupdf_fast_path_passes(clean))
        # A missing outline alone isn't disqualifying: a document with no
        # native TOC and clean text has nothing structural to recover, so
        # there's no point deferring it to Mistral OCR (user decision
        # 2026-07-26; see pymupdf_fast_path_rejection_reason docstring).
        self.assertTrue(pymupdf_fast_path_passes({**clean, "has_outline": False}))
        self.assertFalse(pymupdf_fast_path_passes({**clean, "is_scanned": True}))
        self.assertFalse(pymupdf_fast_path_passes({**clean, "is_corrupted": True}))
        self.assertFalse(pymupdf_fast_path_passes({**clean, "scanned_ratio": 0.1}))
        self.assertFalse(pymupdf_fast_path_passes({**clean, "corrupted_ratio": 0.1}))
        self.assertFalse(pymupdf_fast_path_passes({**clean, "extraction_failure_ratio": 0.1}))

    def test_native_outline_tolerates_sparse_cover_map_and_index_anomalies(self):
        quality = {
            "has_outline": True, "is_scanned": False, "is_corrupted": False,
            "scanned_ratio": 0.01, "corrupted_ratio": 0.014,
            "extraction_failure_ratio": 0.0,
        }
        self.assertTrue(pymupdf_fast_path_passes(quality))
        self.assertIsNone(pymupdf_fast_path_rejection_reason(quality))
        rejected = {**quality, "corrupted_ratio": 0.021}
        self.assertEqual(
            pymupdf_fast_path_rejection_reason(rejected),
            "corrupted_ratio_above_native_outline_tolerance",
        )

    def test_fast_path_accepts_deterministically_recovered_ai_toc(self):
        recovered = {
            "has_outline": False, "ai_toc_recovery_status": "accepted",
            "is_scanned": False, "is_corrupted": False,
            "scanned_ratio": 0.0, "corrupted_ratio": 0.0,
            "extraction_failure_ratio": 0.0,
        }
        self.assertTrue(pymupdf_fast_path_passes(recovered))
        # The no-outline tolerance is unified with the native-outline one
        # (2%, PYMUPDF_NATIVE_OUTLINE_ANOMALY_RATIO_MAX) -- a missing outline
        # is no longer inherently stricter (user decision 2026-07-26): 0.01
        # stays within tolerance, 0.03 doesn't.
        self.assertTrue(pymupdf_fast_path_passes({**recovered, "scanned_ratio": 0.01}))
        self.assertFalse(pymupdf_fast_path_passes({**recovered, "scanned_ratio": 0.03}))

    def test_package_import_path_loads_existing_adapter_modules(self):
        registry = EngineRegistry()
        for module_name in (
            "src.pdf_extract",
            "src.docling_extract",
            "src.ndlocr_extract",
        ):
            self.assertIsNotNone(importlib.import_module(module_name))
        self.assertEqual(registry.get("pymupdf").name, "pymupdf")

    def test_mistral_ocr_requires_key_and_explicit_cloud_opt_in(self):
        engine = MistralOCREngine()
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop("MISTRAL_OCR_API_KEY", None)
            os.environ.pop("OCR_BAKEOFF_ALLOW_CLOUD", None)
            self.assertFalse(inspect_engine_availability(engine).available)

            os.environ["MISTRAL_OCR_API_KEY"] = "sk-test"
            state = inspect_engine_availability(engine)
            self.assertFalse(state.available)
            self.assertIn("OCR_BAKEOFF_ALLOW_CLOUD", state.reason)

            os.environ["OCR_BAKEOFF_ALLOW_CLOUD"] = "1"
            self.assertTrue(inspect_engine_availability(engine).available)

    def test_no_cloud_selection_rejects_mistral_ocr(self):
        registry = EngineRegistry()
        with self.assertRaises(ValueError):
            registry.select(requested="mistral_ocr", no_cloud=True)


if __name__ == "__main__":
    unittest.main()
