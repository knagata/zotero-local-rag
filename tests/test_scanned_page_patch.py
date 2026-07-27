from __future__ import annotations

import os
import unittest
from unittest import mock

from src.extraction_engine import pymupdf_fast_path_rejection_reason
from src.pdf_extract import (
    corrupted_ratio_threshold,
    recompute_scanned_quality_after_patch,
    recompute_corrupted_quality_after_patch,
    scanned_ratio_threshold,
)


class RecomputeScannedQualityAfterPatchTests(unittest.TestCase):
    def test_patched_pages_are_removed_and_ratio_recomputed(self):
        quality = {"scanned_pages": [3, 4, 5], "scanned_ratio": 0.03, "is_scanned": False}
        updated = recompute_scanned_quality_after_patch(quality, {3, 4}, total_pages=100)
        self.assertEqual(updated["scanned_pages"], [5])
        self.assertEqual(updated["scanned_ratio"], 0.01)
        self.assertFalse(updated["is_scanned"])

    def test_fully_patched_document_has_zero_scanned_ratio(self):
        quality = {"scanned_pages": [1, 2], "scanned_ratio": 0.02, "is_scanned": False}
        updated = recompute_scanned_quality_after_patch(quality, {1, 2}, total_pages=100)
        self.assertEqual(updated["scanned_pages"], [])
        self.assertEqual(updated["scanned_ratio"], 0.0)
        self.assertFalse(updated["is_scanned"])

    def test_is_scanned_flips_true_if_remaining_ratio_crosses_threshold(self):
        # 85 of 100 pages still scanned after a partial patch attempt -> is_scanned.
        quality = {"scanned_pages": list(range(1, 91)), "scanned_ratio": 0.9, "is_scanned": True}
        updated = recompute_scanned_quality_after_patch(
            quality, set(range(1, 6)), total_pages=100,
        )
        self.assertEqual(len(updated["scanned_pages"]), 85)
        self.assertGreaterEqual(updated["scanned_ratio"], 0.8)
        self.assertTrue(updated["is_scanned"])

    def test_original_quality_info_is_not_mutated(self):
        quality = {"scanned_pages": [1, 2], "scanned_ratio": 0.02, "is_scanned": False}
        recompute_scanned_quality_after_patch(quality, {1}, total_pages=100)
        self.assertEqual(quality["scanned_pages"], [1, 2])


class RecomputeCorruptedQualityAfterPatchTests(unittest.TestCase):
    # E2d (dev-notes/current/77, user decision 2026-07-26): sister function
    # to recompute_scanned_quality_after_patch, same contract.
    def test_patched_pages_are_removed_and_ratio_recomputed(self):
        quality = {"corrupted_pages": [3, 4, 5], "corrupted_ratio": 0.03, "is_corrupted": False}
        updated = recompute_corrupted_quality_after_patch(quality, {3, 4}, total_pages=100)
        self.assertEqual(updated["corrupted_pages"], [5])
        self.assertEqual(updated["corrupted_ratio"], 0.01)
        self.assertFalse(updated["is_corrupted"])

    def test_fully_patched_document_has_zero_corrupted_ratio(self):
        quality = {"corrupted_pages": [1, 2], "corrupted_ratio": 0.02, "is_corrupted": False}
        updated = recompute_corrupted_quality_after_patch(quality, {1, 2}, total_pages=100)
        self.assertEqual(updated["corrupted_pages"], [])
        self.assertEqual(updated["corrupted_ratio"], 0.0)
        self.assertFalse(updated["is_corrupted"])

    def test_is_corrupted_flips_true_if_remaining_ratio_crosses_threshold(self):
        # 65 of 100 pages still corrupted after a partial patch attempt ->
        # is_corrupted (threshold 0.6, TEXT_QUALITY_CORRUPTION_THRESHOLD default).
        quality = {"corrupted_pages": list(range(1, 71)), "corrupted_ratio": 0.7, "is_corrupted": True}
        updated = recompute_corrupted_quality_after_patch(
            quality, set(range(1, 6)), total_pages=100,
        )
        self.assertEqual(len(updated["corrupted_pages"]), 65)
        self.assertGreaterEqual(updated["corrupted_ratio"], 0.6)
        self.assertTrue(updated["is_corrupted"])

    def test_original_quality_info_is_not_mutated(self):
        quality = {"corrupted_pages": [1, 2], "corrupted_ratio": 0.02, "is_corrupted": False}
        recompute_corrupted_quality_after_patch(quality, {1}, total_pages=100)
        self.assertEqual(quality["corrupted_pages"], [1, 2])

    def test_extraction_failure_breakdown_is_also_recomputed(self):
        # Review finding (2026-07-26): corrupted_pages is the union of
        # extraction_failure_pages + content_corruption_pages, so patching
        # pages out of the union must also patch them out of the breakdowns.
        # extraction_failure_ratio feeds pymupdf_fast_path_rejection_reason's
        # unconditional early-reject; a stale pre-patch value would veto the
        # fast path even after a successful repair.
        quality = {
            "corrupted_pages": [2, 5, 9],
            "extraction_failure_pages": [2, 5],
            "content_corruption_pages": [9],
            "corrupted_ratio": 0.03,
            "extraction_failure_ratio": 0.02,
            "is_corrupted": False,
        }
        updated = recompute_corrupted_quality_after_patch(quality, {2, 5, 9}, total_pages=100)
        self.assertEqual(updated["corrupted_pages"], [])
        self.assertEqual(updated["extraction_failure_pages"], [])
        self.assertEqual(updated["extraction_failure_ratio"], 0.0)
        self.assertEqual(updated["content_corruption_pages"], [])
        # Original untouched.
        self.assertEqual(quality["extraction_failure_pages"], [2, 5])
        self.assertEqual(quality["extraction_failure_ratio"], 0.02)

    def test_partial_patch_keeps_unattempted_breakdown_pages(self):
        quality = {
            "corrupted_pages": [2, 5, 9],
            "extraction_failure_pages": [2, 5],
            "content_corruption_pages": [9],
            "corrupted_ratio": 0.03,
            "extraction_failure_ratio": 0.02,
            "is_corrupted": False,
        }
        updated = recompute_corrupted_quality_after_patch(quality, {2}, total_pages=100)
        self.assertEqual(updated["extraction_failure_pages"], [5])
        self.assertEqual(updated["extraction_failure_ratio"], 0.01)
        self.assertEqual(updated["content_corruption_pages"], [9])

    def test_fully_patched_extraction_failure_document_passes_fast_path_gate(self):
        # End-to-end regression for the review finding: a document whose only
        # anomaly was extraction_failure-type corruption, fully repaired by the
        # E2d patch, must clear pymupdf_fast_path_rejection_reason -- including
        # its unconditional extraction_failure_ratio early-reject.
        quality = {
            "has_outline": True, "is_scanned": False, "is_corrupted": False,
            "scanned_pages": [], "scanned_ratio": 0.0,
            "corrupted_pages": [4, 7],
            "extraction_failure_pages": [4, 7],
            "content_corruption_pages": [],
            "corrupted_ratio": 0.02,
            "extraction_failure_ratio": 0.02,
        }
        # Pre-patch: the early-reject fires on the nonzero ratio.
        self.assertEqual(
            pymupdf_fast_path_rejection_reason(quality),
            "extraction_failure_ratio_nonzero",
        )
        updated = recompute_corrupted_quality_after_patch(quality, {4, 7}, total_pages=100)
        self.assertIsNone(pymupdf_fast_path_rejection_reason(updated))


class RatioThresholdSeparationTests(unittest.TestCase):
    # P3 (dev-notes/current/78, user approval 2026-07-26): document-level
    # ratio thresholds get their own env vars; unset, they inherit the legacy
    # page-level variables so existing .env files behave identically.
    def test_defaults_match_legacy_values(self):
        with mock.patch.dict(os.environ, {}, clear=False):
            for name in (
                "TEXT_QUALITY_SCAN_THRESHOLD", "TEXT_QUALITY_SCANNED_RATIO_THRESHOLD",
                "TEXT_QUALITY_CORRUPTION_THRESHOLD", "TEXT_QUALITY_CORRUPTED_RATIO_THRESHOLD",
            ):
                os.environ.pop(name, None)
            self.assertEqual(scanned_ratio_threshold(), 0.8)
            self.assertEqual(corrupted_ratio_threshold(), 0.6)

    def test_unset_ratio_variable_inherits_legacy_variable(self):
        env = {"TEXT_QUALITY_SCAN_THRESHOLD": "0.7", "TEXT_QUALITY_CORRUPTION_THRESHOLD": "0.5"}
        with mock.patch.dict(os.environ, env, clear=False):
            os.environ.pop("TEXT_QUALITY_SCANNED_RATIO_THRESHOLD", None)
            os.environ.pop("TEXT_QUALITY_CORRUPTED_RATIO_THRESHOLD", None)
            self.assertEqual(scanned_ratio_threshold(), 0.7)
            self.assertEqual(corrupted_ratio_threshold(), 0.5)

    def test_dedicated_ratio_variable_wins_without_moving_page_threshold(self):
        env = {
            "TEXT_QUALITY_SCAN_THRESHOLD": "0.8",
            "TEXT_QUALITY_SCANNED_RATIO_THRESHOLD": "0.9",
            "TEXT_QUALITY_CORRUPTION_THRESHOLD": "0.6",
            "TEXT_QUALITY_CORRUPTED_RATIO_THRESHOLD": "0.4",
        }
        with mock.patch.dict(os.environ, env, clear=False):
            self.assertEqual(scanned_ratio_threshold(), 0.9)
            self.assertEqual(corrupted_ratio_threshold(), 0.4)
            # And the recompute helpers consume the dedicated variable.
            quality = {"corrupted_pages": list(range(1, 51)), "is_corrupted": False}
            updated = recompute_corrupted_quality_after_patch(quality, set(), total_pages=100)
            self.assertTrue(updated["is_corrupted"])  # 0.5 >= 0.4


if __name__ == "__main__":
    unittest.main()
