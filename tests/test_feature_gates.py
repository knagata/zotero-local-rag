"""Tests for feature resolution (note 80).

Two rules are pinned here: everything is off until explicitly enabled, and a
feature switched on without the resource it needs is reported as an error
rather than quietly doing nothing.
"""
from __future__ import annotations

import os
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import feature_gates as fg  # noqa: E402

#: Every gate, so a newly added one cannot quietly skip the default-off rule.
ALL_GATES = (
    fg.ai_toc_enabled, fg.ocr_layer_audit_enabled, fg.query_expansion_enabled,
    fg.llm_summaries_enabled, fg.llm_reference_extraction_enabled,
    fg.mistral_batch_queue_enabled, fg.citation_network_enabled,
    fg.pdf_structure_recovery_enabled,
)

_CLEARED = {
    "LLM_CHEAP": "", "LLM_STANDARD": "", "LLM_REVIEW": "",
    "MISTRAL_OCR_API_KEY": "", "S2_API_KEY": "",
    "PDF_AI_TOC_FAST_PATH_ENABLE": "", "OCR_LAYER_AUDIT_ENABLE": "",
    "QUERY_EXPANSION_ENABLE": "", "LLM_SUMMARIES_ENABLE": "",
    "LLM_REFERENCE_EXTRACTION_ENABLE": "", "PDF_MISTRAL_TOC_QUEUE_ENABLE": "",
    "CITATION_NETWORK_ENABLE": "", "PDF_STRUCTURE_RECOVERY_ENABLE": "",
    # Choice (C). Omitting these let a developer's real .env -- loaded into
    # os.environ by an unrelated test earlier in the run -- decide the engine,
    # so these cases passed alone and failed in the full suite.
    "PDF_STRUCTURE_ENGINE_SHORT": "", "PDF_STRUCTURE_ENGINE_LONG": "",
    "PDF_STRUCTURE_ENGINE_PAGE_BOUNDARY": "",
}


def _env(**overrides):
    """Patch the environment with every relevant variable explicitly cleared.

    ``load_dotenv_native`` is stubbed out so a developer's real .env cannot
    leak into the result.
    """
    values = dict(_CLEARED)
    values.update({key: str(value) for key, value in overrides.items()})
    return patch.dict(os.environ, values, clear=False), patch.object(
        fg, "load_dotenv_native", lambda *_a, **_k: None,
    )


class DefaultOffTests(unittest.TestCase):
    """Nothing runs until it is asked for (user decision 2026-07-27).

    The wizard writes the values it needs, so a default that tries to infer
    intent only makes the resulting .env harder to read.
    """

    def test_every_feature_is_off_when_nothing_is_set(self):
        env, dotenv = _env()
        with env, dotenv:
            for gate in ALL_GATES:
                self.assertFalse(gate(), gate.__name__)

    def test_a_configured_key_alone_does_not_switch_anything_on(self):
        env, dotenv = _env(
            LLM_CHEAP="deepseek:deepseek-v4-flash", MISTRAL_OCR_API_KEY="sk-test",
            S2_API_KEY="s2-test",
        )
        with env, dotenv:
            self.assertTrue(fg.llm_configured())
            for gate in ALL_GATES:
                self.assertFalse(gate(), gate.__name__)

    def test_explicit_one_switches_a_feature_on(self):
        env, dotenv = _env(
            LLM_CHEAP="deepseek:deepseek-v4-flash", PDF_AI_TOC_FAST_PATH_ENABLE="1",
        )
        with env, dotenv:
            self.assertTrue(fg.ai_toc_enabled())
            self.assertFalse(fg.ocr_layer_audit_enabled())

    def test_flag_enabled_recognises_the_usual_spellings(self):
        for raw, expected in (("true", True), ("YES", True), ("on", True),
                              ("false", False), ("No", False), ("off", False)):
            env, dotenv = _env(SOME_FLAG=raw)
            with env, dotenv:
                self.assertEqual(fg.flag_enabled("SOME_FLAG"), expected, raw)


class ResourceDetectionTests(unittest.TestCase):
    def test_llm_roles_are_recognised_by_provider_prefix(self):
        env, dotenv = _env(LLM_CHEAP="deepseek:deepseek-v4-flash")
        with env, dotenv:
            self.assertTrue(fg.llm_configured())

    def test_a_non_provider_value_does_not_count_as_configured(self):
        for spec in ("", "none", "disabled", "deepseek"):
            env, dotenv = _env(LLM_CHEAP=spec)
            with env, dotenv:
                self.assertFalse(fg.llm_configured(), spec)

    def test_local_cli_agent_counts_as_an_llm(self):
        env, dotenv = _env(LLM_STANDARD="codex_cli:gpt-5.6-luna")
        with env, dotenv:
            self.assertTrue(fg.llm_configured())


class VerifyEnabledFeaturesTests(unittest.TestCase):
    """A feature on without its resource is an error, not a silent downgrade."""

    def test_coherent_configuration_reports_nothing(self):
        env, dotenv = _env(
            LLM_CHEAP="deepseek:deepseek-v4-flash", PDF_AI_TOC_FAST_PATH_ENABLE="1",
        )
        with env, dotenv:
            self.assertEqual(fg.verify_enabled_features(), [])

    def test_enabled_without_the_key_is_reported_by_name(self):
        env, dotenv = _env(PDF_AI_TOC_FAST_PATH_ENABLE="1")
        with env, dotenv:
            problems = fg.verify_enabled_features()
        self.assertEqual(len(problems), 1)
        self.assertIn("AI TOC fast path", problems[0])
        self.assertIn("LLM_* role", problems[0])

    def test_each_missing_resource_is_reported_separately(self):
        env, dotenv = _env(
            PDF_AI_TOC_FAST_PATH_ENABLE="1", PDF_MISTRAL_TOC_QUEUE_ENABLE="1",
            CITATION_NETWORK_ENABLE="1",
        )
        with env, dotenv:
            problems = fg.verify_enabled_features()
        self.assertEqual(len(problems), 3)
        self.assertTrue(any("MISTRAL_OCR_API_KEY" in p for p in problems))
        self.assertTrue(any("S2_API_KEY" in p for p in problems))

    def test_a_disabled_feature_without_its_key_is_not_a_problem(self):
        env, dotenv = _env(PDF_AI_TOC_FAST_PATH_ENABLE="0")
        with env, dotenv:
            self.assertEqual(fg.verify_enabled_features(), [])

    def test_structure_recovery_needs_no_resource(self):
        # Choice (A) is local-only, so it is absent from FEATURE_REQUIREMENTS
        # and enabling it can never be a misconfiguration.
        env, dotenv = _env(PDF_STRUCTURE_RECOVERY_ENABLE="1")
        with env, dotenv:
            self.assertTrue(fg.pdf_structure_recovery_enabled())
            self.assertEqual(fg.verify_enabled_features(), [])


class SupersededGateTests(unittest.TestCase):
    """MISTRAL_OCR_FALLBACK_ENABLE was removed entirely (2026-07-27).

    Every remaining route to Mistral OCR is already behind an explicit operator
    action -- passing a queue to ``--reocr-candidates``, or running
    ``--submit`` -- or only records a candidate for a later, separate send.
    Automatic ingestion never calls it synchronously. A second switch in front
    of an already-explicit action produced only the failure it was meant to
    prevent: a configured key silently doing nothing.
    """

    def test_the_flag_no_longer_resolves_to_anything(self):
        self.assertFalse(hasattr(fg, "mistral_sync_fallback_enabled"))
        self.assertNotIn("MISTRAL_OCR_FALLBACK_ENABLE", fg.FEATURE_REQUIREMENTS)


if __name__ == "__main__":
    unittest.main()


class StructureEngineTests(unittest.TestCase):
    """Choice (C): the operator picks an engine per size bucket (note 80)."""

    def test_defaults_to_docling_in_both_buckets(self):
        env, dotenv = _env()
        with env, dotenv:
            self.assertEqual(fg.structure_engine_for(10), "docling")
            self.assertEqual(fg.structure_engine_for(300), "docling")

    def test_each_bucket_is_chosen_independently(self):
        env, dotenv = _env(
            PDF_STRUCTURE_ENGINE_SHORT="docling", PDF_STRUCTURE_ENGINE_LONG="mistral",
        )
        with env, dotenv:
            self.assertEqual(fg.structure_engine_for(29), "docling")
            self.assertEqual(fg.structure_engine_for(30), "mistral")

    def test_the_boundary_is_its_own_setting(self):
        # Deliberately not a reuse of PDF_AI_TOC_MIN_PAGES: that asks whether a
        # document is long enough to have a table of contents, this asks which
        # engine should carry the cost.
        env, dotenv = _env(
            PDF_STRUCTURE_ENGINE_LONG="mistral", PDF_STRUCTURE_ENGINE_PAGE_BOUNDARY="100",
        )
        with env, dotenv:
            self.assertEqual(fg.structure_engine_for(99), "docling")
            self.assertEqual(fg.structure_engine_for(100), "mistral")

    def test_an_unparseable_boundary_falls_back_to_the_default(self):
        env, dotenv = _env(PDF_STRUCTURE_ENGINE_PAGE_BOUNDARY="not-a-number")
        with env, dotenv:
            self.assertEqual(fg.structure_engine_page_boundary(), 30)

    def test_an_unknown_engine_name_is_reported(self):
        env, dotenv = _env(PDF_STRUCTURE_ENGINE_SHORT="tesseract")
        with env, dotenv:
            problems = fg.verify_enabled_features()
        self.assertTrue(any("tesseract" in p for p in problems))

    def test_mistral_without_a_key_is_reported(self):
        env, dotenv = _env(PDF_STRUCTURE_ENGINE_LONG="mistral")
        with env, dotenv:
            problems = fg.verify_enabled_features()
        self.assertTrue(any("MISTRAL_OCR_API_KEY" in p for p in problems))

    def test_granite_needs_both_the_venv_and_the_adapter(self):
        # The bake-off left a venv behind, so checking only for that would let
        # an operator select an engine the pipeline cannot call -- the route
        # would match no branch and the document would come out unstructured
        # with nothing reported.
        env, dotenv = _env(PDF_STRUCTURE_ENGINE_LONG="granite")
        with env, dotenv, patch.object(fg, "granite_configured", return_value=False):
            problems = fg.verify_enabled_features()
        self.assertTrue(any("granite" in p for p in problems))
