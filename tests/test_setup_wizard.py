from __future__ import annotations

import stat
import tempfile
import unittest
from io import StringIO
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from scripts import setup_wizard


class SetupWizardTests(unittest.TestCase):
    def test_grouped_env_round_trip_preserves_unknown_settings(self):
        values = {
            "FEATURE_LEVEL": "citation",
            "ZOTERO_DATA_DIR": "/tmp/Zotero",
            "EMB_PROFILE": "fast",
            "S2_API_KEY": "secret",
            "CUSTOM_SETTING": "kept",
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / ".env"
            setup_wizard.write_env_file(path, values)
            rendered = path.read_text(encoding="utf-8")
            self.assertIn("# Core: local search and indexing", rendered)
            self.assertIn("# Citation network and bibliographic metadata", rendered)
            self.assertIn("# Other existing settings", rendered)
            self.assertEqual(setup_wizard.read_env_file(path), values)
            self.assertEqual(stat.S_IMODE(path.stat().st_mode), 0o600)
            self.assertFalse(path.with_name(".env.tmp").exists())

    def test_env_write_does_not_follow_a_predictable_temp_symlink(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            victim = root / "victim"
            victim.write_text("do not overwrite", encoding="utf-8")
            (root / ".env.tmp").symlink_to(victim)

            path = root / ".env"
            setup_wizard.write_env_file(path, {"FEATURE_LEVEL": "core"})

            self.assertEqual(victim.read_text(encoding="utf-8"), "do not overwrite")
            self.assertFalse(path.is_symlink())
            self.assertEqual(
                setup_wizard.read_env_file(path)["FEATURE_LEVEL"],
                "core",
            )
            self.assertEqual(stat.S_IMODE(path.stat().st_mode), 0o600)

    def test_core_level_does_not_require_network_credentials(self):
        config = {"UNRELATED": "preserved"}
        with patch("builtins.input", side_effect=["1"]), patch("sys.stdout", StringIO()):
            setup_wizard.configure_feature_level(config)
        self.assertEqual(config["FEATURE_LEVEL"], "core")
        self.assertNotIn("S2_API_KEY", config)
        self.assertNotIn("LLM_STANDARD", config)
        self.assertEqual(config["UNRELATED"], "preserved")

    def test_minimal_preset_needs_no_keys_and_indexes_pdfs_flat(self):
        config: dict[str, str] = {}
        with patch("builtins.input", side_effect=["1"]), patch("sys.stdout", StringIO()):
            setup_wizard.configure_feature_level(config)
        self.assertEqual(setup_wizard.describe_preset(config), "minimal")
        self.assertEqual(config["PDF_STRUCTURE_RECOVERY_ENABLE"], "0")
        for flag in setup_wizard.LLM_FLAGS:
            self.assertEqual(config[flag], "0")
        self.assertNotIn("S2_API_KEY", config)
        self.assertEqual(config["INGEST_STRUCTURED_V3_ENABLE"], "1")
        self.assertEqual(config["CHROMA_COLLECTION"], "zotero_paragraphs_v3")

    def test_local_preset_structures_pdfs_without_paid_services(self):
        config: dict[str, str] = {}
        with patch("builtins.input", side_effect=["2", "n"]), patch(
            "getpass.getpass", side_effect=["s2-secret"],
        ), patch("sys.stdout", StringIO()):
            setup_wizard.configure_feature_level(config)
        self.assertEqual(config["PDF_STRUCTURE_RECOVERY_ENABLE"], "1")
        self.assertEqual(config["PDF_STRUCTURE_ENGINE_LONG"], "docling")
        self.assertEqual(config["CITATION_NETWORK_ENABLE"], "1")
        self.assertEqual(config["PDF_MISTRAL_TOC_QUEUE_ENABLE"], "0")
        for flag in setup_wizard.LLM_FLAGS:
            self.assertEqual(config[flag], "0")

    def test_citation_network_is_switched_off_when_no_s2_key_is_given(self):
        # The key is free but the unkeyed rate limit is unusable, so leaving the
        # feature on would only produce a run that refuses to start.
        config: dict[str, str] = {}
        with patch("builtins.input", side_effect=["2", "n"]), patch(
            "getpass.getpass", side_effect=[""],
        ), patch("sys.stdout", StringIO()):
            setup_wizard.configure_feature_level(config)
        self.assertEqual(config["CITATION_NETWORK_ENABLE"], "0")

    def test_full_preset_configures_deepseek_and_mistral(self):
        config: dict[str, str] = {}
        with patch("builtins.input", side_effect=["3", "1", "n"]), patch(
            "getpass.getpass", side_effect=["s2-secret", "deepseek-secret", "mistral-secret"],
        ), patch("sys.stdout", StringIO()):
            setup_wizard.configure_feature_level(config)
        self.assertEqual(config["LLM_CHEAP"], "deepseek:deepseek-v4-flash")
        self.assertEqual(config["DEEPSEEK_API_KEY"], "deepseek-secret")
        self.assertEqual(config["PDF_STRUCTURE_ENGINE_LONG"], "mistral")
        for flag in setup_wizard.LLM_FLAGS:
            self.assertEqual(config[flag], "1")

    def test_full_preset_without_a_mistral_key_falls_back_to_docling(self):
        # Leaving PDF_STRUCTURE_ENGINE_LONG=mistral would stop the next run.
        config: dict[str, str] = {}
        with patch("builtins.input", side_effect=["3", "1", "n"]), patch(
            "getpass.getpass", side_effect=["s2-secret", "deepseek-secret", ""],
        ), patch("sys.stdout", StringIO()):
            setup_wizard.configure_feature_level(config)
        self.assertEqual(config["PDF_STRUCTURE_ENGINE_LONG"], "docling")
        self.assertEqual(config["PDF_MISTRAL_TOC_QUEUE_ENABLE"], "0")

    def test_granite_menu_numbers_match_the_rendered_order(self):
        with patch.object(setup_wizard, "_granite_selectable", return_value=True), patch(
            "builtins.input", return_value="2",
        ), patch("sys.stdout", StringIO()):
            self.assertEqual(setup_wizard._choose_engine("Long", "docling"), "granite")
        with patch.object(setup_wizard, "_granite_selectable", return_value=True), patch(
            "builtins.input", return_value="3",
        ), patch("sys.stdout", StringIO()):
            self.assertEqual(setup_wizard._choose_engine("Long", "docling"), "mistral")

    def test_switching_engines_away_from_mistral_disables_queue(self):
        config = {
            "PDF_STRUCTURE_ENGINE_SHORT": "mistral",
            "PDF_STRUCTURE_ENGINE_LONG": "mistral",
            "PDF_MISTRAL_TOC_QUEUE_ENABLE": "1",
        }
        with patch.object(
            setup_wizard, "_choose_engine", side_effect=["docling", "docling"],
        ), patch("builtins.input", return_value=""), patch("sys.stdout", StringIO()):
            setup_wizard.configure_pdf_engines(config)
        self.assertEqual(config["PDF_MISTRAL_TOC_QUEUE_ENABLE"], "0")

    def test_enforce_v3_configuration_replaces_retired_targets(self):
        config = {
            "INGEST_STRUCTURED_V3_ENABLE": "0",
            "CHROMA_COLLECTION": "zotero_paragraphs",
            "MANIFEST_PATH": "data/manifest.json",
            "LEXICAL_DB_PATH": "data/lexical.sqlite3",
        }
        setup_wizard.enforce_v3_configuration(config)
        self.assertEqual(config, setup_wizard.V3_DATA_PLANE)

    def test_server_reconfiguration_keeps_existing_path_and_never_runs_indexer(self):
        with tempfile.TemporaryDirectory() as directory:
            zotero = Path(directory) / "Zotero Custom"
            zotero.mkdir()
            (zotero / "storage").mkdir()
            (zotero / "zotero.sqlite").write_text("", encoding="utf-8")
            config = {
                **setup_wizard.PRESETS["full"],
                "ZOTERO_DATA_DIR": str(zotero),
                "EMB_PROFILE": "bge",
                "S2_API_KEY": "s2",
                "DEEPSEEK_API_KEY": "deepseek",
                "MISTRAL_OCR_API_KEY": "mistral",
                "LLM_CHEAP": "deepseek:cheap",
                "LLM_STANDARD": "deepseek:standard",
                "LLM_REVIEW": "deepseek:review",
            }
            written = {}
            with patch.object(
                setup_wizard, "read_env_file", return_value=dict(config),
            ), patch.object(
                setup_wizard, "write_env_file",
                side_effect=lambda _path, values: written.update(values),
            ), patch.object(
                setup_wizard, "configure_feature_level",
            ), patch(
                "builtins.input", side_effect=["y", "", ""],
            ), patch.object(
                setup_wizard.shutil, "which", return_value=None,
            ), patch.object(
                setup_wizard.subprocess, "run",
                return_value=SimpleNamespace(stdout="", returncode=0),
            ) as executed, patch("sys.stdout", StringIO()):
                setup_wizard.main(["--server"])

        self.assertEqual(written["ZOTERO_DATA_DIR"], str(zotero))
        self.assertEqual(written["EMB_PROFILE"], "bge")
        self.assertEqual(written["CHROMA_COLLECTION"], "zotero_paragraphs_v3")
        self.assertFalse(any(
            "index_from_zotero.py" in str(call.args[0]) for call in executed.call_args_list
        ))

    def test_keeping_current_settings_changes_nothing(self):
        config = dict(setup_wizard.PRESETS["full"])
        config["LLM_CHEAP"] = "deepseek:deepseek-v4-flash"
        before = dict(config)
        with patch("builtins.input", side_effect=["0"]), patch("sys.stdout", StringIO()):
            setup_wizard.configure_feature_level(config)
        self.assertEqual(config, before)

    def test_removed_cloud_policy_keys_are_cleared_on_any_run(self):
        config = {"EXTRACT_EXCLUDE_TAGS": "stale", "SUMMARY_ALLOW_CLOUD_ALL": "1",
                  "MISTRAL_OCR_FALLBACK_ENABLE": "1"}
        with patch("builtins.input", side_effect=["1"]), patch("sys.stdout", StringIO()):
            setup_wizard.configure_feature_level(config)
        for key in ("EXTRACT_EXCLUDE_TAGS", "SUMMARY_ALLOW_CLOUD_ALL",
                    "MISTRAL_OCR_FALLBACK_ENABLE"):
            self.assertNotIn(key, config)

    def test_status_never_prints_secret_values(self):
        config = {
            "FEATURE_LEVEL": "llm", "S2_API_KEY": "do-not-print",
            "DEEPSEEK_API_KEY": "also-secret", "LLM_STANDARD": "deepseek:model",
        }
        output = StringIO()
        with patch("sys.stdout", output):
            setup_wizard.print_configuration_status(config)
        rendered = output.getvalue()
        self.assertIn("S2 API key     : configured", rendered)
        self.assertIn("preset         :", rendered)
        self.assertNotIn("do-not-print", rendered)
        self.assertNotIn("also-secret", rendered)


if __name__ == "__main__":
    unittest.main()
