from __future__ import annotations

import tempfile
import unittest
from io import StringIO
from pathlib import Path
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

    def test_core_level_does_not_require_network_credentials(self):
        config = {"UNRELATED": "preserved"}
        with patch("builtins.input", side_effect=["1"]), patch("sys.stdout", StringIO()):
            setup_wizard.configure_feature_level(config)
        self.assertEqual(config["FEATURE_LEVEL"], "core")
        self.assertNotIn("S2_API_KEY", config)
        self.assertNotIn("LLM_STANDARD", config)
        self.assertEqual(config["UNRELATED"], "preserved")

    def test_llm_level_configures_deepseek_and_blacklist_policy(self):
        config: dict[str, str] = {}
        with patch("builtins.input", side_effect=["3", "1", "1", "private,no-cloud"]), patch(
            "getpass.getpass", side_effect=["s2-secret", "deepseek-secret"],
        ), patch("sys.stdout", StringIO()):
            setup_wizard.configure_feature_level(config)
        self.assertEqual(config["FEATURE_LEVEL"], "llm")
        self.assertEqual(config["S2_API_KEY"], "s2-secret")
        self.assertEqual(config["LLM_CHEAP"], "deepseek:deepseek-v4-flash")
        self.assertEqual(config["LLM_STANDARD"], "deepseek:deepseek-v4-pro")
        self.assertEqual(config["LLM_REVIEW"], "deepseek:deepseek-v4-pro")
        self.assertEqual(config["DEEPSEEK_API_KEY"], "deepseek-secret")
        self.assertEqual(config["SUMMARY_EXCLUDE_TAGS"], "private,no-cloud")
        self.assertEqual(config["EXTRACT_EXCLUDE_TAGS"], "private,no-cloud")
        self.assertNotIn("SUMMARY_ALLOW_CLOUD_ALL", config)

    def test_citation_level_reprompts_until_s2_key_is_entered(self):
        config: dict[str, str] = {}
        with patch("builtins.input", side_effect=["2"]), patch(
            "getpass.getpass", side_effect=["", "required-key"],
        ), patch("sys.stdout", StringIO()):
            setup_wizard.configure_feature_level(config)
        self.assertEqual(config["FEATURE_LEVEL"], "citation")
        self.assertEqual(config["S2_API_KEY"], "required-key")

    def test_status_never_prints_secret_values(self):
        config = {
            "FEATURE_LEVEL": "llm", "S2_API_KEY": "do-not-print",
            "DEEPSEEK_API_KEY": "also-secret", "LLM_STANDARD": "deepseek:model",
            "EXTRACT_EXCLUDE_TAGS": "private",
        }
        output = StringIO()
        with patch("sys.stdout", output):
            setup_wizard.print_configuration_status(config)
        rendered = output.getvalue()
        self.assertIn("S2 API key     : configured", rendered)
        self.assertIn("cloud policy   : tag blacklist", rendered)
        self.assertNotIn("do-not-print", rendered)
        self.assertNotIn("also-secret", rendered)


if __name__ == "__main__":
    unittest.main()
