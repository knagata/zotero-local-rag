from __future__ import annotations

import stat
import sys
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
            self.assertIn("# 基本設定: ローカル検索と索引", rendered)
            self.assertIn("# 引用ネットワークと書誌情報", rendered)
            self.assertIn("# その他の既存設定", rendered)
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

    def test_setup_menu_offers_minimal_and_custom_only(self):
        output = StringIO()
        with patch("builtins.input", return_value="1"), patch("sys.stdout", output):
            setup_wizard.configure_feature_level({})
        rendered = output.getvalue()
        self.assertIn("[1] Minimal", rendered)
        self.assertIn("[2] Custom", rendered)
        self.assertNotIn("Local —", rendered)
        self.assertNotIn("Full —", rendered)

    def test_custom_allows_granite_for_both_page_buckets(self):
        config: dict[str, str] = {}
        answers = ["2", "n", "y", "", "2", "2", "", "", "", "", ""]
        with patch.object(
            setup_wizard, "_granite_selectable", return_value=True,
        ), patch("builtins.input", side_effect=answers), patch(
            "sys.stdout", StringIO(),
        ):
            setup_wizard.configure_feature_level(config)
        self.assertEqual(config["FEATURE_LEVEL"], "custom")
        self.assertEqual(config["PDF_STRUCTURE_RECOVERY_ENABLE"], "1")
        self.assertEqual(config["PDF_STRUCTURE_ENGINE_SHORT"], "granite")
        self.assertEqual(config["PDF_STRUCTURE_ENGINE_LONG"], "granite")
        self.assertEqual(config["PDF_MISTRAL_TOC_QUEUE_ENABLE"], "0")
        for flag in setup_wizard.LLM_FLAGS:
            self.assertEqual(config[flag], "0")

    def test_custom_can_install_granite_when_it_is_not_ready(self):
        config: dict[str, str] = {}
        answers = ["2", "n", "y", "", "2", "2", "", "", "", "", "", ""]
        with patch.object(
            setup_wizard, "_granite_selectable", return_value=False,
        ), patch.object(
            setup_wizard, "install_granite_environment", return_value=True,
        ) as install, patch(
            "builtins.input", side_effect=answers,
        ), patch("sys.stdout", StringIO()):
            setup_wizard.configure_feature_level(config)
        install.assert_called_once_with(config)
        self.assertEqual(config["PDF_STRUCTURE_ENGINE_SHORT"], "granite")
        self.assertEqual(config["PDF_STRUCTURE_ENGINE_LONG"], "granite")

    def test_custom_citation_network_prompts_for_s2_key(self):
        config: dict[str, str] = {}
        with patch(
            "builtins.input", side_effect=["2", "y", "n", "", "", ""],
        ), patch.object(
            setup_wizard.getpass, "getpass", return_value="s2-secret",
        ), patch("sys.stdout", StringIO()):
            setup_wizard.configure_feature_level(config)
        self.assertEqual(config["CITATION_NETWORK_ENABLE"], "1")
        self.assertEqual(config["S2_API_KEY"], "s2-secret")

    def test_custom_can_enable_only_hierarchical_summaries(self):
        config: dict[str, str] = {}
        with patch(
            "builtins.input",
            side_effect=["2", "n", "n", "n", "y", "n", "2"],
        ), patch("sys.stdout", StringIO()):
            setup_wizard.configure_feature_level(config)
        self.assertEqual(config["LLM_SUMMARIES_ENABLE"], "1")
        self.assertEqual(config["LLM_CHEAP"], "codex_cli:auto")
        self.assertEqual(config["PDF_AI_TOC_FAST_PATH_ENABLE"], "0")
        self.assertEqual(config["OCR_LAYER_AUDIT_ENABLE"], "0")
        self.assertEqual(config["QUERY_EXPANSION_ENABLE"], "0")
        self.assertEqual(config["LLM_REFERENCE_EXTRACTION_ENABLE"], "0")

    def test_custom_mistral_engine_requires_its_api_key(self):
        config: dict[str, str] = {}
        answers = ["2", "n", "y", "", "1", "3", "", "", "", "", ""]
        with patch.object(
            setup_wizard, "_granite_selectable", return_value=False,
        ), patch("builtins.input", side_effect=answers), patch.object(
            setup_wizard.getpass, "getpass", return_value="mistral-secret",
        ), patch("sys.stdout", StringIO()):
            setup_wizard.configure_feature_level(config)
        self.assertEqual(config["PDF_STRUCTURE_ENGINE_SHORT"], "docling")
        self.assertEqual(config["PDF_STRUCTURE_ENGINE_LONG"], "mistral")
        self.assertEqual(config["PDF_MISTRAL_TOC_QUEUE_ENABLE"], "1")
        self.assertEqual(config["MISTRAL_OCR_API_KEY"], "mistral-secret")

    def test_api_key_prompt_explains_that_input_is_hidden(self):
        config: dict[str, str] = {}
        output = StringIO()
        with patch.object(setup_wizard.getpass, "getpass", return_value="secret"), patch.object(
            sys, "stdout", output,
        ):
            setup_wizard._set_required_secret(config, "S2_API_KEY", "S2 key")
        self.assertIn("APIキーは画面に表示されません", output.getvalue())
        self.assertNotIn("secret", output.getvalue())

    def test_granite_menu_numbers_match_the_rendered_order(self):
        with patch.object(setup_wizard, "_granite_selectable", return_value=True), patch(
            "builtins.input", return_value="2",
        ), patch("sys.stdout", StringIO()):
            self.assertEqual(setup_wizard._choose_engine("Long", "docling"), "granite")
        with patch.object(setup_wizard, "_granite_selectable", return_value=True), patch(
            "builtins.input", return_value="3",
        ), patch("sys.stdout", StringIO()):
            self.assertEqual(setup_wizard._choose_engine("Long", "docling"), "mistral")

    def test_granite_remains_selectable_before_its_environment_is_installed(self):
        output = StringIO()
        with patch.object(setup_wizard, "_granite_selectable", return_value=False), patch(
            "builtins.input", return_value="2",
        ), patch("sys.stdout", output):
            selected = setup_wizard._choose_engine("長いPDF", "docling")
        self.assertEqual(selected, "granite")
        self.assertIn("初回選択時に専用環境を導入", output.getvalue())

    def test_granite_installer_creates_and_verifies_an_isolated_environment(self):
        config: dict[str, str] = {}
        with tempfile.TemporaryDirectory() as directory:
            interpreter = Path(directory) / "granite" / "bin" / "python"
            completed = SimpleNamespace(returncode=0)
            with patch.object(
                setup_wizard.platform, "system", return_value="Darwin",
            ), patch.object(
                setup_wizard.platform, "machine", return_value="arm64",
            ), patch.object(
                setup_wizard.shutil, "which", return_value="/usr/local/bin/uv",
            ), patch.object(
                setup_wizard.subprocess, "run", return_value=completed,
            ) as run, patch.object(
                setup_wizard, "_granite_environment_ready", return_value=True,
            ), patch("sys.stdout", StringIO()):
                installed = setup_wizard.install_granite_environment(
                    config, python_path=interpreter,
                )
        self.assertTrue(installed)
        self.assertEqual(run.call_count, 3)
        self.assertIn("--clear", run.call_args_list[0].args[0])
        self.assertEqual(config["GRANITE_VENV_PYTHON"], str(interpreter.absolute()))

    def test_ndlocr_detection_records_the_absolute_executable(self):
        config: dict[str, str] = {}
        with tempfile.TemporaryDirectory() as directory:
            executable = Path(directory) / "ndlocr-lite"
            executable.touch()
            with patch.object(
                setup_wizard.shutil, "which", return_value=str(executable),
            ), patch.object(
                setup_wizard, "_ndlocr_executable_ready", return_value=True,
            ), patch("sys.stdout", StringIO()):
                setup_wizard.configure_ndlocr(config)
        self.assertEqual(config["NDLOCR_BIN"], str(executable.absolute()))

    def test_ndlocr_missing_from_custom_setup_can_be_installed(self):
        config: dict[str, str] = {}
        with patch.object(
            setup_wizard, "_find_ndlocr", return_value=None,
        ), patch.object(
            setup_wizard, "install_ndlocr", return_value=True,
        ) as install, patch(
            "builtins.input", return_value="",
        ), patch("sys.stdout", StringIO()):
            setup_wizard.configure_ndlocr(config)
        install.assert_called_once_with(config)

    def test_ndlocr_installer_uses_an_isolated_uv_tool(self):
        config: dict[str, str] = {}
        with tempfile.TemporaryDirectory() as directory:
            bin_directory = Path(directory)
            installed = SimpleNamespace(returncode=0, stdout="")
            bin_result = SimpleNamespace(returncode=0, stdout=str(bin_directory))
            with patch.object(
                setup_wizard.shutil, "which",
                side_effect=lambda name: "/usr/local/bin/uv" if name == "uv" else None,
            ), patch.object(
                setup_wizard.subprocess, "run",
                side_effect=[installed, bin_result],
            ) as run, patch.object(
                setup_wizard, "_ndlocr_executable_ready", return_value=True,
            ), patch("sys.stdout", StringIO()):
                result = setup_wizard.install_ndlocr(config)
        self.assertTrue(result)
        self.assertEqual(run.call_count, 2)
        self.assertEqual(
            run.call_args_list[0].args[0],
            [
                "/usr/local/bin/uv", "tool", "install", "--force",
                "git+https://github.com/ndl-lab/ndlocr-lite.git@1.0.0",
            ],
        )
        self.assertEqual(
            config["NDLOCR_BIN"],
            str((bin_directory / "ndlocr-lite").absolute()),
        )

    def test_custom_setup_can_install_tesseract_and_japanese_data(self):
        with patch.object(
            setup_wizard, "_find_tesseract", return_value=None,
        ), patch.object(
            setup_wizard, "install_tesseract", return_value=True,
        ) as install, patch(
            "builtins.input", return_value="",
        ), patch("sys.stdout", StringIO()):
            setup_wizard.configure_tesseract(allow_install=True)
        install.assert_called_once_with()

    def test_minimal_setup_does_not_offer_to_install_tesseract(self):
        with patch.object(
            setup_wizard, "_find_tesseract", return_value=None,
        ), patch("builtins.input") as prompt, patch("sys.stdout", StringIO()):
            setup_wizard.configure_tesseract(allow_install=False)
        prompt.assert_not_called()

    def test_tesseract_installer_uses_homebrew_and_verifies_japanese(self):
        executable = Path("/opt/homebrew/bin/tesseract")
        completed = SimpleNamespace(returncode=0)
        with patch.object(
            setup_wizard, "_find_homebrew",
            return_value=Path("/opt/homebrew/bin/brew"),
        ), patch.object(
            setup_wizard.subprocess, "run", return_value=completed,
        ) as run, patch.object(
            setup_wizard, "_find_tesseract", return_value=executable,
        ), patch.object(
            setup_wizard, "_tesseract_languages", return_value={"eng", "jpn"},
        ), patch("sys.stdout", StringIO()):
            installed = setup_wizard.install_tesseract()
        self.assertTrue(installed)
        self.assertEqual(
            run.call_args.args[0],
            [
                "/opt/homebrew/bin/brew", "install",
                "tesseract", "tesseract-lang",
            ],
        )

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
                **setup_wizard.PRESETS["minimal"],
                "FEATURE_LEVEL": "custom",
                "PDF_STRUCTURE_RECOVERY_ENABLE": "1",
                "PDF_AI_TOC_FAST_PATH_ENABLE": "1",
                "OCR_LAYER_AUDIT_ENABLE": "1",
                "QUERY_EXPANSION_ENABLE": "1",
                "LLM_SUMMARIES_ENABLE": "1",
                "LLM_REFERENCE_EXTRACTION_ENABLE": "1",
                "CITATION_NETWORK_ENABLE": "1",
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
            ), patch.object(
                setup_wizard, "configure_ndlocr",
            ), patch.object(
                setup_wizard, "configure_tesseract",
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
        config = dict(setup_wizard.PRESETS["minimal"])
        config["FEATURE_LEVEL"] = "custom"
        config["LLM_SUMMARIES_ENABLE"] = "1"
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
            "FEATURE_LEVEL": "custom", "S2_API_KEY": "do-not-print",
            "DEEPSEEK_API_KEY": "also-secret", "LLM_STANDARD": "deepseek:model",
        }
        output = StringIO()
        with patch("sys.stdout", output):
            setup_wizard.print_configuration_status(config)
        rendered = output.getvalue()
        self.assertIn("S2 APIキー     : 設定済み", rendered)
        self.assertIn("設定種別       :", rendered)
        self.assertNotIn("do-not-print", rendered)
        self.assertNotIn("also-secret", rendered)


if __name__ == "__main__":
    unittest.main()
