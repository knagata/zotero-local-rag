from __future__ import annotations

import os
import unittest
from pathlib import Path
from unittest.mock import patch

from scripts.show_effective_config import (
    NOT_OUR_CONFIG, _is_secret, build_report, discover_settings, format_text,
)

ROOT = Path(__file__).resolve().parents[1]


class SecretRedactionTests(unittest.TestCase):
    """The report is meant to be pasted into an issue or a chat.

    Its whole purpose is to be compared between two machines, so it will be
    copied around; a value printed here is a value leaked.
    """

    def test_credential_names_are_recognised(self):
        for name in (
            "S2_API_KEY", "ZOTERO_API_KEY", "AZURE_TRANSLATOR_KEY",
            "GITHUB_TOKEN", "DB_PASSWORD", "CLIENT_SECRET",
        ):
            self.assertTrue(_is_secret(name), name)

    def test_ordinary_settings_are_not_treated_as_secret(self):
        for name in ("CHROMA_DIR", "MANIFEST_PATH", "PDF_OCR_LANG", "EMB_PROFILE"):
            self.assertFalse(_is_secret(name), name)

    def test_a_secret_value_never_appears_in_any_output(self):
        secret = "sk-do-not-print-me-0123456789"
        with patch.dict(os.environ, {"S2_API_KEY": secret, "CHROMA_DIR": "/tmp/x"}):
            report = build_report()
            text = format_text(report, verbose=True)
        self.assertNotIn(secret, text)
        self.assertNotIn(secret, str(report))

        row = next(r for r in report["settings"] if r["name"] == "S2_API_KEY")
        self.assertEqual(row["state"], "set")
        # Length is still reported: "is it set, and does it look like the same
        # key?" is exactly the question being asked when diffing two machines.
        self.assertIn(str(len(secret)), row["value"])

    def test_non_secret_values_are_shown_verbatim(self):
        with patch.dict(os.environ, {"CHROMA_DIR": "/srv/chroma"}):
            report = build_report()
        row = next(r for r in report["settings"] if r["name"] == "CHROMA_DIR")
        self.assertEqual(row["value"], "/srv/chroma")


class DiscoveryTests(unittest.TestCase):
    """The setting list is derived from source, so it cannot go stale."""

    def setUp(self):
        self.settings = discover_settings(ROOT)

    def test_finds_the_setting_behind_the_incident_this_script_exists_for(self):
        # PDF_SCANNED_PAGE_PATCH_ENABLE differed between two machines and
        # silently cost a week of page content. If the report cannot show it,
        # the report is useless for the case it was written for.
        self.assertIn("PDF_SCANNED_PAGE_PATCH_ENABLE", self.settings)

    def test_finds_the_canonical_paths_and_the_v3_switch(self):
        for name in (
            "CHROMA_DIR", "MANIFEST_PATH", "LEXICAL_DB_PATH", "RELATIONS_DB_PATH",
            "INGEST_STRUCTURED_V3_ENABLE",
        ):
            self.assertIn(name, self.settings, name)

    def test_assignments_to_third_party_variables_are_not_reported_as_settings(self):
        # src/rag_mcp_server.py does os.environ["OMP_NUM_THREADS"] = "1" to
        # configure a library. That is this project writing, not reading, and
        # listing it would put a value nobody can change into the diff.
        for name in ("OMP_NUM_THREADS", "TOKENIZERS_PARALLELISM", "MKL_NUM_THREADS"):
            self.assertNotIn(name, self.settings, name)

    def test_every_reported_setting_names_the_files_that_read_it(self):
        for name, files in self.settings.items():
            self.assertTrue(files, name)
            for path in files:
                self.assertTrue((ROOT / path).is_file(), f"{name}: {path}")

    def test_os_supplied_variables_are_excluded(self):
        self.assertEqual(set(self.settings) & NOT_OUR_CONFIG, set())

    def test_discovery_is_substantial(self):
        # A visitor silently matching nothing would make every assertion above
        # vacuous while the report printed an empty, reassuring list.
        self.assertGreater(len(self.settings), 80)


class ReportShapeTests(unittest.TestCase):
    def test_unset_settings_are_reported_as_unset_not_omitted(self):
        # "Set on that machine, absent on this one" is the drift being hunted;
        # dropping the absent side would hide exactly that.
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("CINII_APP_ID", None)
            report = build_report()
        row = next(r for r in report["settings"] if r["name"] == "CINII_APP_ID")
        self.assertEqual(row["state"], "unset")
        self.assertIsNone(row["value"])
        self.assertIn("(unset)", format_text(report))

    def test_settings_are_sorted_so_two_machines_diff_cleanly(self):
        names = [row["name"] for row in build_report()["settings"]]
        self.assertEqual(names, sorted(names))


if __name__ == "__main__":
    unittest.main()
