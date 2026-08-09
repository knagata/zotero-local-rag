"""An attachment handed to the Mistral OCR batch must still appear in the manifest.

Observed in a `--rebuild` run: it aborted with
``failed=0 deferred=50 missing_manifest_attachments=['BMKYYFCZ','CAXCWCQB','CCHED4LQ']``.
All three were EPUBs that had been correctly queued for OCR (they later
indexed as ``parser=mistral_ocr_epub``). The PDF deferral path wrote a manifest
row and the EPUB path did not, so `--rebuild`'s completeness check listed the
EPUBs as missing while also counting them in ``deferred_extract`` -- the same
queued file reported twice, as though it had been lost.
"""
from __future__ import annotations

import ast
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import index_from_zotero  # noqa: E402


class DeferredManifestEntryTests(unittest.TestCase):
    def _entry(self, previous=None, **over):
        kwargs = {
            "mtime": 1700.0, "size": 4096, "source_path": Path("/lib/book.epub"),
            "title": "Some Book", "quality": {},
        }
        kwargs.update(over)
        return index_from_zotero._deferred_manifest_entry(previous, **kwargs)

    def test_entry_carries_the_source_identity_rebuild_checks_for(self):
        entry = self._entry()
        self.assertEqual(entry["mtime"], 1700.0)
        self.assertEqual(entry["size"], 4096)
        self.assertEqual(entry["pdf_path"], "/lib/book.epub")
        self.assertEqual(entry["title"], "Some Book")

    def test_previous_fields_are_preserved(self):
        # A deferral leaves the existing chunks searchable, so the row it writes
        # must not discard what the last successful ingest recorded.
        entry = self._entry({"content_signature": "sha:old", "extra": 1})
        self.assertEqual(entry["extra"], 1)
        self.assertEqual(entry["content_signature"], "sha:old")

    def test_content_signature_is_written_only_when_present(self):
        self.assertNotIn("content_signature", self._entry())
        self.assertEqual(
            self._entry(content_signature_value="sha:new")["content_signature"], "sha:new",
        )

    def test_pipeline_fingerprint_is_omitted_when_v3_is_off(self):
        self.assertNotIn("pipeline_fingerprint", self._entry(pipeline_fingerprint=None))
        self.assertEqual(
            self._entry(pipeline_fingerprint="fp-1")["pipeline_fingerprint"], "fp-1",
        )

    def test_a_non_dict_previous_entry_is_tolerated(self):
        self.assertEqual(self._entry("corrupt-row")["size"], 4096)


class CompletedEntryTests(unittest.TestCase):
    """The row a finished ingest writes, from the same single definition."""

    def _entry(self, **over):
        kwargs = {
            "mtime": 1700.0, "size": 4096, "source_path": Path("/lib/book.pdf"),
            "title": "Some Book", "quality": {"parser": "pymupdf"},
        }
        kwargs.update(over)
        return index_from_zotero._manifest_entry(**kwargs)

    def test_completed_entry_carries_the_quality_as_measured(self):
        # Unlike a deferral, a completed ingest replaces the row outright, so
        # the fresh quality stands on its own rather than being merged.
        self.assertEqual(self._entry()["quality"], {"parser": "pymupdf"})

    def test_completed_entry_does_not_carry_a_previous_row_forward(self):
        # _manifest_entry takes no previous row at all: a finished extraction
        # supersedes whatever was there.
        self.assertEqual(
            set(self._entry()),
            {"mtime", "size", "pdf_path", "title", "quality"},
        )

    def test_optional_fields_follow_the_same_rules_on_both_paths(self):
        self.assertNotIn("content_signature", self._entry(content_signature_value=""))
        self.assertNotIn("pipeline_fingerprint", self._entry(pipeline_fingerprint=None))
        entry = self._entry(content_signature_value="sha:new", pipeline_fingerprint="fp-1")
        self.assertEqual(entry["content_signature"], "sha:new")
        self.assertEqual(entry["pipeline_fingerprint"], "fp-1")


class EveryManifestRowGoesThroughOneBuilderTests(unittest.TestCase):
    """Four sites produce a row; they drifted once, so pin that they cannot."""

    def test_no_site_spells_the_row_out_inline(self):
        source = (SRC / "index_from_zotero.py").read_text()
        tree = ast.parse(source)
        builders = {"_manifest_entry", "_deferred_manifest_entry"}
        for node in ast.walk(tree):
            if not isinstance(node, ast.Assign):
                continue
            for target in node.targets:
                if not (isinstance(target, ast.Subscript)
                        and isinstance(target.value, ast.Name)
                        and target.value.id in {"files_manifest", "pending_manifest_updates"}):
                    continue
                self.assertIsInstance(
                    node.value, ast.Call,
                    f"line {node.lineno}: manifest row built inline instead of via a builder",
                )
                self.assertIn(node.value.func.id, builders, f"line {node.lineno}")


class EveryDeferralPathWritesToTheManifestTests(unittest.TestCase):
    """The PDF and EPUB paths drifted apart once; pin that they cannot again."""

    @staticmethod
    def _decides_to_defer(statement):
        """Whether this statement is a path choosing to hand a file to the batch.

        Two shapes, because the PDF route now lives in its own function. It
        reports the decision by returning ``PdfExtraction(deferred=True)`` and
        the caller does the counting; the EPUB route still increments in place.
        The counter alone is no longer the marker: the increment at the call
        site is a relay of a decision made elsewhere, and demanding a manifest
        write beside it would be asking the wrong statement for it.
        """
        if (isinstance(statement, ast.AugAssign)
                and isinstance(statement.target, ast.Name)
                and statement.target.id == "deferred_extract"):
            return True
        return (
            isinstance(statement, ast.Return)
            and isinstance(statement.value, ast.Call)
            and isinstance(statement.value.func, ast.Name)
            and statement.value.func.id == "PdfExtraction"
            and any(kw.arg == "deferred" and getattr(kw.value, "value", False) is True
                    for kw in statement.value.keywords)
        )

    def _deferral_sites(self):
        """(block, guard) for every block where a file is sent to the batch.

        ``guard`` is the condition the block sits under, when it sits under one.
        It is what tells a decision apart from the call site that merely counts
        one: the latter is guarded by the returned flag.
        """
        tree = ast.parse((SRC / "index_from_zotero.py").read_text())
        sites = []
        for node in ast.walk(tree):
            for attribute in ("body", "orelse", "finalbody"):
                body = getattr(node, attribute, None)
                if not isinstance(body, list):
                    continue
                if not any(self._decides_to_defer(s) for s in body):
                    continue
                guard = (
                    ast.unparse(node.test)
                    if isinstance(node, ast.If) and attribute == "body" else ""
                )
                sites.append((body, guard))
        return sites

    def test_both_deferral_sites_were_found(self):
        # The PDF route's decision, the EPUB route's, and the call site that
        # relays the first of them into the counter.
        self.assertEqual(len(self._deferral_sites()), 3)

    def test_exactly_one_site_is_a_relay(self):
        relays = [guard for _body, guard in self._deferral_sites()
                  if ".deferred" in guard]
        self.assertEqual(relays, ["extraction.deferred"])

    def test_each_deferral_records_a_manifest_row(self):
        checked = 0
        for block, guard in self._deferral_sites():
            if ".deferred" in guard:
                continue    # the relay; the decision it counts is a site of its own
            names = {
                child.func.id
                for statement in block
                for child in ast.walk(statement)
                if isinstance(child, ast.Call) and isinstance(child.func, ast.Name)
            }
            self.assertIn("_deferred_manifest_entry", names)
            self.assertIn("save_manifest", names)
            checked += 1
        # Both deciding paths reached, not one of them plus the relay twice.
        self.assertEqual(checked, 2)


if __name__ == "__main__":
    unittest.main()
