from __future__ import annotations

import tempfile
import unittest
import zipfile
from pathlib import Path

from src.epub_reference_extractor import extract_epub_reference_candidates


class EpubReferenceExtractorTests(unittest.TestCase):
    def _epub(self, files: dict[str, str]) -> Path:
        path = Path(self.tempdir.name) / "sample.epub"
        with zipfile.ZipFile(path, "w") as archive:
            for name, content in files.items():
                archive.writestr(name, content)
        return path

    def setUp(self):
        self.tempdir = tempfile.TemporaryDirectory()

    def tearDown(self):
        self.tempdir.cleanup()

    def test_explicit_noteref_extracts_non_numeric_footnote(self):
        path = self._epub({
            "text.xhtml": '''<html xmlns:epub="http://www.idpf.org/2007/ops"><body>
                <p>Claim<a epub:type="noteref" href="notes.xhtml#fn1">*</a></p>
            </body></html>''',
            "notes.xhtml": '''<html xmlns:epub="http://www.idpf.org/2007/ops"><body>
                <aside id="fn1" epub:type="footnote">Alice Smith (2020). A Work.</aside>
            </body></html>''',
        })
        rows = extract_epub_reference_candidates(str(path))
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["source_zone"], "footnote")
        self.assertIn("Alice Smith", rows[0]["raw_reference_text"])

    def test_generic_numeric_link_to_body_is_not_a_reference(self):
        path = self._epub({
            "text.xhtml": '''<html><body>
                <p>See section <a href="#section1">1</a>.</p>
                <p id="section1">This is an ordinary body paragraph with enough text.</p>
            </body></html>''',
        })
        self.assertEqual(extract_epub_reference_candidates(str(path)), [])

    def test_bibliography_entries_are_extracted_without_inline_links(self):
        path = self._epub({
            "references.xhtml": '''<html><body><section>
                <h2>References</h2><ol>
                  <li>Alice Smith (2020). First Work.</li>
                  <li>Bob Jones (2021). Second Work.</li>
                </ol>
            </section></body></html>''',
        })
        rows = extract_epub_reference_candidates(str(path))
        self.assertEqual(len(rows), 2)
        self.assertEqual({row["source_zone"] for row in rows}, {"bibliography"})

    def test_linked_bibliography_entry_is_deduplicated_and_keeps_context(self):
        path = self._epub({
            "text.xhtml": '''<html><body>
                <p>Body citation <a href="references.xhtml#ref1">[1]</a>.</p>
            </body></html>''',
            "references.xhtml": '''<html><body><section>
                <h2>Bibliography</h2><ol>
                  <li id="ref1">Alice Smith (2020). First Work.</li>
                </ol>
            </section></body></html>''',
        })
        rows = extract_epub_reference_candidates(str(path))
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["source_zone"], "bibliography")
        self.assertIn("Body citation", rows[0]["context_snippet"])

    def test_notes_heading_extracts_one_table_row_not_entire_notes_container(self):
        path = self._epub({
            "chapter.xhtml": '''<html><body>
                <p>First claim<a href="notes.xhtml#note1">1</a>.</p>
                <p>Second claim<a href="notes.xhtml#note2">2</a>.</p>
            </body></html>''',
            "notes.xhtml": '''<html><body><h2>Notes</h2><table>
                <tr id="note1"><td>1</td><td>Alice Smith (2020). First Work.</td></tr>
                <tr id="note2"><td>2</td><td>Bob Jones (2021). Second Work.</td></tr>
            </table></body></html>''',
        })
        rows = extract_epub_reference_candidates(str(path))
        self.assertEqual(len(rows), 2)
        self.assertEqual({row["source_zone"] for row in rows}, {"endnote"})
        self.assertTrue(all("First Work" not in row["raw_reference_text"] or "Second Work" not in row["raw_reference_text"] for row in rows))

    def test_reciprocal_backlink_identifies_unmarked_endnote(self):
        path = self._epub({
            "chapter.xhtml": '''<html><body>
                <p id="claim">Claim<a href="notes.xhtml#n1">1</a>.</p>
            </body></html>''',
            "notes.xhtml": '''<html><body><ol>
                <li id="n1">Alice Smith (2020). First Work. <a href="chapter.xhtml#claim">↩</a></li>
            </ol></body></html>''',
        })
        rows = extract_epub_reference_candidates(str(path))
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["source_zone"], "endnote")

    def test_plain_note_label_identifies_following_note_paragraphs(self):
        path = self._epub({
            "chapter.xhtml": '''<html><body>
                <p>Claim<a href="#n1">1</a> and another<a href="#n2">2</a>.</p>
                <p>Notes</p>
                <p id="n1">1 Alice Smith (2020). First Work.</p>
                <p id="n2">2 Bob Jones (2021). Second Work.</p>
            </body></html>''',
        })
        rows = extract_epub_reference_candidates(str(path))
        self.assertEqual(len(rows), 2)
        self.assertEqual({row["source_zone"] for row in rows}, {"endnote"})

    def test_bibliography_is_ordered_before_notes(self):
        path = self._epub({
            "chapter.xhtml": '''<html xmlns:epub="http://www.idpf.org/2007/ops"><body>
                <p>Claim<a epub:type="noteref" href="#n1">1</a>.</p>
                <aside id="n1" epub:type="footnote">Alice Smith (2020). Footnote Work.</aside>
                <section><h2>References</h2><p>Bob Jones (2021). Listed Work.</p></section>
            </body></html>''',
        })
        rows = extract_epub_reference_candidates(str(path))
        self.assertEqual([row["source_zone"] for row in rows], ["bibliography", "footnote"])


if __name__ == "__main__":
    unittest.main()
