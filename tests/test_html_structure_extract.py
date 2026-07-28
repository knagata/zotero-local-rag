from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from src.html_extract import (
    _extract_epub_document_blocks,
    _resolve_note_citations,
    extract_chunks_from_html_snapshot,
    extract_dom_blocks,
)


class HtmlStructureExtractionTests(unittest.TestCase):
    def test_epub_document_falls_back_when_semantic_tags_only_hold_a_caption(self):
        # Some EPUB articles use leaf divs for every body paragraph while a
        # figure caption makes the semantic extractor non-empty.  The fallback
        # must be selected per spine document, not only when the whole book is
        # empty, otherwise the full article is silently lost.
        prose = " ".join(["full body prose"] * 120)
        blocks = _extract_epub_document_blocks(
            f"<body><figcaption>Only semantic caption.</figcaption>"
            f"<div><div>{prose}</div></div></body>"
        )
        self.assertEqual(len(blocks), 1)
        self.assertEqual(blocks[0]["block_type"], "div")
        self.assertGreater(len(blocks[0]["text"]), 1_000)

    def test_epub_document_retains_substantive_semantic_blocks(self):
        prose = " ".join(["semantic paragraph"] * 160)
        blocks = _extract_epub_document_blocks(f"<body><p>{prose}</p></body>")
        self.assertEqual(len(blocks), 1)
        self.assertEqual(blocks[0]["block_type"], "p")

    def test_malformed_long_heading_is_retained_as_body(self):
        prose = " ".join(["publisher-mislabelled prose"] * 30)
        blocks = extract_dom_blocks(f"<body><h2>Chapter</h2><h3>{prose}</h3></body>")
        self.assertEqual(len(blocks), 1)
        self.assertEqual(blocks[0]["block_type"], "malformed_heading_body")
        self.assertEqual(blocks[0]["structure_path"], ["Chapter"])
        self.assertIn("publisher-mislabelled prose", blocks[0]["text"])

    def test_preserves_heading_path_table_and_semantic_zones(self):
        blocks = extract_dom_blocks('''
            <html xmlns:epub="http://www.idpf.org/2007/ops"><body>
              <h1>Part I</h1><h2>Chapter One</h2>
              <p>Body text that belongs to the first chapter.</p>
              <table><tr><td>A</td><td>B</td></tr></table>
              <aside epub:type="footnote">A source note.</aside>
              <section epub:type="bibliography"><h2>References</h2>
                <ol><li>Author (2020). Work.</li></ol>
              </section>
            </body></html>
        ''')
        self.assertEqual(blocks[0]["structure_path"], ["Part I", "Chapter One"])
        self.assertEqual(blocks[1]["block_type"], "table")
        self.assertIn("A B", blocks[1]["text"])
        self.assertEqual(blocks[2]["zone"], "footnote")
        self.assertEqual(blocks[3]["zone"], "bibliography")

    def test_uses_initial_toc_path_when_file_has_no_heading(self):
        blocks = extract_dom_blocks("<body><p>Content.</p></body>", initial_path=["Part", "Chapter"])
        self.assertEqual(blocks[0]["structure_path"], ["Part", "Chapter"])

    def test_br_separated_references_split_into_one_block_per_entry(self):
        # A bibliography packed into one <p> with <br/> separators must yield one
        # block per reference so the boundary survives chunking (Part A, note 77).
        blocks = extract_dom_blocks('''
            <html xmlns:epub="http://www.idpf.org/2007/ops"><body>
              <section epub:type="bibliography"><h2>References</h2>
                <p>Barrat, J. (2013). <em>Our Final Invention</em>, Press.<br/>
                   Bostrom, N. (2014). Superintelligence. Oxford University Press.<br/>
                   Dennett, D. C. (2017). From Bacteria to Bach and Back. Norton.</p>
              </section>
            </body></html>
        ''')
        bib = [block for block in blocks if block["zone"] == "bibliography"]
        self.assertEqual(len(bib), 3)
        self.assertTrue(bib[0]["text"].startswith("Barrat, J. (2013)."))
        # Inline <em> must not become its own boundary.
        self.assertIn("Our Final Invention", bib[0]["text"])
        self.assertTrue(bib[1]["text"].startswith("Bostrom, N. (2014)."))

    def test_body_paragraph_with_inline_markup_is_not_over_split(self):
        blocks = extract_dom_blocks(
            "<body><h1>Intro</h1><p>This is <em>body</em> prose with an "
            "<a href='x'>inline link</a> and more.</p></body>"
        )
        body = [block for block in blocks if block["zone"] == "body"]
        self.assertEqual(len(body), 1)
        self.assertEqual(body[0]["text"], "This is body prose with an inline link and more.")

    def test_noteref_and_note_body_ids_are_recorded_on_blocks(self):
        # Part C (dev-notes/current/77): a body <p> noteref records the target
        # fragment id; the note body block records the matching element id.
        blocks = extract_dom_blocks('''
            <html xmlns:epub="http://www.idpf.org/2007/ops"><body>
              <p>Body sentence with a marker<a epub:type="noteref"
                 href="chapter.xhtml#note1">1</a>.</p>
              <aside epub:type="footnote" id="note1">A source note body text.</aside>
            </body></html>
        ''')
        body = next(b for b in blocks if b["zone"] == "body")
        note = next(b for b in blocks if b["zone"] == "footnote")
        self.assertEqual(body["noteref_targets"], ["note1"])
        self.assertIn("note1", note["element_ids"])
        # The note body carries no outgoing noteref.
        self.assertEqual(note["noteref_targets"], [])

    def test_role_doc_noteref_is_detected_and_plain_links_ignored(self):
        blocks = extract_dom_blocks('''
            <body>
              <p>Cite<a role="doc-noteref" href="#n2">2</a> and a
                 <a href="http://example.com#frag">plain link</a>.</p>
            </body>
        ''')
        body = next(b for b in blocks if b["zone"] == "body")
        self.assertEqual(body["noteref_targets"], ["n2"])

    def test_classifies_japanese_paratext_headings(self):
        blocks = extract_dom_blocks('''<body>
          <h1>目 次</h1><p>第一章　1</p>
          <h1>巻末注</h1><p>注記本文</p>
          <h1>参考文献</h1><p>著者、書名</p>
          <h1>人名索引</h1><p>人物名</p>
          <h1>奥付</h1><p>発行情報</p>
          <h1>あとがき</h1><p>後書き本文</p>
        </body>''')
        self.assertEqual(
            [block["zone"] for block in blocks],
            ["toc", "endnote", "bibliography", "index", "colophon", "back_matter"],
        )


class NoteCitationResolutionTests(unittest.TestCase):
    def _chunks_for(self, html: str):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "snapshot.html"
            path.write_text(html, encoding="utf-8")
            chunks, _quality = extract_chunks_from_html_snapshot(path, "ITEMKEY", {})
        return chunks

    _BODY = (
        "This is a body paragraph long enough to survive chunking and it carries "
        "an inline note marker"
    )
    _NOTE = (
        "This is the footnote body text and it is long enough to survive chunk "
        "merging without being dropped as too short."
    )

    def test_footnote_chunk_gets_citing_chunk_id(self):
        html = (
            '<html xmlns:epub="http://www.idpf.org/2007/ops"><body>'
            f'<p>{self._BODY}<a epub:type="noteref" href="#note1">1</a>.</p>'
            f'<aside epub:type="footnote" id="note1">{self._NOTE}</aside>'
            "</body></html>"
        )
        chunks = self._chunks_for(html)
        body = next((c for c in chunks if c[2].get("zone") == "body"), None)
        note = next((c for c in chunks if c[2].get("zone") == "footnote"), None)
        self.assertIsNotNone(body)
        self.assertIsNotNone(note)
        self.assertEqual(note[2].get("citing_chunk_id"), body[0])
        # Body chunk itself is not a note target.
        self.assertNotIn("citing_chunk_id", body[2])
        # Intermediate list evidence must never reach the DB metadata.
        for _cid, _text, md in chunks:
            self.assertNotIn("_element_ids", md)
            self.assertNotIn("_noteref_targets", md)
            self.assertNotIsInstance(md.get("citing_chunk_id", ""), (list, set))
        from chromadb.api.types import validate_metadata
        for _cid, _text, md in chunks:
            validate_metadata(md)

    def test_no_noteref_leaves_no_citing_chunk_id(self):
        html = (
            '<html xmlns:epub="http://www.idpf.org/2007/ops"><body>'
            f'<p>{self._BODY}.</p>'
            f'<aside epub:type="footnote" id="note1">{self._NOTE}</aside>'
            "</body></html>"
        )
        chunks = self._chunks_for(html)
        self.assertTrue(chunks)
        for _cid, _text, md in chunks:
            self.assertNotIn("citing_chunk_id", md)

    def test_cross_file_endnote_uses_href_document_component(self):
        # EPUB endnotes commonly live in a dedicated spine document.  Resolve
        # relative to the citing document, while retaining same-file behavior.
        chunks = _resolve_note_citations([
            ("body", "Body", {
                "chapter_index": 0,
                "_spine_path": "Text/chapter-01.xhtml",
                "_noteref_hrefs": ["../Notes/endnotes.xhtml#n1"],
            }),
            ("note", "Endnote", {
                "chapter_index": 1,
                "_spine_path": "Notes/endnotes.xhtml",
                "_element_ids": ["n1"],
            }),
        ])
        note = next(row for row in chunks if row[0] == "note")
        self.assertEqual(note[2].get("citing_chunk_id"), "body")
        for _cid, _text, md in chunks:
            self.assertNotIn("_spine_path", md)
            self.assertNotIn("_noteref_hrefs", md)


if __name__ == "__main__":
    unittest.main()


def test_paratext_headings_the_vocabulary_used_to_miss():
    """Vocabulary gaps left real paratext filed as body.

    "Sources" appeared in none of the patterns, so a 72,150-character source
    list stayed in the body zone (GI85JWZH). The per-chapter form matters as
    much as the bare word: matching only "Notes" misses every "Notes to
    Chapter 3" in a book that numbers its note sections (2026-07-28).
    """
    from src.html_extract import _BIBLIOGRAPHY_RE, _BACK_MATTER_RE, _INDEX_RE, _NOTE_RE

    assert _BIBLIOGRAPHY_RE.match("Sources")
    assert _BIBLIOGRAPHY_RE.match("Select Bibliography")
    assert _BIBLIOGRAPHY_RE.match("Further Reading")
    assert _BIBLIOGRAPHY_RE.match("参考資料")
    assert _NOTE_RE.match("Notes to Chapter 3")
    assert _NOTE_RE.match("Notes")
    assert _INDEX_RE.match("General Index")
    assert _INDEX_RE.match("地名索引")
    assert _BACK_MATTER_RE.match("Glossary")


def test_a_body_heading_beginning_with_a_paratext_word_stays_body():
    """The looser patterns must not swallow ordinary chapter titles.

    "Sources of Power" is a chapter, not a source list, and the trailing-form
    allowance is restricted to "to"/"for" so it cannot absorb arbitrary titles.
    """
    from src.html_extract import _BIBLIOGRAPHY_RE, _NOTE_RE

    assert not _BIBLIOGRAPHY_RE.match("Sources of Power")
    assert not _BIBLIOGRAPHY_RE.match("References and Realities")
    assert not _NOTE_RE.match("Notes on the State of Virginia")


def test_zone_for_element_checks_the_whole_ancestor_path_not_just_the_leaf():
    """"Notes" containing "Chapter 3" must classify as endnote.

    Checking only the immediate heading (heading_path[-1]) missed this: "Chapter
    3" carries no paratext vocabulary of its own, but everything nested under
    "Notes" is a note regardless. 2,715 chunks were affected (2026-07-28).
    """
    from bs4 import BeautifulSoup
    from src.html_extract import _zone_for_element

    soup = BeautifulSoup("<html><body><div>text</div></body></html>", "html.parser")
    tag = soup.find("div")

    assert _zone_for_element(tag, ["Notes", "Chapter 3"]) == "endnote"
    assert _zone_for_element(tag, ["Part One", "Chapter 3"]) == "body"
    assert _zone_for_element(tag, ["Sources"]) == "bibliography"
