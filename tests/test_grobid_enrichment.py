import os
from unittest import mock

import pytest

from src.grobid_enrichment import parse_tei, should_enrich

TEI = '''<TEI xmlns="http://www.tei-c.org/ns/1.0"><teiHeader><fileDesc><titleStmt>
<title>Example Article</title></titleStmt></fileDesc></teiHeader><text><body><div><head>Introduction</head>
<p>Prior work <ref type="bibr" target="#b1">[1]</ref>.</p></div></body><back><listBibl>
<biblStruct xml:id="b1"><analytic><author><persName><surname>Smith</surname></persName></author>
<title level="a">A Study</title></analytic><monogr><title level="j">Journal</title>
<imprint><date when="2020"/></imprint></monogr><idno type="DOI">10.1/example</idno>
<note type="raw_reference">Smith. A Study. 2020.</note></biblStruct></listBibl></back></text></TEI>'''


def test_parse_tei_preserves_reference_and_inline_link():
    result = parse_tei(TEI)
    assert result.title == "Example Article"
    assert result.headings == ["Introduction"]
    assert result.citation_marker_count == 1
    assert result.linked_citation_count == 1
    assert result.references[0]["title"] == "A Study"
    assert result.references[0]["doi"] == "10.1/example"
    assert result.references[0]["year"] == 2020
    assert result.references[0]["source_kind"] == "grobid_bibliography"


def test_parse_tei_rejects_malformed_xml():
    with pytest.raises(Exception):
        parse_tei("<TEI>")


def test_should_enrich_is_opt_in_english_scholarly_pdf_only():
    with mock.patch.dict(os.environ, {"GROBID_ENRICHMENT_ENABLE": "1"}, clear=False):
        assert should_enrich(item_type="journalArticle", language="en-US", source_type="pdf")
        assert not should_enrich(item_type="book", language="en", source_type="pdf")
        assert not should_enrich(item_type="journalArticle", language="ja", source_type="pdf")
        assert not should_enrich(item_type="journalArticle", language="en", source_type="html")
