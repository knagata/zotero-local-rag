from __future__ import annotations

import unittest
import xml.etree.ElementTree as ET

from src.cinii_client import _normalize_item
from src.ndl_client import _normalize_record


class BibliographicClientTests(unittest.TestCase):
    def test_normalizes_cinii_json_ld(self):
        row = _normalize_item({
            "@id": "https://cir.nii.ac.jp/crid/123",
            "title": {"@value": "文化人類学の論文"},
            "creator": [{"name": "山田 太郎"}],
            "date": "2020-01-01",
            "identifier": ["10.1234/example"],
        })
        self.assertEqual(row["cinii_crid"], "123")
        self.assertEqual(row["year"], 2020)
        self.assertEqual(row["doi"], "10.1234/example")

    def test_normalizes_ndl_dc_record(self):
        record = ET.fromstring('''
            <recordData xmlns:dc="http://purl.org/dc/elements/1.1/"
                        xmlns:dcterms="http://purl.org/dc/terms/">
              <dc:title>贈与論</dc:title><dc:creator>モース</dc:creator>
              <dc:date>2014</dc:date>
              <dc:identifier>https://ndlsearch.ndl.go.jp/books/R100000002-I123</dc:identifier>
              <dc:identifier>ISBN 978-4-00-000000-1</dc:identifier>
              <dcterms:alternative>Essai sur le don</dcterms:alternative>
            </recordData>
        ''')
        row = _normalize_record(record)
        self.assertEqual(row["title"], "贈与論")
        self.assertEqual(row["year"], 2014)
        self.assertEqual(row["isbn"], "9784000000001")
        self.assertIn("Essai sur le don", row["alternative_titles"])


if __name__ == "__main__":
    unittest.main()
