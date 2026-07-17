import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
import sqlite3

from src.reference_text import s2_candidate_is_supported
from scripts.repair_unverified_epub_mappings import repair


class S2CandidateValidationTests(unittest.TestCase):
    def test_accepts_exact_title_year_and_author(self):
        raw = "Catherine Malabou, What Should We Do with Our Brain? (Fordham, 2008)."
        paper = {"title": "What Should We Do with Our Brain?", "year": 2008,
                 "authors": [{"name": "C. Malabou"}]}
        self.assertTrue(s2_candidate_is_supported(raw, paper))

    def test_rejects_unrelated_first_search_result(self):
        raw = "Hal Foster, The Return of the Real (MIT Press, 1996)."
        paper = {"title": "Euclid Q1 Early Release Observations", "year": 2025,
                 "authors": [{"name": "A. Astronomer"}]}
        self.assertFalse(s2_candidate_is_supported(raw, paper))

    def test_rejects_year_mismatch(self):
        raw = "Jane Doe, A Distinctive Long Title (2001)."
        paper = {"title": "A Distinctive Long Title", "year": 2020,
                 "authors": [{"name": "Jane Doe"}]}
        self.assertFalse(s2_candidate_is_supported(raw, paper))

    def test_rejects_author_mismatch_even_when_title_and_year_match(self):
        raw = "Lauren Berlant, Cruel Optimism (2011)."
        paper = {"title": "Cruel Optimism", "year": 2011,
                 "authors": [{"name": "Unrelated Reviewer"}]}
        self.assertFalse(s2_candidate_is_supported(raw, paper))

    def test_accepts_matching_doi(self):
        raw = "Doe, article title, doi:10.1234/ABC.5"
        paper = {"title": "Different formatting", "externalIds": {"DOI": "10.1234/abc.5"}}
        self.assertTrue(s2_candidate_is_supported(raw, paper))


class RepairUnverifiedMappingsTests(unittest.TestCase):
    def test_repair_preserves_evidence_and_removes_only_unverified_edge(self):
        with TemporaryDirectory() as directory:
            db_path = Path(directory) / "relations.db"
            backup = Path(directory) / "backup.db"
            connection = sqlite3.connect(db_path)
            connection.executescript('''
                CREATE TABLE global_references (
                    id INTEGER PRIMARY KEY, citing_item_key TEXT, source TEXT,
                    raw_reference_text TEXT, context_snippet TEXT, s2_status TEXT,
                    cited_paper_id TEXT, cited_title TEXT, cited_year INTEGER,
                    cited_doi TEXT, cited_authors TEXT,
                    cited_citation_count INTEGER, cited_influential_count INTEGER
                );
                CREATE TABLE works (work_id INTEGER PRIMARY KEY, zotero_item_key TEXT);
                CREATE TABLE work_edges (
                    id INTEGER PRIMARY KEY, citing_work_id INTEGER, source TEXT,
                    raw_reference TEXT
                );
                INSERT INTO works VALUES (1, 'OWN');
                INSERT INTO global_references VALUES
                    (1, 'OWN', 'epub', 'Hal Foster, The Return of the Real (1996).',
                     'kept context', 'mapped', 'BAD', 'Euclid Q1', 2025, NULL,
                     'A. Astronomer', 10, 1);
                INSERT INTO work_edges VALUES
                    (10, 1, 'epub', 'Hal Foster, The Return of the Real (1996).');
            ''')
            connection.commit()
            connection.close()

            result = repair(db_path, commit=True, backup_path=backup)
            self.assertEqual(result["unverified_references"], 1)
            connection = sqlite3.connect(db_path)
            row = connection.execute(
                "SELECT raw_reference_text, context_snippet, cited_title, s2_status FROM global_references"
            ).fetchone()
            self.assertEqual(row, (
                "Hal Foster, The Return of the Real (1996).", "kept context", None, "unverified"
            ))
            self.assertEqual(connection.execute("SELECT COUNT(*) FROM work_edges").fetchone()[0], 0)
            connection.close()
            self.assertTrue(backup.exists())


if __name__ == "__main__":
    unittest.main()
