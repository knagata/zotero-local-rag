from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from src import db_relations
from scripts import backfill_works


class CanonicalWorksTests(unittest.TestCase):
    def setUp(self):
        self.tempdir = tempfile.TemporaryDirectory()
        self.db_patch = patch.object(
            db_relations, "DB_PATH", str(Path(self.tempdir.name) / "relations.db")
        )
        self.db_patch.start()
        db_relations._db_initialized = False

    def tearDown(self):
        self.db_patch.stop()
        db_relations._db_initialized = False
        self.tempdir.cleanup()

    def test_stable_identifier_resolves_existing_work(self):
        first = db_relations.resolve_work(
            doi="https://doi.org/10.1234/ABC", title="The Gift", year=1925
        )
        second = db_relations.resolve_work(
            doi="doi:10.1234/abc", authors="Marcel Mauss"
        )
        self.assertEqual(first, second)

    def test_conflicting_secondary_identifier_does_not_abort_resolution(self):
        first = db_relations.resolve_work(doi="10.1234/one", s2_paper_id="S2-ONE", title="One")
        second = db_relations.resolve_work(doi="10.1234/two", s2_paper_id="S2-TWO", title="Two")
        resolved = db_relations.resolve_work(
            doi="10.1234/one", s2_paper_id="S2-TWO", container="Journal"
        )
        self.assertEqual(resolved, second)
        self.assertNotEqual(first, second)
        connection = db_relations.get_db_connection()
        try:
            row = connection.execute("SELECT * FROM works WHERE work_id = ?", (second,)).fetchone()
        finally:
            connection.close()
        self.assertEqual(row["doi"], "10.1234/two")
        self.assertEqual(row["s2_paper_id"], "s2-two")
        self.assertEqual(row["container"], "Journal")

    def test_purge_removed_item_cleans_owned_work_and_edges(self):
        connection = db_relations.get_db_connection()
        connection.execute(
            "INSERT INTO item_citation_status (item_key, s2_status) VALUES ('REMOVED', 'mapped')"
        )
        connection.commit()
        connection.close()
        owned = db_relations.resolve_work(zotero_item_key="REMOVED", title="Removed")
        external = db_relations.resolve_work(title="External")
        db_relations.save_work_edge(owned, external, source="test", raw_reference="ref")
        counts = db_relations.purge_removed_items(set())
        self.assertEqual(counts["works"], 1)
        self.assertEqual(counts["work_edges"], 1)
        connection = db_relations.get_db_connection()
        try:
            self.assertIsNone(
                connection.execute("SELECT 1 FROM works WHERE work_id = ?", (owned,)).fetchone()
            )
        finally:
            connection.close()

    def test_conservative_title_match_and_year(self):
        first = db_relations.resolve_work(title="Essai sur le don", authors="Mauss", year=1925)
        second = db_relations.resolve_work(title="Essai sur le don!", authors="Mauss", year=1926)
        far_year = db_relations.resolve_work(title="Essai sur le don", authors="Mauss", year=2000)
        self.assertEqual(first, second)
        self.assertNotEqual(first, far_year)

    def test_title_only_works_are_never_merged(self):
        first = db_relations.resolve_work(title="贈与")
        second = db_relations.resolve_work(title="贈与")
        self.assertNotEqual(first, second)

    def test_title_and_author_can_merge(self):
        first = db_relations.resolve_work(title="贈与", authors="Marcel Mauss")
        second = db_relations.resolve_work(title="贈与!", authors="Marcel Mauss")
        self.assertEqual(first, second)

    def test_title_only_does_not_imply_ownership(self):
        db_relations.resolve_work(zotero_item_key="OWNED", title="家族")
        self.assertFalse(db_relations.is_owned_work(title="家族"))

    def test_title_plus_author_can_establish_ownership(self):
        db_relations.resolve_work(
            zotero_item_key="OWNED", title="家族", authors="Author A"
        )
        self.assertTrue(db_relations.is_owned_work(title="家族", authors="Author A"))

    def test_work_cluster_is_undirected_and_transitive(self):
        original = db_relations.resolve_work(title="Original")
        translation = db_relations.resolve_work(title="Translation")
        edition = db_relations.resolve_work(title="New Edition")
        db_relations.save_work_link(translation, original, "translation_of", source="manual")
        db_relations.save_work_link(edition, translation, "edition_of", source="manual")
        self.assertEqual(db_relations.get_work_cluster(original), sorted([original, translation, edition]))

    def test_translation_cluster_counts_as_owned(self):
        owned_translation = db_relations.resolve_work(
            zotero_item_key="OWNED", title="贈与論"
        )
        original = db_relations.resolve_work(
            title="Essai sur le don", doi="10.1234/original"
        )
        db_relations.save_work_link(
            owned_translation, original, "translation_of", source="manual"
        )
        self.assertTrue(db_relations.is_owned_work(doi="https://doi.org/10.1234/original"))

    def test_child_works_can_share_zotero_item_key(self):
        parent = db_relations.resolve_work(zotero_item_key="ITEM1", title="Collected Volume")
        chapter1 = db_relations.resolve_work(
            zotero_item_key="ITEM1", title="Chapter One", container_work_id=parent, section_id="c1"
        )
        chapter2 = db_relations.resolve_work(
            zotero_item_key="ITEM1", title="Chapter Two", container_work_id=parent, section_id="c2"
        )
        self.assertEqual(len({parent, chapter1, chapter2}), 3)

    def test_edge_upsert_is_idempotent(self):
        citing = db_relations.resolve_work(title="Citing")
        cited = db_relations.resolve_work(title="Cited")
        first_id = db_relations.save_work_edge(citing, cited, source="test", raw_reference="ref")
        second_id = db_relations.save_work_edge(citing, cited, source="test", raw_reference="ref")
        connection = db_relations.get_db_connection()
        try:
            count = connection.execute("SELECT COUNT(*) FROM work_edges").fetchone()[0]
        finally:
            connection.close()
        self.assertEqual(count, 1)
        self.assertGreater(first_id, 0)
        self.assertEqual(first_id, second_id)

    def test_legacy_backfill_is_idempotent(self):
        connection = db_relations.get_db_connection()
        connection.execute(
            "INSERT INTO item_citation_status (item_key, s2_status) VALUES ('OWN1', 'mapped')"
        )
        connection.execute('''
            INSERT INTO global_references
                (cited_paper_id, cited_title, citing_item_key, context_snippet)
            VALUES ('S2-1', 'External', 'OWN1', 'context')
        ''')
        connection.commit()
        connection.close()
        first = backfill_works.backfill()
        second = backfill_works.backfill()
        self.assertEqual(first["edges"], 1)
        self.assertEqual(second["edges"], 1)
        connection = db_relations.get_db_connection()
        try:
            self.assertEqual(connection.execute("SELECT COUNT(*) FROM works").fetchone()[0], 2)
            self.assertEqual(connection.execute("SELECT COUNT(*) FROM work_edges").fetchone()[0], 1)
        finally:
            connection.close()

    def test_title_only_legacy_reference_stays_separate_and_idempotent(self):
        connection = db_relations.get_db_connection()
        connection.execute(
            "INSERT INTO item_citation_status (item_key, s2_status) VALUES ('OWN1', 'mapped')"
        )
        connection.executemany('''
            INSERT INTO global_references
                (cited_title, citing_item_key, context_snippet, raw_reference_text)
            VALUES ('家族', 'OWN1', ?, ?)
        ''', [("context-a", "ref-a"), ("context-b", "ref-b")])
        connection.commit()
        connection.close()
        backfill_works.backfill()
        backfill_works.backfill()
        connection = db_relations.get_db_connection()
        try:
            title_works = connection.execute(
                "SELECT COUNT(*) FROM works WHERE title_norm = ?",
                (db_relations.normalize_work_title("家族"),),
            ).fetchone()[0]
            edges = connection.execute("SELECT COUNT(*) FROM work_edges").fetchone()[0]
        finally:
            connection.close()
        self.assertEqual(title_works, 2)
        self.assertEqual(edges, 2)

    def test_summary_and_case_crud(self):
        db_relations.save_item_summary(
            "ITEM1", "日本語要約", "extractive", summary_en="English summary",
            keywords="gift; 贈与", chunk_count=12, source_mtime=1.5,
        )
        db_relations.save_section_summary(
            "ITEM1", "c1", "章要約", chapter="第一章", model="extractive",
            chunk_count=4, chapter_authors="A. Author",
        )
        db_relations.replace_case_annotations(
            "ITEM1", "c1", [{
                "description": "交換の事例", "region": "Melanesia",
                "practices": ["kula"], "phenomena": ["reciprocity"],
                "evidence_quote": "Kula exchange was observed in Melanesia.",
            }], model="test",
        )
        self.assertEqual(db_relations.get_item_summary("ITEM1")["summary_en"], "English summary")
        self.assertEqual(db_relations.get_section_summaries("ITEM1")[0]["chapter"], "第一章")
        self.assertEqual(db_relations.get_case_annotations("ITEM1")[0]["practices"], "kula")
        self.assertEqual(
            db_relations.get_case_annotations("ITEM1")[0]["evidence_quote"],
            "Kula exchange was observed in Melanesia.",
        )
        db_relations.delete_section_summary("ITEM1", "c1")
        self.assertEqual(db_relations.get_section_summaries("ITEM1"), [])
        self.assertEqual(db_relations.get_case_annotations("ITEM1"), [])


if __name__ == "__main__":
    unittest.main()
