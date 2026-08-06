"""Regression tests for S2 work identification and the status it is recorded under.

Background (2026-08-06): every one of the 573 indexed items carried
``s2_status='mapped'``, because callers overwrote the mapper's own outcome
unconditionally.  373 of those had no ``s2_paper_id`` at all -- S2 had never
identified them -- and 303 were consequently invisible in the Citation Graph.
The lookup itself was the root cause: it queried S2 with the pre-colon title
plus the author plus Zotero's year, which returns unrelated papers.
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import citation_mapper  # noqa: E402
import db_relations  # noqa: E402


def _paper(title, authors=(), cc=0, pid="PID"):
    return {
        "paperId": pid, "title": title, "citationCount": cc,
        "authors": [{"name": name} for name in authors],
    }


class SelectS2TitleMatchTests(unittest.TestCase):
    def _select(self, results, full, main=None, creators=""):
        return citation_mapper._select_s2_title_match(
            results, full, main if main is not None else full, creators,
        )

    def test_review_of_the_work_is_not_adopted_as_the_work_itself(self):
        # S2 indexes book reviews under near-identical titles but credits the
        # reviewer. Adopting one would attach the review's citation graph to
        # the book. Observed live for Fedman "Seeds of Control" (all five
        # candidates were reviews) and Pelluchon "Nourishment".
        results = [
            _paper("Seeds of Control: Japan's Empire of Forestry in Colonial Korea",
                   ["Chizuko Allen"], cc=19),
            _paper("Seeds of Control: Japan's Empire of Forestry in Colonial Korea",
                   ["Owen Miller"], cc=3),
        ]
        self.assertIsNone(self._select(
            results, "Seeds of Control Japan s Empire of Forestry in Colonial Korea",
            creators="Fedman David",
        ))

    def test_work_whose_author_matches_is_accepted(self):
        results = [_paper("The Stack: On Software and Sovereignty", ["B. Bratton"], cc=636)]
        match = self._select(
            results, "The stack on software and sovereignty",
            main="The stack", creators="Bratton Benjamin H.",
        )
        self.assertIsNotNone(match)
        self.assertEqual(match["citationCount"], 636)

    def test_record_without_any_author_stays_eligible(self):
        # S2 lists no author for many book records; that is missing evidence,
        # not contrary evidence, so it must not be treated as a mismatch.
        results = [_paper("Video Theories: A Transdisciplinary Reader", [], cc=0)]
        self.assertIsNotNone(self._select(
            results, "Video theories a transdisciplinary reader",
            main="Video theories", creators="Daniels Dieter",
        ))

    def test_subtitleless_s2_record_still_matches_via_the_main_title(self):
        results = [_paper("After the End of Art", ["Arthur Danto"], cc=77)]
        self.assertIsNotNone(self._select(
            results, "After the end of art contemporary art and the pale of history",
            main="After the end of art", creators="Danto Arthur Coleman",
        ))

    def test_unrelated_titles_are_rejected_even_with_a_matching_author(self):
        results = [_paper("Planck 2015 results XXIII", ["Benjamin Bratton"], cc=3594)]
        self.assertIsNone(self._select(
            results, "The stack on software and sovereignty",
            main="The stack", creators="Bratton Benjamin H.",
        ))

    def test_closer_title_wins_over_a_more_cited_looser_match(self):
        results = [
            _paper("The Logic of Practice", ["Pierre Bourdieu"], cc=6271, pid="EXACT"),
            _paper("CRM in Call Centres: The Logic of Practice",
                   ["Pierre Bourdieu"], cc=90000, pid="LOOSE"),
        ]
        match = self._select(results, "The logic of practice", creators="Bourdieu Pierre")
        self.assertEqual(match["paperId"], "EXACT")

    def test_two_letter_romanized_surname_still_verifies(self):
        # Dropping <=2-character tokens emptied the token set for names like
        # "Li Xu"/"L. Xu", which made the `if wanted:` guard unreachable and let
        # a reviewer's record through unchecked.
        review = _paper("Chinese Ritual and Politics", ["Margaret Wong"], cc=2, pid="REVIEW")
        real = _paper("Chinese Ritual and Politics", ["L. Xu"], cc=80, pid="REAL")

        self.assertIsNone(self._select([review], "Chinese ritual and politics", creators="Li Xu"))
        picked = self._select([review, real], "Chinese ritual and politics", creators="Li Xu")
        self.assertEqual(picked["paperId"], "REAL")

    def test_a_shared_given_name_is_not_treated_as_a_match(self):
        # A review of Pratt's "Imperial Eyes" by Mary Baine Campbell shares
        # "Mary" with the Zotero creator "Pratt Mary Louise"; comparing every
        # token let that through, so only surnames are compared.
        review = _paper("Imperial Eyes: Travel Writing and Transculturation",
                        ["Mary Baine Campbell"], cc=4)
        self.assertIsNone(self._select(
            review and [review], "Imperial Eyes Travel Writing and Transculturation",
            creators="Pratt Mary Louise",
        ))

    def test_generational_suffix_is_not_mistaken_for_the_surname(self):
        self.assertEqual(
            citation_mapper._external_surnames(["John Smith Jr."]), {"smith"},
        )

    def test_single_letter_initials_are_still_ignored(self):
        # "B." in "B. Bratton" is an initial, not identity: it must not let an
        # unrelated author named e.g. "B. Someone" satisfy the check.
        self.assertEqual(
            citation_mapper._s2_name_tokens(_paper("t", ["B. Bratton"])), {"bratton"},
        )

    def test_compound_surname_split_differently_by_s2_still_verifies(self):
        # Zotero stores "Hartman Davies Oscar"; S2 renders it "O. H. Davies".
        # Comparing only the leading token would reject the author's own work.
        results = [_paper("Digital Ecologies", ["O. H. Davies"], cc=12)]
        self.assertIsNotNone(self._select(
            results, "Digital ecologies", creators="Hartman Davies Oscar",
        ))

    def test_more_cited_record_wins_among_equally_close_duplicates(self):
        results = [
            _paper("Seeing Like a State", ["James C. Scott"], cc=1327, pid="STUB"),
            _paper("Seeing Like a State", ["James C. Scott"], cc=7753, pid="CANON"),
        ]
        match = self._select(results, "Seeing like a state", creators="Scott James C.")
        self.assertEqual(match["paperId"], "CANON")


class FindS2PaperIdQueryTests(unittest.TestCase):
    """The query text itself was the bug; these pin its shape."""

    def _queries_for(self, title, creators="", results_by_query=None):
        seen = []

        def _fake_request(url):
            seen.append(url)
            if results_by_query is None:
                return {"data": []}
            for needle, payload in results_by_query.items():
                if citation_mapper.urllib.parse.quote(needle) in url:
                    return {"data": payload}
            return {"data": []}

        with patch.object(citation_mapper, "s2_request", side_effect=_fake_request):
            match = citation_mapper.find_s2_paper_id(title, "2015", creators)
        return match, seen

    def test_year_is_never_part_of_the_query(self):
        # Zotero stores the edition year; S2 stores the original. Tannahill's
        # "Food in History" is 1989 in Zotero and 1974 in S2, Bourdieu's
        # "Logic of Practice" 2008 vs 1990. A year term matches thousands of
        # unrelated papers of that year instead of narrowing anything.
        _match, urls = self._queries_for("The stack: on software and sovereignty",
                                         "Bratton Benjamin H.")
        self.assertTrue(urls)
        for url in urls:
            self.assertNotIn("2015", url)

    def test_full_title_including_the_subtitle_is_queried_first(self):
        _match, urls = self._queries_for("The stack: on software and sovereignty",
                                         "Bratton Benjamin H.")
        self.assertIn(
            citation_mapper.urllib.parse.quote("The stack on software and sovereignty"),
            urls[0],
        )

    def test_author_is_only_appended_after_title_only_queries_fail(self):
        # "Biopiracy" alone gives S2's ranking nothing to discriminate on, and
        # that item has neither DOI nor ISBN, so the author query is its only
        # route to an identity.
        _match, urls = self._queries_for("Biopiracy", "Shiva Vandana")
        self.assertEqual(len(urls), 2)
        self.assertNotIn(citation_mapper.urllib.parse.quote("Shiva"), urls[0])
        self.assertIn(citation_mapper.urllib.parse.quote("Biopiracy Shiva"), urls[1])

    def test_short_title_still_issues_a_query(self):
        # A previous iteration skipped any query shorter than 10 characters,
        # which silently dropped every short-titled work.
        _match, urls = self._queries_for("Biopiracy", "")
        self.assertEqual(len(urls), 1)

    def test_no_further_queries_once_a_work_is_identified(self):
        match, urls = self._queries_for(
            "The stack: on software and sovereignty", "Bratton Benjamin H.",
            results_by_query={
                "The stack on software and sovereignty": [
                    _paper("The Stack: On Software and Sovereignty", ["B. Bratton"], cc=636),
                ],
            },
        )
        self.assertIsNotNone(match)
        self.assertEqual(len(urls), 1)

    def test_non_english_title_skips_the_search_entirely(self):
        _match, urls = self._queries_for("日本語のタイトルのみの資料", "山田 太郎")
        self.assertEqual(urls, [])


class UnresolvedItemStatusTests(unittest.TestCase):
    """An item S2 cannot identify must not be recorded as 'mapped'."""

    def _run_without_s2_match(self):
        statuses = []
        with patch.object(citation_mapper, "find_s2_paper_id", return_value=None), \
             patch.object(citation_mapper, "get_s2_lookup_candidates", return_value=[]), \
             patch.object(citation_mapper, "update_item_citation_status",
                          side_effect=lambda *a, **kw: statuses.append(a)):
            result = citation_mapper.map_item_global_citations("ITEM", title="Unknown Work")
        return result, statuses

    def test_unidentified_item_is_recorded_as_not_found(self):
        result, statuses = self._run_without_s2_match()
        self.assertEqual(statuses[-1], ("ITEM", "not_found"))
        self.assertNotIn(("ITEM", "mapped"), statuses)

    def test_unidentified_item_reports_s2_resolved_false(self):
        # "status": "success" only means the step ran; callers must branch on
        # s2_resolved, or they re-label the item 'mapped' and it is skipped by
        # every later --all run, losing the chance to ever re-search it.
        result, _statuses = self._run_without_s2_match()
        self.assertEqual(result["status"], "success")
        self.assertIs(result["s2_resolved"], False)

    def test_s2_done_is_reserved_for_items_that_were_identified(self):
        result, statuses = self._run_without_s2_match()
        self.assertNotIn(("ITEM", "s2_done"), statuses)


class StaleRelationsAreClearedOnIdentityChangeTests(unittest.TestCase):
    """A re-run that lands on a different S2 paper must not keep the old one's rows."""

    def _map_resolving_to(self, new_pid, stored_pid):
        cleared = []
        with patch.object(citation_mapper, "find_s2_paper_id",
                          return_value={"paperId": new_pid, "year": 2016, "citationCount": 1}), \
             patch.object(citation_mapper, "get_item_s2_paper_id", return_value=stored_pid), \
             patch.object(citation_mapper, "clear_s2_relations_for_item",
                          side_effect=lambda k: (cleared.append(k),
                                                 {"global_citations": 6, "global_references": 0})[1]), \
             patch.object(citation_mapper, "s2_request", side_effect=[{"data": []}, {"data": []}]), \
             patch.object(citation_mapper, "insert_citation"), \
             patch.object(db_relations, "insert_reference"), \
             patch.object(citation_mapper, "update_item_citation_status"):
            citation_mapper.map_item_global_citations("ITEM", title="Some Work")
        return cleared

    def test_previous_identitys_rows_are_dropped_when_the_paper_changes(self):
        # global_citations is UNIQUE(citing_paper_id, cited_item_key,
        # context_snippet), so rows fetched under the old paper never collide
        # with the new ones and would otherwise accumulate.
        self.assertEqual(self._map_resolving_to("NEW", "OLD"), ["ITEM"])

    def test_nothing_is_dropped_when_the_paper_is_unchanged(self):
        self.assertEqual(self._map_resolving_to("SAME", "SAME"), [])

    def test_nothing_is_dropped_on_a_first_ever_resolution(self):
        self.assertEqual(self._map_resolving_to("NEW", None), [])


class AbstractCachingDoesNotRestateS2StatusTests(unittest.TestCase):
    """Fetching an abstract is unrelated to whether S2 identified the work."""

    def _fetch_abstract_for(self, current_status):
        import json
        import urllib.request

        import citation_graph.server as server
        # The route imports "src.db_relations", which Python holds as a module
        # object distinct from the top-level "db_relations" imported above.
        import src.db_relations as route_db

        written = []

        class _Resp:
            def read(self_inner):
                return json.dumps({"data": {"abstractNote": "An abstract."}}).encode()

            def __enter__(self_inner):
                return self_inner

            def __exit__(self_inner, *_a):
                return False

        with patch.object(route_db, "get_item_abstract", return_value=None), \
             patch.object(route_db, "get_item_citation_status", return_value=current_status), \
             patch.object(route_db, "update_item_citation_status",
                          side_effect=lambda *a, **kw: written.append(a)), \
             patch.object(urllib.request, "urlopen", return_value=_Resp()):
            server._route_fetch_abstract(server._FetchAbstractRequest(item_key="ITEM"))
        return written

    def test_not_found_item_is_not_promoted_to_mapped(self):
        written = self._fetch_abstract_for("not_found")
        self.assertEqual(written[-1], ("ITEM", "not_found"))

    def test_existing_mapped_status_is_preserved(self):
        written = self._fetch_abstract_for("mapped")
        self.assertEqual(written[-1], ("ITEM", "mapped"))


if __name__ == "__main__":
    unittest.main()
