import asyncio
import pytest
from src import zotero_import as imports
from tests.test_zotero_import import candidate


@pytest.mark.parametrize('changes,match', [
    ({'item_type': 'journalArticle'}, 'resource_type'),
    ({'publicationTitle': 'Journal'}, 'container_title'),
    ({'creators': 'Someone'}, 'list'),
    ({'creators': [42]}, 'creator'),
    ({'creators': ["{'lastName': 'Name'}"]}, 'serialized'),
    ({'creators': [{'lastName': 'Name', 'typo': 'x'}]}, 'unknown'),
    ({'creators': [{'name': "{'lastName': 'Name'}"}]}, 'serialized'),
    ({'creators': ['']}, 'empty'),
    ({'creators': [{'name': ''}]}, 'empty'),
    ({'resource_type': 'typo'}, 'unsupported'),
    ({'resource_type': 'conferencePaper'}, 'issue'),
    ({'resource_type': 'book', 'container_title': 'Journal'}, 'journal fields'),
    ({'creators': [{'name': 'Name', 'firstName': 'Other'}]}, 'either'),
    ({'creators': [{'firstName': 'Only'}]}, 'lastName'),
    ({'creators': [{'lastName': 'Name', 'creatorType': 'typo'}]}, 'creatorType'),
    ({'creators': [{'lastName': 42}]}, 'string'),
    ({'title': {'bad': 'title'}}, 'string'),
    ({'language': ['ja']}, 'string'),
    ({'identifiers': []}, 'object'),
    ({'identifiers': {'doi': {'bad': 'doi'}}}, 'string'),
    ({'download_urls': 'https://example.test/file.pdf'}, 'list'),
    ({'download_urls': [42]}, 'string'),
    ({'resource_type': 'text'}, 'journal fields'),
])
def test_bad_candidate_is_rejected_before_proposal_or_write(tmp_path, changes, match):
    path = tmp_path / 'proposal.json'
    with pytest.raises((TypeError, ValueError), match=match):
        imports.create_import_proposal(candidate(**changes), [], path=path)
    assert not path.exists()
    class NoWrites:
        async def create_items(self, items):
            pytest.fail('invalid candidate must never reach Zotero')
    with pytest.raises((TypeError, ValueError), match=match):
        asyncio.run(imports.execute_import({'candidate': candidate(**changes), 'import_mode': 'metadata'}, writer=NoWrites()))


def test_split_names_single_names_language_and_extra_are_preserved():
    creators = [{'lastName': '久保', 'firstName': '文克'}, {'lastName': '王', 'firstName': '明'},
                {'name': 'Research Institute'}, 'Doe, Jane', '久保 文克,久保 文克', '久保 文克']
    preview = imports.preview_import_candidate(candidate(creators=creators, language='ja', extra='Original note'))
    parent = preview['zotero_preview']
    assert parent['creators'] == [
        {'creatorType': 'author', 'lastName': '久保', 'firstName': '文克'},
        {'creatorType': 'author', 'lastName': '王', 'firstName': '明'},
        {'creatorType': 'author', 'name': 'Research Institute'},
        {'creatorType': 'author', 'name': 'Doe, Jane'},
        {'creatorType': 'author', 'name': '久保 文克'},
    ]
    assert (parent['language'], parent['extra']) == ('ja', 'Original note')
    assert parent['itemType'] == 'journalArticle'
    assert (parent['publicationTitle'], parent['volume'], parent['issue'], parent['pages']) == ('Journal of Rice Research', '12', '3', '101-119')
    assert imports.preview_import_candidate(preview['candidate']) == preview


def test_catalog_annotations_are_allowed_but_not_written():
    raw = candidate(textual_resource=True, search_strategy='source_native', matched_query='rice')
    result = imports.preview_import_candidate(raw)
    assert 'matched_query' not in result['candidate']
    assert 'matched_query' not in result['zotero_preview']


def test_proposal_preview_matches_actual_payload(tmp_path):
    proposal = imports.create_import_proposal(candidate(), [], path=tmp_path/'proposal.json')
    assert proposal['zotero_preview'] == imports._zotero_items({**proposal, 'collection_key': ''})[0]


def test_minimal_book_document_and_conference_previews():
    minimal = {"title": "Example", "landing_url": "https://example.test"}
    assert imports.preview_import_candidate(minimal)["zotero_preview"]["itemType"] == "document"
    book = imports.preview_import_candidate({**minimal, "resource_type": "book", "volume": "2"})
    assert book["zotero_preview"]["volume"] == "2"
    conference = imports.preview_import_candidate(candidate(resource_type="conferencePaper", issue="", identifiers={"doi": ["10.123/example"]}))
    assert conference["zotero_preview"]["proceedingsTitle"] == "Journal of Rice Research"
    assert conference["zotero_preview"]["DOI"] == "10.123/example"
    assert "issue" not in conference["zotero_preview"]
    with pytest.raises(ValueError, match="volume"):
        imports.preview_import_candidate({**minimal, "volume": "2"})
    with pytest.raises(TypeError, match="object"):
        imports.preview_import_candidate([])
