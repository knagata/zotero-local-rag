# Zotero import metadata validation

The public inspection tool, proposal creation, and `_zotero_items` before external
writes all use `src/zotero_import.py` candidate validation. Inspection rejects bad
metadata before duplicate lookups. Both inspection and proposal responses include
`zotero_preview`, the parent payload used for bibliographic writes. Collection
selection appears in the proposal preview. A preview validates representation,
not the historical truth of an author's name or a publication date.

`CANDIDATE_TEXT_FIELDS` defines the string fields. `title` and HTTPS `landing_url`
are required. `resource_type` controls the Zotero type; `item_type`, `itemType`,
`publicationTitle`, `authors` and other unknown fields fail with repair guidance.
`container_title` is the journal/proceedings title. Wrong scalar, list, nested
identifier or creator types are rejected rather than stringified. Discovery and
inspection annotations are explicitly allowed but never written as bibliography.
Fields that would disappear because the declared type cannot carry journal
metadata are rejected. Existing catalog type classification is retained; unknown
unclassifiable types are rejected rather than silently treated as documents.

Creators may be name strings or objects containing `name` for a single/corporate
name, or explicit `firstName`/`lastName`. Name and split fields are mutually
exclusive, split names need a nonempty surname, and creatorType defaults to
author (author/editor/translator/contributor supported). No surname inference is
performed. Whitespace and exact duplicate creator entries are normalized. A comma
is preserved unless all comma-separated components are the exact same name.
`Doe, Jane` therefore stays intact. Serialized object strings are rejected.

`language` and multiline `extra` pass through. Metadata-only import continues to
create a Source URL linked attachment. No existing Zotero items are repaired by
this input validation change. The regression suite uses fake writers and checks
that rejected input never reaches an external write, including saved proposals
validated again at execution. Relevant tests are tests/test_zotero_import.py,
tests/test_zotero_import_validation.py and tests/test_mcp_app_factory.py.
