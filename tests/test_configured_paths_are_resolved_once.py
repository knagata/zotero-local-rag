"""A configured path is resolved in one place, not in each module that needs it.

``CHROMA_DIR`` names the database every entry point reads and writes. Reading it
straight out of the environment looks harmless and is not: ``~/chroma`` becomes
a directory literally named ``~``, and a relative value resolves against
whatever directory the process happened to start in, so one configuration names
different databases depending on how it was invoked.

``v3_data_plane.resolve_configured_path`` is where that rule is written, and its
own docstring records the rule drifting apart once already, in 2026-07-30, when
only one of two copies expanded ``~``. ``chunk_store`` carries a second note
from 2026-08-04, when the expansion had been added there but the relative case
had not. Both were repaired by hand, module by module, and seven modules were
still reading the variable directly when this test was written -- the same fault
regrowing because nothing stopped it.

So it is a rule now rather than a habit. A module that needs the configured
location asks for it; only the place that defines the resolution reads the raw
value.
"""
from __future__ import annotations

import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

#: Variables whose raw value must pass through the resolver before use.
CONFIGURED_PATH_VARIABLES = frozenset({
    "CHROMA_DIR", "MANIFEST_PATH", "LEXICAL_DB_PATH",
})

#: Where the resolution rule lives, and so the only module allowed to read the
#: raw value. ``enforce_environment`` also writes resolved values back into the
#: environment for child processes, which is the same module's business.
RESOLVER_MODULE = "src/v3_data_plane.py"

SEARCHED = ("src", "scripts", "citation_graph")


def _environment_reads(tree: ast.AST) -> set[str]:
    """Names read out of ``os.environ`` anywhere in this module."""
    found: set[str] = set()
    for node in ast.walk(tree):
        name = None
        # os.environ.get("X") / os.environ.get("X", default)
        if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                and node.func.attr in {"get", "setdefault"}
                and isinstance(node.func.value, ast.Attribute)
                and node.func.value.attr == "environ" and node.args
                and isinstance(node.args[0], ast.Constant)):
            name = node.args[0].value
        # os.environ["X"]
        elif (isinstance(node, ast.Subscript)
                and isinstance(node.value, ast.Attribute)
                and node.value.attr == "environ"
                and isinstance(node.slice, ast.Constant)):
            name = node.slice.value
        if isinstance(name, str):
            found.add(name)
    return found


def test_only_the_resolver_reads_a_configured_path_from_the_environment():
    offenders: list[str] = []
    for directory in SEARCHED:
        for path in sorted((ROOT / directory).rglob("*.py")):
            relative = path.relative_to(ROOT).as_posix()
            if relative == RESOLVER_MODULE or "attic" in relative:
                continue
            try:
                tree = ast.parse(path.read_text(encoding="utf-8"))
            except SyntaxError:  # pragma: no cover -- compileall covers this
                continue
            for name in sorted(_environment_reads(tree) & CONFIGURED_PATH_VARIABLES):
                offenders.append(f"  {relative}: reads {name} directly")
    assert not offenders, (
        "a configured path is being read straight from the environment. Pass the "
        "raw value through v3_data_plane.resolve_configured_path, which expands "
        "~ and resolves a relative value against the project root -- both halves "
        "of that rule have been lost once already by a copy that looked "
        "harmless:\n" + "\n".join(offenders)
    )


def test_the_resolver_handles_both_halves_of_the_rule():
    # The two failures this guards against, stated as the behaviour rather than
    # as an absence of copies.
    from src.v3_data_plane import resolve_configured_path

    assert resolve_configured_path(ROOT, "~/chroma") == Path.home() / "chroma"
    assert resolve_configured_path(ROOT, "data/chroma") == ROOT / "data" / "chroma"
    assert resolve_configured_path(ROOT, "/tmp/chroma") == Path("/tmp/chroma")


#: Where Zotero lives and how a request to it is addressed. Three modules each
#: worked it out for themselves and disagreed: one sent Accept, the version
#: header and the key; one omitted Accept; and update_citations -- the module
#: that writes back to the library -- sent only the key, so it parsed whatever
#: schema Zotero chose to answer with. Its own test file records a third
#: variant, a hardcoded 127.0.0.1:8080 that pointed at nothing for a whole run
#: and reported 0 items as success (2026-08-01).
ZOTERO_ADDRESS_VARIABLES = frozenset({
    "ZOTERO_LOCAL_API_BASE", "ZOTERO_LOCAL_API_PREFIX", "ZOTERO_API_VERSION",
})
ZOTERO_CLIENT_MODULE = "src/zotero_source_localapi.py"


def test_only_the_zotero_client_works_out_how_to_address_zotero():
    offenders: list[str] = []
    for directory in SEARCHED:
        for path in sorted((ROOT / directory).rglob("*.py")):
            relative = path.relative_to(ROOT).as_posix()
            if relative == ZOTERO_CLIENT_MODULE or "attic" in relative:
                continue
            text = path.read_text(encoding="utf-8")
            try:
                tree = ast.parse(text)
            except SyntaxError:  # pragma: no cover -- compileall covers this
                continue
            for name in sorted(_environment_reads(tree) & ZOTERO_ADDRESS_VARIABLES):
                offenders.append(f"  {relative}: reads {name} directly")
            if '"Zotero-API-Version"' in text:
                offenders.append(f"  {relative}: names the Zotero-API-Version header")
    assert not offenders, (
        "the address of, or the headers for, Zotero are being worked out "
        "somewhere other than zotero_source_localapi. Use local_api_url() and "
        "zotero_api_headers(): the three copies that existed disagreed about "
        "what to send, and the least pinned of them was the one that writes "
        "back to the library.\n" + "\n".join(offenders)
    )


def test_every_caller_sends_the_same_headers():
    from src.zotero_source_localapi import ZoteroLocalAPI, zotero_api_headers

    shared = set(zotero_api_headers("KEY"))
    assert shared == {"Accept", "Zotero-API-Key", "Zotero-API-Version"}
    assert set(ZoteroLocalAPI(api_key="KEY").headers) == shared
    # Extra headers a caller needs are added to the shared set, not instead
    # of it -- the write-back path needs Content-Type and lost the rest.
    assert set(zotero_api_headers("KEY", **{"Content-Type": "application/json"})) == (
        shared | {"Content-Type"}
    )


def test_the_zones_carrying_references_are_named_once():
    """Which zones hold reference entries is a policy, not a literal.

    Three extractors each kept the same three zone names under a name of their
    own -- _REFERENCE_ZONES, _REFERENCE_HARVEST_ZONES, _ENTRY_ZONES. A zone
    added to or removed from the policy table would have reached none of them,
    and a search for any one of those names finds neither of the others.
    """
    from src.document_structure import CITATION_EXTRACT_ZONES, ZONE_POLICIES

    assert CITATION_EXTRACT_ZONES == frozenset(
        zone for zone, (_s, _r, citation) in ZONE_POLICIES.items()
        if citation == "extract"
    )

    literal = '{"bibliography", "endnote", "footnote"}'
    offenders = [
        path.relative_to(ROOT).as_posix()
        for directory in SEARCHED
        for path in sorted((ROOT / directory).rglob("*.py"))
        if "attic" not in path.as_posix()
        and literal in path.read_text(encoding="utf-8")
    ]
    assert not offenders, (
        "the answer is written out instead of being asked of ZONE_POLICIES; "
        "use document_structure.CITATION_EXTRACT_ZONES:\n  "
        + "\n  ".join(offenders)
    )


def test_one_definition_of_what_a_heading_block_is():
    """The module that flags a document and the one that fixes it must agree.

    flat_structure_diagnostics counted section_header, title, chapter and
    chapter_title as heading blocks; source_structure_refresh looked only at
    heading and page_furniture. So an attachment could be listed as a recovery
    candidate on the strength of blocks the recovery then ignored -- which is
    the same shape as the page_furniture gap that left books flat while their
    openers sat in the chunks. Sharing the names costs nothing and closed it.
    """
    from src.document_structure import (
        HEADING_BLOCK_TYPES, HEADING_CANDIDATE_BLOCK_TYPES,
    )
    from src.flat_structure_diagnostics import _HEADING_BLOCK_TYPES
    from src.source_structure_refresh import _HEADING_BLOCKS

    assert _HEADING_BLOCK_TYPES is HEADING_BLOCK_TYPES
    assert _HEADING_BLOCKS is HEADING_CANDIDATE_BLOCK_TYPES
    # Furniture is a candidate, never trusted on its own: the running header
    # that repeats a chapter's name on every page arrives labelled this way.
    assert "page_furniture" in HEADING_CANDIDATE_BLOCK_TYPES
    assert "page_furniture" not in HEADING_BLOCK_TYPES
