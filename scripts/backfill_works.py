#!/usr/bin/env python3
"""Backfill canonical works and work edges from the legacy relations tables."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any
from difflib import SequenceMatcher


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src import db_relations
from src.reference_text import is_short_form_reference


def _resolve_in_connection(connection, cache: dict[tuple[Any, ...], int], **values: Any) -> int:
    s2 = db_relations._normalize_identifier(values.get("s2_paper_id"), "s2")
    doi = db_relations._normalize_identifier(values.get("doi"), "doi")
    isbn = db_relations._normalize_identifier(values.get("isbn"), "isbn")
    zotero = (values.get("zotero_item_key") or "").strip() or None
    title = (values.get("title") or "").strip() or None
    title_norm = db_relations.normalize_work_title(title)
    year = values.get("year")
    authors = (values.get("authors") or "").strip() or None
    authors_norm = db_relations.normalize_work_title(authors)
    key = (s2, doi, isbn, zotero, title_norm, authors_norm, year)
    cacheable = bool(s2 or doi or isbn or zotero or (title_norm and (authors_norm or year is not None)))
    if cacheable and key in cache:
        return cache[key]

    row = None
    for column, value in (
        ("s2_paper_id", s2), ("doi", doi), ("isbn", isbn), ("zotero_item_key", zotero)
    ):
        if value:
            row = connection.execute(
                f"SELECT work_id FROM works WHERE {column} = ? LIMIT 1", (value,)
            ).fetchone()
            if row:
                break
    if row is None and title_norm:
        candidates = connection.execute(
            "SELECT work_id, authors, year FROM works WHERE title_norm = ?", (title_norm,)
        ).fetchall()
        for candidate in candidates:
            years_comparable = year is not None and candidate["year"] is not None
            year_match = years_comparable and abs(int(year) - int(candidate["year"])) <= 1
            year_conflict = years_comparable and not year_match
            authors_comparable = bool(authors_norm and candidate["authors"])
            author_match = bool(
                authors_comparable
                and SequenceMatcher(
                    None, authors_norm, db_relations.normalize_work_title(candidate["authors"])
                ).ratio() >= 0.65
            )
            author_conflict = authors_comparable and not author_match
            if (year_match or author_match) and not year_conflict and not author_conflict:
                row = candidate
                break
    if row is not None:
        work_id = int(row[0])
    else:
        cursor = connection.execute('''
            INSERT INTO works
                (s2_paper_id, doi, isbn, zotero_item_key, title, title_norm,
                 authors, year, citation_count, abstract, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
        ''', (
            s2, doi, isbn, zotero, title, title_norm,
            authors, year, values.get("citation_count"), values.get("abstract"),
        ))
        work_id = int(cursor.lastrowid)
    if cacheable:
        cache[key] = work_id
    return work_id


def backfill(*, limit: int | None = None) -> dict[str, int]:
    """Populate canonical tables idempotently and return processed row counts."""
    connection = db_relations.get_db_connection()
    cache: dict[tuple[Any, ...], int] = {}
    counts = {"owned_works": 0, "external_works": 0, "edges": 0}
    try:
        statuses = connection.execute('''
            SELECT item_key, s2_paper_id, doi, isbn, s2_year, s2_citation_count, abstract
            FROM item_citation_status ORDER BY item_key
        ''').fetchall()
        owned: dict[str, int] = {}
        for row in statuses:
            owned[row["item_key"]] = _resolve_in_connection(
                connection, cache, zotero_item_key=row["item_key"],
                s2_paper_id=row["s2_paper_id"], doi=row["doi"], isbn=row["isbn"],
                year=row["s2_year"], citation_count=row["s2_citation_count"], abstract=row["abstract"],
            )
            counts["owned_works"] += 1

        suffix = " LIMIT ?" if limit is not None else ""
        params = (max(limit or 0, 0),) if limit is not None else ()
        references = connection.execute('''
            SELECT id, cited_paper_id, cited_doi, cited_title, cited_authors, cited_year,
                   cited_citation_count, citing_item_key, raw_reference_text,
                   context_snippet, citing_chunk_id, similarity_distance, source, s2_status
            FROM global_references ORDER BY id
        ''' + suffix, params).fetchall()
        for row in references:
            if is_short_form_reference(row["raw_reference_text"]):
                continue
            if not (row["cited_paper_id"] or row["cited_doi"] or row["cited_title"]):
                continue
            citing = owned.get(row["citing_item_key"])
            if citing is None:
                citing = _resolve_in_connection(
                    connection, cache, zotero_item_key=row["citing_item_key"]
                )
                owned[row["citing_item_key"]] = citing
            raw = row["raw_reference_text"] or f"legacy-reference:{row['id']}"
            source = row["source"] or "legacy-reference"
            existing_edge = connection.execute('''
                SELECT cited_work_id FROM work_edges
                WHERE citing_work_id = ? AND source = ? AND raw_reference = ? LIMIT 1
            ''', (citing, source, raw)).fetchone()
            cited = int(existing_edge[0]) if existing_edge else _resolve_in_connection(
                connection, cache, s2_paper_id=row["cited_paper_id"], doi=row["cited_doi"],
                title=row["cited_title"], authors=row["cited_authors"], year=row["cited_year"],
                citation_count=row["cited_citation_count"],
            )
            connection.execute('''
                INSERT INTO work_edges
                    (citing_work_id, cited_work_id, source, confidence, raw_reference,
                     citing_chunk_id, context_snippet)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(citing_work_id, cited_work_id, source, raw_reference) DO NOTHING
            ''', (
                citing, cited, source,
                max(0.0, 1.0 - float(row["similarity_distance"] or 0.0)), raw,
                row["citing_chunk_id"], row["context_snippet"],
            ))
            counts["edges"] += 1

        citations = connection.execute('''
            SELECT id, citing_paper_id, citing_doi, citing_title, citing_authors, citing_year,
                   citing_citation_count, cited_item_key, context_snippet,
                   cited_chunk_id, similarity_distance
            FROM global_citations ORDER BY id
        ''' + suffix, params).fetchall()
        for row in citations:
            cited = owned.get(row["cited_item_key"])
            if cited is None:
                cited = _resolve_in_connection(
                    connection, cache, zotero_item_key=row["cited_item_key"]
                )
                owned[row["cited_item_key"]] = cited
            raw = f"legacy-citation:{row['id']}"
            existing_edge = connection.execute('''
                SELECT citing_work_id FROM work_edges
                WHERE cited_work_id = ? AND source = 's2-citation' AND raw_reference = ? LIMIT 1
            ''', (cited, raw)).fetchone()
            citing = int(existing_edge[0]) if existing_edge else _resolve_in_connection(
                connection, cache, s2_paper_id=row["citing_paper_id"], doi=row["citing_doi"],
                title=row["citing_title"], authors=row["citing_authors"], year=row["citing_year"],
                citation_count=row["citing_citation_count"],
            )
            connection.execute('''
                INSERT INTO work_edges
                    (citing_work_id, cited_work_id, source, confidence, raw_reference,
                     citing_chunk_id, context_snippet)
                VALUES (?, ?, 's2-citation', ?, ?, ?, ?)
                ON CONFLICT(citing_work_id, cited_work_id, source, raw_reference) DO NOTHING
            ''', (
                citing, cited, max(0.0, 1.0 - float(row["similarity_distance"] or 0.0)),
                raw, row["cited_chunk_id"], row["context_snippet"],
            ))
            counts["edges"] += 1

        counts["external_works"] = max(len(cache) - len(owned), 0)
        connection.commit()
        return counts
    except Exception:
        connection.rollback()
        raise
    finally:
        connection.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, help="Limit each legacy edge table (for smoke tests).")
    args = parser.parse_args()
    counts = backfill(limit=args.limit)
    print(
        f"Backfilled {counts['owned_works']:,} owned rows, "
        f"{counts['external_works']:,} external identities, {counts['edges']:,} edges."
    )


if __name__ == "__main__":
    main()
