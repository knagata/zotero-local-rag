import sqlite3
import os
import re
import unicodedata
import hashlib
import json
from difflib import SequenceMatcher
from pathlib import Path
from typing import List, Dict, Any, Optional

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB_PATH = os.environ.get("RELATIONS_DB_PATH", os.path.join(ROOT, "data", "relations.db"))
_db_initialized = False


def get_db_connection():
    global _db_initialized
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    if not _db_initialized:
        _init_db(conn)
        _db_initialized = True
    return conn

def _init_db(conn: sqlite3.Connection) -> None:
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS global_citations (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            citing_paper_id TEXT,
            citing_title TEXT,
            citing_year INTEGER,
            context_snippet TEXT,
            cited_item_key TEXT,
            cited_chunk_id TEXT,
            similarity_distance REAL,
            page_hint TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            citing_citation_count INTEGER DEFAULT 0,
            citing_influential_count INTEGER DEFAULT 0,
            chunk_status TEXT DEFAULT 'matched',
            UNIQUE(citing_paper_id, cited_item_key, context_snippet)
        )
    ''')
    
    # Indexes for fast lookup
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_cited_chunk_id ON global_citations(cited_chunk_id)')
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_cited_item_key ON global_citations(cited_item_key)')
    
    # Create global_references table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS global_references (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            cited_paper_id TEXT,
            cited_title TEXT,
            cited_year INTEGER,
            context_snippet TEXT,
            citing_item_key TEXT,
            citing_chunk_id TEXT,
            similarity_distance REAL,
            page_hint TEXT,
            source TEXT DEFAULT 's2',
            raw_reference_text TEXT,
            s2_status TEXT DEFAULT 'matched',
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            cited_citation_count INTEGER DEFAULT 0,
            cited_influential_count INTEGER DEFAULT 0,
            UNIQUE(cited_paper_id, citing_item_key, context_snippet, raw_reference_text)
        )
    ''')

    # Must exist before its backward-compatible ALTER migrations on a fresh DB.
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS item_citation_status (
            item_key TEXT PRIMARY KEY,
            s2_status TEXT,
            last_checked_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    
    # Migrations（既存DBへの後方互換カラム追加）
    _migrations = [
        "ALTER TABLE global_references ADD COLUMN s2_status TEXT DEFAULT 'matched'",
        "ALTER TABLE global_citations ADD COLUMN citing_citation_count INTEGER DEFAULT 0",
        "ALTER TABLE global_citations ADD COLUMN citing_influential_count INTEGER DEFAULT 0",
        "ALTER TABLE global_citations ADD COLUMN chunk_status TEXT DEFAULT 'matched'",
        "ALTER TABLE global_citations ADD COLUMN citing_doi TEXT",
        "ALTER TABLE global_references ADD COLUMN cited_citation_count INTEGER DEFAULT 0",
        "ALTER TABLE global_references ADD COLUMN cited_influential_count INTEGER DEFAULT 0",
        "ALTER TABLE global_references ADD COLUMN cited_doi TEXT",
        "ALTER TABLE item_citation_status ADD COLUMN s2_paper_id TEXT",
        "ALTER TABLE item_citation_status ADD COLUMN s2_year INTEGER",
        "ALTER TABLE item_citation_status ADD COLUMN s2_citation_count INTEGER",
        "ALTER TABLE item_citation_status ADD COLUMN doi TEXT",
        "ALTER TABLE item_citation_status ADD COLUMN isbn TEXT",
        "ALTER TABLE global_citations ADD COLUMN citing_authors TEXT",
        "ALTER TABLE global_references ADD COLUMN cited_authors TEXT",
        "ALTER TABLE item_citation_status ADD COLUMN abstract TEXT",
    ]
    for sql in _migrations:
        try:
            cursor.execute(sql)
        except sqlite3.OperationalError:
            pass
        
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_citing_chunk_id ON global_references(citing_chunk_id)')
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_citing_item_key ON global_references(citing_item_key)')

    cursor.execute('''
        CREATE TABLE IF NOT EXISTS item_summaries (
            item_key    TEXT PRIMARY KEY,
            summary     TEXT NOT NULL,
            model       TEXT,
            updated_at  TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    for sql in (
        "ALTER TABLE item_summaries ADD COLUMN summary_en TEXT",
        "ALTER TABLE item_summaries ADD COLUMN keywords TEXT",
        "ALTER TABLE item_summaries ADD COLUMN chunk_count INTEGER",
        "ALTER TABLE item_summaries ADD COLUMN source_mtime REAL",
    ):
        try:
            cursor.execute(sql)
        except sqlite3.OperationalError:
            pass
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS section_summaries (
            item_key TEXT NOT NULL,
            section_id TEXT NOT NULL,
            chapter TEXT,
            summary TEXT NOT NULL,
            model TEXT,
            chunk_count INTEGER,
            chapter_authors TEXT,
            first_publication_note TEXT,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            PRIMARY KEY(item_key, section_id)
        )
    ''')
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS case_annotations (
            case_id INTEGER PRIMARY KEY AUTOINCREMENT,
            item_key TEXT NOT NULL,
            section_id TEXT,
            description TEXT NOT NULL,
            region TEXT,
            grp TEXT,
            practices TEXT,
            phenomena TEXT,
            period TEXT,
            chunk_id TEXT,
            source_kind TEXT,
            model TEXT,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_case_item ON case_annotations(item_key)")

    # 外部論文（S2 paperId）の概要キャッシュ。
    # status='found' は abstract か tldr のいずれかが取得できた状態、
    # status='none' は S2 に概要情報が無いと確定した状態（再取得を避ける）。
    # 取得エラー時は行を作らず、次回クリックで再試行させる。
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS external_abstracts (
            paper_id    TEXT PRIMARY KEY,
            abstract    TEXT,
            tldr        TEXT,
            status      TEXT,
            updated_at  TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')

    cursor.execute('''
        CREATE TABLE IF NOT EXISTS query_expansion_cache (
            query_hash  TEXT PRIMARY KEY,
            expansions  TEXT NOT NULL,
            created_at  TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')

    cursor.execute('''
        CREATE TABLE IF NOT EXISTS works (
            work_id INTEGER PRIMARY KEY AUTOINCREMENT,
            s2_paper_id TEXT, doi TEXT, isbn TEXT,
            openalex_id TEXT, cinii_crid TEXT, ndl_bibid TEXT,
            zotero_item_key TEXT,
            title TEXT, title_norm TEXT, authors TEXT, year INTEGER, lang TEXT,
            container TEXT, work_type TEXT, citation_count INTEGER, abstract TEXT,
            container_work_id INTEGER REFERENCES works(work_id),
            section_id TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP
        )
    ''')
    cursor.execute("CREATE UNIQUE INDEX IF NOT EXISTS idx_works_s2 ON works(s2_paper_id) WHERE s2_paper_id IS NOT NULL")
    cursor.execute("CREATE UNIQUE INDEX IF NOT EXISTS idx_works_doi ON works(doi) WHERE doi IS NOT NULL")
    cursor.execute("CREATE UNIQUE INDEX IF NOT EXISTS idx_works_isbn ON works(isbn) WHERE isbn IS NOT NULL")
    cursor.execute("CREATE UNIQUE INDEX IF NOT EXISTS idx_works_openalex ON works(openalex_id) WHERE openalex_id IS NOT NULL")
    cursor.execute("CREATE UNIQUE INDEX IF NOT EXISTS idx_works_cinii ON works(cinii_crid) WHERE cinii_crid IS NOT NULL")
    cursor.execute("CREATE UNIQUE INDEX IF NOT EXISTS idx_works_ndl ON works(ndl_bibid) WHERE ndl_bibid IS NOT NULL")
    cursor.execute('''CREATE UNIQUE INDEX IF NOT EXISTS idx_works_zot ON works(zotero_item_key)
                      WHERE zotero_item_key IS NOT NULL AND container_work_id IS NULL''')
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_works_title_norm ON works(title_norm)")

    cursor.execute('''
        CREATE TABLE IF NOT EXISTS work_edges (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            citing_work_id INTEGER NOT NULL REFERENCES works(work_id),
            cited_work_id INTEGER NOT NULL REFERENCES works(work_id),
            source TEXT NOT NULL,
            confidence REAL DEFAULT 1.0,
            raw_reference TEXT,
            citing_chunk_id TEXT,
            context_snippet TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(citing_work_id, cited_work_id, source, raw_reference)
        )
    ''')
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_work_edges_citing ON work_edges(citing_work_id)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_work_edges_cited ON work_edges(cited_work_id)")
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS work_links (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            work_id_a INTEGER NOT NULL REFERENCES works(work_id),
            work_id_b INTEGER NOT NULL REFERENCES works(work_id),
            relation TEXT NOT NULL CHECK(relation IN ('translation_of', 'reprint_of', 'edition_of')),
            confidence REAL DEFAULT 1.0,
            source TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(work_id_a, work_id_b, relation)
        )
    ''')
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS resolver_cache (
            query_hash TEXT NOT NULL,
            source TEXT NOT NULL,
            response_json TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            PRIMARY KEY(query_hash, source)
        )
    ''')

    cursor.execute('''
        CREATE TABLE IF NOT EXISTS reference_review_queue (
            review_id INTEGER PRIMARY KEY AUTOINCREMENT,
            item_key TEXT NOT NULL,
            raw_hash TEXT NOT NULL,
            raw_reference TEXT NOT NULL,
            title TEXT,
            authors_json TEXT,
            year INTEGER,
            doi TEXT,
            isbn TEXT,
            container TEXT,
            lang TEXT,
            work_type TEXT,
            extraction_model TEXT,
            status TEXT NOT NULL DEFAULT 'pending'
                CHECK(status IN ('pending', 'approved', 'rejected')),
            reviewer_note TEXT,
            committed_edge_id INTEGER,
            committed_at TIMESTAMP,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(item_key, raw_hash)
        )
    ''')
    for sql in (
        "ALTER TABLE reference_review_queue ADD COLUMN committed_edge_id INTEGER",
        "ALTER TABLE reference_review_queue ADD COLUMN committed_at TIMESTAMP",
    ):
        try:
            cursor.execute(sql)
        except sqlite3.OperationalError:
            pass
    cursor.execute(
        "CREATE INDEX IF NOT EXISTS idx_reference_review_status "
        "ON reference_review_queue(status, item_key)"
    )
    
    conn.commit()


def _normalize_identifier(value: Optional[str], kind: str) -> Optional[str]:
    normalized = unicodedata.normalize("NFKC", value or "").strip().casefold()
    if not normalized:
        return None
    if kind == "doi":
        normalized = re.sub(r"^(?:https?://(?:dx\.)?doi\.org/|doi:\s*)", "", normalized)
    elif kind == "isbn":
        normalized = re.sub(r"[^0-9x]", "", normalized)
    return normalized or None


def resolve_work(
    *,
    s2_paper_id: Optional[str] = None,
    doi: Optional[str] = None,
    isbn: Optional[str] = None,
    openalex_id: Optional[str] = None,
    cinii_crid: Optional[str] = None,
    ndl_bibid: Optional[str] = None,
    zotero_item_key: Optional[str] = None,
    title: Optional[str] = None,
    authors: Optional[str] = None,
    year: Optional[int] = None,
    lang: Optional[str] = None,
    container: Optional[str] = None,
    work_type: Optional[str] = None,
    citation_count: Optional[int] = None,
    abstract: Optional[str] = None,
    container_work_id: Optional[int] = None,
    section_id: Optional[str] = None,
) -> int:
    """Resolve a canonical work by stable ID or conservative fuzzy matching."""
    values: Dict[str, Any] = {
        "s2_paper_id": _normalize_identifier(s2_paper_id, "s2"),
        "doi": _normalize_identifier(doi, "doi"),
        "isbn": _normalize_identifier(isbn, "isbn"),
        "openalex_id": _normalize_identifier(openalex_id, "openalex"),
        "cinii_crid": _normalize_identifier(cinii_crid, "cinii"),
        "ndl_bibid": _normalize_identifier(ndl_bibid, "ndl"),
        "zotero_item_key": (zotero_item_key or "").strip() or None,
        "title": (title or "").strip() or None,
        "title_norm": normalize_work_title(title),
        "authors": (authors or "").strip() or None,
        "year": year, "lang": lang, "container": container,
        "work_type": work_type, "citation_count": citation_count,
        "abstract": abstract, "container_work_id": container_work_id,
        "section_id": section_id,
    }
    conn = get_db_connection()
    try:
        row = None
        for column in ("s2_paper_id", "doi", "isbn", "openalex_id", "cinii_crid", "ndl_bibid", "zotero_item_key"):
            value = values[column]
            if value is None or (column == "zotero_item_key" and container_work_id is not None):
                continue
            row = conn.execute(f"SELECT * FROM works WHERE {column} = ? LIMIT 1", (value,)).fetchone()
            if row:
                break

        if row is None and values["title_norm"]:
            candidates = conn.execute(
                "SELECT * FROM works WHERE title_norm = ? OR title_norm LIKE ? LIMIT 100",
                (values["title_norm"], f"{values['title_norm'][:80]}%"),
            ).fetchall()
            for candidate in candidates:
                similarity = SequenceMatcher(None, values["title_norm"], candidate["title_norm"] or "").ratio()
                years_comparable = year is not None and candidate["year"] is not None
                year_match = years_comparable and abs(int(year) - int(candidate["year"])) <= 1
                year_conflict = years_comparable and not year_match
                author_match = False
                authors_comparable = bool(authors and candidate["authors"])
                if authors and candidate["authors"]:
                    author_match = SequenceMatcher(
                        None, normalize_work_title(authors), normalize_work_title(candidate["authors"])
                    ).ratio() >= 0.65
                author_conflict = authors_comparable and not author_match
                # A title alone is never sufficient evidence of identity.
                if (
                    similarity >= 0.90 and (year_match or author_match)
                    and not year_conflict and not author_conflict
                ):
                    row = candidate
                    break

        columns = list(values)
        if row is not None:
            assignments = ", ".join(f"{column} = COALESCE({column}, ?)" for column in columns)
            try:
                conn.execute(
                    f"UPDATE works SET {assignments}, updated_at = CURRENT_TIMESTAMP WHERE work_id = ?",
                    [values[column] for column in columns] + [row["work_id"]],
                )
            except sqlite3.IntegrityError:
                # A second stable identifier may already belong to another work.
                # Preserve both identities and update only non-unique metadata.
                conn.rollback()
                unique_columns = {
                    "s2_paper_id", "doi", "isbn", "openalex_id", "cinii_crid",
                    "ndl_bibid", "zotero_item_key",
                }
                safe_columns = [column for column in columns if column not in unique_columns]
                safe_assignments = ", ".join(
                    f"{column} = COALESCE({column}, ?)" for column in safe_columns
                )
                conn.execute(
                    f"UPDATE works SET {safe_assignments}, updated_at = CURRENT_TIMESTAMP WHERE work_id = ?",
                    [values[column] for column in safe_columns] + [row["work_id"]],
                )
            conn.commit()
            return int(row["work_id"])

        placeholders = ",".join("?" for _ in columns)
        cursor = conn.execute(
            f"INSERT INTO works ({','.join(columns)}, updated_at) VALUES ({placeholders}, CURRENT_TIMESTAMP)",
            [values[column] for column in columns],
        )
        conn.commit()
        return int(cursor.lastrowid)
    finally:
        conn.close()


def save_work_edge(
    citing_work_id: int, cited_work_id: int, *, source: str,
    confidence: float = 1.0, raw_reference: Optional[str] = None,
    citing_chunk_id: Optional[str] = None, context_snippet: Optional[str] = None,
) -> int:
    conn = get_db_connection()
    try:
        conn.execute('''
            INSERT INTO work_edges
                (citing_work_id, cited_work_id, source, confidence, raw_reference,
                 citing_chunk_id, context_snippet)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(citing_work_id, cited_work_id, source, raw_reference) DO UPDATE SET
                confidence = MAX(confidence, excluded.confidence),
                citing_chunk_id = COALESCE(excluded.citing_chunk_id, citing_chunk_id),
                context_snippet = COALESCE(excluded.context_snippet, context_snippet)
        ''', (citing_work_id, cited_work_id, source, confidence, raw_reference or "", citing_chunk_id, context_snippet))
        row = conn.execute('''
            SELECT id FROM work_edges
            WHERE citing_work_id = ? AND cited_work_id = ? AND source = ? AND raw_reference = ?
        ''', (citing_work_id, cited_work_id, source, raw_reference or "")).fetchone()
        conn.commit()
        return int(row[0])
    finally:
        conn.close()


def confirm_work_edge(edge_id: int, work_id: Optional[int]) -> bool:
    """Reassign a low-confidence edge to a reviewed work, or reject it with None."""
    conn = get_db_connection()
    try:
        if work_id is None:
            cursor = conn.execute("DELETE FROM work_edges WHERE id = ?", (edge_id,))
        else:
            cursor = conn.execute('''
                UPDATE work_edges SET cited_work_id = ?, confidence = 1.0, source = 'manual'
                WHERE id = ?
            ''', (work_id, edge_id))
        conn.commit()
        return cursor.rowcount > 0
    finally:
        conn.close()


def save_work_link(
    work_id_a: int, work_id_b: int, relation: str, *, confidence: float = 1.0,
    source: Optional[str] = None,
) -> int:
    if relation not in {"translation_of", "reprint_of", "edition_of"}:
        raise ValueError("Unsupported work relation.")
    if work_id_a == work_id_b:
        raise ValueError("A work cannot link to itself.")
    conn = get_db_connection()
    try:
        cursor = conn.execute('''
            INSERT INTO work_links (work_id_a, work_id_b, relation, confidence, source)
            VALUES (?, ?, ?, ?, ?)
            ON CONFLICT(work_id_a, work_id_b, relation) DO UPDATE SET
                confidence = excluded.confidence, source = COALESCE(excluded.source, source)
        ''', (work_id_a, work_id_b, relation, confidence, source))
        conn.commit()
        return int(cursor.lastrowid or 0)
    finally:
        conn.close()


def get_work_cluster(work_id: int) -> List[int]:
    """Return the connected component of translation/reprint/edition links."""
    conn = get_db_connection()
    try:
        rows = conn.execute('''
            WITH RECURSIVE cluster(work_id) AS (
                SELECT ?
                UNION
                SELECT CASE WHEN wl.work_id_a = cluster.work_id THEN wl.work_id_b ELSE wl.work_id_a END
                FROM work_links wl JOIN cluster
                  ON wl.work_id_a = cluster.work_id OR wl.work_id_b = cluster.work_id
            ) SELECT work_id FROM cluster ORDER BY work_id
        ''', (work_id,)).fetchall()
        return [int(row[0]) for row in rows]
    finally:
        conn.close()


def get_canonical_work_id(work_id: int) -> int:
    """Follow directed manifestation links toward the original/earlier work."""
    conn = get_db_connection()
    try:
        row = conn.execute('''
            WITH RECURSIVE origins(work_id, depth) AS (
                SELECT ?, 0
                UNION ALL
                SELECT wl.work_id_b, origins.depth + 1
                FROM work_links wl JOIN origins ON wl.work_id_a = origins.work_id
                WHERE origins.depth < 50
            )
            SELECT work_id FROM origins ORDER BY depth DESC, work_id ASC LIMIT 1
        ''', (work_id,)).fetchone()
        return int(row[0]) if row else work_id
    finally:
        conn.close()


def get_s2_lookup_candidates(item_key: str) -> List[Dict[str, Any]]:
    """Return equivalent manifestations and promoted chapters for S2 lookup fallback."""
    conn = get_db_connection()
    try:
        rows = conn.execute('''
            WITH RECURSIVE equivalent(work_id) AS (
                SELECT work_id FROM works
                WHERE zotero_item_key = ? AND container_work_id IS NULL
                UNION
                SELECT CASE WHEN wl.work_id_a = equivalent.work_id THEN wl.work_id_b ELSE wl.work_id_a END
                FROM work_links wl JOIN equivalent
                  ON wl.work_id_a = equivalent.work_id OR wl.work_id_b = equivalent.work_id
            )
            SELECT work_id, title, authors, year, doi, isbn, work_type, section_id
            FROM works
            WHERE work_id IN (SELECT work_id FROM equivalent)
               OR container_work_id IN (SELECT work_id FROM equivalent)
            ORDER BY CASE WHEN doi IS NOT NULL THEN 0 WHEN isbn IS NOT NULL THEN 1 ELSE 2 END,
                     container_work_id IS NOT NULL, work_id
        ''', (item_key,)).fetchall()
        return [dict(row) for row in rows]
    finally:
        conn.close()


def get_query_expansion(query_hash: str) -> Optional[str]:
    """Return cached query-expansion JSON, or ``None`` on a cache miss."""
    conn = get_db_connection()
    try:
        row = conn.execute(
            "SELECT expansions FROM query_expansion_cache WHERE query_hash = ?",
            (query_hash,),
        ).fetchone()
        return row[0] if row else None
    finally:
        conn.close()


def save_query_expansion(query_hash: str, expansions: str) -> None:
    """Persist a successful query expansion for reuse across MCP sessions."""
    conn = get_db_connection()
    try:
        conn.execute('''
            INSERT INTO query_expansion_cache (query_hash, expansions, created_at)
            VALUES (?, ?, CURRENT_TIMESTAMP)
            ON CONFLICT(query_hash) DO UPDATE SET
                expansions = excluded.expansions,
                created_at = CURRENT_TIMESTAMP
        ''', (query_hash, expansions))
        conn.commit()
    finally:
        conn.close()


def get_resolver_cache(query_hash: str, source: str) -> Optional[str]:
    conn = get_db_connection()
    try:
        row = conn.execute(
            "SELECT response_json FROM resolver_cache WHERE query_hash = ? AND source = ?",
            (query_hash, source),
        ).fetchone()
        return row[0] if row else None
    finally:
        conn.close()


def save_resolver_cache(query_hash: str, source: str, response_json: str) -> None:
    conn = get_db_connection()
    try:
        conn.execute('''
            INSERT INTO resolver_cache (query_hash, source, response_json, created_at)
            VALUES (?, ?, ?, CURRENT_TIMESTAMP)
            ON CONFLICT(query_hash, source) DO UPDATE SET
                response_json = excluded.response_json, created_at = CURRENT_TIMESTAMP
        ''', (query_hash, source, response_json))
        conn.commit()
    finally:
        conn.close()


def stage_reference_candidates(
    item_key: str, model: str, references: List[Dict[str, Any]],
) -> Dict[str, int]:
    """Upsert extracted references into a review-only queue, never the works graph."""
    conn = get_db_connection()
    counts = {"staged": 0, "updated": 0}
    try:
        for reference in references:
            raw = str(reference.get("raw") or "").strip()
            if not raw:
                continue
            raw_hash = hashlib.sha256(raw.encode("utf-8")).hexdigest()
            existed = conn.execute(
                "SELECT 1 FROM reference_review_queue WHERE item_key = ? AND raw_hash = ?",
                (item_key, raw_hash),
            ).fetchone()
            authors = reference.get("authors") or []
            conn.execute('''
                INSERT INTO reference_review_queue
                    (item_key, raw_hash, raw_reference, title, authors_json, year,
                     doi, isbn, container, lang, work_type, extraction_model, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
                ON CONFLICT(item_key, raw_hash) DO UPDATE SET
                    raw_reference = excluded.raw_reference,
                    title = excluded.title,
                    authors_json = excluded.authors_json,
                    year = excluded.year,
                    doi = excluded.doi,
                    isbn = excluded.isbn,
                    container = excluded.container,
                    lang = excluded.lang,
                    work_type = excluded.work_type,
                    extraction_model = excluded.extraction_model,
                    updated_at = CURRENT_TIMESTAMP
            ''', (
                item_key, raw_hash, raw, reference.get("title"),
                json.dumps(authors, ensure_ascii=False), reference.get("year"),
                reference.get("doi"), reference.get("isbn"), reference.get("container"),
                reference.get("lang"), reference.get("type"), model,
            ))
            counts["updated" if existed else "staged"] += 1
        conn.commit()
        return counts
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def get_reference_review_candidates(status: Optional[str] = None) -> List[Dict[str, Any]]:
    conn = get_db_connection()
    try:
        if status is not None and status not in {"pending", "approved", "rejected"}:
            raise ValueError("invalid review status")
        sql = "SELECT * FROM reference_review_queue"
        params: tuple[Any, ...] = ()
        if status is not None:
            sql += " WHERE status = ?"
            params = (status,)
        sql += " ORDER BY item_key, review_id"
        rows = [dict(row) for row in conn.execute(sql, params).fetchall()]
        for row in rows:
            try:
                row["authors"] = json.loads(row.pop("authors_json") or "[]")
            except (json.JSONDecodeError, TypeError):
                row["authors"] = []
        return rows
    finally:
        conn.close()


def set_reference_review_status(review_id: int, status: str, note: str | None = None) -> bool:
    if status not in {"approved", "rejected", "pending"}:
        raise ValueError("invalid review status")
    conn = get_db_connection()
    try:
        cursor = conn.execute('''
            UPDATE reference_review_queue
            SET status = ?, reviewer_note = ?, updated_at = CURRENT_TIMESTAMP
            WHERE review_id = ?
        ''', (status, note, review_id))
        conn.commit()
        return cursor.rowcount > 0
    finally:
        conn.close()


def mark_reference_review_committed(review_id: int, edge_id: int) -> bool:
    conn = get_db_connection()
    try:
        cursor = conn.execute('''
            UPDATE reference_review_queue
            SET committed_edge_id = ?, committed_at = CURRENT_TIMESTAMP,
                updated_at = CURRENT_TIMESTAMP
            WHERE review_id = ? AND status = 'approved' AND committed_edge_id IS NULL
        ''', (edge_id, review_id))
        conn.commit()
        return cursor.rowcount > 0
    finally:
        conn.close()


def update_item_citation_status(
    item_key: str,
    s2_status: str,
    s2_paper_id: Optional[str] = None,
    s2_year: Optional[int] = None,
    s2_citation_count: Optional[int] = None,
    doi: Optional[str] = None,
    isbn: Optional[str] = None,
    abstract: Optional[str] = None,
):
    conn = get_db_connection()
    try:
        cursor = conn.cursor()
        cursor.execute('''
            INSERT INTO item_citation_status
                (item_key, s2_status, last_checked_at, s2_paper_id, s2_year, s2_citation_count, doi, isbn, abstract)
            VALUES (?, ?, CURRENT_TIMESTAMP, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(item_key) DO UPDATE SET
                s2_status         = excluded.s2_status,
                last_checked_at   = CURRENT_TIMESTAMP,
                s2_paper_id       = COALESCE(excluded.s2_paper_id,       s2_paper_id),
                s2_year           = COALESCE(excluded.s2_year,           s2_year),
                s2_citation_count = COALESCE(excluded.s2_citation_count, s2_citation_count),
                doi               = COALESCE(excluded.doi,               doi),
                isbn              = COALESCE(excluded.isbn,              isbn),
                abstract          = COALESCE(excluded.abstract,          abstract)
        ''', (item_key, s2_status, s2_paper_id, s2_year, s2_citation_count, doi, isbn, abstract))
        conn.commit()
    finally:
        conn.close()


def get_item_abstract(item_key: str) -> Optional[str]:
    conn = get_db_connection()
    try:
        cursor = conn.cursor()
        cursor.execute('SELECT abstract FROM item_citation_status WHERE item_key = ?', (item_key,))
        row = cursor.fetchone()
        return row[0] if row else None
    finally:
        conn.close()


def get_item_summary(item_key: str) -> Optional[dict]:
    conn = get_db_connection()
    try:
        cursor = conn.cursor()
        cursor.execute('''
            SELECT summary, summary_en, keywords, model, chunk_count, source_mtime, updated_at
            FROM item_summaries WHERE item_key = ?
        ''', (item_key,))
        row = cursor.fetchone()
        return dict(row) if row else None
    finally:
        conn.close()


def save_item_summary(
    item_key: str, summary: str, model: str = "", *, summary_en: Optional[str] = None,
    keywords: Optional[str] = None, chunk_count: Optional[int] = None,
    source_mtime: Optional[float] = None,
) -> None:
    conn = get_db_connection()
    try:
        cursor = conn.cursor()
        cursor.execute('''
            INSERT INTO item_summaries
                (item_key, summary, summary_en, keywords, model, chunk_count, source_mtime, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
            ON CONFLICT(item_key) DO UPDATE SET
                summary    = excluded.summary,
                summary_en = excluded.summary_en,
                keywords   = excluded.keywords,
                model      = excluded.model,
                chunk_count = excluded.chunk_count,
                source_mtime = excluded.source_mtime,
                updated_at = CURRENT_TIMESTAMP
        ''', (item_key, summary, summary_en, keywords, model, chunk_count, source_mtime))
        conn.commit()
    finally:
        conn.close()


def get_section_summaries(item_key: str) -> List[Dict[str, Any]]:
    conn = get_db_connection()
    try:
        rows = conn.execute('''
            SELECT * FROM section_summaries WHERE item_key = ? ORDER BY section_id
        ''', (item_key,)).fetchall()
        return [dict(row) for row in rows]
    finally:
        conn.close()


def save_section_summary(
    item_key: str, section_id: str, summary: str, *, chapter: Optional[str] = None,
    model: str = "", chunk_count: Optional[int] = None,
    chapter_authors: Optional[str] = None, first_publication_note: Optional[str] = None,
) -> None:
    conn = get_db_connection()
    try:
        conn.execute('''
            INSERT INTO section_summaries
                (item_key, section_id, chapter, summary, model, chunk_count,
                 chapter_authors, first_publication_note, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
            ON CONFLICT(item_key, section_id) DO UPDATE SET
                chapter = excluded.chapter, summary = excluded.summary,
                model = excluded.model, chunk_count = excluded.chunk_count,
                chapter_authors = excluded.chapter_authors,
                first_publication_note = excluded.first_publication_note,
                updated_at = CURRENT_TIMESTAMP
        ''', (
            item_key, section_id, chapter, summary, model, chunk_count,
            chapter_authors, first_publication_note,
        ))
        conn.commit()
    finally:
        conn.close()


def replace_case_annotations(
    item_key: str, section_id: str, cases: List[Dict[str, Any]], *, model: str = "",
) -> None:
    conn = get_db_connection()
    try:
        conn.execute(
            "DELETE FROM case_annotations WHERE item_key = ? AND section_id = ?",
            (item_key, section_id),
        )
        conn.executemany('''
            INSERT INTO case_annotations
                (item_key, section_id, description, region, grp, practices, phenomena,
                 period, chunk_id, source_kind, model, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
        ''', [(
            item_key, section_id, case.get("description") or "", case.get("region"),
            case.get("group") or case.get("grp"),
            "; ".join(case.get("practices") or []) if isinstance(case.get("practices"), list) else case.get("practices"),
            "; ".join(case.get("phenomena") or []) if isinstance(case.get("phenomena"), list) else case.get("phenomena"),
            case.get("period"), case.get("chunk_id"), case.get("source_kind"), model,
        ) for case in cases if case.get("description")])
        conn.commit()
    finally:
        conn.close()


def get_case_annotations(item_key: Optional[str] = None) -> List[Dict[str, Any]]:
    conn = get_db_connection()
    try:
        if item_key:
            rows = conn.execute(
                "SELECT * FROM case_annotations WHERE item_key = ? ORDER BY case_id", (item_key,)
            ).fetchall()
        else:
            rows = conn.execute("SELECT * FROM case_annotations ORDER BY case_id").fetchall()
        return [dict(row) for row in rows]
    finally:
        conn.close()


def update_case_chunk(case_id: int, chunk_id: Optional[str]) -> None:
    conn = get_db_connection()
    try:
        conn.execute(
            "UPDATE case_annotations SET chunk_id = ?, updated_at = CURRENT_TIMESTAMP WHERE case_id = ?",
            (chunk_id, case_id),
        )
        conn.commit()
    finally:
        conn.close()

def get_external_abstract(paper_id: str) -> Optional[dict]:
    """外部論文の概要キャッシュを返す。未取得なら None。"""
    conn = get_db_connection()
    try:
        cursor = conn.cursor()
        cursor.execute(
            'SELECT abstract, tldr, status, updated_at FROM external_abstracts WHERE paper_id = ?',
            (paper_id,),
        )
        row = cursor.fetchone()
        if not row:
            return None
        return {
            "abstract":   row[0],
            "tldr":       row[1],
            "status":     row[2],
            "updated_at": row[3],
        }
    finally:
        conn.close()


def save_external_abstract(paper_id: str, abstract: Optional[str], tldr: Optional[str], status: str) -> None:
    """外部論文の概要をキャッシュに保存する。"""
    conn = get_db_connection()
    try:
        cursor = conn.cursor()
        cursor.execute('''
            INSERT INTO external_abstracts (paper_id, abstract, tldr, status, updated_at)
            VALUES (?, ?, ?, ?, CURRENT_TIMESTAMP)
            ON CONFLICT(paper_id) DO UPDATE SET
                abstract   = excluded.abstract,
                tldr       = excluded.tldr,
                status     = excluded.status,
                updated_at = CURRENT_TIMESTAMP
        ''', (paper_id, abstract, tldr, status))
        conn.commit()
    finally:
        conn.close()


def get_item_citation_status(item_key: str) -> Optional[str]:
    conn = get_db_connection()
    try:
        cursor = conn.cursor()
        cursor.execute('SELECT s2_status FROM item_citation_status WHERE item_key = ?', (item_key,))
        row = cursor.fetchone()
        return row['s2_status'] if row else None
    finally:
        conn.close()

def insert_citation(
    citing_paper_id: str,
    citing_title: str,
    citing_year: Optional[int],
    context_snippet: str,
    cited_item_key: str,
    cited_chunk_id: Optional[str],
    similarity_distance: Optional[float],
    page_hint: Optional[str],
    citing_citation_count: int = 0,
    citing_influential_count: int = 0,
    chunk_status: str = 'matched',
    citing_doi: Optional[str] = None,
    citing_authors: Optional[str] = None,
):
    conn = get_db_connection()
    cursor = conn.cursor()
    try:
        cursor.execute('''
            INSERT OR REPLACE INTO global_citations
            (citing_paper_id, citing_title, citing_year, context_snippet, cited_item_key,
             cited_chunk_id, similarity_distance, page_hint,
             citing_citation_count, citing_influential_count, chunk_status, citing_doi,
             citing_authors)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (citing_paper_id, citing_title, citing_year, context_snippet, cited_item_key,
              cited_chunk_id, similarity_distance, page_hint,
              citing_citation_count, citing_influential_count, chunk_status, citing_doi,
              citing_authors))
        conn.commit()
    finally:
        conn.close()

def get_citations_for_chunk(chunk_id: str, limit: int = 3) -> List[Dict[str, Any]]:
    conn = get_db_connection()
    try:
        cursor = conn.cursor()
        query = '''
            SELECT * FROM global_citations
            WHERE cited_chunk_id = ?
            ORDER BY citing_influential_count DESC, citing_citation_count DESC, similarity_distance ASC
        '''
        params = [chunk_id]
        if limit is not None and limit > 0:
            query += " LIMIT ?"
            params.append(limit)
        cursor.execute(query, tuple(params))
        rows = cursor.fetchall()
        return [dict(row) for row in rows]
    finally:
        conn.close()

def get_cited_chunks_for_item(item_key: str) -> List[Dict[str, Any]]:
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute('''
        SELECT cited_chunk_id, COUNT(*) as citation_count, MIN(similarity_distance) as best_distance
        FROM global_citations
        WHERE cited_item_key = ?
          AND cited_chunk_id IS NOT NULL
        GROUP BY cited_chunk_id
        ORDER BY citation_count DESC, best_distance ASC
    ''', (item_key,))
    rows = cursor.fetchall()
    conn.close()
    return [dict(row) for row in rows]

def insert_reference(cited_paper_id: Optional[str], cited_title: Optional[str], cited_year: Optional[int], context_snippet: Optional[str], citing_item_key: str, citing_chunk_id: Optional[str], similarity_distance: Optional[float], page_hint: Optional[str] = None, source: str = 's2', raw_reference_text: Optional[str] = None, s2_status: str = 'matched', cited_citation_count: int = 0, cited_influential_count: int = 0, cited_doi: Optional[str] = None, cited_authors: Optional[str] = None):
    conn = get_db_connection()
    try:
        cursor = conn.cursor()
        cursor.execute('''
            INSERT OR REPLACE INTO global_references
            (cited_paper_id, cited_title, cited_year, context_snippet, citing_item_key, citing_chunk_id,
             similarity_distance, page_hint, source, raw_reference_text, s2_status,
             cited_citation_count, cited_influential_count, cited_doi, cited_authors)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (cited_paper_id, cited_title, cited_year, context_snippet, citing_item_key, citing_chunk_id,
              similarity_distance, page_hint, source, raw_reference_text, s2_status,
              cited_citation_count, cited_influential_count, cited_doi, cited_authors))
        conn.commit()
    finally:
        conn.close()

def get_references_for_chunk(chunk_id: str) -> List[Dict[str, Any]]:
    conn = get_db_connection()
    try:
        cursor = conn.cursor()
        cursor.execute('''
            SELECT id, cited_paper_id, cited_title, cited_year, context_snippet,
                   citing_item_key, citing_chunk_id, similarity_distance, page_hint,
                   source, raw_reference_text, s2_status, created_at,
                   cited_citation_count, cited_influential_count
            FROM global_references
            WHERE citing_chunk_id = ?
            ORDER BY cited_influential_count DESC, cited_citation_count DESC, similarity_distance ASC
        ''', (chunk_id,))
        rows = cursor.fetchall()
        return [dict(row) for row in rows]
    finally:
        conn.close()

def get_references_for_item(item_key: str) -> List[Dict[str, Any]]:
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute('''
        SELECT citing_chunk_id, COUNT(*) as reference_count, MIN(similarity_distance) as best_distance
        FROM global_references 
        WHERE citing_item_key = ? 
        GROUP BY citing_chunk_id
        ORDER BY reference_count DESC, best_distance ASC
    ''', (item_key,))
    rows = cursor.fetchall()
    conn.close()
    return [dict(row) for row in rows]


def normalize_work_title(title: Optional[str]) -> str:
    """Normalize a title for conservative owned/unowned comparison."""
    text = unicodedata.normalize("NFKC", title or "").casefold()
    text = re.sub(r"[^\w]+", "", text, flags=re.UNICODE)
    return text


def get_owned_work_identifiers() -> Dict[str, Any]:
    """Return identifiers for owned works and every equivalent manifestation."""
    conn = get_db_connection()
    try:
        rows = conn.execute('''
            WITH RECURSIVE owned(work_id) AS (
                SELECT work_id FROM works WHERE zotero_item_key IS NOT NULL
                UNION
                SELECT CASE WHEN wl.work_id_a = owned.work_id THEN wl.work_id_b ELSE wl.work_id_a END
                FROM work_links wl JOIN owned
                  ON wl.work_id_a = owned.work_id OR wl.work_id_b = owned.work_id
            )
            SELECT s2_paper_id, doi, isbn, title_norm, authors, year FROM works
            WHERE work_id IN (SELECT work_id FROM owned)
        ''').fetchall()
        if not rows:
            rows = conn.execute(
                "SELECT s2_paper_id, doi, isbn, NULL AS title_norm, NULL AS authors, "
                "s2_year AS year FROM item_citation_status"
            ).fetchall()
        return {
            "s2": {str(row[0]).strip().casefold() for row in rows if row[0]},
            "doi": {_normalize_identifier(row[1], "doi") for row in rows if row[1]},
            "isbn": {_normalize_identifier(row[2], "isbn") for row in rows if row[2]},
            "title": {str(row[3]) for row in rows if row[3]},
            "records": [
                {"title_norm": str(row[3]), "authors": row[4], "year": row[5]}
                for row in rows if row[3]
            ],
        }
    finally:
        conn.close()


def is_owned_work(
    *,
    s2_paper_id: Optional[str] = None,
    doi: Optional[str] = None,
    isbn: Optional[str] = None,
    title: Optional[str] = None,
    authors: Optional[str] = None,
    year: Optional[int] = None,
    identifiers: Optional[Dict[str, Any]] = None,
    normalized_titles: Optional[set[str]] = None,
) -> bool:
    """Check ownership using stable IDs or title plus corroborating metadata."""
    owned = identifiers if identifiers is not None else get_owned_work_identifiers()
    candidates = {
        "s2": (s2_paper_id or "").strip().casefold(),
        "doi": _normalize_identifier(doi, "doi") or "",
        "isbn": _normalize_identifier(isbn, "isbn") or "",
    }
    if any(value and value in owned.get(kind, set()) for kind, value in candidates.items()):
        return True
    title_key = normalize_work_title(title)
    if not title_key:
        return False
    for record in owned.get("records", []):
        if title_key != record.get("title_norm"):
            continue
        years_comparable = year is not None and record.get("year") is not None
        year_match = years_comparable and abs(int(year) - int(record["year"])) <= 1
        year_conflict = years_comparable and not year_match
        author_match = False
        authors_comparable = bool(authors and record.get("authors"))
        if authors and record.get("authors"):
            author_match = SequenceMatcher(
                None, normalize_work_title(authors), normalize_work_title(record["authors"])
            ).ratio() >= 0.65
        author_conflict = authors_comparable and not author_match
        if (year_match or author_match) and not year_conflict and not author_conflict:
            return True
    return False


def aggregate_unowned_works(
    scope_item_keys: Optional[List[str]] = None,
    *,
    direction: str = "references",
    min_citing_items: int = 2,
    limit: int = 100,
    normalized_owned_titles: Optional[set[str]] = None,
) -> List[Dict[str, Any]]:
    """Aggregate external works and exclude owned equivalent manifestations."""
    if direction not in {"references", "citations"}:
        raise ValueError("direction must be 'references' or 'citations'.")
    if min_citing_items < 1:
        raise ValueError("min_citing_items must be at least 1.")
    if limit < 1:
        return []

    if direction == "references":
        table = "global_references"
        s2_col, doi_col = "cited_paper_id", "cited_doi"
        title_col, year_col, authors_col = "cited_title", "cited_year", "cited_authors"
        item_col, citation_count_col = "citing_item_key", "cited_citation_count"
    else:
        table = "global_citations"
        s2_col, doi_col = "citing_paper_id", "citing_doi"
        title_col, year_col, authors_col = "citing_title", "citing_year", "citing_authors"
        item_col, citation_count_col = "cited_item_key", "citing_citation_count"

    identity_sql = f'''CASE
        WHEN NULLIF(TRIM({s2_col}), '') IS NOT NULL THEN 's2:' || LOWER(TRIM({s2_col}))
        WHEN NULLIF(TRIM({doi_col}), '') IS NOT NULL THEN 'doi:' || LOWER(TRIM({doi_col}))
        ELSE 'title:' || LOWER(TRIM({title_col})) END'''
    where = [f"NULLIF(TRIM({title_col}), '') IS NOT NULL"]
    params: list[Any] = []
    if scope_item_keys:
        placeholders = ",".join("?" for _ in scope_item_keys)
        where.append(f"{item_col} IN ({placeholders})")
        params.extend(scope_item_keys)

    sql = f'''
        SELECT {identity_sql} AS identity_key,
               MAX({title_col}) AS title,
               MAX({authors_col}) AS authors,
               MAX({year_col}) AS year,
               MAX({doi_col}) AS doi,
               MAX({s2_col}) AS s2_paper_id,
               COUNT(DISTINCT {item_col}) AS adjacent_item_count,
               GROUP_CONCAT(DISTINCT {item_col}) AS adjacent_item_keys,
               MAX(COALESCE({citation_count_col}, 0)) AS total_citation_count,
               COUNT(*) AS mention_count
        FROM {table}
        WHERE {' AND '.join(where)}
        GROUP BY identity_key
        HAVING COUNT(DISTINCT {item_col}) >= ?
        ORDER BY adjacent_item_count DESC, mention_count DESC,
                 total_citation_count DESC, title COLLATE NOCASE ASC
    '''
    params.append(min_citing_items)
    conn = get_db_connection()
    try:
        rows = [dict(row) for row in conn.execute(sql, params).fetchall()]
    finally:
        conn.close()

    identifiers = get_owned_work_identifiers()
    results: List[Dict[str, Any]] = []
    for row in rows:
        if is_owned_work(
            s2_paper_id=row.get("s2_paper_id"),
            doi=row.get("doi"),
            title=row.get("title"),
            authors=row.get("authors"),
            year=row.get("year"),
            identifiers=identifiers,
            normalized_titles=normalized_owned_titles,
        ):
            continue
        if not row.get("s2_paper_id") and not row.get("doi"):
            row["identity_status"] = (
                "corroborated_metadata" if row.get("authors") or row.get("year")
                else "title_only_unverified"
            )
        keys = [key for key in (row.pop("adjacent_item_keys", "") or "").split(",") if key]
        row["adjacent_item_keys"] = keys[:5]
        results.append(row)
        if len(results) >= limit:
            break
    return results


def _reference_identity_sql(alias: str) -> str:
    return f'''CASE
        WHEN NULLIF(TRIM({alias}.cited_paper_id), '') IS NOT NULL
            THEN 's2:' || LOWER(TRIM({alias}.cited_paper_id))
        WHEN NULLIF(TRIM({alias}.cited_doi), '') IS NOT NULL
            THEN 'doi:' || LOWER(TRIM({alias}.cited_doi))
        ELSE 'title:' || LOWER(TRIM({alias}.cited_title)) END'''


def get_coupling_pairs(item_key: str, limit: int = 100) -> List[Dict[str, Any]]:
    """Rank owned items sharing outgoing references with ``item_key``."""
    identity = _reference_identity_sql("r")
    candidate_identity = _reference_identity_sql("c")
    conn = get_db_connection()
    try:
        rows = conn.execute(
            f'''
            WITH target_refs AS (
                SELECT DISTINCT {identity} AS work_key
                FROM global_references r
                WHERE r.citing_item_key = ?
                  AND NULLIF(TRIM(r.cited_title), '') IS NOT NULL
            ), candidate_refs AS (
                SELECT DISTINCT c.citing_item_key AS item_key,
                       {candidate_identity} AS work_key
                FROM global_references c
                WHERE c.citing_item_key <> ?
                  AND NULLIF(TRIM(c.cited_title), '') IS NOT NULL
            )
            SELECT candidate_refs.item_key,
                   COUNT(*) AS shared_reference_count
            FROM candidate_refs
            JOIN target_refs USING (work_key)
            GROUP BY candidate_refs.item_key
            ORDER BY shared_reference_count DESC, candidate_refs.item_key ASC
            LIMIT ?
            ''',
            (item_key, item_key, max(limit, 0)),
        ).fetchall()
        return [dict(row) for row in rows]
    finally:
        conn.close()


def get_cocitation_pairs(item_key: str, limit: int = 100) -> List[Dict[str, Any]]:
    """Rank owned items cited alongside ``item_key`` by the same external works."""
    conn = get_db_connection()
    try:
        rows = conn.execute(
            '''
            WITH target_citers AS (
                SELECT DISTINCT citing_paper_id
                FROM global_citations
                WHERE cited_item_key = ?
                  AND NULLIF(TRIM(citing_paper_id), '') IS NOT NULL
            ), candidate_pairs AS (
                SELECT DISTINCT c.cited_item_key AS item_key, c.citing_paper_id
                FROM global_citations c
                JOIN target_citers t ON t.citing_paper_id = c.citing_paper_id
                WHERE c.cited_item_key <> ?
            )
            SELECT item_key, COUNT(*) AS shared_citer_count
            FROM candidate_pairs
            GROUP BY item_key
            ORDER BY shared_citer_count DESC, item_key ASC
            LIMIT ?
            ''',
            (item_key, item_key, max(limit, 0)),
        ).fetchall()
        return [dict(row) for row in rows]
    finally:
        conn.close()


def get_network_item_keys() -> List[str]:
    """Return all owned item keys represented in citation or status tables."""
    conn = get_db_connection()
    try:
        rows = conn.execute('''
            SELECT item_key FROM item_citation_status
            UNION SELECT citing_item_key FROM global_references
            UNION SELECT cited_item_key FROM global_citations
        ''').fetchall()
        return sorted(row[0] for row in rows if row[0])
    finally:
        conn.close()


def get_case_overlap_pairs(item_key: str, limit: int = 100) -> List[Dict[str, Any]]:
    """Rank items by overlap of structured case region/practice/phenomenon terms."""
    conn = get_db_connection()
    try:
        rows = conn.execute('''
            SELECT item_key, region, practices, phenomena FROM case_annotations
            WHERE item_key = ? OR item_key IN (
                SELECT DISTINCT item_key FROM case_annotations WHERE item_key <> ?
            )
        ''', (item_key, item_key)).fetchall()
    finally:
        conn.close()
    terms: Dict[str, set[str]] = {}
    for row in rows:
        bucket = terms.setdefault(row["item_key"], set())
        for value in (row["region"], row["practices"], row["phenomena"]):
            bucket.update(
                token.strip().casefold() for token in re.split(r"[;,、/|]", value or "") if token.strip()
            )
    target = terms.get(item_key, set())
    if not target:
        return []
    output = []
    for candidate, candidate_terms in terms.items():
        if candidate == item_key:
            continue
        shared = sorted(target & candidate_terms)
        if shared:
            output.append({
                "item_key": candidate, "shared_case_terms": shared,
                "case_overlap_score": len(shared) / max(len(target | candidate_terms), 1),
            })
    return sorted(
        output, key=lambda row: (-row["case_overlap_score"], -len(row["shared_case_terms"]), row["item_key"])
    )[: max(limit, 0)]

def purge_removed_items(current_item_keys: set[str]) -> dict[str, int]:
    """Zoteroから削除されたアイテムキーに関連するDBレコードを削除する。

    Returns:
        削除件数の辞書 {"item_citation_status": n, "global_citations": n, "global_references": n}
    """
    conn = get_db_connection()
    try:
        cursor = conn.cursor()

        cursor.execute("SELECT item_key FROM item_citation_status")
        db_keys = {row[0] for row in cursor.fetchall()}
        removed_keys = db_keys - current_item_keys

        counts: dict[str, int] = {
            "item_citation_status": 0, "global_citations": 0, "global_references": 0,
            "item_summaries": 0, "section_summaries": 0, "case_annotations": 0,
            "works": 0, "work_edges": 0, "work_links": 0,
        }
        if not removed_keys:
            return counts

        placeholders = ",".join("?" * len(removed_keys))
        params = list(removed_keys)

        cursor.execute(
            f"DELETE FROM item_citation_status WHERE item_key IN ({placeholders})", params
        )
        counts["item_citation_status"] = cursor.rowcount

        cursor.execute(
            f"DELETE FROM global_citations WHERE cited_item_key IN ({placeholders})", params
        )
        counts["global_citations"] = cursor.rowcount

        cursor.execute(
            f"DELETE FROM global_references WHERE citing_item_key IN ({placeholders})", params
        )
        counts["global_references"] = cursor.rowcount

        for table in ("item_summaries", "section_summaries", "case_annotations"):
            cursor.execute(
                f"DELETE FROM {table} WHERE item_key IN ({placeholders})", params
            )
            counts[table] = cursor.rowcount

        work_rows = cursor.execute(
            f"SELECT work_id FROM works WHERE zotero_item_key IN ({placeholders})", params
        ).fetchall()
        work_ids = [int(row[0]) for row in work_rows]
        if work_ids:
            work_placeholders = ",".join("?" * len(work_ids))
            cursor.execute(
                f"DELETE FROM work_edges WHERE citing_work_id IN ({work_placeholders}) "
                f"OR cited_work_id IN ({work_placeholders})",
                [*work_ids, *work_ids],
            )
            counts["work_edges"] = cursor.rowcount
            cursor.execute(
                f"DELETE FROM work_links WHERE work_id_a IN ({work_placeholders}) "
                f"OR work_id_b IN ({work_placeholders})",
                [*work_ids, *work_ids],
            )
            counts["work_links"] = cursor.rowcount
            cursor.execute(f"DELETE FROM works WHERE work_id IN ({work_placeholders})", work_ids)
            counts["works"] = cursor.rowcount

        conn.commit()
        return counts
    finally:
        conn.close()


def get_skipped_epub_refs(item_key: str, statuses: tuple = ('skipped',)) -> List[Dict[str, Any]]:
    """指定 s2_status の EPUB 参照行を返す（resume-skipped / エラー再試行用）。"""
    conn = get_db_connection()
    try:
        cursor = conn.cursor()
        placeholders = ",".join("?" * len(statuses))
        cursor.execute(
            "SELECT id, raw_reference_text, context_snippet, citing_chunk_id, similarity_distance"
            f" FROM global_references WHERE citing_item_key = ? AND s2_status IN ({placeholders})"
            " ORDER BY id ASC",
            (item_key, *statuses),
        )
        return [dict(row) for row in cursor.fetchall()]
    finally:
        conn.close()


def get_items_with_skipped_epub_refs(statuses: tuple = ('skipped',)) -> List[str]:
    """指定 s2_status の参照を持つアイテムキー一覧を返す。"""
    conn = get_db_connection()
    try:
        cursor = conn.cursor()
        placeholders = ",".join("?" * len(statuses))
        cursor.execute(
            f"SELECT DISTINCT citing_item_key FROM global_references WHERE s2_status IN ({placeholders})",
            statuses,
        )
        return [row[0] for row in cursor.fetchall()]
    finally:
        conn.close()


def update_reference_s2_data(
    ref_id: int,
    cited_paper_id: Optional[str],
    cited_title: Optional[str],
    cited_year: Optional[int],
    cited_citation_count: int,
    cited_influential_count: int,
    cited_doi: Optional[str],
    cited_authors: Optional[str],
    s2_status: str,
) -> None:
    """global_references の既存行を S2 解決結果で更新する（resume-skipped 用）。"""
    conn = get_db_connection()
    try:
        cursor = conn.cursor()
        cursor.execute(
            """UPDATE global_references
               SET cited_paper_id=?, cited_title=?, cited_year=?,
                   cited_citation_count=?, cited_influential_count=?,
                   cited_doi=?, cited_authors=?, s2_status=?
               WHERE id=?""",
            (cited_paper_id, cited_title, cited_year,
             cited_citation_count, cited_influential_count,
             cited_doi, cited_authors, s2_status, ref_id),
        )
        conn.commit()
    finally:
        conn.close()
