import sqlite3
import os
import re
import unicodedata
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
                year_ok = year is None or candidate["year"] is None or abs(int(year) - int(candidate["year"])) <= 1
                author_ok = True
                if authors and candidate["authors"]:
                    author_ok = SequenceMatcher(
                        None, normalize_work_title(authors), normalize_work_title(candidate["authors"])
                    ).ratio() >= 0.65
                if similarity >= 0.90 and year_ok and author_ok:
                    row = candidate
                    break

        columns = list(values)
        if row is not None:
            assignments = ", ".join(f"{column} = COALESCE({column}, ?)" for column in columns)
            conn.execute(
                f"UPDATE works SET {assignments}, updated_at = CURRENT_TIMESTAMP WHERE work_id = ?",
                [values[column] for column in columns] + [row["work_id"]],
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
        cursor = conn.execute('''
            INSERT INTO work_edges
                (citing_work_id, cited_work_id, source, confidence, raw_reference,
                 citing_chunk_id, context_snippet)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(citing_work_id, cited_work_id, source, raw_reference) DO UPDATE SET
                confidence = MAX(confidence, excluded.confidence),
                citing_chunk_id = COALESCE(excluded.citing_chunk_id, citing_chunk_id),
                context_snippet = COALESCE(excluded.context_snippet, context_snippet)
        ''', (citing_work_id, cited_work_id, source, confidence, raw_reference or "", citing_chunk_id, context_snippet))
        conn.commit()
        return int(cursor.lastrowid or 0)
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
        cursor.execute('SELECT summary, model, updated_at FROM item_summaries WHERE item_key = ?', (item_key,))
        row = cursor.fetchone()
        return {"summary": row[0], "model": row[1], "updated_at": row[2]} if row else None
    finally:
        conn.close()


def save_item_summary(item_key: str, summary: str, model: str = "") -> None:
    conn = get_db_connection()
    try:
        cursor = conn.cursor()
        cursor.execute('''
            INSERT INTO item_summaries (item_key, summary, model, updated_at)
            VALUES (?, ?, ?, CURRENT_TIMESTAMP)
            ON CONFLICT(item_key) DO UPDATE SET
                summary    = excluded.summary,
                model      = excluded.model,
                updated_at = CURRENT_TIMESTAMP
        ''', (item_key, summary, model))
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


def get_owned_work_identifiers() -> Dict[str, set[str]]:
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
            SELECT s2_paper_id, doi, isbn, title_norm FROM works
            WHERE work_id IN (SELECT work_id FROM owned)
        ''').fetchall()
        if not rows:
            rows = conn.execute(
                "SELECT s2_paper_id, doi, isbn, NULL AS title_norm FROM item_citation_status"
            ).fetchall()
        return {
            "s2": {str(row[0]).strip().casefold() for row in rows if row[0]},
            "doi": {_normalize_identifier(row[1], "doi") for row in rows if row[1]},
            "isbn": {_normalize_identifier(row[2], "isbn") for row in rows if row[2]},
            "title": {str(row[3]) for row in rows if row[3]},
        }
    finally:
        conn.close()


def is_owned_work(
    *,
    s2_paper_id: Optional[str] = None,
    doi: Optional[str] = None,
    isbn: Optional[str] = None,
    title: Optional[str] = None,
    identifiers: Optional[Dict[str, set[str]]] = None,
    normalized_titles: Optional[set[str]] = None,
) -> bool:
    """Check ownership using stable identifiers, then exact normalized title."""
    owned = identifiers if identifiers is not None else get_owned_work_identifiers()
    candidates = {
        "s2": (s2_paper_id or "").strip().casefold(),
        "doi": _normalize_identifier(doi, "doi") or "",
        "isbn": _normalize_identifier(isbn, "isbn") or "",
    }
    if any(value and value in owned.get(kind, set()) for kind, value in candidates.items()):
        return True
    title_key = normalize_work_title(title)
    owned_titles = set(normalized_titles or ()) | owned.get("title", set())
    return bool(title_key and title_key in owned_titles)


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
            identifiers=identifiers,
            normalized_titles=normalized_owned_titles,
        ):
            continue
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

        counts: dict[str, int] = {"item_citation_status": 0, "global_citations": 0, "global_references": 0}
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
