import sqlite3
import os
import re
import unicodedata
import hashlib
import json
import threading
from difflib import SequenceMatcher
from typing import List, Dict, Any, Iterable, Optional

try:
    from .reference_text import strip_unicode_format_characters
    from .db_schema import add_column
    from .cache_repository import CacheRepository, SummaryRepository
    from .artifact_repository import ArtifactRepository
    from .structure_repository import StructureRepository
except ImportError:  # pragma: no cover - direct src/ entrypoints
    from reference_text import strip_unicode_format_characters
    from db_schema import add_column
    from cache_repository import CacheRepository, SummaryRepository
    from artifact_repository import ArtifactRepository
    from structure_repository import StructureRepository

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB_PATH = os.environ.get("RELATIONS_DB_PATH", os.path.join(ROOT, "data", "relations.db"))
_db_initialized = False
_initialized_db_path: str | None = None
_db_init_lock = threading.Lock()


def _cache_repository() -> CacheRepository:
    return CacheRepository(get_db_connection)


def _summary_repository() -> SummaryRepository:
    return SummaryRepository(get_db_connection)


def _artifact_repository() -> ArtifactRepository:
    return ArtifactRepository(
        get_db_connection,
        artifact_types=_ARTIFACT_TYPES,
        artifact_statuses=_ARTIFACT_STATUSES,
    )


def _structure_repository() -> StructureRepository:
    return StructureRepository(get_db_connection, statuses=_STRUCTURE_STATUSES)


def get_db_connection():
    global _db_initialized, _initialized_db_path
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = sqlite3.connect(DB_PATH, timeout=30)
    try:
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA busy_timeout = 30000")
        conn.execute("PRAGMA foreign_keys = ON")
        if not _db_initialized or _initialized_db_path != DB_PATH:
            with _db_init_lock:
                if not _db_initialized or _initialized_db_path != DB_PATH:
                    # journal_mode is persistent database state. Running this on
                    # several first-use connections before taking the init lock
                    # makes those connections contend with the schema migration.
                    conn.execute("PRAGMA journal_mode = WAL")
                    _init_db(conn)
                    _db_initialized = True
                    _initialized_db_path = DB_PATH
        return conn
    except BaseException:
        conn.close()
        raise

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
        add_column(cursor, sql)

    # SQLite considers every NULL distinct in a UNIQUE constraint.  The
    # original table-level constraints therefore let retries accumulate rows
    # whenever S2 did not provide a context or a paper id.  These expression
    # indexes make the intended relation identity explicit: absent optional
    # values are a stable empty value, rather than a new relation on each run.
    #
    # A pre-existing database can already contain such duplicates.  Compact
    # only exact logical duplicates before adding the index (keep the oldest
    # row, rather than rebuilding or dropping either table).  This migration is
    # deliberately narrow and is safe to run repeatedly.
    citation_work = """
        COALESCE(
            NULLIF(citing_paper_id, ''), NULLIF(citing_doi, ''),
            CASE WHEN NULLIF(TRIM(citing_title), '') IS NOT NULL THEN
                'title:' || LOWER(TRIM(citing_title)) || ':' ||
                COALESCE(CAST(citing_year AS TEXT), '') || ':' ||
                LOWER(COALESCE(citing_authors, ''))
            END, ''
        )
    """
    reference_work = """
        COALESCE(
            NULLIF(cited_paper_id, ''), NULLIF(cited_doi, ''),
            CASE WHEN NULLIF(TRIM(cited_title), '') IS NOT NULL THEN
                'title:' || LOWER(TRIM(cited_title)) || ':' ||
                COALESCE(CAST(cited_year AS TEXT), '') || ':' ||
                LOWER(COALESCE(cited_authors, ''))
            END, ''
        )
    """
    citation_identity = (
        citation_work, "COALESCE(cited_item_key, '')",
        "COALESCE(context_snippet, '')",
    )
    reference_identity = (
        reference_work, "COALESCE(citing_item_key, '')",
        "COALESCE(context_snippet, '')", "COALESCE(raw_reference_text, '')",
    )
    citation_eligible = (
        f"(({citation_work}) <> '' OR COALESCE(context_snippet, '') <> '')"
    )
    reference_eligible = (
        f"(({reference_work}) <> '' OR COALESCE(context_snippet, '') <> '' "
        "OR COALESCE(raw_reference_text, '') <> '')"
    )
    _deduplicate_relation_rows(
        cursor, "global_citations", citation_identity, citation_eligible,
    )
    _deduplicate_relation_rows(
        cursor, "global_references", reference_identity, reference_eligible,
    )
    # Replace the uncommitted first-generation expression indexes if a process
    # initialized the database before this stronger bibliographic identity was
    # installed.
    cursor.execute("DROP INDEX IF EXISTS uq_global_citations_identity")
    cursor.execute("DROP INDEX IF EXISTS uq_global_references_identity")
    cursor.execute('''
        CREATE UNIQUE INDEX uq_global_citations_identity
        ON global_citations (
            COALESCE(
                NULLIF(citing_paper_id, ''), NULLIF(citing_doi, ''),
                CASE WHEN NULLIF(TRIM(citing_title), '') IS NOT NULL THEN
                    'title:' || LOWER(TRIM(citing_title)) || ':' ||
                    COALESCE(CAST(citing_year AS TEXT), '') || ':' ||
                    LOWER(COALESCE(citing_authors, ''))
                END, ''
            ),
            COALESCE(cited_item_key, ''),
            COALESCE(context_snippet, '')
        )
        WHERE COALESCE(NULLIF(citing_paper_id, ''), NULLIF(citing_doi, ''),
                      NULLIF(TRIM(citing_title), ''), '') <> ''
           OR COALESCE(context_snippet, '') <> ''
    ''')
    cursor.execute('''
        CREATE UNIQUE INDEX uq_global_references_identity
        ON global_references (
            COALESCE(
                NULLIF(cited_paper_id, ''), NULLIF(cited_doi, ''),
                CASE WHEN NULLIF(TRIM(cited_title), '') IS NOT NULL THEN
                    'title:' || LOWER(TRIM(cited_title)) || ':' ||
                    COALESCE(CAST(cited_year AS TEXT), '') || ':' ||
                    LOWER(COALESCE(cited_authors, ''))
                END, ''
            ),
            COALESCE(citing_item_key, ''),
            COALESCE(context_snippet, ''),
            COALESCE(raw_reference_text, '')
        )
        WHERE COALESCE(NULLIF(cited_paper_id, ''), NULLIF(cited_doi, ''),
                      NULLIF(TRIM(cited_title), ''), '') <> ''
           OR COALESCE(context_snippet, '') <> ''
           OR COALESCE(raw_reference_text, '') <> ''
    ''')
        
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
        add_column(cursor, sql)
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
        CREATE TABLE IF NOT EXISTS insight_generation_status (
            item_key TEXT NOT NULL,
            kind TEXT NOT NULL CHECK(kind = 'sections'),
            status TEXT NOT NULL CHECK(status IN ('processed_empty', 'available')),
            row_count INTEGER NOT NULL DEFAULT 0,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            PRIMARY KEY(item_key, kind)
        )
    ''')

    # v2 document structure.  These tables are deliberately separate from the
    # legacy item/section summary tables so a failed or partial migration never
    # removes an existing usable index.
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS document_structures (
            item_key TEXT PRIMARY KEY,
            source_fingerprint TEXT NOT NULL,
            structure_version TEXT NOT NULL,
            status TEXT NOT NULL CHECK(status IN
                ('exact', 'recovered', 'flat_fallback', 'unavailable')),
            confidence REAL,
            node_count INTEGER NOT NULL DEFAULT 0,
            leaf_count INTEGER NOT NULL DEFAULT 0,
            diagnostics_json TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS document_nodes (
            node_id TEXT PRIMARY KEY,
            item_key TEXT NOT NULL,
            attachment_key TEXT,
            parent_node_id TEXT REFERENCES document_nodes(node_id) ON DELETE CASCADE,
            node_type TEXT NOT NULL,
            depth INTEGER NOT NULL,
            ordinal INTEGER NOT NULL,
            title TEXT,
            normalized_title TEXT,
            source_kind TEXT NOT NULL,
            source_locator_json TEXT,
            confidence REAL,
            content_chars INTEGER NOT NULL DEFAULT 0,
            first_chunk_id TEXT,
            last_chunk_id TEXT,
            UNIQUE(item_key, parent_node_id, ordinal)
        )
    ''')
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_document_nodes_item ON document_nodes(item_key, depth, ordinal)')
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_document_nodes_parent ON document_nodes(parent_node_id, ordinal)')
    # V3 keeps source-zone policy with the canonical tree, rather than trying
    # to infer it again when summaries or retrieval are built.  ALTERs make
    # this safe for the existing relations.db as well as fresh databases.
    for sql in (
        "ALTER TABLE document_nodes ADD COLUMN zone TEXT NOT NULL DEFAULT 'body'",
        "ALTER TABLE document_nodes ADD COLUMN summary_policy TEXT NOT NULL DEFAULT 'include'",
        "ALTER TABLE document_nodes ADD COLUMN retrieval_policy TEXT NOT NULL DEFAULT 'normal'",
        "ALTER TABLE document_nodes ADD COLUMN citation_policy TEXT NOT NULL DEFAULT 'none'",
        "ALTER TABLE document_nodes ADD COLUMN extraction_engine TEXT",
        "ALTER TABLE document_nodes ADD COLUMN extraction_version TEXT",
    ):
        add_column(cursor, sql)
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS document_node_chunks (
            node_id TEXT NOT NULL REFERENCES document_nodes(node_id) ON DELETE CASCADE,
            chunk_id TEXT NOT NULL UNIQUE,
            ordinal INTEGER NOT NULL,
            PRIMARY KEY(node_id, chunk_id)
        )
    ''')
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_document_node_chunks_node ON document_node_chunks(node_id, ordinal)')
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS document_node_summaries (
            node_id TEXT PRIMARY KEY REFERENCES document_nodes(node_id) ON DELETE CASCADE,
            item_key TEXT NOT NULL,
            summary TEXT NOT NULL,
            summary_kind TEXT NOT NULL CHECK(summary_kind IN ('llm', 'extractive')),
            model TEXT,
            prompt_version TEXT NOT NULL,
            source_fingerprint TEXT NOT NULL,
            source_chunk_count INTEGER NOT NULL,
            source_chars INTEGER NOT NULL,
            searchable INTEGER NOT NULL DEFAULT 0,
            quality_status TEXT NOT NULL CHECK(quality_status IN
                ('accepted', 'candidate', 'degraded', 'disabled')),
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_document_node_summaries_item ON document_node_summaries(item_key, searchable)')
    add_column(cursor, "ALTER TABLE document_node_summaries ADD COLUMN input_scope_json TEXT")
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS document_node_summary_parts (
            node_id TEXT NOT NULL REFERENCES document_nodes(node_id) ON DELETE CASCADE,
            part_ordinal INTEGER NOT NULL,
            child_node_ids_json TEXT NOT NULL,
            summary TEXT NOT NULL,
            model TEXT,
            prompt_version TEXT NOT NULL,
            source_fingerprint TEXT NOT NULL,
            PRIMARY KEY(node_id, part_ordinal)
        )
    ''')
    # A structure rebuild replaces document_nodes atomically, which would
    # otherwise cascade-delete their summaries before the V3 summary worker can
    # decide whether the prompt input is unchanged.  This small, non-FK cache
    # retains the prior generation solely for content-fingerprint reuse.
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS document_node_summary_reuse_cache (
            item_key TEXT NOT NULL,
            node_id TEXT NOT NULL,
            title TEXT,
            summary TEXT NOT NULL,
            summary_kind TEXT NOT NULL,
            model TEXT,
            prompt_version TEXT NOT NULL,
            source_fingerprint TEXT NOT NULL,
            source_chunk_count INTEGER NOT NULL,
            source_chars INTEGER NOT NULL,
            quality_status TEXT NOT NULL,
            input_scope_json TEXT,
            cached_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            PRIMARY KEY (item_key, node_id)
        )
    ''')
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_node_summary_reuse_item ON document_node_summary_reuse_cache(item_key)')
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS artifact_processing_status (
            item_key TEXT NOT NULL,
            attachment_key TEXT NOT NULL DEFAULT '',
            artifact_type TEXT NOT NULL CHECK(artifact_type IN
                ('extraction', 'structure', 'summary', 'references', 'embeddings',
                 'summary_index')),
            status TEXT NOT NULL CHECK(status IN
                ('pending', 'running', 'success', 'empty', 'degraded', 'blocked',
                 'failed', 'stale', 'excluded')),
            reason_code TEXT,
            message TEXT,
            retryable INTEGER NOT NULL DEFAULT 0,
            attempt_count INTEGER NOT NULL DEFAULT 0,
            source_fingerprint TEXT,
            processor_version TEXT,
            model TEXT,
            counts_json TEXT,
            fallback_kind TEXT,
            started_at TIMESTAMP,
            finished_at TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            PRIMARY KEY(item_key, attachment_key, artifact_type)
        )
    ''')
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_artifact_processing_item ON artifact_processing_status(item_key, artifact_type)')
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS artifact_processing_events (
            event_id INTEGER PRIMARY KEY AUTOINCREMENT,
            item_key TEXT NOT NULL,
            attachment_key TEXT NOT NULL DEFAULT '',
            artifact_type TEXT NOT NULL,
            from_status TEXT,
            to_status TEXT NOT NULL,
            reason_code TEXT,
            message TEXT,
            run_id TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')

    # 既存DBのartifact_processing_status.CHECK制約に 'summary_index' を追加する。
    # node要約索引の状態をチャンク埋め込み状態('embeddings')から分離するため
    # （R3）。CHECK制約は後付け変更できないので、旧CHECKのままなら表を作り直す。
    # 移行先CHECKには 'cases' も含める: 旧CHECK追加前に投入されgrandfatherされた
    # 事例レコード（R15で `retire_case_database.py` が正式退避する予定）が存在するため、
    # これを含めないとINSERTがCHECK違反で失敗し全行を取りこぼす。新規書込みは
    # `_ARTIFACT_TYPES`（'cases'を含まない）でPython側が拒否するので安全。
    existing_sql = cursor.execute(
        "SELECT sql FROM sqlite_master WHERE type='table' AND name='artifact_processing_status'"
    ).fetchone()
    orphan_old = cursor.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name='artifact_processing_status_old'"
    ).fetchone()
    needs_migration = existing_sql and "summary_index" not in (existing_sql[0] or "")
    # 過去の失敗migrationで live table が空・データが _old に残った状態を自己修復する。
    if orphan_old and not needs_migration:
        live_rows = cursor.execute("SELECT count(*) FROM artifact_processing_status").fetchone()[0]
        old_rows = cursor.execute("SELECT count(*) FROM artifact_processing_status_old").fetchone()[0]
        if live_rows == 0 and old_rows > 0:
            cursor.execute("DROP TABLE artifact_processing_status")
            cursor.execute("ALTER TABLE artifact_processing_status_old RENAME TO artifact_processing_status")
            existing_sql = cursor.execute(
                "SELECT sql FROM sqlite_master WHERE type='table' AND name='artifact_processing_status'"
            ).fetchone()
            needs_migration = existing_sql and "summary_index" not in (existing_sql[0] or "")
            orphan_old = None
    if needs_migration:
        cursor.execute("DROP TABLE IF EXISTS artifact_processing_status_old")
        cursor.execute("ALTER TABLE artifact_processing_status RENAME TO artifact_processing_status_old")
        cursor.execute('''
            CREATE TABLE artifact_processing_status (
                item_key TEXT NOT NULL,
                attachment_key TEXT NOT NULL DEFAULT '',
                artifact_type TEXT NOT NULL CHECK(artifact_type IN
                    ('extraction', 'structure', 'summary', 'references', 'embeddings',
                     'summary_index', 'cases')),
                status TEXT NOT NULL CHECK(status IN
                    ('pending', 'running', 'success', 'empty', 'degraded', 'blocked',
                     'failed', 'stale', 'excluded')),
                reason_code TEXT,
                message TEXT,
                retryable INTEGER NOT NULL DEFAULT 0,
                attempt_count INTEGER NOT NULL DEFAULT 0,
                source_fingerprint TEXT,
                processor_version TEXT,
                model TEXT,
                counts_json TEXT,
                fallback_kind TEXT,
                started_at TIMESTAMP,
                finished_at TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                PRIMARY KEY(item_key, attachment_key, artifact_type)
            )
        ''')
        cursor.execute('''
            INSERT INTO artifact_processing_status
            SELECT item_key, attachment_key, artifact_type, status, reason_code, message,
                   retryable, attempt_count, source_fingerprint, processor_version, model,
                   counts_json, fallback_kind, started_at, finished_at, updated_at
            FROM artifact_processing_status_old
        ''')
        cursor.execute("DROP TABLE artifact_processing_status_old")
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_artifact_processing_item ON artifact_processing_status(item_key, artifact_type)')

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
            contributors_json TEXT,
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
            source_reference_id INTEGER,
            source_context TEXT,
            source_kind TEXT,
            resolution_source TEXT,
            resolution_confidence REAL,
            resolution_metadata_json TEXT,
            resolution_evidence_json TEXT,
            structure_classification TEXT,
            structure_model TEXT,
            structure_evidence_json TEXT,
            parent_review_id INTEGER,
            compound_split_model TEXT,
            compound_split_evidence_json TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(item_key, raw_hash)
        )
    ''')
    for sql in (
        "ALTER TABLE reference_review_queue ADD COLUMN contributors_json TEXT",
        "ALTER TABLE reference_review_queue ADD COLUMN committed_edge_id INTEGER",
        "ALTER TABLE reference_review_queue ADD COLUMN committed_at TIMESTAMP",
        "ALTER TABLE reference_review_queue ADD COLUMN source_reference_id INTEGER",
        "ALTER TABLE reference_review_queue ADD COLUMN source_context TEXT",
        "ALTER TABLE reference_review_queue ADD COLUMN source_kind TEXT",
        "ALTER TABLE reference_review_queue ADD COLUMN resolution_source TEXT",
        "ALTER TABLE reference_review_queue ADD COLUMN resolution_confidence REAL",
        "ALTER TABLE reference_review_queue ADD COLUMN resolution_metadata_json TEXT",
        "ALTER TABLE reference_review_queue ADD COLUMN resolution_evidence_json TEXT",
        "ALTER TABLE reference_review_queue ADD COLUMN structure_classification TEXT",
        "ALTER TABLE reference_review_queue ADD COLUMN structure_model TEXT",
        "ALTER TABLE reference_review_queue ADD COLUMN structure_evidence_json TEXT",
        "ALTER TABLE reference_review_queue ADD COLUMN parent_review_id INTEGER",
        "ALTER TABLE reference_review_queue ADD COLUMN compound_split_model TEXT",
        "ALTER TABLE reference_review_queue ADD COLUMN compound_split_evidence_json TEXT",
    ):
        add_column(cursor, sql)
    cursor.execute(
        "CREATE INDEX IF NOT EXISTS idx_reference_review_status "
        "ON reference_review_queue(status, item_key)"
    )

    # Human-in-the-loop exception layer for S2 citation/reference relations.
    # The stable identity is direction + local item key + external paper ID, so
    # a disabled relation stays disabled even when S2 data is refreshed.
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS relation_reports (
            report_id INTEGER PRIMARY KEY AUTOINCREMENT,
            direction TEXT NOT NULL CHECK(direction IN ('references', 'citations')),
            item_key TEXT NOT NULL,
            external_paper_id TEXT NOT NULL,
            external_title TEXT,
            reason TEXT NOT NULL,
            details TEXT,
            reporter TEXT NOT NULL DEFAULT 'mcp',
            status TEXT NOT NULL DEFAULT 'pending'
                CHECK(status IN ('pending', 'disabled', 'kept')),
            report_count INTEGER NOT NULL DEFAULT 1,
            triage_status TEXT NOT NULL DEFAULT 'unreviewed',
            triage_model TEXT,
            triage_evidence_json TEXT,
            reviewer_note TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            reviewed_at TIMESTAMP,
            UNIQUE(direction, item_key, external_paper_id)
        )
    ''')
    cursor.execute(
        "CREATE INDEX IF NOT EXISTS idx_relation_reports_status "
        "ON relation_reports(status, direction, item_key)"
    )
    for sql in (
        "ALTER TABLE relation_reports ADD COLUMN triage_status TEXT NOT NULL DEFAULT 'unreviewed'",
        "ALTER TABLE relation_reports ADD COLUMN triage_model TEXT",
        "ALTER TABLE relation_reports ADD COLUMN triage_evidence_json TEXT",
    ):
        add_column(cursor, sql)

    # Runtime quality reports for LLM summary routing. Reports are tied to a
    # summary fingerprint so regeneration automatically retires stale decisions.
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS summary_quality_reports (
            report_id INTEGER PRIMARY KEY AUTOINCREMENT,
            item_key TEXT NOT NULL,
            section_id TEXT NOT NULL DEFAULT '',
            summary_node_id TEXT,
            prior_quality_status TEXT,
            summary_hash TEXT NOT NULL,
            summary_model TEXT,
            reason TEXT NOT NULL,
            details TEXT,
            evidence_chunk_ids_json TEXT,
            reporter TEXT NOT NULL DEFAULT 'mcp:claude',
            status TEXT NOT NULL DEFAULT 'pending'
                CHECK(status IN ('pending', 'disabled', 'kept')),
            triage_status TEXT NOT NULL DEFAULT 'unreviewed'
                CHECK(triage_status IN ('unreviewed', 'confirmed', 'dismissed', 'uncertain')),
            triage_model TEXT,
            triage_evidence_json TEXT,
            report_count INTEGER NOT NULL DEFAULT 1,
            reviewer_note TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            reviewed_at TIMESTAMP,
            UNIQUE(item_key, section_id, summary_hash)
        )
    ''')
    cursor.execute(
        "CREATE INDEX IF NOT EXISTS idx_summary_quality_reports_status "
        "ON summary_quality_reports(status, triage_status, item_key)"
    )
    for sql in (
        "ALTER TABLE summary_quality_reports ADD COLUMN summary_node_id TEXT",
        "ALTER TABLE summary_quality_reports ADD COLUMN prior_quality_status TEXT",
    ):
        add_column(cursor, sql)

    # Runtime quality reports for *source text* rather than summaries (note 79,
    # U0-b). Degraded OCR that reaches a reader is reported here and surfaces as
    # a re-OCR candidate. Scoped by the chunk's own text hash so a re-extraction
    # retires the report automatically, mirroring how summary_hash retires a
    # summary report.
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS chunk_quality_reports (
            report_id INTEGER PRIMARY KEY AUTOINCREMENT,
            item_key TEXT NOT NULL,
            attachment_key TEXT NOT NULL DEFAULT '',
            chunk_id TEXT NOT NULL,
            chunk_hash TEXT NOT NULL,
            page INTEGER,
            reason TEXT NOT NULL,
            details TEXT,
            reporter TEXT NOT NULL DEFAULT 'mcp:claude',
            status TEXT NOT NULL DEFAULT 'pending'
                CHECK(status IN ('pending', 'resolved', 'dismissed')),
            report_count INTEGER NOT NULL DEFAULT 1,
            reviewer_note TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(chunk_id, chunk_hash)
        )
    ''')
    cursor.execute(
        "CREATE INDEX IF NOT EXISTS idx_chunk_quality_reports_item "
        "ON chunk_quality_reports(status, item_key)"
    )

    conn.commit()


def _deduplicate_relation_rows(
    cursor: sqlite3.Cursor, table: str, identity_expressions: tuple[str, ...],
    eligible_sql: str,
) -> None:
    """Keep one row per NULL-safe relation identity during schema migration.

    Inputs are module constants, never user input.
    Retaining the smallest id also makes a second initialization a no-op.
    """
    identity = ", ".join(identity_expressions)
    cursor.execute(f'''
        DELETE FROM {table}
        WHERE ({eligible_sql}) AND id NOT IN (
            SELECT retained_id FROM (
                SELECT MIN(id) AS retained_id
                FROM {table}
                WHERE {eligible_sql}
                GROUP BY {identity}
            )
        )
    ''')


def _normalize_identifier(value: Optional[str], kind: str) -> Optional[str]:
    normalized = unicodedata.normalize(
        "NFKC", strip_unicode_format_characters(value),
    ).strip().casefold()
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
            # ORDER BY work_id: candidates is scanned in full below rather than
            # stopped at the first match, but a stable enumeration order still
            # matters for the score-tie fallback (below) to be deterministic
            # across runs rather than depend on SQLite's unordered scan order
            # (found in code review, fixed 2026-07-30).
            candidates = conn.execute(
                "SELECT * FROM works WHERE title_norm = ? OR title_norm LIKE ? "
                "ORDER BY work_id LIMIT 100",
                (values["title_norm"], f"{values['title_norm'][:80]}%"),
            ).fetchall()
            # Take the best-scoring candidate across the whole set, not the
            # first one that clears the threshold: two distinct works (e.g. a
            # book and a later edition/reprint by the same author) can both
            # qualify, and stopping at the first previously meant the match
            # depended on SQLite's unordered scan order rather than which
            # candidate is actually the better match (found in code review,
            # fixed 2026-07-30).
            best_score = -1.0
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
                    and similarity > best_score
                ):
                    row = candidate
                    best_score = similarity

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
    return _cache_repository().get_query_expansion(query_hash)


def save_query_expansion(query_hash: str, expansions: str) -> None:
    """Persist a successful query expansion for reuse across MCP sessions."""
    _cache_repository().save_query_expansion(query_hash, expansions)


def get_resolver_cache(query_hash: str, source: str) -> Optional[str]:
    return _cache_repository().get_resolver(query_hash, source)


def save_resolver_cache(query_hash: str, source: str, response_json: str) -> None:
    _cache_repository().save_resolver(query_hash, source, response_json)


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
                    (item_key, raw_hash, raw_reference, title, authors_json, contributors_json, year,
                     doi, isbn, container, lang, work_type, extraction_model,
                     source_reference_id, source_context, source_kind, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
                ON CONFLICT(item_key, raw_hash) DO UPDATE SET
                    raw_reference = excluded.raw_reference,
                    title = excluded.title,
                    authors_json = excluded.authors_json,
                    contributors_json = excluded.contributors_json,
                    year = excluded.year,
                    doi = excluded.doi,
                    isbn = excluded.isbn,
                    container = excluded.container,
                    lang = excluded.lang,
                    work_type = excluded.work_type,
                    extraction_model = excluded.extraction_model,
                    source_reference_id = COALESCE(
                        excluded.source_reference_id, reference_review_queue.source_reference_id
                    ),
                    source_context = COALESCE(
                        excluded.source_context, reference_review_queue.source_context
                    ),
                    source_kind = COALESCE(excluded.source_kind, reference_review_queue.source_kind),
                    updated_at = CURRENT_TIMESTAMP
            ''', (
                item_key, raw_hash, raw, reference.get("title"),
                json.dumps(authors, ensure_ascii=False),
                json.dumps(reference.get("contributors") or [], ensure_ascii=False),
                reference.get("year"),
                reference.get("doi"), reference.get("isbn"), reference.get("container"),
                reference.get("lang"), reference.get("type"), model,
                reference.get("source_reference_id"), reference.get("source_context"),
                reference.get("source_kind"),
            ))
            counts["updated" if existed else "staged"] += 1
        conn.commit()
        return counts
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def get_reference_review_candidates(
    status: Optional[str] = None, *, source_kind: Optional[str] = None,
) -> List[Dict[str, Any]]:
    conn = get_db_connection()
    try:
        if status is not None and status not in {"pending", "approved", "rejected"}:
            raise ValueError("invalid review status")
        sql = "SELECT * FROM reference_review_queue"
        clauses: list[str] = []
        params_list: list[Any] = []
        if status is not None:
            clauses.append("status = ?")
            params_list.append(status)
        if source_kind is not None:
            clauses.append("source_kind = ?")
            params_list.append(source_kind)
        if clauses:
            sql += " WHERE " + " AND ".join(clauses)
        sql += " ORDER BY item_key, review_id"
        rows = [dict(row) for row in conn.execute(sql, tuple(params_list)).fetchall()]
        for row in rows:
            try:
                row["authors"] = json.loads(row.pop("authors_json") or "[]")
            except (json.JSONDecodeError, TypeError):
                row["authors"] = []
            try:
                row["contributors"] = json.loads(row.pop("contributors_json") or "[]")
            except (json.JSONDecodeError, TypeError):
                row["contributors"] = []
            for source, target in (
                ("resolution_metadata_json", "resolution_metadata"),
                ("resolution_evidence_json", "resolution_evidence"),
                ("structure_evidence_json", "structure_evidence"),
                ("compound_split_evidence_json", "compound_split_evidence"),
            ):
                try:
                    row[target] = json.loads(row.pop(source) or "null")
                except (json.JSONDecodeError, TypeError):
                    row[target] = None
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


def _prepare_reference_review_decision(
    conn: sqlite3.Connection, decision: Dict[str, Any],
) -> tuple[Any, ...]:
    review_id = int(decision["review_id"])
    status = str(decision.get("status") or "pending")
    if status not in {"approved", "rejected", "pending"}:
        raise ValueError("invalid review status")
    row = conn.execute(
        "SELECT raw_reference FROM reference_review_queue WHERE review_id = ?", (review_id,)
    ).fetchone()
    if row is None:
        raise ValueError(f"review {review_id}: not found")
    raw = unicodedata.normalize("NFKC", str(row["raw_reference"] or ""))
    identifier_raw = strip_unicode_format_characters(raw)
    raw_folded = raw.casefold()
    identifier_raw_folded = identifier_raw.casefold()
    raw_compact = re.sub(r"[^0-9x]", "", identifier_raw_folded)

    title = str(decision.get("title") or "").strip() or None
    if title:
        normalized_title = normalize_work_title(title)
        if len(normalized_title) < 8 or normalized_title not in normalize_work_title(raw):
            raise ValueError(f"review {review_id}: title is not supported by raw reference")
    year = decision.get("year")
    if year is not None:
        year = int(year)
        if str(year) not in raw:
            raise ValueError(f"review {review_id}: year is not present in raw reference")
    doi = str(decision.get("doi") or "").strip() or None
    doi_norm = _normalize_identifier(doi, "doi") if doi else None
    doi_verified = bool(doi_norm and doi_norm in identifier_raw_folded)
    if doi and not doi_verified:
        raise ValueError(f"review {review_id}: DOI is not present in raw reference")
    isbn = str(decision.get("isbn") or "").strip() or None
    isbn_norm = _normalize_identifier(isbn, "isbn") if isbn else None
    isbn_verified = bool(
        isbn_norm and len(isbn_norm) in {10, 13} and isbn_norm in raw_compact
    )
    if isbn and not isbn_verified:
        raise ValueError(f"review {review_id}: ISBN is not present in raw reference")
    if status == "approved" and not (doi_verified or isbn_verified):
        raise ValueError(f"review {review_id}: approval requires a literal DOI or ISBN")

    authors = decision.get("authors")
    authors_json = None
    if authors is not None:
        if not isinstance(authors, list) or not all(isinstance(value, str) for value in authors):
            raise ValueError(f"review {review_id}: authors must be a string array")
        raw_tokens = set(re.findall(r"\w+", raw_folded, flags=re.UNICODE))
        for author in authors:
            author_tokens = re.findall(
                r"\w+", unicodedata.normalize("NFKC", author).casefold(), flags=re.UNICODE,
            )
            if author_tokens and (
                len(author_tokens[-1]) < 3 or author_tokens[-1] not in raw_tokens
            ):
                raise ValueError(f"review {review_id}: author is not supported by raw reference")
        authors_json = json.dumps(authors, ensure_ascii=False)
    return (
        status, decision.get("note"), title, authors_json, year,
        doi_norm, isbn_norm, review_id,
    )


def apply_reference_review_decisions(decisions: List[Dict[str, Any]]) -> int:
    """Validate an entire reviewer batch first, then apply it atomically."""
    conn = get_db_connection()
    try:
        review_ids = [int(decision["review_id"]) for decision in decisions]
        if len(review_ids) != len(set(review_ids)):
            raise ValueError("duplicate review_id in decision batch")
        prepared = [_prepare_reference_review_decision(conn, decision) for decision in decisions]
        applied = 0
        for params in prepared:
            cursor = conn.execute('''
            UPDATE reference_review_queue
            SET status=?, reviewer_note=?,
                title=COALESCE(?, title), authors_json=COALESCE(?, authors_json),
                year=COALESCE(?, year), doi=COALESCE(?, doi), isbn=COALESCE(?, isbn),
                updated_at=CURRENT_TIMESTAMP
            WHERE review_id=?
            ''', params)
            applied += cursor.rowcount
        conn.commit()
        return applied
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def apply_reference_review_decision(decision: Dict[str, Any]) -> bool:
    """Apply one reviewer decision through the atomic batch validator."""
    return apply_reference_review_decisions([decision]) == 1


def _prepare_reference_structure_decision(
    conn: sqlite3.Connection, decision: Dict[str, Any], model: str,
) -> tuple[Any, ...]:
    review_id = int(decision["review_id"])
    classification = str(decision.get("classification") or "").strip()
    if classification not in {"full_reference", "short_citation", "commentary_or_body"}:
        raise ValueError(f"review {review_id}: unsupported structure classification")
    row = conn.execute('''
        SELECT raw_reference, status, source_kind, committed_edge_id
        FROM reference_review_queue WHERE review_id = ?
    ''', (review_id,)).fetchone()
    if row is None:
        raise ValueError(f"review {review_id}: not found")
    if row["status"] != "rejected" or row["source_kind"] != "epub-unverified" or row["committed_edge_id"]:
        raise ValueError(f"review {review_id}: row is not eligible for structure classification")
    raw = unicodedata.normalize("NFKC", str(row["raw_reference"] or ""))
    raw_normalized = normalize_work_title(raw)
    raw_tokens = set(re.findall(r"\w+", raw.casefold(), flags=re.UNICODE))
    raw_identifier = strip_unicode_format_characters(raw).casefold()
    raw_compact = re.sub(r"[^0-9x]", "", raw_identifier)

    title = authors_json = year = doi = isbn = container = work_type = None
    if classification == "full_reference":
        title = str(decision.get("title") or "").strip() or None
        if not title or normalize_work_title(title) not in raw_normalized:
            raise ValueError(f"review {review_id}: structured title is not supported by raw reference")
        authors = decision.get("authors")
        if not isinstance(authors, list) or not authors or not all(
            isinstance(author, str) and author.strip() for author in authors
        ):
            raise ValueError(f"review {review_id}: full reference requires authors")
        for author in authors:
            author_tokens = re.findall(r"\w+", author.casefold(), flags=re.UNICODE)
            family = author_tokens[0] if "," in author else author_tokens[-1]
            if family not in raw_tokens:
                raise ValueError(f"review {review_id}: structured author is not supported by raw reference")
        authors_json = json.dumps(authors, ensure_ascii=False)
        if decision.get("year") is not None:
            year = int(decision["year"])
            if str(year) not in raw:
                raise ValueError(f"review {review_id}: structured year is not present in raw reference")
        doi_value = str(decision.get("doi") or "").strip()
        if doi_value:
            doi = _normalize_identifier(doi_value, "doi")
            if doi not in raw_identifier:
                raise ValueError(f"review {review_id}: structured DOI is not present in raw reference")
        isbn_value = str(decision.get("isbn") or "").strip()
        if isbn_value:
            isbn = _normalize_identifier(isbn_value, "isbn")
            if len(isbn) not in {10, 13} or isbn not in raw_compact:
                raise ValueError(f"review {review_id}: structured ISBN is not present in raw reference")
        container = str(decision.get("container") or "").strip() or None
        work_type = str(decision.get("type") or "").strip() or None
        note = "unresolved: insufficient stable identifier; structured from epub note"
    elif classification == "short_citation":
        note = "short citation: requires bibliography or preceding-note linkage"
    else:
        note = "not a bibliographic reference: structure classification"
    return (
        note, title, authors_json, year, doi, isbn, container, work_type,
        classification, model, json.dumps(decision, ensure_ascii=False), review_id,
    )


def apply_reference_structure_decisions(
    decisions: List[Dict[str, Any]], *, model: str,
) -> int:
    """Atomically store validated DeepSeek classification without writing Work edges."""
    review_ids = [int(decision["review_id"]) for decision in decisions]
    if len(review_ids) != len(set(review_ids)):
        raise ValueError("duplicate review_id in structure classification batch")
    conn = get_db_connection()
    try:
        prepared = [
            _prepare_reference_structure_decision(conn, decision, model)
            for decision in decisions
        ]
        applied = 0
        for params in prepared:
            cursor = conn.execute('''
                UPDATE reference_review_queue
                SET reviewer_note = ?, title = COALESCE(?, title),
                    authors_json = COALESCE(?, authors_json), year = COALESCE(?, year),
                    doi = COALESCE(?, doi), isbn = COALESCE(?, isbn),
                    container = COALESCE(?, container), work_type = COALESCE(?, work_type),
                    structure_classification = ?, structure_model = ?,
                    structure_evidence_json = ?, updated_at = CURRENT_TIMESTAMP
                WHERE review_id = ? AND status = 'rejected' AND committed_edge_id IS NULL
            ''', params)
            if cursor.rowcount != 1:
                raise ValueError(f"review {params[-1]}: row changed during structure classification")
            applied += 1
        conn.commit()
        return applied
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def apply_compound_reference_splits(
    splits: List[Dict[str, Any]], *, model: str,
) -> Dict[str, int]:
    """Atomically stage full-reference children while retaining compound parents."""
    parent_ids = [int(split["review_id"]) for split in splits]
    if len(parent_ids) != len(set(parent_ids)):
        raise ValueError("duplicate parent review_id in compound split batch")
    conn = get_db_connection()
    try:
        prepared: list[tuple[sqlite3.Row, Dict[str, Any], list[Dict[str, Any]]]] = []
        for split in splits:
            review_id = int(split["review_id"])
            if split.get("classification") != "multiple_full_references":
                raise ValueError(f"review {review_id}: split is not multiple_full_references")
            parent = conn.execute('''
                SELECT * FROM reference_review_queue WHERE review_id = ?
            ''', (review_id,)).fetchone()
            if (
                parent is None or parent["status"] != "rejected"
                or parent["source_kind"] != "epub-unverified"
                or not str(parent["reviewer_note"] or "").startswith("compound reference")
                or parent["committed_edge_id"]
            ):
                raise ValueError(f"review {review_id}: parent is not eligible for compound split")
            children = split.get("references")
            if not isinstance(children, list) or len(children) < 2:
                raise ValueError(f"review {review_id}: split requires at least two references")
            parent_text = " ".join(str(parent["raw_reference"] or "").split())
            child_texts: set[str] = set()
            child_cursor = 0
            for child in children:
                raw = " ".join(str(child.get("raw") or "").split())
                position = parent_text.find(raw, child_cursor) if raw else -1
                if position < 0:
                    raise ValueError(
                        f"review {review_id}: child raw is not sequentially present in parent"
                    )
                child_cursor = position + len(raw)
                if raw in child_texts:
                    raise ValueError(f"review {review_id}: duplicate child raw")
                child_texts.add(raw)
                title = str(child.get("title") or "").strip()
                if not title or normalize_work_title(title) not in normalize_work_title(raw):
                    raise ValueError(f"review {review_id}: child title is not supported by child raw")
                authors = child.get("authors")
                if not isinstance(authors, list) or not authors:
                    raise ValueError(f"review {review_id}: child has no authors")
                raw_tokens = set(re.findall(r"\w+", unicodedata.normalize("NFKC", raw).casefold()))
                for author in authors:
                    author_tokens = re.findall(r"\w+", str(author).casefold())
                    if not author_tokens:
                        raise ValueError(f"review {review_id}: child has an empty author")
                    family = author_tokens[0] if "," in str(author) else author_tokens[-1]
                    if family not in raw_tokens:
                        raise ValueError(f"review {review_id}: child author is not supported by child raw")
                year = child.get("year")
                if year is None or str(int(year)) not in unicodedata.normalize("NFKC", raw):
                    raise ValueError(f"review {review_id}: child year is not supported by child raw")
                identifier_raw = strip_unicode_format_characters(
                    unicodedata.normalize("NFKC", raw),
                ).casefold()
                doi = _normalize_identifier(str(child.get("doi") or ""), "doi")
                if doi and doi not in identifier_raw:
                    raise ValueError(f"review {review_id}: child DOI is not supported by child raw")
                isbn = _normalize_identifier(str(child.get("isbn") or ""), "isbn")
                if isbn:
                    raw_compact = re.sub(r"[^0-9x]", "", identifier_raw)
                    if len(isbn) not in {10, 13} or isbn not in raw_compact:
                        raise ValueError(f"review {review_id}: child ISBN is not supported by child raw")
            prepared.append((parent, split, children))

        totals = {"parents": 0, "children_staged": 0, "children_existing": 0}
        for parent, split, children in prepared:
            review_id = int(split["review_id"])
            for child in children:
                raw = " ".join(str(child["raw"]).split())
                raw_hash = hashlib.sha256(raw.encode("utf-8")).hexdigest()
                existing = conn.execute(
                    "SELECT review_id FROM reference_review_queue WHERE item_key = ? AND raw_hash = ?",
                    (parent["item_key"], raw_hash),
                ).fetchone()
                if existing:
                    totals["children_existing"] += 1
                    continue
                conn.execute('''
                    INSERT INTO reference_review_queue
                        (item_key, raw_hash, raw_reference, title, authors_json,
                         contributors_json, year, doi, isbn, container, work_type,
                         extraction_model, status, reviewer_note, source_reference_id,
                         source_context, source_kind, structure_classification,
                         structure_model, structure_evidence_json, parent_review_id,
                         updated_at)
                    VALUES (?, ?, ?, ?, ?, '[]', ?, ?, ?, ?, ?, ?, 'rejected', ?, ?, ?,
                            'epub-compound-child', 'full_reference', ?, ?, ?, CURRENT_TIMESTAMP)
                ''', (
                    parent["item_key"], raw_hash, raw, child.get("title"),
                    json.dumps(child.get("authors") or [], ensure_ascii=False),
                    child.get("year"),
                    _normalize_identifier(str(child.get("doi") or ""), "doi"),
                    _normalize_identifier(str(child.get("isbn") or ""), "isbn"),
                    child.get("container"), child.get("type"), model,
                    "unresolved: insufficient stable identifier; split from compound reference",
                    parent["source_reference_id"], parent["source_context"], model,
                    json.dumps(child, ensure_ascii=False), review_id,
                ))
                totals["children_staged"] += 1
            cursor = conn.execute('''
                UPDATE reference_review_queue
                SET reviewer_note = ?, structure_classification = 'compound_parent',
                    compound_split_model = ?, compound_split_evidence_json = ?,
                    updated_at = CURRENT_TIMESTAMP
                WHERE review_id = ? AND status = 'rejected' AND committed_edge_id IS NULL
            ''', (
                f"compound reference: split into {len(children)} full references",
                model, json.dumps(split, ensure_ascii=False), review_id,
            ))
            if cursor.rowcount != 1:
                raise ValueError(f"review {review_id}: parent changed during compound split")
            totals["parents"] += 1
        conn.commit()
        return totals
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def _prepare_metadata_resolution(
    review_id: int, candidate: Dict[str, Any], evidence: Dict[str, Any],
) -> tuple[Any, ...]:
    if not evidence.get("accepted"):
        raise ValueError(f"review {review_id}: metadata evidence is not accepted")
    stable_fields = (
        "doi", "isbn", "cinii_crid", "ndl_bibid", "openalex_id", "s2_paper_id",
    )
    if not any(str(candidate.get(field) or "").strip() for field in stable_fields):
        raise ValueError(f"review {review_id}: metadata candidate has no stable identifier")
    source = str(candidate.get("source") or "").strip()
    if source not in {"crossref", "cinii", "ndl", "openalex", "s2"}:
        raise ValueError(f"review {review_id}: unsupported metadata source")
    return (
        f"metadata-resolved:{source}", candidate.get("title"),
        candidate.get("doi"), candidate.get("isbn"), source,
        float(evidence.get("score") or 0),
        json.dumps(candidate, ensure_ascii=False),
        json.dumps(evidence, ensure_ascii=False), review_id,
    )


def apply_reference_metadata_resolutions(resolutions: List[Dict[str, Any]]) -> int:
    """Atomically approve rows using prevalidated external-metadata evidence."""
    review_ids = [int(row["review_id"]) for row in resolutions]
    if len(review_ids) != len(set(review_ids)):
        raise ValueError("duplicate review_id in metadata resolution batch")
    prepared = [
        _prepare_metadata_resolution(
            int(row["review_id"]), row["candidate"], row["evidence"],
        )
        for row in resolutions
    ]
    conn = get_db_connection()
    try:
        applied = 0
        for params in prepared:
            cursor = conn.execute('''
                UPDATE reference_review_queue
                SET status = 'approved',
                    reviewer_note = ?,
                    title = COALESCE(title, ?),
                    doi = COALESCE(?, doi), isbn = COALESCE(?, isbn),
                    resolution_source = ?, resolution_confidence = ?,
                    resolution_metadata_json = ?, resolution_evidence_json = ?,
                    updated_at = CURRENT_TIMESTAMP
                WHERE review_id = ? AND status = 'rejected' AND committed_edge_id IS NULL
            ''', params)
            if cursor.rowcount != 1:
                raise ValueError(f"review {params[-1]}: row is not eligible for metadata resolution")
            applied += 1
        conn.commit()
        return applied
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def apply_reference_metadata_resolution(
    review_id: int, candidate: Dict[str, Any], evidence: Dict[str, Any],
) -> bool:
    return apply_reference_metadata_resolutions([{
        "review_id": review_id, "candidate": candidate, "evidence": evidence,
    }]) == 1


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
    return _summary_repository().get_item_abstract(item_key)


def get_item_summary(item_key: str) -> Optional[dict]:
    return _summary_repository().get_item_summary(item_key)


def get_section_summaries(item_key: str) -> List[Dict[str, Any]]:
    return _summary_repository().get_section_summaries(item_key)


def reset_ingestion_derived_state() -> Dict[str, int]:
    """Remove corpus-derived structure/summary state for an explicit rebuild.

    Chunk-linked citation/reference mappings are included because their chunk
    IDs belong to the old corpus. Bibliographic work metadata and human
    relation reports remain; they do not point at canonical chunks.
    """
    conn = get_db_connection()
    try:
        conn.execute("BEGIN IMMEDIATE")
        counts: Dict[str, int] = {}
        statements = (
            ("global_citations", "DELETE FROM global_citations", ()),
            ("global_references", "DELETE FROM global_references", ()),
            ("item_citation_status", "DELETE FROM item_citation_status", ()),
            ("document_node_summary_reuse_cache",
             "DELETE FROM document_node_summary_reuse_cache", ()),
            ("document_nodes", "DELETE FROM document_nodes", ()),
            ("document_structures", "DELETE FROM document_structures", ()),
            ("section_summaries", "DELETE FROM section_summaries", ()),
            ("item_summaries", "DELETE FROM item_summaries", ()),
            ("insight_generation_status", "DELETE FROM insight_generation_status", ()),
            (
                "artifact_processing_status",
                "DELETE FROM artifact_processing_status "
                "WHERE artifact_type IN ('extraction','structure','summary','embeddings','summary_index')",
                (),
            ),
            (
                "artifact_processing_events",
                "DELETE FROM artifact_processing_events "
                "WHERE artifact_type IN ('extraction','structure','summary','embeddings','summary_index')",
                (),
            ),
        )
        for name, sql, params in statements:
            cursor = conn.execute(sql, params)
            counts[name] = max(0, int(cursor.rowcount))
        conn.commit()
        return counts
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def get_external_abstract(paper_id: str) -> Optional[dict]:
    """外部論文の概要キャッシュを返す。未取得なら None。"""
    return _cache_repository().get_external_abstract(paper_id)


def save_external_abstract(paper_id: str, abstract: Optional[str], tldr: Optional[str], status: str) -> None:
    """外部論文の概要をキャッシュに保存する。"""
    _cache_repository().save_external_abstract(paper_id, abstract, tldr, status)


def get_item_citation_status(item_key: str) -> Optional[str]:
    conn = get_db_connection()
    try:
        cursor = conn.cursor()
        cursor.execute('SELECT s2_status FROM item_citation_status WHERE item_key = ?', (item_key,))
        row = cursor.fetchone()
        return row['s2_status'] if row else None
    finally:
        conn.close()


def get_item_s2_paper_id(item_key: str) -> Optional[str]:
    conn = get_db_connection()
    try:
        cursor = conn.cursor()
        cursor.execute('SELECT s2_paper_id FROM item_citation_status WHERE item_key = ?', (item_key,))
        row = cursor.fetchone()
        return (row['s2_paper_id'] or None) if row else None
    finally:
        conn.close()


def clear_s2_relations_for_item(
    item_key: str, *, citations: bool = True, references: bool = True,
) -> Dict[str, int]:
    """Drop the S2-derived relations of one item, so a re-fetch can replace them.

    Needed because the rows are keyed by the *external* paper -- citations by
    ``UNIQUE(citing_paper_id, cited_item_key, context_snippet)`` -- so a re-run
    never overwrites rows fetched under a different identity, or under the same
    identity for citers S2 has since stopped reporting. They would otherwise
    survive and mix with the new ones.

    ``citations`` and ``references`` select which half to clear. Callers replace
    the two at different points and must not delete either before the data that
    replaces it is in hand, or a failed fetch leaves the item with nothing.

    Only S2-sourced rows are removed. References extracted locally from the
    item's own EPUB/PDF (``source='epub'``) are independent evidence and are
    left untouched.
    """
    conn = get_db_connection()
    try:
        cursor = conn.cursor()
        removed_citations = removed_references = 0
        if citations:
            cursor.execute('DELETE FROM global_citations WHERE cited_item_key = ?', (item_key,))
            removed_citations = cursor.rowcount
        if references:
            cursor.execute(
                "DELETE FROM global_references WHERE citing_item_key = ? AND source = 's2'",
                (item_key,),
            )
            removed_references = cursor.rowcount
        conn.commit()
        return {
            "global_citations": removed_citations,
            "global_references": removed_references,
        }
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
            INSERT INTO global_citations
            (citing_paper_id, citing_title, citing_year, context_snippet, cited_item_key,
             cited_chunk_id, similarity_distance, page_hint,
             citing_citation_count, citing_influential_count, chunk_status, citing_doi,
             citing_authors)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT DO UPDATE SET
                citing_title = excluded.citing_title,
                citing_year = excluded.citing_year,
                cited_chunk_id = excluded.cited_chunk_id,
                similarity_distance = excluded.similarity_distance,
                page_hint = excluded.page_hint,
                citing_citation_count = excluded.citing_citation_count,
                citing_influential_count = excluded.citing_influential_count,
                chunk_status = excluded.chunk_status,
                citing_doi = excluded.citing_doi,
                citing_authors = excluded.citing_authors
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
              AND NOT EXISTS (
                  SELECT 1 FROM relation_reports rr
                  WHERE rr.direction = 'citations'
                    AND rr.item_key = global_citations.cited_item_key
                    AND rr.external_paper_id = global_citations.citing_paper_id
                    AND rr.status = 'disabled'
              )
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
          AND NOT EXISTS (
              SELECT 1 FROM relation_reports rr
              WHERE rr.direction = 'citations'
                AND rr.item_key = global_citations.cited_item_key
                AND rr.external_paper_id = global_citations.citing_paper_id
                AND rr.status = 'disabled'
          )
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
            INSERT INTO global_references
            (cited_paper_id, cited_title, cited_year, context_snippet, citing_item_key, citing_chunk_id,
             similarity_distance, page_hint, source, raw_reference_text, s2_status,
             cited_citation_count, cited_influential_count, cited_doi, cited_authors)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT DO UPDATE SET
                cited_title = excluded.cited_title,
                cited_year = excluded.cited_year,
                citing_chunk_id = excluded.citing_chunk_id,
                similarity_distance = excluded.similarity_distance,
                page_hint = excluded.page_hint,
                source = excluded.source,
                s2_status = excluded.s2_status,
                cited_citation_count = excluded.cited_citation_count,
                cited_influential_count = excluded.cited_influential_count,
                cited_doi = excluded.cited_doi,
                cited_authors = excluded.cited_authors
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
              AND NOT EXISTS (
                  SELECT 1 FROM relation_reports rr
                  WHERE rr.direction = 'references'
                    AND rr.item_key = global_references.citing_item_key
                    AND rr.external_paper_id = global_references.cited_paper_id
                    AND rr.status = 'disabled'
              )
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
          AND NOT EXISTS (
              SELECT 1 FROM relation_reports rr
              WHERE rr.direction = 'references'
                AND rr.item_key = global_references.citing_item_key
                AND rr.external_paper_id = global_references.cited_paper_id
                AND rr.status = 'disabled'
          )
        GROUP BY citing_chunk_id
        ORDER BY reference_count DESC, best_distance ASC
    ''', (item_key,))
    rows = cursor.fetchall()
    conn.close()
    return [dict(row) for row in rows]


RELATION_REPORT_REASONS = {
    "not_in_source", "wrong_work", "wrong_direction", "metadata_error", "other",
}

SUMMARY_REPORT_REASONS = {
    "unsupported_claim", "wrong_number", "wrong_work", "missing_context",
    "misleading_summary", "other",
}

#: Defects in *extracted source text*, not in a summary (note 79, U0-b).
CHUNK_REPORT_REASONS = {
    "ocr_garbled", "encoding_broken", "missing_text", "wrong_reading_order",
    "figure_noise", "other",
}


def submit_chunk_quality_report(
    *, item_key: str, chunk_id: str, chunk_text: str, reason: str, details: str,
    attachment_key: str = "", page: Optional[int] = None,
    reporter: str = "mcp:claude",
) -> Dict[str, Any]:
    """Record that a source chunk's own text is unusable.

    Keyed by the chunk's text hash, so re-extracting or re-OCRing the document
    retires the report automatically instead of leaving a stale complaint about
    text that no longer exists. Repeat reports of the same text increment
    ``report_count`` rather than piling up rows -- a passage several readers
    stumble over ranks higher as a re-OCR candidate.
    """
    item_key = (item_key or "").strip()
    chunk_id = (chunk_id or "").strip()
    reason = (reason or "").strip().lower()
    details = (details or "").strip()
    if not item_key:
        raise ValueError("item_key is required.")
    if not chunk_id:
        raise ValueError("chunk_id is required.")
    if reason not in CHUNK_REPORT_REASONS:
        raise ValueError(
            "reason must be one of: " + ", ".join(sorted(CHUNK_REPORT_REASONS))
        )
    if len(details) < 10:
        raise ValueError("details must contain concrete evidence (at least 10 characters).")
    chunk_hash = hashlib.sha256((chunk_text or "").encode("utf-8")).hexdigest()
    conn = get_db_connection()
    try:
        conn.execute('''
            INSERT INTO chunk_quality_reports
                (item_key, attachment_key, chunk_id, chunk_hash, page, reason,
                 details, reporter, status)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, 'pending')
            ON CONFLICT(chunk_id, chunk_hash) DO UPDATE SET
                reason = excluded.reason,
                details = excluded.details,
                reporter = excluded.reporter,
                status = CASE WHEN chunk_quality_reports.status = 'dismissed'
                              THEN 'dismissed' ELSE 'pending' END,
                report_count = chunk_quality_reports.report_count + 1,
                updated_at = CURRENT_TIMESTAMP
        ''', (item_key, (attachment_key or "").strip(), chunk_id, chunk_hash,
              page, reason, details, reporter))
        conn.commit()
        row = conn.execute(
            "SELECT report_id, report_count, status FROM chunk_quality_reports "
            "WHERE chunk_id = ? AND chunk_hash = ?", (chunk_id, chunk_hash),
        ).fetchone()
    finally:
        conn.close()
    return {
        "report_id": row["report_id"] if row else None,
        "report_count": row["report_count"] if row else 1,
        "status": row["status"] if row else "pending",
        "chunk_id": chunk_id,
        "item_key": item_key,
        "reason": reason,
    }


def get_chunk_quality_reports(
    status: Optional[str] = "pending", item_key: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Return chunk-quality reports, newest and most-reported first."""
    clauses: List[str] = []
    params: List[Any] = []
    if status:
        clauses.append("status = ?")
        params.append(status)
    if item_key:
        clauses.append("item_key = ?")
        params.append(item_key)
    where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
    conn = get_db_connection()
    try:
        rows = conn.execute(
            f"SELECT * FROM chunk_quality_reports {where} "
            "ORDER BY report_count DESC, updated_at DESC", params,
        ).fetchall()
    finally:
        conn.close()
    return [dict(row) for row in rows]


def _summary_fingerprint(summary: str, model: Optional[str]) -> str:
    return hashlib.sha256(
        f"{model or ''}\n{summary or ''}".encode("utf-8")
    ).hexdigest()


def _current_summary(
    conn: sqlite3.Connection, item_key: str, section_id: str,
) -> Optional[Dict[str, Any]]:
    if section_id:
        row = conn.execute(
            """
            SELECT s.node_id, s.summary, s.model, s.summary_kind,
                   s.quality_status, s.searchable, s.updated_at
            FROM document_node_summaries s
            JOIN document_nodes n ON n.node_id = s.node_id
            WHERE s.item_key = ? AND s.node_id = ?
              AND n.node_type != 'item_root'
            """,
            (item_key, section_id),
        ).fetchone()
    else:
        row = conn.execute(
            """
            SELECT s.node_id, s.summary, s.model, s.summary_kind,
                   s.quality_status, s.searchable, s.updated_at
            FROM document_node_summaries s
            JOIN document_nodes n ON n.node_id = s.node_id
            WHERE s.item_key = ? AND n.node_type = 'item_root'
            ORDER BY s.updated_at DESC LIMIT 1
            """,
            (item_key,),
        ).fetchone()
    return dict(row) if row else None


def submit_summary_quality_report(
    *, item_key: str, section_id: Optional[str], reason: str, details: str,
    evidence_chunk_ids: Optional[List[str]] = None, reporter: str = "mcp:claude",
) -> Dict[str, Any]:
    """Report a concrete problem in the current summary without deleting it."""
    item_key = (item_key or "").strip()
    section_id = (section_id or "").strip()
    reason = (reason or "").strip().lower()
    details = (details or "").strip()
    if not item_key:
        raise ValueError("item_key is required.")
    if reason not in SUMMARY_REPORT_REASONS:
        raise ValueError(
            "reason must be one of: " + ", ".join(sorted(SUMMARY_REPORT_REASONS))
        )
    if len(details) < 10:
        raise ValueError("details must contain concrete evidence (at least 10 characters).")
    chunk_ids = list(dict.fromkeys(
        str(value or "").strip() for value in (evidence_chunk_ids or [])
        if str(value or "").strip()
    ))
    conn = get_db_connection()
    try:
        current = _current_summary(conn, item_key, section_id)
        if current is None:
            raise ValueError("The specified current item/section summary does not exist.")
        summary_hash = _summary_fingerprint(current["summary"], current.get("model"))
        conn.execute('''
            INSERT INTO summary_quality_reports
                (item_key, section_id, summary_node_id, prior_quality_status,
                 summary_hash, summary_model, reason, details,
                 evidence_chunk_ids_json, reporter, status, triage_status)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'pending', 'unreviewed')
            ON CONFLICT(item_key, section_id, summary_hash) DO UPDATE SET
                summary_node_id = excluded.summary_node_id,
                reason = excluded.reason,
                details = excluded.details,
                evidence_chunk_ids_json = excluded.evidence_chunk_ids_json,
                reporter = excluded.reporter,
                status = CASE WHEN summary_quality_reports.status = 'disabled'
                              THEN 'disabled' ELSE 'pending' END,
                triage_status = CASE WHEN summary_quality_reports.status = 'disabled'
                                     THEN summary_quality_reports.triage_status ELSE 'unreviewed' END,
                report_count = summary_quality_reports.report_count + 1,
                updated_at = CURRENT_TIMESTAMP
        ''', (
            item_key, section_id, current.get("node_id"),
            current.get("quality_status"), summary_hash, current.get("model"), reason, details,
            json.dumps(chunk_ids, ensure_ascii=False), (reporter or "mcp:claude").strip(),
        ))
        conn.commit()
        row = conn.execute(
            "SELECT * FROM summary_quality_reports "
            "WHERE item_key = ? AND section_id = ? AND summary_hash = ?",
            (item_key, section_id, summary_hash),
        ).fetchone()
        result = dict(row)
        result["evidence_chunk_ids"] = json.loads(
            result.pop("evidence_chunk_ids_json") or "[]"
        )
        result["summary_key"] = (
            f"section:{item_key}:{section_id}:{summary_hash[:12]}"
            if section_id else f"item:{item_key}:{summary_hash[:12]}"
        )
        return result
    finally:
        conn.close()


def get_summary_quality_reports(status: Optional[str] = "pending") -> List[Dict[str, Any]]:
    if status is not None and status not in {"pending", "disabled", "kept"}:
        raise ValueError("status must be pending, disabled, kept, or None.")
    conn = get_db_connection()
    try:
        where = "WHERE status = ?" if status is not None else ""
        params = (status,) if status is not None else ()
        rows = conn.execute(
            f"SELECT * FROM summary_quality_reports {where} "
            "ORDER BY updated_at, report_id", params,
        ).fetchall()
        output = []
        for row in rows:
            item = dict(row)
            item["triage_evidence"] = json.loads(
                item.pop("triage_evidence_json", None) or "null"
            )
            item["evidence_chunk_ids"] = json.loads(
                item.pop("evidence_chunk_ids_json") or "[]"
            )
            item["summary_key"] = (
                f"section:{item['item_key']}:{item['section_id']}:{item['summary_hash'][:12]}"
                if item["section_id"]
                else f"item:{item['item_key']}:{item['summary_hash'][:12]}"
            )
            output.append(item)
        return output
    finally:
        conn.close()


def resolve_summary_quality_report(
    report_id: int, decision: str, *, triage_model: Optional[str] = None,
    triage_evidence: Optional[Dict[str, Any]] = None,
    reviewer_note: Optional[str] = None,
) -> bool:
    """Apply a reversible keep/disable decision or mark a report uncertain."""
    normalized = (decision or "").strip().lower()
    mapping = {
        "disable": ("disabled", "confirmed"),
        "keep": ("kept", "dismissed"),
        "uncertain": ("pending", "uncertain"),
    }
    if normalized not in mapping:
        raise ValueError("decision must be disable, keep, or uncertain.")
    status, triage_status = mapping[normalized]
    conn = get_db_connection()
    try:
        report = conn.execute(
            "SELECT * FROM summary_quality_reports WHERE report_id = ?",
            (int(report_id),),
        ).fetchone()
        if report is None:
            return False
        current = _current_summary(conn, report["item_key"], report["section_id"])
        is_current = bool(
            current
            and _summary_fingerprint(current["summary"], current.get("model"))
            == report["summary_hash"]
        )
        if is_current and normalized == "disable":
            conn.execute(
                "UPDATE document_node_summaries "
                "SET searchable = 0, quality_status = 'disabled', updated_at = CURRENT_TIMESTAMP "
                "WHERE node_id = ?",
                (current["node_id"],),
            )
        elif is_current and normalized == "keep" and current.get("quality_status") == "disabled":
            restored = str(report["prior_quality_status"] or "accepted")
            if restored not in {"accepted", "candidate", "degraded"}:
                restored = "accepted"
            searchable = int(
                current.get("summary_kind") == "llm"
                and restored in {"accepted", "candidate"}
            )
            conn.execute(
                "UPDATE document_node_summaries "
                "SET searchable = ?, quality_status = ?, updated_at = CURRENT_TIMESTAMP "
                "WHERE node_id = ?",
                (searchable, restored, current["node_id"]),
            )
        cursor = conn.execute('''
            UPDATE summary_quality_reports
            SET status = ?, triage_status = ?, triage_model = ?,
                triage_evidence_json = ?, reviewer_note = ?,
                reviewed_at = CASE WHEN ? = 'pending' THEN NULL ELSE CURRENT_TIMESTAMP END,
                updated_at = CURRENT_TIMESTAMP
            WHERE report_id = ?
        ''', (
            status, triage_status, (triage_model or "").strip() or None,
            json.dumps(triage_evidence, ensure_ascii=False) if triage_evidence else None,
            (reviewer_note or "").strip() or None, status, int(report_id),
        ))
        conn.commit()
        return cursor.rowcount > 0
    finally:
        conn.close()


def get_disabled_summary_keys() -> set[tuple[str, str]]:
    """Return disabled current summaries as ``(item_key, section_id)`` pairs."""
    conn = get_db_connection()
    try:
        rows = conn.execute(
            "SELECT item_key, section_id, summary_hash FROM summary_quality_reports "
            "WHERE status = 'disabled'"
        ).fetchall()
        disabled: set[tuple[str, str]] = set()
        for row in rows:
            current = _current_summary(conn, row["item_key"], row["section_id"])
            if current and _summary_fingerprint(
                current["summary"], current.get("model")
            ) == row["summary_hash"]:
                disabled.add((row["item_key"], row["section_id"]))
        return disabled
    finally:
        conn.close()


def relation_key(direction: str, item_key: str, external_paper_id: str) -> str:
    """Return the stable public identifier used by the UI and MCP tools."""
    return f"{direction}:{item_key.strip()}:{external_paper_id.strip()}"


def _validate_relation_identity(
    direction: str, item_key: str, external_paper_id: str,
) -> tuple[str, str, str]:
    direction = (direction or "").strip().lower()
    item_key = (item_key or "").strip()
    external_paper_id = (external_paper_id or "").strip()
    if direction not in {"references", "citations"}:
        raise ValueError("direction must be 'references' or 'citations'.")
    if not item_key or not external_paper_id:
        raise ValueError("item_key and external_paper_id are required.")
    return direction, item_key, external_paper_id


def _relation_exists(conn: sqlite3.Connection, direction: str, item_key: str,
                     external_paper_id: str) -> bool:
    if direction == "references":
        row = conn.execute(
            "SELECT 1 FROM global_references WHERE citing_item_key = ? "
            "AND cited_paper_id = ? LIMIT 1",
            (item_key, external_paper_id),
        ).fetchone()
    else:
        row = conn.execute(
            "SELECT 1 FROM global_citations WHERE cited_item_key = ? "
            "AND citing_paper_id = ? LIMIT 1",
            (item_key, external_paper_id),
        ).fetchone()
    return row is not None


def submit_relation_report(
    *, direction: str, item_key: str, external_paper_id: str,
    reason: str, details: Optional[str] = None, reporter: str = "mcp",
    external_title: Optional[str] = None,
) -> Dict[str, Any]:
    """Create or reopen a report without hiding the relation before review."""
    direction, item_key, external_paper_id = _validate_relation_identity(
        direction, item_key, external_paper_id,
    )
    reason = (reason or "").strip().lower()
    if reason not in RELATION_REPORT_REASONS:
        raise ValueError(
            "reason must be one of: " + ", ".join(sorted(RELATION_REPORT_REASONS))
        )
    details = (details or "").strip() or None
    reporter = (reporter or "mcp").strip() or "mcp"
    external_title = (external_title or "").strip() or None
    conn = get_db_connection()
    try:
        if not _relation_exists(conn, direction, item_key, external_paper_id):
            raise ValueError("The specified citation/reference relation does not exist.")
        conn.execute('''
            INSERT INTO relation_reports
                (direction, item_key, external_paper_id, external_title,
                 reason, details, reporter, status)
            VALUES (?, ?, ?, ?, ?, ?, ?, 'pending')
            ON CONFLICT(direction, item_key, external_paper_id) DO UPDATE SET
                external_title = COALESCE(excluded.external_title, relation_reports.external_title),
                reason = excluded.reason,
                details = excluded.details,
                reporter = excluded.reporter,
                status = CASE WHEN relation_reports.status = 'disabled'
                              THEN 'disabled' ELSE 'pending' END,
                report_count = relation_reports.report_count + 1,
                updated_at = CURRENT_TIMESTAMP,
                reviewed_at = CASE WHEN relation_reports.status = 'disabled'
                                   THEN relation_reports.reviewed_at ELSE NULL END
        ''', (direction, item_key, external_paper_id, external_title,
              reason, details, reporter))
        conn.commit()
        row = conn.execute('''
            SELECT * FROM relation_reports
            WHERE direction = ? AND item_key = ? AND external_paper_id = ?
        ''', (direction, item_key, external_paper_id)).fetchone()
        result = dict(row)
        result["relation_key"] = relation_key(direction, item_key, external_paper_id)
        return result
    finally:
        conn.close()


def get_relation_reports(status: Optional[str] = "pending") -> List[Dict[str, Any]]:
    """List relation reports, enriched with the latest stored relation metadata."""
    if status is not None and status not in {"pending", "disabled", "kept"}:
        raise ValueError("status must be pending, disabled, kept, or None.")
    conn = get_db_connection()
    try:
        where = "WHERE rr.status = ?" if status is not None else ""
        params = (status,) if status is not None else ()
        rows = conn.execute(f'''
            SELECT rr.*,
                   COALESCE(rr.external_title,
                       CASE rr.direction
                           WHEN 'references' THEN (
                               SELECT MAX(r.cited_title) FROM global_references r
                               WHERE r.citing_item_key = rr.item_key
                                 AND r.cited_paper_id = rr.external_paper_id)
                           ELSE (
                               SELECT MAX(c.citing_title) FROM global_citations c
                               WHERE c.cited_item_key = rr.item_key
                                 AND c.citing_paper_id = rr.external_paper_id)
                       END) AS relation_title,
                   CASE rr.direction
                       WHEN 'references' THEN (
                           SELECT COUNT(*) FROM global_references r
                           WHERE r.citing_item_key = rr.item_key
                             AND r.cited_paper_id = rr.external_paper_id)
                       ELSE (
                           SELECT COUNT(*) FROM global_citations c
                           WHERE c.cited_item_key = rr.item_key
                             AND c.citing_paper_id = rr.external_paper_id)
                   END AS record_count,
                   CASE rr.direction
                       WHEN 'references' THEN (
                           SELECT COUNT(*) FROM global_references r
                           WHERE r.citing_item_key = rr.item_key
                             AND r.cited_paper_id = rr.external_paper_id
                             AND NULLIF(TRIM(r.context_snippet), '') IS NOT NULL)
                       ELSE (
                           SELECT COUNT(*) FROM global_citations c
                           WHERE c.cited_item_key = rr.item_key
                             AND c.citing_paper_id = rr.external_paper_id
                             AND NULLIF(TRIM(c.context_snippet), '') IS NOT NULL)
                   END AS context_count,
                   CASE rr.direction
                       WHEN 'references' THEN (
                           SELECT MAX(r.raw_reference_text) FROM global_references r
                           WHERE r.citing_item_key = rr.item_key
                             AND r.cited_paper_id = rr.external_paper_id)
                       ELSE NULL
                   END AS sample_raw_reference,
                   CASE rr.direction
                       WHEN 'references' THEN (
                           SELECT MAX(r.context_snippet) FROM global_references r
                           WHERE r.citing_item_key = rr.item_key
                             AND r.cited_paper_id = rr.external_paper_id)
                       ELSE (
                           SELECT MAX(c.context_snippet) FROM global_citations c
                           WHERE c.cited_item_key = rr.item_key
                             AND c.citing_paper_id = rr.external_paper_id)
                   END AS sample_context
            FROM relation_reports rr
            {where}
            ORDER BY rr.updated_at, rr.report_id
        ''', params).fetchall()
        output = []
        for row in rows:
            item = dict(row)
            item["triage_evidence"] = json.loads(
                item.pop("triage_evidence_json", None) or "null"
            )
            item["relation_key"] = relation_key(
                item["direction"], item["item_key"], item["external_paper_id"],
            )
            output.append(item)
        return output
    finally:
        conn.close()


def review_relation_report(report_id: int, decision: str,
                           reviewer_note: Optional[str] = None, *,
                           triage_model: Optional[str] = None,
                           triage_evidence: Optional[Dict[str, Any]] = None) -> bool:
    """Resolve a pending report. ``disable`` hides it; ``keep`` retains it."""
    status = {"disable": "disabled", "keep": "kept"}.get((decision or "").lower())
    if status is None:
        raise ValueError("decision must be 'disable' or 'keep'.")
    conn = get_db_connection()
    try:
        cursor = conn.execute('''
            UPDATE relation_reports
            SET status = ?, reviewer_note = ?,
                triage_status = CASE WHEN ? IS NULL THEN triage_status
                                     WHEN ? = 'disabled' THEN 'confirmed' ELSE 'dismissed' END,
                triage_model = COALESCE(?, triage_model),
                triage_evidence_json = COALESCE(?, triage_evidence_json),
                reviewed_at = CURRENT_TIMESTAMP,
                updated_at = CURRENT_TIMESTAMP
            WHERE report_id = ?
        ''', (
            status, (reviewer_note or "").strip() or None,
            (triage_model or "").strip() or None, status,
            (triage_model or "").strip() or None,
            json.dumps(triage_evidence, ensure_ascii=False) if triage_evidence else None,
            int(report_id),
        ))
        conn.commit()
        return cursor.rowcount > 0
    finally:
        conn.close()


def mark_relation_report_uncertain(
    report_id: int, *, triage_model: Optional[str] = None,
    triage_evidence: Optional[Dict[str, Any]] = None,
) -> bool:
    """Leave a relation pending while recording why automation could not decide."""
    conn = get_db_connection()
    try:
        cursor = conn.execute('''
            UPDATE relation_reports
            SET status = 'pending', triage_status = 'uncertain', triage_model = ?,
                triage_evidence_json = ?, updated_at = CURRENT_TIMESTAMP
            WHERE report_id = ?
        ''', (
            (triage_model or "").strip() or None,
            json.dumps(triage_evidence, ensure_ascii=False) if triage_evidence else None,
            int(report_id),
        ))
        conn.commit()
        return cursor.rowcount > 0
    finally:
        conn.close()


def get_reference_relations_for_item(item_key: str,
                                     include_disabled: bool = False) -> List[Dict[str, Any]]:
    """Return one evidence-aware row per outgoing external relation."""
    conn = get_db_connection()
    try:
        disabled = "" if include_disabled else "AND COALESCE(rr.status, '') <> 'disabled'"
        rows = conn.execute(f'''
            SELECT r.citing_item_key AS item_key, r.cited_paper_id AS external_paper_id,
                   MAX(r.cited_title) AS external_title, MAX(r.cited_authors) AS external_authors,
                   MAX(r.cited_year) AS external_year, MAX(r.cited_doi) AS external_doi,
                   COUNT(*) AS record_count,
                   SUM(CASE WHEN NULLIF(TRIM(r.context_snippet), '') IS NOT NULL THEN 1 ELSE 0 END)
                       AS context_count,
                   SUM(CASE WHEN NULLIF(TRIM(r.raw_reference_text), '') IS NOT NULL THEN 1 ELSE 0 END)
                       AS raw_reference_count,
                   GROUP_CONCAT(DISTINCT r.source) AS sources,
                   rr.status AS review_status, rr.report_id
            FROM global_references r
            LEFT JOIN relation_reports rr
              ON rr.direction = 'references' AND rr.item_key = r.citing_item_key
             AND rr.external_paper_id = r.cited_paper_id
            WHERE r.citing_item_key = ? AND NULLIF(TRIM(r.cited_paper_id), '') IS NOT NULL
              {disabled}
            GROUP BY r.citing_item_key, r.cited_paper_id
            ORDER BY MAX(r.cited_citation_count) DESC, external_title COLLATE NOCASE
        ''', (item_key,)).fetchall()
        output = []
        for row in rows:
            item = dict(row)
            item["direction"] = "references"
            item["relation_key"] = relation_key(
                "references", item_key, item["external_paper_id"],
            )
            output.append(item)
        return output
    finally:
        conn.close()


def get_citation_relations_for_item(item_key: str,
                                    include_disabled: bool = False) -> List[Dict[str, Any]]:
    """Return one evidence-aware row per incoming external relation."""
    conn = get_db_connection()
    try:
        disabled = "" if include_disabled else "AND COALESCE(rr.status, '') <> 'disabled'"
        rows = conn.execute(f'''
            SELECT c.cited_item_key AS item_key, c.citing_paper_id AS external_paper_id,
                   MAX(c.citing_title) AS external_title, MAX(c.citing_authors) AS external_authors,
                   MAX(c.citing_year) AS external_year, MAX(c.citing_doi) AS external_doi,
                   COUNT(*) AS record_count,
                   SUM(CASE WHEN NULLIF(TRIM(c.context_snippet), '') IS NOT NULL THEN 1 ELSE 0 END)
                       AS context_count,
                   0 AS raw_reference_count, 's2' AS sources,
                   rr.status AS review_status, rr.report_id
            FROM global_citations c
            LEFT JOIN relation_reports rr
              ON rr.direction = 'citations' AND rr.item_key = c.cited_item_key
             AND rr.external_paper_id = c.citing_paper_id
            WHERE c.cited_item_key = ? AND NULLIF(TRIM(c.citing_paper_id), '') IS NOT NULL
              {disabled}
            GROUP BY c.cited_item_key, c.citing_paper_id
            ORDER BY MAX(c.citing_citation_count) DESC, external_title COLLATE NOCASE
        ''', (item_key,)).fetchall()
        output = []
        for row in rows:
            item = dict(row)
            item["direction"] = "citations"
            item["relation_key"] = relation_key(
                "citations", item_key, item["external_paper_id"],
            )
            output.append(item)
        return output
    finally:
        conn.close()


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
    if direction == "references":
        where.append(
            "NOT EXISTS (SELECT 1 FROM relation_reports rr "
            "WHERE rr.direction = 'references' "
            f"AND rr.item_key = {item_col} AND rr.external_paper_id = {s2_col} "
            "AND rr.status = 'disabled')"
        )
    else:
        where.append(
            "NOT EXISTS (SELECT 1 FROM relation_reports rr "
            "WHERE rr.direction = 'citations' "
            f"AND rr.item_key = {item_col} AND rr.external_paper_id = {s2_col} "
            "AND rr.status = 'disabled')"
        )
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
                  AND NOT EXISTS (
                      SELECT 1 FROM relation_reports rr
                      WHERE rr.direction = 'references'
                        AND rr.item_key = r.citing_item_key
                        AND rr.external_paper_id = r.cited_paper_id
                        AND rr.status = 'disabled'
                  )
            ), candidate_refs AS (
                SELECT DISTINCT c.citing_item_key AS item_key,
                       {candidate_identity} AS work_key
                FROM global_references c
                WHERE c.citing_item_key <> ?
                  AND NULLIF(TRIM(c.cited_title), '') IS NOT NULL
                  AND NOT EXISTS (
                      SELECT 1 FROM relation_reports rr
                      WHERE rr.direction = 'references'
                        AND rr.item_key = c.citing_item_key
                        AND rr.external_paper_id = c.cited_paper_id
                        AND rr.status = 'disabled'
                  )
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
                  AND NOT EXISTS (
                      SELECT 1 FROM relation_reports rr
                      WHERE rr.direction = 'citations'
                        AND rr.item_key = global_citations.cited_item_key
                        AND rr.external_paper_id = global_citations.citing_paper_id
                        AND rr.status = 'disabled'
                  )
            ), candidate_pairs AS (
                SELECT DISTINCT c.cited_item_key AS item_key, c.citing_paper_id
                FROM global_citations c
                JOIN target_citers t ON t.citing_paper_id = c.citing_paper_id
                WHERE c.cited_item_key <> ?
                  AND NOT EXISTS (
                      SELECT 1 FROM relation_reports rr
                      WHERE rr.direction = 'citations'
                        AND rr.item_key = c.cited_item_key
                        AND rr.external_paper_id = c.citing_paper_id
                        AND rr.status = 'disabled'
                  )
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


def _ledger_populated_item_keys() -> set[str]:
    """Every item key this project's V3/citation tables currently know about."""
    conn = get_db_connection()
    try:
        cursor = conn.cursor()
        keys: set[str] = set()
        for table, column in (
            ("item_citation_status", "item_key"),
            ("global_references", "citing_item_key"),
            ("global_citations", "cited_item_key"),
            ("document_structures", "item_key"),
            ("document_nodes", "item_key"),
            ("artifact_processing_status", "item_key"),
        ):
            try:
                cursor.execute(f"SELECT DISTINCT {column} FROM {table}")
            except sqlite3.Error:
                continue
            keys |= {str(row[0]).strip() for row in cursor.fetchall() if row[0]}
        return keys
    finally:
        conn.close()


def ledger_keys_pending_removal(current_item_keys: set[str]) -> set[str]:
    """What ``purge_removed_items`` would delete, without deleting it.

    Exposed so a caller can size-check the removal before committing to it --
    ``purge_removed_items`` itself takes no confirmation and deletes everything
    this returns, so a caller whose evidence is only an enumeration (as
    ``current_item_keys`` from Zotero can be) should look at the size first.
    """
    return _ledger_populated_item_keys() - current_item_keys


def purge_removed_items(current_item_keys: set[str]) -> dict[str, int]:
    """Zoteroから削除されたアイテムキーに関連するDBレコードを削除する。

    citation系・summary系・works系に加え、V3の文書構造
    (document_nodes / document_structures) と artifact状態
    (artifact_processing_status / artifact_processing_events) もパージする（R7）。

    Returns:
        削除件数の辞書。キー例: item_citation_status / global_citations /
        global_references / item_summaries / section_summaries / works /
        work_edges / work_links / document_nodes / document_structures /
        artifact_processing_status / artifact_processing_events。

    Note:
        本関数はdry-runフラグを持たない（呼び出し側 index_from_zotero.py が
        current_item_keys を与えて実削除する設計）。削除予定のitem_keyを事前確認したい
        場合は、実削除せずに差分だけ列挙できる。例::

            import sqlite3
            from src import db_relations
            conn = sqlite3.connect(db_relations.DB_PATH)
            # 候補は本関数がpurgeする全テーブルから集める（citation台帳だけでは
            # V3の構造・状態しか持たないitemを取りこぼす）。
            db_keys = set()
            for table, column in (
                ("item_citation_status", "item_key"),
                ("document_structures", "item_key"),
                ("artifact_processing_status", "item_key"),
            ):
                db_keys |= {r[0] for r in conn.execute(f"SELECT DISTINCT {column} FROM {table}")}
            removed = db_keys - current_item_keys  # ← これがpurge対象
            print(sorted(removed))
    """
    conn = get_db_connection()
    try:
        cursor = conn.cursor()

        # Candidates must be gathered from every table this function purges, not
        # just the citation ledger. Reading only ``item_citation_status`` meant
        # an item that never went through citation mapping was invisible here,
        # so the V3 structure and status rows this function is supposed to clean
        # were never reached: five items deleted from Zotero still held their
        # document_structures, document_nodes and artifact_processing_status
        # rows after a purge reported success (2026-07-27).
        db_keys: set[str] = set()
        for table, column in (
            ("item_citation_status", "item_key"),
            ("global_references", "citing_item_key"),
            ("global_citations", "cited_item_key"),
            ("document_structures", "item_key"),
            ("document_nodes", "item_key"),
            ("artifact_processing_status", "item_key"),
        ):
            try:
                cursor.execute(f"SELECT DISTINCT {column} FROM {table}")
            except sqlite3.Error:
                continue
            db_keys |= {str(row[0]).strip() for row in cursor.fetchall() if row[0]}
        removed_keys = db_keys - current_item_keys

        counts: dict[str, int] = {
            "item_citation_status": 0, "global_citations": 0, "global_references": 0,
            "item_summaries": 0, "section_summaries": 0,
            "works": 0, "work_edges": 0, "work_links": 0,
            "document_nodes": 0, "document_structures": 0,
            "artifact_processing_status": 0, "artifact_processing_events": 0,
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

        for table in ("item_summaries", "section_summaries"):
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

        # V3 文書構造・artifact状態レコードのパージ（R7）。
        # document_nodes は document_structures への FK を持たないため
        # document_structures 削除では cascade しない。item_key で document_nodes を
        # 直接一括削除する必要がある。get_db_connection() が PRAGMA foreign_keys=ON を
        # 有効化しているので、node_id を参照する document_node_summaries /
        # document_node_chunks / document_node_summary_parts は ON DELETE CASCADE により
        # 自動的に消える（parent_node_id の自己参照 CASCADE には依存しない）。
        cursor.execute(
            f"DELETE FROM document_nodes WHERE item_key IN ({placeholders})", params
        )
        counts["document_nodes"] = cursor.rowcount

        cursor.execute(
            f"DELETE FROM document_structures WHERE item_key IN ({placeholders})", params
        )
        counts["document_structures"] = cursor.rowcount

        cursor.execute(
            f"DELETE FROM artifact_processing_status WHERE item_key IN ({placeholders})", params
        )
        counts["artifact_processing_status"] = cursor.rowcount

        cursor.execute(
            f"DELETE FROM artifact_processing_events WHERE item_key IN ({placeholders})", params
        )
        counts["artifact_processing_events"] = cursor.rowcount

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


# ---------------------------------------------------------------------------
# Document structure v2 and processing-state ledger

_ARTIFACT_TYPES = {
    "extraction", "structure", "summary", "references", "embeddings", "summary_index",
}
_ARTIFACT_STATUSES = {
    "pending", "running", "success", "empty", "degraded", "blocked", "failed", "stale", "excluded",
}
_STRUCTURE_STATUSES = {"exact", "recovered", "flat_fallback", "unavailable"}


def replace_document_structure(
    item_key: str, *, source_fingerprint: str, structure_version: str,
    status: str, nodes: List[Dict[str, Any]], diagnostics: Optional[Dict[str, Any]] = None,
    confidence: Optional[float] = None,
) -> None:
    """Atomically replace one item's derived document tree."""
    _structure_repository().replace(
        item_key, source_fingerprint=source_fingerprint,
        structure_version=structure_version, status=status, nodes=nodes,
        diagnostics=diagnostics, confidence=confidence,
    )
def get_document_structure(item_key: str) -> Optional[Dict[str, Any]]:
    return _structure_repository().get_structure(item_key)
def get_document_nodes(item_key: str, *, include_chunks: bool = False) -> List[Dict[str, Any]]:
    """Return the canonical tree in parent-before-display order."""
    return _structure_repository().get_nodes(item_key, include_chunks=include_chunks)
def get_node_descendant_chunks(node_ids: List[str]) -> List[str]:
    """Expand structure node IDs to source chunks in stable document order."""
    return _structure_repository().descendant_chunks(node_ids)
def get_node_descendant_leaf_ids(node_ids: List[str]) -> List[str]:
    """Return descendant leaf node IDs that directly own source chunks."""
    return _structure_repository().descendant_leaf_ids(node_ids)
def save_document_node_summary(
    node_id: str, item_key: str, summary: str, *, summary_kind: str, model: str = "",
    prompt_version: str = "structure-v2", source_fingerprint: str,
    source_chunk_count: int, source_chars: int, quality_status: str = "accepted",
    input_scope: Optional[Dict[str, Any]] = None,
) -> None:
    summary = str(summary or "").strip()
    if not summary:
        raise ValueError("summary must not be empty")
    if summary_kind not in {"llm", "extractive"}:
        raise ValueError("summary_kind must be llm or extractive")
    if quality_status not in {"accepted", "candidate", "degraded", "disabled"}:
        raise ValueError("invalid node summary quality status")
    searchable = int(summary_kind == "llm" and quality_status in {"accepted", "candidate"})
    conn = get_db_connection()
    try:
        conn.execute('''
            INSERT INTO document_node_summaries
                (node_id, item_key, summary, summary_kind, model, prompt_version,
                 source_fingerprint, source_chunk_count, source_chars, searchable,
                 quality_status, input_scope_json, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
            ON CONFLICT(node_id) DO UPDATE SET
                item_key=excluded.item_key, summary=excluded.summary,
                summary_kind=excluded.summary_kind, model=excluded.model,
                prompt_version=excluded.prompt_version,
                source_fingerprint=excluded.source_fingerprint,
                source_chunk_count=excluded.source_chunk_count, source_chars=excluded.source_chars,
                searchable=excluded.searchable, quality_status=excluded.quality_status,
                input_scope_json=excluded.input_scope_json,
                updated_at=CURRENT_TIMESTAMP
        ''', (
            node_id, item_key, summary, summary_kind, model, prompt_version, source_fingerprint,
            int(source_chunk_count), int(source_chars), searchable, quality_status,
            json.dumps(input_scope or {}, ensure_ascii=False, sort_keys=True),
        ))
        conn.commit()
    finally:
        conn.close()


def replace_document_node_summary_parts(
    node_id: str, parts: List[Dict[str, Any]], *, prompt_version: str,
    source_fingerprint: str,
) -> None:
    """Persist ordered intermediate reductions used to make a node summary.

    These rows are audit material, not retrieval inputs.  Replacing them with
    the final node write keeps retries idempotent and prevents stale reduction
    traces from being shown after a source or prompt change.
    """
    conn = get_db_connection()
    try:
        conn.execute("DELETE FROM document_node_summary_parts WHERE node_id = ?", (node_id,))
        for ordinal, part in enumerate(parts):
            conn.execute('''
                INSERT INTO document_node_summary_parts
                    (node_id, part_ordinal, child_node_ids_json, summary, model,
                     prompt_version, source_fingerprint)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (
                node_id, ordinal,
                json.dumps(list(part.get("child_node_ids") or []), ensure_ascii=False),
                str(part.get("summary") or ""), str(part.get("model") or ""),
                prompt_version, source_fingerprint,
            ))
        conn.commit()
    finally:
        conn.close()


def get_document_node_summary_parts(node_id: str) -> List[Dict[str, Any]]:
    """Return a node's retained reduction inputs in document order."""
    normalized = str(node_id or "").strip()
    if not normalized:
        return []
    return get_document_node_summary_parts_for_nodes([normalized]).get(normalized, [])


def get_document_node_summary_parts_for_nodes(
    node_ids: List[str],
) -> Dict[str, List[Dict[str, Any]]]:
    """Return retained reduction inputs for several nodes in one query."""
    ids = list(dict.fromkeys(str(value or "").strip() for value in node_ids if value))
    if not ids:
        return {}
    placeholders = ",".join("?" for _ in ids)
    conn = get_db_connection()
    try:
        rows = conn.execute(f'''
            SELECT * FROM document_node_summary_parts
            WHERE node_id IN ({placeholders})
            ORDER BY node_id ASC, part_ordinal ASC
        ''', ids).fetchall()
        result: Dict[str, List[Dict[str, Any]]] = {}
        for row in rows:
            value = dict(row)
            try:
                value["child_node_ids"] = json.loads(value.pop("child_node_ids_json") or "[]")
            except (TypeError, ValueError):
                value["child_node_ids"] = []
            result.setdefault(str(value["node_id"]), []).append(value)
        return result
    finally:
        conn.close()


def get_document_node_summaries(
    item_key: str, *, searchable_only: bool = False,
) -> List[Dict[str, Any]]:
    conn = get_db_connection()
    try:
        rows = conn.execute('''
            SELECT s.*, n.parent_node_id, n.attachment_key, n.node_type, n.depth,
                   n.title, n.first_chunk_id, n.last_chunk_id
            FROM document_node_summaries s JOIN document_nodes n ON n.node_id = s.node_id
            WHERE s.item_key = ? AND (? = 0 OR s.searchable = 1)
            ORDER BY n.depth ASC, n.first_chunk_id ASC, n.ordinal ASC
        ''', (item_key, int(searchable_only))).fetchall()
        result = []
        for row in rows:
            value = dict(row)
            try:
                value["input_scope"] = json.loads(value.pop("input_scope_json") or "{}")
            except (TypeError, ValueError):
                value["input_scope"] = {}
            result.append(value)
        return result
    finally:
        conn.close()


def get_item_root_summaries(
    item_keys: List[str], *, searchable_only: bool = True,
) -> Dict[str, Dict[str, Any]]:
    """Return at most one V3 item-root summary per requested item."""
    keys = list(dict.fromkeys(str(value or "").strip() for value in item_keys if value))
    if not keys:
        return {}
    placeholders = ",".join("?" for _ in keys)
    conn = get_db_connection()
    try:
        rows = conn.execute(f'''
            SELECT s.*, n.parent_node_id, n.attachment_key, n.node_type, n.depth,
                   n.title, n.first_chunk_id, n.last_chunk_id
            FROM document_node_summaries s
            JOIN document_nodes n ON n.node_id = s.node_id
            WHERE s.item_key IN ({placeholders})
              AND n.node_type = 'item_root'
              AND (? = 0 OR s.searchable = 1)
            ORDER BY s.item_key, s.updated_at DESC
        ''', [*keys, int(searchable_only)]).fetchall()
        result: Dict[str, Dict[str, Any]] = {}
        for row in rows:
            value = dict(row)
            if value["item_key"] in result:
                continue
            try:
                value["input_scope"] = json.loads(value.pop("input_scope_json") or "{}")
            except (TypeError, ValueError):
                value["input_scope"] = {}
            result[str(value["item_key"])] = value
        return result
    finally:
        conn.close()


def get_item_root_summary(
    item_key: str, *, searchable_only: bool = True,
) -> Optional[Dict[str, Any]]:
    """Return the current V3 item-root summary without loading child nodes."""
    return get_item_root_summaries(
        [item_key], searchable_only=searchable_only,
    ).get(str(item_key or "").strip())


def get_document_node_summary(
    node_id: str, *, searchable_only: bool = False,
) -> Optional[Dict[str, Any]]:
    """Return one V3 node summary with its structure metadata."""
    conn = get_db_connection()
    try:
        row = conn.execute('''
            SELECT s.*, n.parent_node_id, n.attachment_key, n.node_type, n.depth,
                   n.title, n.first_chunk_id, n.last_chunk_id
            FROM document_node_summaries s
            JOIN document_nodes n ON n.node_id = s.node_id
            WHERE s.node_id = ? AND (? = 0 OR s.searchable = 1)
        ''', ((node_id or "").strip(), int(searchable_only))).fetchone()
        if row is None:
            return None
        value = dict(row)
        try:
            value["input_scope"] = json.loads(value.pop("input_scope_json") or "{}")
        except (TypeError, ValueError):
            value["input_scope"] = {}
        return value
    finally:
        conn.close()


def get_searchable_document_node_ids(node_ids: List[str]) -> set[str]:
    """Return the requested node IDs still eligible for summary routing."""
    ids = list(dict.fromkeys(str(value or "").strip() for value in node_ids if value))
    if not ids:
        return set()
    placeholders = ",".join("?" for _ in ids)
    conn = get_db_connection()
    try:
        rows = conn.execute(
            f"SELECT node_id FROM document_node_summaries "
            f"WHERE node_id IN ({placeholders}) AND searchable = 1",
            ids,
        ).fetchall()
        return {str(row["node_id"]) for row in rows}
    finally:
        conn.close()


def get_all_document_node_summaries(
    *, searchable_only: bool = False, item_key: Optional[str] = None,
) -> List[Dict[str, Any]]:
    conn = get_db_connection()
    try:
        params: List[Any] = [int(searchable_only)]
        clause = ""
        if item_key is not None:
            clause = " AND s.item_key = ?"
            params.append(item_key)
        rows = conn.execute(f'''
            SELECT s.*, n.parent_node_id, n.attachment_key, n.node_type, n.depth,
                   n.title, n.first_chunk_id, n.last_chunk_id
            FROM document_node_summaries s JOIN document_nodes n ON n.node_id = s.node_id
            WHERE (? = 0 OR s.searchable = 1){clause}
            ORDER BY s.item_key, n.depth ASC, n.first_chunk_id ASC, n.ordinal ASC
        ''', params).fetchall()
        return [dict(row) for row in rows]
    finally:
        conn.close()


def get_document_node_summary_reuse_cache(item_key: str) -> List[Dict[str, Any]]:
    """Return summaries retained across a destructive structure replacement.

    Callers must still verify their exact prompt-input fingerprint before using
    a row.  Keeping this separate from live summaries makes a failed rebuild
    recoverable without exposing stale rows to retrieval.
    """
    conn = get_db_connection()
    try:
        rows = conn.execute('''
            SELECT * FROM document_node_summary_reuse_cache
            WHERE item_key = ? ORDER BY cached_at DESC, node_id ASC
        ''', (item_key,)).fetchall()
        result = []
        for row in rows:
            value = dict(row)
            try:
                value["input_scope"] = json.loads(value.pop("input_scope_json") or "{}")
            except (TypeError, ValueError):
                value["input_scope"] = {}
            result.append(value)
        return result
    finally:
        conn.close()


def delete_document_node_summaries(item_key: str) -> int:
    conn = get_db_connection()
    try:
        cursor = conn.execute("DELETE FROM document_node_summaries WHERE item_key = ?", (item_key,))
        conn.commit()
        return max(0, int(cursor.rowcount))
    finally:
        conn.close()


def _status_counts_json(counts: Optional[Dict[str, Any]]) -> Optional[str]:
    return json.dumps(counts, ensure_ascii=False, sort_keys=True) if counts is not None else None


def mark_artifact_status(
    item_key: str, artifact_type: str, status: str, *, attachment_key: Optional[str] = None,
    reason_code: Optional[str] = None, message: Optional[str] = None,
    retryable: bool = False, source_fingerprint: Optional[str] = None,
    processor_version: Optional[str] = None, model: Optional[str] = None,
    counts: Optional[Dict[str, Any]] = None, fallback_kind: Optional[str] = None,
    run_id: Optional[str] = None,
) -> Dict[str, Any]:
    """Record a processing state transition and append an audit event."""
    return _artifact_repository().mark_status(
        item_key, artifact_type, status, attachment_key=attachment_key,
        reason_code=reason_code, message=message, retryable=retryable,
        source_fingerprint=source_fingerprint, processor_version=processor_version,
        model=model, counts=counts, fallback_kind=fallback_kind, run_id=run_id,
    )


def drop_stale_identity_rows(stale_item_key: str, current_item_key: str) -> int:
    """Retire ledger rows left under an attachment key that now has a parent.

    A top-level PDF is tracked under its attachment key (``scope_item_key``
    falls back to it). Once the user files it under a parent item, every
    subsequent write goes to the parent key and the old rows describe an
    identity that no longer exists, showing as permanently unresolved. They
    describe the same physical file, so they are retired rather than migrated:
    migrating would collide with the rows the current run is already writing
    under the correct key.

    Only status and structure bookkeeping is touched -- never chunks. The
    content is alive and correctly indexed under ``current_item_key``.

    Returns the number of ledger rows removed.
    """
    stale_item_key = (stale_item_key or "").strip()
    current_item_key = (current_item_key or "").strip()
    if not stale_item_key or not current_item_key or stale_item_key == current_item_key:
        return 0
    conn = get_db_connection()
    try:
        removed = conn.execute(
            "DELETE FROM artifact_processing_status WHERE item_key = ?",
            (stale_item_key,),
        ).rowcount or 0
        # document_nodes has no foreign key to document_structures (there is
        # none to have -- PRAGMA foreign_key_list confirms it), and nothing
        # else deletes it when only document_structures was retired here. A
        # re-parented attachment therefore left orphaned nodes and their
        # summaries live under the dead key, still feeding the summary index
        # and leaf routing (2026-07-28, found in code review).
        try:
            conn.execute(
                "DELETE FROM document_node_chunks WHERE node_id IN "
                "(SELECT node_id FROM document_nodes WHERE item_key = ?)",
                (stale_item_key,),
            )
        except sqlite3.Error:
            pass
        for table in (
            "artifact_processing_events", "document_structures",
            "document_node_summaries", "document_node_summary_parts",
            "document_node_summary_reuse_cache", "document_nodes",
        ):
            try:
                conn.execute(f"DELETE FROM {table} WHERE item_key = ?", (stale_item_key,))
            except sqlite3.Error:
                pass
        conn.commit()
    finally:
        conn.close()
    return int(removed)


def purge_artifact_status_for_attachments(attachment_keys: Iterable[str]) -> int:
    """Remove attachment-scoped ledger rows for attachments no longer in Zotero.

    purge_removed_items() only acts at item-key granularity, and
    drop_stale_identity_rows() only handles an attachment gaining a parent --
    neither covers an attachment being deleted and replaced by a new one
    under the *same*, still-live parent item. The old attachment_key's
    extraction row then has nothing left to ever revisit it: the item is
    live, so it is never a purge candidate, and the new attachment is tracked
    under its own key. Called from the same call site that already confirmed
    each key is gone from Zotero (index_from_zotero.py's stale-attachment
    deletion), so no second confirmation is needed here (2026-08-03).

    Only rows with a non-empty ``attachment_key`` can match, so item-scoped
    bookkeeping (structure/summary/embeddings, stored under ``''``) is never
    touched.
    """
    keys = sorted({str(key or "").strip() for key in attachment_keys if str(key or "").strip()})
    if not keys:
        return 0
    conn = get_db_connection()
    try:
        placeholders = ",".join("?" * len(keys))
        cursor = conn.cursor()
        cursor.execute(
            f"DELETE FROM artifact_processing_status WHERE attachment_key IN ({placeholders})",
            keys,
        )
        removed = cursor.rowcount or 0
        cursor.execute(
            f"DELETE FROM artifact_processing_events WHERE attachment_key IN ({placeholders})",
            keys,
        )
        conn.commit()
        return int(removed)
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def get_item_processing_status(item_key: str) -> List[Dict[str, Any]]:
    return _artifact_repository().list_for_item(item_key)


def get_artifact_processing_statuses(
    *, artifact_type: Optional[str] = None, reason_code: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Return artifact states across items for deterministic maintenance queues."""
    return _artifact_repository().list_statuses(
        artifact_type=artifact_type, reason_code=reason_code,
    )


def get_processing_status_summary() -> List[Dict[str, Any]]:
    return _artifact_repository().status_summary()


def recover_interrupted_artifacts(*, older_than_seconds: int = 3600) -> int:
    """Turn abandoned ``running`` rows into retryable failures on startup."""
    return _artifact_repository().recover_interrupted(
        older_than_seconds=older_than_seconds,
    )
