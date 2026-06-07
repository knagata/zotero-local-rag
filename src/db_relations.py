import sqlite3
import os
from pathlib import Path
from typing import List, Dict, Any, Optional

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB_PATH = os.environ.get("RELATIONS_DB_PATH", os.path.join(ROOT, "data", "relations.db"))

def get_db_connection():
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    conn = get_db_connection()
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
            UNIQUE(citing_paper_id, cited_chunk_id, context_snippet)
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
            s2_status TEXT DEFAULT 'mapped',
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(cited_paper_id, citing_chunk_id, context_snippet, raw_reference_text)
        )
    ''')
    
    try:
        cursor.execute("ALTER TABLE global_references ADD COLUMN s2_status TEXT DEFAULT 'mapped'")
    except sqlite3.OperationalError:
        pass
        
    # Migrations for citation counts
    try:
        cursor.execute("ALTER TABLE global_citations ADD COLUMN citing_citation_count INTEGER DEFAULT 0")
        cursor.execute("ALTER TABLE global_citations ADD COLUMN citing_influential_count INTEGER DEFAULT 0")
    except sqlite3.OperationalError:
        pass

    try:
        cursor.execute("ALTER TABLE global_references ADD COLUMN cited_citation_count INTEGER DEFAULT 0")
        cursor.execute("ALTER TABLE global_references ADD COLUMN cited_influential_count INTEGER DEFAULT 0")
    except sqlite3.OperationalError:
        pass
        
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_citing_chunk_id ON global_references(citing_chunk_id)')
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_citing_item_key ON global_references(citing_item_key)')
    
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS item_citation_status (
            item_key TEXT PRIMARY KEY,
            s2_status TEXT,
            last_checked_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    
    conn.commit()
    conn.close()

def update_item_citation_status(item_key: str, s2_status: str):
    conn = get_db_connection()
    try:
        cursor = conn.cursor()
        cursor.execute('''
            INSERT INTO item_citation_status (item_key, s2_status, last_checked_at)
            VALUES (?, ?, CURRENT_TIMESTAMP)
            ON CONFLICT(item_key) DO UPDATE SET 
            s2_status = excluded.s2_status,
            last_checked_at = CURRENT_TIMESTAMP
        ''', (item_key, s2_status))
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
    cited_chunk_id: str,
    similarity_distance: float,
    page_hint: Optional[str],
    citing_citation_count: int = 0,
    citing_influential_count: int = 0
):
    conn = get_db_connection()
    cursor = conn.cursor()
    try:
        cursor.execute('''
            INSERT OR REPLACE INTO global_citations 
            (citing_paper_id, citing_title, citing_year, context_snippet, cited_item_key, cited_chunk_id, similarity_distance, page_hint, citing_citation_count, citing_influential_count)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (citing_paper_id, citing_title, citing_year, context_snippet, cited_item_key, cited_chunk_id, similarity_distance, page_hint, citing_citation_count, citing_influential_count))
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
        GROUP BY cited_chunk_id
        ORDER BY citation_count DESC, best_distance ASC
    ''', (item_key,))
    rows = cursor.fetchall()
    conn.close()
    return [dict(row) for row in rows]

def insert_reference(cited_paper_id: str, cited_title: str, cited_year: int, context_snippet: str, citing_item_key: str, citing_chunk_id: str, similarity_distance: float, page_hint: Optional[str] = None, source: str = 's2', raw_reference_text: Optional[str] = None, s2_status: str = 'mapped', cited_citation_count: int = 0, cited_influential_count: int = 0):
    conn = get_db_connection()
    try:
        cursor = conn.cursor()
        cursor.execute('''
            INSERT OR REPLACE INTO global_references 
            (cited_paper_id, cited_title, cited_year, context_snippet, citing_item_key, citing_chunk_id, similarity_distance, page_hint, source, raw_reference_text, s2_status, cited_citation_count, cited_influential_count)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (cited_paper_id, cited_title, cited_year, context_snippet, citing_item_key, citing_chunk_id, similarity_distance, page_hint, source, raw_reference_text, s2_status, cited_citation_count, cited_influential_count))
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

# Initialize DB on import
init_db()
