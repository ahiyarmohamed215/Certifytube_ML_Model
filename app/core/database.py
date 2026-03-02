"""MySQL database helpers for transcript caching.

Provides connection pooling and CRUD operations for the `transcripts` table.
The table is created automatically on first import.

If MySQL is unavailable, all operations gracefully degrade to no-ops
(cache misses return None, saves are silently skipped).
"""

from __future__ import annotations

import logging
from typing import Optional

import mysql.connector
from mysql.connector import pooling

from app.core.settings import settings

log = logging.getLogger(__name__)

_CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS transcripts (
    video_id VARCHAR(20) PRIMARY KEY,
    transcript LONGTEXT NOT NULL,
    fetched_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
"""

# ---------------------------------------------------------------------------
# Connection pool (lazy-initialized)
# ---------------------------------------------------------------------------
_pool: Optional[pooling.MySQLConnectionPool] = None
_pool_failed: bool = False  # Tracks if pool creation already failed


def _get_pool() -> Optional[pooling.MySQLConnectionPool]:
    """Return the shared connection pool, creating it on first call.

    Returns ``None`` if MySQL is not available (graceful degradation).
    """
    global _pool, _pool_failed
    if _pool_failed:
        return None
    if _pool is not None:
        return _pool
    try:
        _pool = pooling.MySQLConnectionPool(
            pool_name="certifytube_pool",
            pool_size=5,
            host=settings.db_host,
            port=settings.db_port,
            user=settings.db_user,
            password=settings.db_password,
            database=settings.db_name,
            charset="utf8mb4",
            collation="utf8mb4_general_ci",
            autocommit=True,
        )
        # Ensure the transcripts table exists.
        _init_table()
        return _pool
    except mysql.connector.Error:
        log.warning(
            "MySQL is not available (host=%s, port=%s, user=%s, db=%s). "
            "Transcript caching is DISABLED — transcripts will be fetched from YouTube every time.",
            settings.db_host, settings.db_port, settings.db_user, settings.db_name,
        )
        _pool_failed = True
        return None


def _init_table() -> None:
    """Create the transcripts table if it does not exist."""
    try:
        conn = _pool.get_connection()  # type: ignore[union-attr]
        try:
            cursor = conn.cursor()
            cursor.execute(_CREATE_TABLE_SQL)
            cursor.close()
        finally:
            conn.close()
        log.info("transcripts table ready")
    except mysql.connector.Error:
        log.exception("Failed to initialize transcripts table")
        raise


# ---------------------------------------------------------------------------
# Public helpers
# ---------------------------------------------------------------------------

def get_cached_transcript(video_id: str) -> Optional[str]:
    """Return the cached transcript for *video_id*, or ``None`` if not found."""
    pool = _get_pool()
    if pool is None:
        return None
    conn = pool.get_connection()
    try:
        cursor = conn.cursor()
        cursor.execute(
            "SELECT transcript FROM transcripts WHERE video_id = %s",
            (video_id,),
        )
        row = cursor.fetchone()
        cursor.close()
        if row:
            log.info("Transcript cache HIT for video_id=%s", video_id)
            return row[0]
        log.info("Transcript cache MISS for video_id=%s", video_id)
        return None
    except mysql.connector.Error:
        log.exception("Error reading transcript cache for video_id=%s", video_id)
        return None
    finally:
        conn.close()


def save_transcript(video_id: str, transcript_text: str) -> None:
    """Insert or update the cached transcript for *video_id*."""
    pool = _get_pool()
    if pool is None:
        return
    conn = pool.get_connection()
    try:
        cursor = conn.cursor()
        cursor.execute(
            """
            INSERT INTO transcripts (video_id, transcript)
            VALUES (%s, %s)
            ON DUPLICATE KEY UPDATE transcript = VALUES(transcript),
                                    fetched_at = CURRENT_TIMESTAMP
            """,
            (video_id, transcript_text),
        )
        cursor.close()
        log.info("Transcript cached for video_id=%s", video_id)
    except mysql.connector.Error:
        log.exception("Error saving transcript for video_id=%s", video_id)
    finally:
        conn.close()
