"""MySQL database helpers for transcript, engagement, and quiz persistence.

Provides connection pooling and CRUD operations for transcript caching,
engagement-session persistence, and quiz generation/grading state.
Tables are created automatically on first import.

If MySQL is unavailable, transcript cache operations degrade to no-ops
(cache misses return None, saves are silently skipped). Engagement and quiz
flows require durable state and therefore raise explicit persistence errors.
"""

from __future__ import annotations

import json
import logging
from datetime import date, datetime
from decimal import Decimal
from typing import Any, Dict, List, Optional
from uuid import uuid4

import mysql.connector
from mysql.connector import pooling

from app.core.settings import settings

log = logging.getLogger(__name__)

_CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS transcripts (
    video_id VARCHAR(20) PRIMARY KEY,
    transcript LONGTEXT NOT NULL,
    processed_transcript LONGTEXT,
    fetched_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
"""

_CHECK_COLUMN_EXISTS_SQL = """
SELECT COUNT(*) FROM INFORMATION_SCHEMA.COLUMNS
WHERE TABLE_SCHEMA = DATABASE()
  AND TABLE_NAME = 'transcripts'
  AND COLUMN_NAME = 'processed_transcript';
"""

_ADD_PROCESSED_COLUMN_SQL = """
ALTER TABLE transcripts
ADD COLUMN processed_transcript LONGTEXT AFTER transcript;
"""

_CREATE_ENGAGEMENT_SESSIONS_SQL = """
CREATE TABLE IF NOT EXISTS engagement_sessions (
    session_id VARCHAR(255) PRIMARY KEY,
    feature_version VARCHAR(50) NOT NULL,
    input_source VARCHAR(20) NOT NULL,
    feature_count INT NOT NULL DEFAULT 0,
    event_count INT NOT NULL DEFAULT 0,
    features_json LONGTEXT NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
"""

_CREATE_ENGAGEMENT_EVENTS_SQL = """
CREATE TABLE IF NOT EXISTS engagement_events (
    id BIGINT AUTO_INCREMENT PRIMARY KEY,
    session_id VARCHAR(255) NOT NULL,
    event_order INT NOT NULL,
    event_id VARCHAR(255) NULL,
    event_type VARCHAR(100) NULL,
    payload_json LONGTEXT NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE KEY uq_engagement_event_order (session_id, event_order),
    KEY idx_engagement_events_session (session_id),
    CONSTRAINT fk_engagement_events_session
        FOREIGN KEY (session_id) REFERENCES engagement_sessions(session_id)
        ON DELETE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
"""

_CREATE_ENGAGEMENT_FEATURES_SQL = """
CREATE TABLE IF NOT EXISTS engagement_features (
    id BIGINT AUTO_INCREMENT PRIMARY KEY,
    session_id VARCHAR(255) NOT NULL,
    feature_name VARCHAR(255) NOT NULL,
    feature_value DOUBLE NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE KEY uq_engagement_feature (session_id, feature_name),
    KEY idx_engagement_features_session (session_id),
    CONSTRAINT fk_engagement_features_session
        FOREIGN KEY (session_id) REFERENCES engagement_sessions(session_id)
        ON DELETE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
"""

_CREATE_ENGAGEMENT_ANALYSES_SQL = """
CREATE TABLE IF NOT EXISTS engagement_analyses (
    analysis_id CHAR(36) PRIMARY KEY,
    session_id VARCHAR(255) NOT NULL,
    feature_version VARCHAR(50) NOT NULL,
    model_name VARCHAR(20) NOT NULL,
    input_source VARCHAR(20) NOT NULL,
    feature_count INT NOT NULL DEFAULT 0,
    event_count INT NOT NULL DEFAULT 0,
    engagement_score DECIMAL(8,6) NOT NULL,
    explanation LONGTEXT NOT NULL,
    top_negative_json LONGTEXT NOT NULL,
    top_positive_json LONGTEXT NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    KEY idx_engagement_analyses_session (session_id),
    KEY idx_engagement_analyses_model (model_name),
    CONSTRAINT fk_engagement_analyses_session
        FOREIGN KEY (session_id) REFERENCES engagement_sessions(session_id)
        ON DELETE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
"""

_CREATE_QUIZ_ATTEMPTS_SQL = """
CREATE TABLE IF NOT EXISTS quiz_attempts (
    quiz_id CHAR(36) PRIMARY KEY,
    session_id VARCHAR(255) NOT NULL,
    video_id VARCHAR(50) NOT NULL,
    total_questions INT NOT NULL,
    quiz_score_percent DECIMAL(5,2) NULL,
    correct_answers INT NULL,
    incorrect_answers INT NULL,
    unanswered_questions INT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    graded_at TIMESTAMP NULL DEFAULT NULL
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
"""

_CREATE_QUIZ_QUESTIONS_SQL = """
CREATE TABLE IF NOT EXISTS quiz_questions (
    id BIGINT AUTO_INCREMENT PRIMARY KEY,
    quiz_id CHAR(36) NOT NULL,
    question_id VARCHAR(50) NOT NULL,
    question_type VARCHAR(20) NOT NULL,
    question_text LONGTEXT NOT NULL,
    options_json LONGTEXT,
    correct_answer LONGTEXT NOT NULL,
    explanation LONGTEXT NOT NULL,
    difficulty VARCHAR(20) NOT NULL,
    source_segment LONGTEXT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE KEY uq_quiz_question (quiz_id, question_id),
    CONSTRAINT fk_quiz_questions_attempt
        FOREIGN KEY (quiz_id) REFERENCES quiz_attempts(quiz_id)
        ON DELETE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
"""

_CREATE_QUIZ_SUBMISSIONS_SQL = """
CREATE TABLE IF NOT EXISTS quiz_submissions (
    id BIGINT AUTO_INCREMENT PRIMARY KEY,
    quiz_id CHAR(36) NOT NULL,
    question_id VARCHAR(50) NOT NULL,
    submitted_answer LONGTEXT NOT NULL,
    is_correct TINYINT(1) NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE KEY uq_quiz_submission (quiz_id, question_id),
    CONSTRAINT fk_quiz_submissions_attempt
        FOREIGN KEY (quiz_id) REFERENCES quiz_attempts(quiz_id)
        ON DELETE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
"""


class QuizPersistenceError(Exception):
    pass


class EngagementPersistenceError(Exception):
    pass


# ---------------------------------------------------------------------------
# Connection pool (lazy-initialized)
# ---------------------------------------------------------------------------
_pool: Optional[pooling.MySQLConnectionPool] = None
_pool_failed: bool = False  # Tracks if pool creation already failed


def _ensure_database_exists() -> None:
    """Create the configured database if the MySQL server is reachable."""
    conn = mysql.connector.connect(
        host=settings.db_host,
        port=settings.db_port,
        user=settings.db_user,
        password=settings.db_password,
        autocommit=True,
    )
    try:
        cursor = conn.cursor()
        db_name = settings.db_name.replace("`", "``")
        cursor.execute(
            f"CREATE DATABASE IF NOT EXISTS `{db_name}` "
            "CHARACTER SET utf8mb4 COLLATE utf8mb4_general_ci"
        )
        cursor.close()
    finally:
        conn.close()


def _json_default(value: Any) -> Any:
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    if isinstance(value, Decimal):
        return float(value)
    item_method = getattr(value, "item", None)
    if callable(item_method):
        try:
            return item_method()
        except Exception:
            pass
    return str(value)


def _json_dumps(payload: Any) -> str:
    return json.dumps(payload, default=_json_default, ensure_ascii=False)


def _normalize_feature_map(features: Dict[str, Any]) -> Dict[str, float]:
    return {str(name): float(value) for name, value in features.items()}


def _nullable_str(value: Any) -> Optional[str]:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


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
        _ensure_database_exists()
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
        # Ensure all MySQL-backed tables exist.
        _init_tables()
        return _pool
    except mysql.connector.Error:
        log.warning(
            "MySQL is not available (host=%s, port=%s, user=%s, db=%s). "
            "Transcript caching is disabled and engagement/quiz persistence is unavailable.",
            settings.db_host,
            settings.db_port,
            settings.db_user,
            settings.db_name,
        )
        _pool_failed = True
        return None


def _init_tables() -> None:
    """Create transcript, engagement, and quiz tables if they do not exist.

    Also migrates existing tables by adding the ``processed_transcript``
    column when it is missing.
    """
    try:
        conn = _pool.get_connection()  # type: ignore[union-attr]
        try:
            cursor = conn.cursor()
            cursor.execute(_CREATE_TABLE_SQL)
            cursor.execute(_CREATE_ENGAGEMENT_SESSIONS_SQL)
            cursor.execute(_CREATE_ENGAGEMENT_EVENTS_SQL)
            cursor.execute(_CREATE_ENGAGEMENT_FEATURES_SQL)
            cursor.execute(_CREATE_ENGAGEMENT_ANALYSES_SQL)
            cursor.execute(_CREATE_QUIZ_ATTEMPTS_SQL)
            cursor.execute(_CREATE_QUIZ_QUESTIONS_SQL)
            cursor.execute(_CREATE_QUIZ_SUBMISSIONS_SQL)
            # Migration: add processed_transcript column to existing tables
            cursor.execute(_CHECK_COLUMN_EXISTS_SQL)
            (col_exists,) = cursor.fetchone()
            if col_exists == 0:
                try:
                    cursor.execute(_ADD_PROCESSED_COLUMN_SQL)
                    log.info("Added processed_transcript column to transcripts table")
                except mysql.connector.Error:
                    log.warning("Could not add processed_transcript column (may already exist)")
            cursor.close()
        finally:
            conn.close()
        log.info("MySQL tables ready (transcripts + engagement + quiz storage)")
    except mysql.connector.Error:
        log.exception("Failed to initialize MySQL tables")
        raise


def _require_pool_for_quiz() -> pooling.MySQLConnectionPool:
    pool = _get_pool()
    if pool is None:
        raise QuizPersistenceError(
            "MySQL is unavailable. Quiz generation/grading requires database persistence."
        )
    return pool


def _require_pool_for_engagement() -> pooling.MySQLConnectionPool:
    pool = _get_pool()
    if pool is None:
        raise EngagementPersistenceError(
            "MySQL is unavailable. Engagement analysis requires database persistence."
        )
    return pool


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


def get_cached_processed_transcript(video_id: str) -> Optional[str]:
    """Return the cached *processed* transcript for *video_id*, or ``None``."""
    pool = _get_pool()
    if pool is None:
        return None
    conn = pool.get_connection()
    try:
        cursor = conn.cursor()
        cursor.execute(
            "SELECT processed_transcript FROM transcripts WHERE video_id = %s",
            (video_id,),
        )
        row = cursor.fetchone()
        cursor.close()
        if row and row[0]:
            log.info("Processed transcript cache HIT for video_id=%s", video_id)
            return row[0]
        log.info("Processed transcript cache MISS for video_id=%s", video_id)
        return None
    except mysql.connector.Error:
        log.exception("Error reading processed transcript cache for video_id=%s", video_id)
        return None
    finally:
        conn.close()


def save_processed_transcript(video_id: str, processed_text: str) -> None:
    """Update the processed transcript column for *video_id*."""
    pool = _get_pool()
    if pool is None:
        return
    conn = pool.get_connection()
    try:
        cursor = conn.cursor()
        cursor.execute(
            """
            UPDATE transcripts
            SET processed_transcript = %s
            WHERE video_id = %s
            """,
            (processed_text, video_id),
        )
        cursor.close()
        log.info("Processed transcript cached for video_id=%s", video_id)
    except mysql.connector.Error:
        log.exception("Error saving processed transcript for video_id=%s", video_id)
    finally:
        conn.close()


def save_engagement_analysis(
    session_id: str,
    feature_version: str,
    model_name: str,
    input_source: str,
    features: Dict[str, Any],
    events: Optional[List[Dict[str, Any]]],
    engagement_score: float,
    explanation: str,
    top_negative: List[Dict[str, Any]],
    top_positive: List[Dict[str, Any]],
) -> None:
    pool = _require_pool_for_engagement()
    normalized_features = _normalize_feature_map(features)
    event_rows = list(events or [])
    analysis_id = str(uuid4())
    conn = pool.get_connection()
    try:
        conn.autocommit = False
        cursor = conn.cursor()
        cursor.execute(
            """
            INSERT INTO engagement_sessions (
                session_id, feature_version, input_source, feature_count, event_count, features_json
            )
            VALUES (%s, %s, %s, %s, %s, %s)
            ON DUPLICATE KEY UPDATE
                feature_version = VALUES(feature_version),
                input_source = VALUES(input_source),
                feature_count = VALUES(feature_count),
                event_count = VALUES(event_count),
                features_json = VALUES(features_json),
                updated_at = CURRENT_TIMESTAMP
            """,
            (
                session_id,
                feature_version,
                input_source,
                len(normalized_features),
                len(event_rows),
                _json_dumps(normalized_features),
            ),
        )

        cursor.execute(
            "DELETE FROM engagement_events WHERE session_id = %s",
            (session_id,),
        )
        for index, event in enumerate(event_rows, start=1):
            event_payload = dict(event)
            event_payload["session_id"] = session_id
            cursor.execute(
                """
                INSERT INTO engagement_events (
                    session_id, event_order, event_id, event_type, payload_json
                )
                VALUES (%s, %s, %s, %s, %s)
                """,
                (
                    session_id,
                    index,
                    _nullable_str(event_payload.get("event_id")),
                    _nullable_str(event_payload.get("event_type")),
                    _json_dumps(event_payload),
                ),
            )

        cursor.execute(
            "DELETE FROM engagement_features WHERE session_id = %s",
            (session_id,),
        )
        for feature_name, feature_value in normalized_features.items():
            cursor.execute(
                """
                INSERT INTO engagement_features (session_id, feature_name, feature_value)
                VALUES (%s, %s, %s)
                """,
                (session_id, feature_name, feature_value),
            )

        cursor.execute(
            """
            INSERT INTO engagement_analyses (
                analysis_id, session_id, feature_version, model_name, input_source,
                feature_count, event_count, engagement_score, explanation,
                top_negative_json, top_positive_json
            )
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            """,
            (
                analysis_id,
                session_id,
                feature_version,
                model_name,
                input_source,
                len(normalized_features),
                len(event_rows),
                float(engagement_score),
                explanation,
                _json_dumps(top_negative),
                _json_dumps(top_positive),
            ),
        )

        conn.commit()
        cursor.close()
        log.info(
            "Engagement analysis persisted: analysis_id=%s session_id=%s model=%s",
            analysis_id,
            session_id,
            model_name,
        )
    except mysql.connector.Error as exc:
        conn.rollback()
        log.exception(
            "Error saving engagement analysis session_id=%s model=%s",
            session_id,
            model_name,
        )
        raise EngagementPersistenceError("Failed to persist engagement analysis.") from exc
    finally:
        conn.autocommit = True
        conn.close()


def save_generated_quiz(
    quiz_id: str,
    session_id: str,
    video_id: str,
    questions: List[Dict[str, Any]],
) -> None:
    pool = _require_pool_for_quiz()
    conn = pool.get_connection()
    try:
        conn.autocommit = False
        cursor = conn.cursor()
        cursor.execute(
            """
            INSERT INTO quiz_attempts (quiz_id, session_id, video_id, total_questions)
            VALUES (%s, %s, %s, %s)
            """,
            (quiz_id, session_id, video_id, len(questions)),
        )

        for question in questions:
            options = question.get("options")
            cursor.execute(
                """
                INSERT INTO quiz_questions (
                    quiz_id, question_id, question_type, question_text, options_json,
                    correct_answer, explanation, difficulty, source_segment
                )
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                """,
                (
                    quiz_id,
                    str(question["question_id"]),
                    str(question["type"]),
                    str(question["question"]),
                    _json_dumps(options) if options is not None else None,
                    str(question["correct_answer"]),
                    str(question["explanation"]),
                    str(question["difficulty"]),
                    str(question.get("source_segment", "")) or None,
                ),
            )

        conn.commit()
        cursor.close()
        log.info("Generated quiz persisted: quiz_id=%s session_id=%s", quiz_id, session_id)
    except mysql.connector.Error as exc:
        conn.rollback()
        log.exception("Error saving generated quiz quiz_id=%s", quiz_id)
        raise QuizPersistenceError("Failed to persist generated quiz.") from exc
    finally:
        conn.autocommit = True
        conn.close()


def get_quiz_answer_key(quiz_id: str) -> Optional[Dict[str, Any]]:
    pool = _require_pool_for_quiz()
    conn = pool.get_connection()
    try:
        cursor = conn.cursor(dictionary=True)
        cursor.execute(
            """
            SELECT quiz_id, session_id, video_id, total_questions
            FROM quiz_attempts
            WHERE quiz_id = %s
            """,
            (quiz_id,),
        )
        attempt = cursor.fetchone()
        if not attempt:
            cursor.close()
            return None

        cursor.execute(
            """
            SELECT question_id, correct_answer, explanation
            FROM quiz_questions
            WHERE quiz_id = %s
            ORDER BY id ASC
            """,
            (quiz_id,),
        )
        rows = cursor.fetchall()
        cursor.close()
        return {
            "quiz_id": attempt["quiz_id"],
            "session_id": attempt["session_id"],
            "video_id": attempt["video_id"],
            "total_questions": int(attempt["total_questions"]),
            "answer_key": [
                {
                    "question_id": row["question_id"],
                    "correct_answer": row["correct_answer"],
                    "explanation": row["explanation"],
                }
                for row in rows
            ],
        }
    except mysql.connector.Error as exc:
        log.exception("Error loading quiz answer key quiz_id=%s", quiz_id)
        raise QuizPersistenceError("Failed to load quiz answer key.") from exc
    finally:
        conn.close()


def save_quiz_grading_result(
    quiz_id: str,
    quiz_score_percent: float,
    correct_answers: int,
    incorrect_answers: int,
    unanswered_questions: int,
    results: List[Dict[str, Any]],
) -> None:
    pool = _require_pool_for_quiz()
    conn = pool.get_connection()
    try:
        conn.autocommit = False
        cursor = conn.cursor()
        cursor.execute(
            """
            UPDATE quiz_attempts
            SET quiz_score_percent = %s,
                correct_answers = %s,
                incorrect_answers = %s,
                unanswered_questions = %s,
                graded_at = CURRENT_TIMESTAMP
            WHERE quiz_id = %s
            """,
            (
                quiz_score_percent,
                correct_answers,
                incorrect_answers,
                unanswered_questions,
                quiz_id,
            ),
        )

        for row in results:
            cursor.execute(
                """
                INSERT INTO quiz_submissions (quiz_id, question_id, submitted_answer, is_correct)
                VALUES (%s, %s, %s, %s)
                ON DUPLICATE KEY UPDATE
                    submitted_answer = VALUES(submitted_answer),
                    is_correct = VALUES(is_correct),
                    created_at = CURRENT_TIMESTAMP
                """,
                (
                    quiz_id,
                    str(row["question_id"]),
                    str(row["submitted_answer"]),
                    1 if row["is_correct"] else 0,
                ),
            )

        conn.commit()
        cursor.close()
        log.info("Quiz grading persisted: quiz_id=%s score=%.2f", quiz_id, quiz_score_percent)
    except mysql.connector.Error as exc:
        conn.rollback()
        log.exception("Error saving quiz grading result quiz_id=%s", quiz_id)
        raise QuizPersistenceError("Failed to persist quiz grading result.") from exc
    finally:
        conn.autocommit = True
        conn.close()
