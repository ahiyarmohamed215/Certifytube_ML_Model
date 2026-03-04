"""Fetch YouTube video transcripts with MySQL caching, then validate and process.

Flow:
1. Check MySQL cache for the video_id (raw + processed)
2. If processed cached → return ProcessedTranscript immediately
3. If raw cached → process it, save processed to MySQL, return
4. If not cached → fetch from YouTube via youtube_transcript_api
5. Validate the raw transcript
6. Save raw transcript to MySQL cache
7. Process (clean + chunk) the transcript
8. Save processed transcript to MySQL cache
9. Return ProcessedTranscript ready for LLM
"""

from __future__ import annotations

import logging
import re
from typing import Optional

from youtube_transcript_api import YouTubeTranscriptApi

from app.core.database import (
    get_cached_processed_transcript,
    get_cached_transcript,
    save_processed_transcript,
    save_transcript,
)
from verification.quiz.transcript.processor import (
    ProcessedTranscript,
    clean_transcript,
    process_transcript,
)

log = logging.getLogger(__name__)


def _extract_video_id(video_id_or_url: str) -> str:
    """Extract a YouTube video ID from a URL or return the raw ID.

    Supports:
      - Plain 11-char IDs: ``dQw4w9WgXcQ``
      - Full URLs: ``https://www.youtube.com/watch?v=dQw4w9WgXcQ``
      - Short URLs: ``https://youtu.be/dQw4w9WgXcQ``
    """
    raw = video_id_or_url.strip()
    # Full YouTube URL
    match = re.search(r"[?&]v=([A-Za-z0-9_-]{11})", raw)
    if match:
        return match.group(1)
    # Short youtu.be URL
    match = re.search(r"youtu\.be/([A-Za-z0-9_-]{11})", raw)
    if match:
        return match.group(1)
    # Assume it's already a video ID
    return raw


def _fetch_from_youtube(video_id: str) -> str:
    """Fetch transcript from YouTube and concatenate all segments.

    Uses youtube_transcript_api v1.2.2+ API:
    - ``YouTubeTranscriptApi().fetch(video_id)`` returns a ``FetchedTranscript``
    - Iterate to get ``FetchedTranscriptSnippet`` objects with ``.text`` attribute
    """
    ytt = YouTubeTranscriptApi()
    fetched = ytt.fetch(video_id)
    parts = [snippet.text for snippet in fetched if snippet.text]
    full_text = " ".join(parts)
    # Basic cleanup — collapse whitespace
    full_text = re.sub(r"\s+", " ", full_text).strip()
    return full_text


def _validate_raw_transcript(video_id: str, text: str) -> None:
    """Raise ValueError if the raw transcript is too short or empty."""
    if not text or len(text.strip()) < 20:
        raise ValueError(
            f"Transcript for video '{video_id}' is too short or empty."
        )


def fetch_and_process_transcript(video_id_or_url: str) -> ProcessedTranscript:
    """Fetch, validate, process, cache, and return a ProcessedTranscript.

    Pipeline:
    1. Extract video ID from URL (if needed)
    2. Check MySQL for cached **processed** transcript → return immediately
    3. Otherwise, check MySQL for cached **raw** transcript
    4. If no cache at all, fetch from YouTube and cache the raw text
    5. Validate the raw transcript (length, emptiness)
    6. Process (clean + chunk) the raw transcript
    7. Cache the processed (cleaned) text in MySQL
    8. Return ProcessedTranscript ready for quiz generation

    Args:
        video_id_or_url: A YouTube video ID or full URL.

    Returns:
        A ProcessedTranscript with cleaned, chunked text.

    Raises:
        ValueError: If the transcript cannot be fetched or is too short.
    """
    video_id = _extract_video_id(video_id_or_url)
    if not video_id:
        raise ValueError("Invalid video_id or URL provided.")

    # ── 1. Check for cached processed transcript ──────────────────────
    cached_processed = get_cached_processed_transcript(video_id)
    if cached_processed is not None:
        log.info("Using cached processed transcript for video_id=%s", video_id)
        return process_transcript(cached_processed)

    # ── 2. Get raw transcript (cache or YouTube) ──────────────────────
    raw_transcript: Optional[str] = get_cached_transcript(video_id)

    if raw_transcript is None:
        # Fetch from YouTube
        try:
            raw_transcript = _fetch_from_youtube(video_id)
        except Exception as exc:
            log.exception("Failed to fetch transcript for video_id=%s", video_id)
            raise ValueError(
                f"Could not fetch transcript for video '{video_id}'. "
                "Make sure the video exists and has captions/subtitles enabled."
            ) from exc

        # Validate
        _validate_raw_transcript(video_id, raw_transcript)

        # Save raw to cache
        save_transcript(video_id, raw_transcript)

    # ── 3. Process (clean + chunk) ────────────────────────────────────
    cleaned_text = clean_transcript(raw_transcript)
    processed = process_transcript(raw_transcript)

    # ── 4. Save processed text to cache ───────────────────────────────
    if cleaned_text:
        save_processed_transcript(video_id, cleaned_text)

    return processed
