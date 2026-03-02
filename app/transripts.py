"""Fetch YouTube video transcripts with MySQL caching.

Flow:
1. Check MySQL cache for the video_id
2. If cached → return immediately
3. If not cached → fetch from YouTube via youtube_transcript_api
4. Save to MySQL cache
5. Return the transcript text
"""

from __future__ import annotations

import logging
import re
from typing import Optional

from youtube_transcript_api import YouTubeTranscriptApi

from app.core.database import get_cached_transcript, save_transcript

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


def fetch_transcript(video_id_or_url: str) -> str:
    """Return the transcript for a YouTube video.

    Checks MySQL cache first. If not found, fetches from YouTube,
    caches the result, and returns it.

    Args:
        video_id_or_url: A YouTube video ID or full URL.

    Returns:
        The full transcript text.

    Raises:
        ValueError: If the transcript cannot be fetched.
    """
    video_id = _extract_video_id(video_id_or_url)
    if not video_id:
        raise ValueError("Invalid video_id or URL provided.")

    # 1) Check cache
    cached = get_cached_transcript(video_id)
    if cached is not None:
        return cached

    # 2) Fetch from YouTube
    try:
        transcript_text = _fetch_from_youtube(video_id)
    except Exception as exc:
        log.exception("Failed to fetch transcript for video_id=%s", video_id)
        raise ValueError(
            f"Could not fetch transcript for video '{video_id}'. "
            "Make sure the video exists and has captions/subtitles enabled."
        ) from exc

    if not transcript_text or len(transcript_text.strip()) < 20:
        raise ValueError(
            f"Transcript for video '{video_id}' is too short or empty."
        )

    # 3) Save to cache
    save_transcript(video_id, transcript_text)

    return transcript_text
