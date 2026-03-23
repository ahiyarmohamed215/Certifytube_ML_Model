"""Fetch YouTube video transcripts with MySQL caching, then validate and process."""

from __future__ import annotations

import logging
import re
from typing import Optional

from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api._errors import (
    AgeRestricted,
    CookieError,
    CookieInvalid,
    CookiePathInvalid,
    CouldNotRetrieveTranscript,
    FailedToCreateConsentCookie,
    HTTPError,
    InvalidVideoId,
    IpBlocked,
    NoTranscriptFound,
    PoTokenRequired,
    RequestBlocked,
    TranscriptsDisabled,
    VideoUnavailable,
    VideoUnplayable,
    YouTubeDataUnparsable,
    YouTubeRequestFailed,
)

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


class TranscriptError(Exception):
    """Base class for transcript retrieval failures."""


class TranscriptBadRequestError(TranscriptError):
    """Raised when the caller provides an invalid video identifier."""


class TranscriptNotFoundError(TranscriptError):
    """Raised when a transcript truly does not exist for the video."""


class TranscriptUpstreamUnavailableError(TranscriptError):
    """Raised when YouTube is blocked or unavailable from this deployment."""


def _extract_video_id(video_id_or_url: str) -> str:
    """Extract a YouTube video ID from a URL or return the raw ID."""
    raw = video_id_or_url.strip()
    match = re.search(r"[?&]v=([A-Za-z0-9_-]{11})", raw)
    if match:
        return match.group(1)

    match = re.search(r"youtu\.be/([A-Za-z0-9_-]{11})", raw)
    if match:
        return match.group(1)

    return raw


def _raise_transcript_fetch_error(video_id: str, exc: Exception) -> None:
    if isinstance(exc, InvalidVideoId):
        raise TranscriptBadRequestError(
            f"Invalid YouTube video ID '{video_id}'."
        ) from exc

    if isinstance(exc, (NoTranscriptFound, TranscriptsDisabled, VideoUnavailable)):
        raise TranscriptNotFoundError(
            f"Transcript does not exist for video '{video_id}'."
        ) from exc

    if isinstance(exc, (RequestBlocked, IpBlocked)):
        raise TranscriptUpstreamUnavailableError(
            "Transcript upstream is blocked. Retry later."
        ) from exc

    if isinstance(
        exc,
        (
            AgeRestricted,
            CookieError,
            CookieInvalid,
            CookiePathInvalid,
            CouldNotRetrieveTranscript,
            FailedToCreateConsentCookie,
            HTTPError,
            PoTokenRequired,
            VideoUnplayable,
            YouTubeDataUnparsable,
            YouTubeRequestFailed,
        ),
    ):
        raise TranscriptUpstreamUnavailableError(
            "Transcript upstream is currently unavailable."
        ) from exc

    raise exc


def _fetch_from_youtube(video_id: str) -> str:
    """Fetch a transcript from YouTube and concatenate all text segments."""
    try:
        ytt = YouTubeTranscriptApi()
        fetched = ytt.fetch(video_id)
        parts = [snippet.text for snippet in fetched if snippet.text]
        full_text = " ".join(parts)
        return re.sub(r"\s+", " ", full_text).strip()
    except Exception as exc:
        _raise_transcript_fetch_error(video_id, exc)
        raise


def _validate_raw_transcript(video_id: str, text: str) -> None:
    """Ensure the raw transcript has usable content."""
    if not text or len(text.strip()) < 20:
        raise TranscriptNotFoundError(
            f"Transcript does not exist for video '{video_id}'."
        )


def fetch_and_process_transcript(video_id_or_url: str) -> ProcessedTranscript:
    """Fetch, validate, process, cache, and return a ProcessedTranscript."""
    video_id = _extract_video_id(video_id_or_url)
    if not video_id:
        raise TranscriptBadRequestError("Invalid video_id or URL provided.")

    cached_processed = get_cached_processed_transcript(video_id)
    if cached_processed is not None:
        log.info("Using cached processed transcript for video_id=%s", video_id)
        return process_transcript(cached_processed)

    raw_transcript: Optional[str] = get_cached_transcript(video_id)
    if raw_transcript is None:
        try:
            raw_transcript = _fetch_from_youtube(video_id)
        except TranscriptError:
            raise
        except Exception as exc:
            log.exception("Unexpected transcript fetch failure for video_id=%s", video_id)
            raise TranscriptUpstreamUnavailableError(
                "Transcript upstream is currently unavailable."
            ) from exc

        _validate_raw_transcript(video_id, raw_transcript)
        save_transcript(video_id, raw_transcript)

    cleaned_text = clean_transcript(raw_transcript)
    processed = process_transcript(raw_transcript)

    if cleaned_text:
        save_processed_transcript(video_id, cleaned_text)

    return processed
