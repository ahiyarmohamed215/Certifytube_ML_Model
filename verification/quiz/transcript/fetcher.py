"""Fetch YouTube video transcripts with caching and production proxy support."""

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
from youtube_transcript_api.proxies import GenericProxyConfig, ProxyConfig, WebshareProxyConfig

from app.core.database import (
    get_cached_processed_transcript,
    get_cached_transcript,
    save_processed_transcript,
    save_transcript,
)
from app.core.settings import settings
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


class TranscriptConfigurationError(Exception):
    """Raised when transcript proxy configuration is invalid."""


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


def _parse_csv(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def _build_proxy_config() -> Optional[ProxyConfig]:
    """Build a youtube-transcript-api proxy config from env settings."""
    mode = settings.transcript_proxy_mode.strip().lower()
    if not mode or mode == "none":
        return None

    if mode == "generic":
        http_url = settings.transcript_proxy_http_url.strip() or None
        https_url = settings.transcript_proxy_https_url.strip() or None
        if not http_url and not https_url:
            raise TranscriptConfigurationError(
                "TRANSCRIPT_PROXY_MODE=generic requires TRANSCRIPT_PROXY_HTTP_URL "
                "or TRANSCRIPT_PROXY_HTTPS_URL."
            )
        return GenericProxyConfig(http_url=http_url, https_url=https_url)

    if mode == "webshare":
        username = settings.transcript_webshare_proxy_username.strip()
        password = settings.transcript_webshare_proxy_password.strip()
        if not username or not password:
            raise TranscriptConfigurationError(
                "TRANSCRIPT_PROXY_MODE=webshare requires "
                "TRANSCRIPT_WEBSHARE_PROXY_USERNAME and "
                "TRANSCRIPT_WEBSHARE_PROXY_PASSWORD."
            )
        locations = _parse_csv(settings.transcript_webshare_proxy_locations)
        return WebshareProxyConfig(
            proxy_username=username,
            proxy_password=password,
            filter_ip_locations=locations or None,
        )

    raise TranscriptConfigurationError(
        f"Unsupported TRANSCRIPT_PROXY_MODE '{settings.transcript_proxy_mode}'."
    )


def _build_youtube_client() -> YouTubeTranscriptApi:
    proxy_config = _build_proxy_config()
    return YouTubeTranscriptApi(proxy_config=proxy_config)


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
            "Transcript upstream is blocked. Retry later or enable a supported proxy."
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
            "Transcript upstream is currently unavailable. Retry later or use a "
            "proxy-backed transcript fetcher."
        ) from exc

    raise exc


def _fetch_from_youtube(video_id: str) -> str:
    """Fetch a transcript from YouTube and concatenate all text segments."""
    try:
        ytt = _build_youtube_client()
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
