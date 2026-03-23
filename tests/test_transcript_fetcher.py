from youtube_transcript_api._errors import InvalidVideoId, RequestBlocked, TranscriptsDisabled
from youtube_transcript_api.proxies import GenericProxyConfig, WebshareProxyConfig

from app.core.settings import settings
from verification.quiz.transcript.fetcher import (
    TranscriptBadRequestError,
    TranscriptNotFoundError,
    TranscriptUpstreamUnavailableError,
    _build_proxy_config,
    _raise_transcript_fetch_error,
)


def test_build_proxy_config_returns_generic_proxy(monkeypatch):
    monkeypatch.setattr(settings, "transcript_proxy_mode", "generic")
    monkeypatch.setattr(settings, "transcript_proxy_http_url", "http://user:pass@proxy:8080")
    monkeypatch.setattr(settings, "transcript_proxy_https_url", "")

    proxy_config = _build_proxy_config()

    assert isinstance(proxy_config, GenericProxyConfig)


def test_build_proxy_config_returns_webshare_proxy(monkeypatch):
    monkeypatch.setattr(settings, "transcript_proxy_mode", "webshare")
    monkeypatch.setattr(settings, "transcript_webshare_proxy_username", "user")
    monkeypatch.setattr(settings, "transcript_webshare_proxy_password", "pass")
    monkeypatch.setattr(settings, "transcript_webshare_proxy_locations", "us,sg")

    proxy_config = _build_proxy_config()

    assert isinstance(proxy_config, WebshareProxyConfig)


def test_raise_transcript_fetch_error_maps_invalid_video_id():
    try:
        _raise_transcript_fetch_error("bad-id", InvalidVideoId("bad-id"))
        assert False, "Expected TranscriptBadRequestError"
    except TranscriptBadRequestError:
        pass


def test_raise_transcript_fetch_error_maps_transcript_missing():
    try:
        _raise_transcript_fetch_error("video1234567", TranscriptsDisabled("video1234567"))
        assert False, "Expected TranscriptNotFoundError"
    except TranscriptNotFoundError:
        pass


def test_raise_transcript_fetch_error_maps_request_blocked():
    try:
        _raise_transcript_fetch_error("video1234567", RequestBlocked("video1234567"))
        assert False, "Expected TranscriptUpstreamUnavailableError"
    except TranscriptUpstreamUnavailableError:
        pass
