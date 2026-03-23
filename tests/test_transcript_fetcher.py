from youtube_transcript_api._errors import InvalidVideoId, RequestBlocked, TranscriptsDisabled

from verification.quiz.transcript.fetcher import (
    TranscriptBadRequestError,
    TranscriptNotFoundError,
    TranscriptUpstreamUnavailableError,
    _raise_transcript_fetch_error,
)


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
