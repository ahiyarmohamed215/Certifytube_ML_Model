from fastapi import HTTPException

from app.api.quiz_routes import generate_quiz_endpoint
from app.api.quiz_schemas import GenerateQuizRequest
from verification.quiz.transcript.fetcher import (
    TranscriptBadRequestError,
    TranscriptNotFoundError,
    TranscriptUpstreamUnavailableError,
)


def _request() -> GenerateQuizRequest:
    return GenerateQuizRequest(
        session_id="session-1",
        video_id="dQw4w9WgXcQ",
        video_duration_sec=120.0,
    )


def test_generate_quiz_maps_transcript_bad_request_to_400(monkeypatch):
    def fake_fetch(_: str):
        raise TranscriptBadRequestError("Invalid YouTube video ID.")

    monkeypatch.setattr("app.api.quiz_routes.fetch_and_process_transcript", fake_fetch)

    try:
        generate_quiz_endpoint(_request())
        assert False, "Expected HTTPException"
    except HTTPException as exc:
        assert exc.status_code == 400
        assert exc.detail == "Invalid YouTube video ID."


def test_generate_quiz_maps_missing_transcript_to_404(monkeypatch):
    def fake_fetch(_: str):
        raise TranscriptNotFoundError("Transcript does not exist.")

    monkeypatch.setattr("app.api.quiz_routes.fetch_and_process_transcript", fake_fetch)

    try:
        generate_quiz_endpoint(_request())
        assert False, "Expected HTTPException"
    except HTTPException as exc:
        assert exc.status_code == 404
        assert exc.detail == "Transcript does not exist."


def test_generate_quiz_maps_blocked_upstream_to_503(monkeypatch):
    def fake_fetch(_: str):
        raise TranscriptUpstreamUnavailableError("Transcript upstream is blocked.")

    monkeypatch.setattr("app.api.quiz_routes.fetch_and_process_transcript", fake_fetch)

    try:
        generate_quiz_endpoint(_request())
        assert False, "Expected HTTPException"
    except HTTPException as exc:
        assert exc.status_code == 503
        assert exc.detail == "Transcript upstream is blocked."


def test_generate_quiz_keeps_unexpected_failures_as_500(monkeypatch):
    def fake_fetch(_: str):
        raise RuntimeError("boom")

    monkeypatch.setattr("app.api.quiz_routes.fetch_and_process_transcript", fake_fetch)

    try:
        generate_quiz_endpoint(_request())
        assert False, "Expected HTTPException"
    except HTTPException as exc:
        assert exc.status_code == 500
        assert exc.detail == "Internal quiz processing error."
