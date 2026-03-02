from __future__ import annotations

from fastapi import APIRouter, HTTPException

from app.api.quiz_schemas import GenerateQuizRequest, GenerateQuizResponse, QuizQuestion
from app.core.logging import get_logger
from app.transripts import fetch_transcript
from verification.quiz.generator.quiz_gen import QuizGenerationError, generate_quiz
from verification.quiz.transcript.processor import process_transcript

router = APIRouter(prefix="/quiz", tags=["quiz"])
log = get_logger(__name__)


@router.post("/generate", response_model=GenerateQuizResponse)
def generate_quiz_endpoint(req: GenerateQuizRequest) -> GenerateQuizResponse:
    try:
        # ---- Resolve transcript ----
        raw_transcript: str
        if req.transcript:
            # Explicit transcript provided — use it directly
            raw_transcript = req.transcript
        elif req.video_id:
            # Fetch transcript by video_id (checks MySQL cache first)
            raw_transcript = fetch_transcript(req.video_id)
        else:
            raise HTTPException(
                status_code=400,
                detail="Either 'video_id' or 'transcript' must be provided.",
            )

        # ---- Process and generate quiz ----
        transcript = process_transcript(raw_transcript)
        allow_coding = req.include_coding and transcript.has_code_content

        generated_questions = generate_quiz(
            session_id=req.session_id,
            video_title=req.video_title,
            transcript=transcript,
            video_duration_sec=req.video_duration_sec,
            requested_questions=req.num_questions,
            max_questions=req.max_questions,
            include_coding=allow_coding,
        )

        questions = [QuizQuestion(**question) for question in generated_questions]

        return GenerateQuizResponse(
            session_id=req.session_id,
            questions=questions,
            total_questions=len(questions),
            has_coding_questions=any(question.type == "coding" for question in questions),
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except QuizGenerationError as exc:
        log.exception("Quiz generation upstream/provider failure")
        raise HTTPException(status_code=502, detail="Quiz generation failed. Please retry.") from exc
    except HTTPException:
        raise
    except Exception as exc:
        log.exception("Internal error in /quiz/generate")
        raise HTTPException(status_code=500, detail="Internal quiz processing error.") from exc
