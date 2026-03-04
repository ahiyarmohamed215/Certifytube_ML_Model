from __future__ import annotations

from fastapi import APIRouter, HTTPException

from app.api.quiz_schemas import GenerateQuizRequest, GenerateQuizResponse, QuizQuestion
from app.core.logging import get_logger
from verification.quiz.generator.quiz_gen import QuizGenerationError, generate_quiz
from verification.quiz.transcript.fetcher import fetch_and_process_transcript

router = APIRouter(prefix="/quiz", tags=["quiz"])
log = get_logger(__name__)


@router.post("/generate", response_model=GenerateQuizResponse)
def generate_quiz_endpoint(req: GenerateQuizRequest) -> GenerateQuizResponse:
    try:
        # ---- Fetch, validate, process, and cache transcript ----
        transcript = fetch_and_process_transcript(req.video_id)

        # ---- Generate quiz (LLM decides question count, max from env) ----
        generated_questions = generate_quiz(
            session_id=req.session_id,
            video_title="YouTube Video",
            transcript=transcript,
            video_duration_sec=req.video_duration_sec,
            include_coding=transcript.has_code_content,
        )

        # ---- Map to response (only fields the backend needs) ----
        questions = [
            QuizQuestion(
                question_id=q["question_id"],
                type=q["type"],
                question=q["question"],
                options=q.get("options"),
                correct_answer=q["correct_answer"],
                explanation=q["explanation"],
                difficulty=q["difficulty"],
            )
            for q in generated_questions
        ]

        return GenerateQuizResponse(
            session_id=req.session_id,
            video_id=req.video_id,
            questions=questions,
            total_questions=len(questions),
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
