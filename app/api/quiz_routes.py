from __future__ import annotations

from uuid import uuid4

from fastapi import APIRouter, HTTPException

from app.api.quiz_schemas import (
    GenerateQuizRequest,
    GenerateQuizResponse,
    GradeQuizRequest,
    GradeQuizResponse,
    QuizQuestion,
    QuizQuestionGrade,
)
from app.core.database import (
    QuizPersistenceError,
    get_quiz_answer_key,
    save_generated_quiz,
    save_quiz_grading_result,
)
from app.core.logging import get_logger
from verification.quiz.grader import QuizGradingError, grade_quiz
from verification.quiz.generator.quiz_gen import QuizGenerationError, generate_quiz
from verification.quiz.transcript.fetcher import (
    TranscriptBadRequestError,
    TranscriptNotFoundError,
    TranscriptUpstreamUnavailableError,
    fetch_and_process_transcript,
)

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
        quiz_id = str(uuid4())
        save_generated_quiz(
            quiz_id=quiz_id,
            session_id=req.session_id,
            video_id=req.video_id,
            questions=generated_questions,
        )

        # ---- Map to response (public questions only) ----
        questions = [
            QuizQuestion(
                question_id=q["question_id"],
                type=q["type"],
                question=q["question"],
                options=q.get("options"),
                difficulty=q["difficulty"],
            )
            for q in generated_questions
        ]

        return GenerateQuizResponse(
            quiz_id=quiz_id,
            session_id=req.session_id,
            video_id=req.video_id,
            questions=questions,
            total_questions=len(questions),
        )
    except TranscriptBadRequestError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except TranscriptNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except TranscriptUpstreamUnavailableError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except QuizPersistenceError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except QuizGenerationError as exc:
        log.exception("Quiz generation upstream/provider failure")
        raise HTTPException(status_code=502, detail="Quiz generation failed. Please retry.") from exc
    except HTTPException:
        raise
    except Exception as exc:
        log.exception("Internal error in /quiz/generate")
        raise HTTPException(status_code=500, detail="Internal quiz processing error.") from exc


@router.post("/grade", response_model=GradeQuizResponse)
def grade_quiz_endpoint(req: GradeQuizRequest) -> GradeQuizResponse:
    try:
        stored_quiz = get_quiz_answer_key(req.quiz_id)
        if stored_quiz is None:
            raise HTTPException(status_code=404, detail="quiz_id not found.")
        if stored_quiz["session_id"] != req.session_id:
            raise HTTPException(status_code=400, detail="session_id does not match the stored quiz.")
        if stored_quiz["video_id"] != req.video_id:
            raise HTTPException(status_code=400, detail="video_id does not match the stored quiz.")

        graded = grade_quiz(
            answer_key=stored_quiz["answer_key"],
            answers=[item.model_dump() for item in req.answers],
        )
        save_quiz_grading_result(
            quiz_id=req.quiz_id,
            quiz_score_percent=graded["quiz_score_percent"],
            correct_answers=graded["correct_answers"],
            incorrect_answers=graded["incorrect_answers"],
            unanswered_questions=graded["unanswered_questions"],
            results=graded["results"],
        )

        return GradeQuizResponse(
            quiz_id=req.quiz_id,
            session_id=req.session_id,
            video_id=req.video_id,
            total_questions=graded["total_questions"],
            answered_questions=graded["answered_questions"],
            correct_answers=graded["correct_answers"],
            incorrect_answers=graded["incorrect_answers"],
            unanswered_questions=graded["unanswered_questions"],
            quiz_score_percent=graded["quiz_score_percent"],
            results=[QuizQuestionGrade(**row) for row in graded["results"]],
        )
    except QuizPersistenceError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except QuizGradingError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except HTTPException:
        raise
    except Exception as exc:
        log.exception("Internal error in /quiz/grade")
        raise HTTPException(status_code=500, detail="Internal quiz grading error.") from exc
