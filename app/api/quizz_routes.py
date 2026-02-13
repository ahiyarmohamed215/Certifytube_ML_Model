"""
Quiz API routes — video-ID-based quiz generation via Gemini.
"""
from __future__ import annotations

from fastapi import APIRouter, HTTPException

from app.api.schemas import QuizGenerateRequest, QuizGenerateResponse
from ml.quiz.generate import generate_quiz

router = APIRouter(prefix="/quiz", tags=["quiz"])


@router.post("/generate", response_model=QuizGenerateResponse)
def generate(req: QuizGenerateRequest):
    """
    Generate a quiz from a YouTube video.

    Send the video_id (e.g. "dQw4w9WgXcQ") and Gemini will watch the
    video and produce structured quiz questions.
    """
    try:
        quiz_data = generate_quiz(
            video_id=req.video_id,
            difficulty=req.difficulty,
            num_questions=req.num_questions,
        )
        return quiz_data

    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        print(f"Error generating quiz: {e}")
        raise HTTPException(status_code=500, detail=str(e))