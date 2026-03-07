from __future__ import annotations

from typing import List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field

QuestionType = Literal["mcq", "true_false", "fill_blank"]
DifficultyLevel = Literal["easy", "medium", "hard"]


class GenerateQuizRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    session_id: str
    video_id: str = Field(
        ...,
        description="YouTube video ID or URL. The ML service fetches and processes the transcript automatically.",
    )
    video_duration_sec: float = Field(..., gt=0)


class QuizQuestion(BaseModel):
    question_id: str
    type: QuestionType
    question: str
    options: Optional[List[str]] = None
    correct_answer: str
    explanation: str
    difficulty: DifficultyLevel


class GenerateQuizResponse(BaseModel):
    session_id: str
    video_id: str
    questions: List[QuizQuestion]
    total_questions: int
