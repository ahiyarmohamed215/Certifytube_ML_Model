from __future__ import annotations

from typing import List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, model_validator

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
    difficulty: DifficultyLevel


class GenerateQuizResponse(BaseModel):
    quiz_id: str
    session_id: str
    video_id: str
    questions: List[QuizQuestion]
    total_questions: int


class QuizAnswerSubmission(BaseModel):
    question_id: str
    answer: str


class GradeQuizRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    quiz_id: str
    session_id: str
    video_id: str
    answers: List[QuizAnswerSubmission] = Field(..., min_length=1)

    @model_validator(mode="after")
    def validate_unique_question_ids(self) -> "GradeQuizRequest":
        answer_ids = [item.question_id for item in self.answers]

        if len(answer_ids) != len(set(answer_ids)):
            raise ValueError("answers contains duplicate question_id values.")

        return self


class QuizQuestionGrade(BaseModel):
    question_id: str
    submitted_answer: str
    correct_answer: str
    explanation: str
    is_correct: bool


class GradeQuizResponse(BaseModel):
    quiz_id: str
    session_id: str
    video_id: str
    total_questions: int
    answered_questions: int
    correct_answers: int
    incorrect_answers: int
    unanswered_questions: int
    quiz_score_percent: float = Field(..., ge=0.0, le=100.0)
    results: List[QuizQuestionGrade]
