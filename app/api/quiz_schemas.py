from __future__ import annotations

from typing import List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, model_validator

QuestionType = Literal["mcq", "true_false", "fill_blank", "short_answer", "coding"]
DifficultyLevel = Literal["easy", "medium", "hard"]
BloomLevel = Literal["remember", "understand", "apply", "analyze"]


class GenerateQuizRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    session_id: str
    video_id: Optional[str] = Field(
        default=None,
        description="YouTube video ID or URL. The transcript will be fetched from YouTube and cached in the DB.",
    )
    transcript: Optional[str] = Field(
        default=None,
        min_length=20,
        description="Raw transcript text. If provided, this takes priority over video_id.",
    )
    video_title: str = Field(..., min_length=2)
    video_duration_sec: float = Field(..., gt=0)
    num_questions: Optional[int] = Field(
        default=None,
        ge=1,
        le=20,
        description="Optional manual override. If omitted, the LLM plans count from transcript and duration.",
    )
    max_questions: int = Field(default=20, ge=1, le=20)
    include_coding: bool = False

    @model_validator(mode="after")
    def _require_video_id_or_transcript(self) -> "GenerateQuizRequest":
        if not self.video_id and not self.transcript:
            raise ValueError("Either 'video_id' or 'transcript' must be provided.")
        return self


class QuizQuestion(BaseModel):
    model_config = ConfigDict(extra="forbid")

    question_id: str
    type: QuestionType
    question: str
    options: Optional[List[str]] = None
    correct_answer: str
    explanation: str
    source_segment: str
    difficulty: DifficultyLevel
    bloom_level: BloomLevel


class GenerateQuizResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    session_id: str
    questions: List[QuizQuestion]
    total_questions: int
    has_coding_questions: bool
