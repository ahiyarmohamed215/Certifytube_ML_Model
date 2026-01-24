from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional
from pydantic import BaseModel, Field, ConfigDict

EngagementStatus = Literal["ENGAGED", "NOT_ENGAGED"]

QuizQType = Literal["mcq", "tf"]
Difficulty = Literal["easy", "medium", "hard"]

video_duration_sec: float = Field(..., ge=1, description="Video length in seconds")


class AnalyzeRequest(BaseModel):
    # STRICT: request should not contain response fields like reason_codes
    session_id: str = Field(..., description="Unique session identifier from backend")
    feature_version: str = Field(..., description="Feature contract version, e.g., v1.0")

    features: Dict[str, float] = Field(
        ...,
        description="Feature vector computed by backend. Keys must match feature_contract.",
        min_length=1,
    )


class ShapContributor(BaseModel):
    feature: str
    shap: float
    value: float
    behavior: str


class CounterfactualSuggestion(BaseModel):
    feature: str
    current: float
    suggested: float
    action: str


class CounterfactualResponse(BaseModel):
    target_threshold: float
    suggestions: List[CounterfactualSuggestion]


class AnalyzeResponse(BaseModel):
    # Optional: forbid extra fields to catch bugs early
    model_config = ConfigDict(extra="forbid")

    session_id: str
    feature_version: str

    engagement_score: float = Field(..., ge=0.0, le=1.0)
    threshold: float = Field(..., ge=0.0, le=1.0, description="Engagement threshold")
    status: EngagementStatus

    explanation_text: str

    # ✅ Option B: safe categorical reason codes
    reason_codes: List[str] = Field(default_factory=list)

    shap_top_negative: List[ShapContributor] = Field(default_factory=list)
    shap_top_positive: List[ShapContributor] = Field(default_factory=list)

    # Internal only (must be gated)
    counterfactual: Optional[CounterfactualResponse] = None
    debug: Optional[Dict[str, Any]] = None




class QuizGenerateRequest(BaseModel):
    video_id: str
    video_duration_sec: float = Field(..., ge=1, description="Video length in seconds")
    transcript: str
    difficulty: Difficulty = Field(default="medium")
    num_questions: Optional[int] = Field(default=None, ge=1, le=20)

class QuizQuestion(BaseModel):
    qid: str
    type: QuizQType
    stem: str

    # For MCQ: choices must exist
    choices: Optional[List[str]] = None

    # Correct answer:
    # - for MCQ: exact matching choice string
    # - for TF: "True" or "False"
    answer: str

    explanation: str
    source_sentence: str
    difficulty: Optional[Difficulty] = None


class QuizGenerateResponse(BaseModel):
    quiz_id: str
    video_id: str
    cleaned_transcript_word_count: int
    questions: List[QuizQuestion]

    # Optional debug for audit
    debug: Optional[Dict[str, Any]] = None


class QuizGradeRequest(BaseModel):
    quiz_id: str
    # user answers: { qid: "answer string" }
    answers: Dict[str, str]


class QuizFeedbackItem(BaseModel):
    qid: str
    correct: bool
    correct_answer: str
    user_answer: Optional[str] = None
    explanation: Optional[str] = None


class QuizGradeResponse(BaseModel):
    quiz_id: str
    score: float = Field(..., ge=0.0, le=1.0)
    threshold: float = Field(..., ge=0.0, le=1.0)
    passed_threshold: bool
    feedback: List[QuizFeedbackItem]