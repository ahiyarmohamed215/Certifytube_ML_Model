from __future__ import annotations

from typing import Dict, List, Literal, Optional
from pydantic import BaseModel, Field, ConfigDict

EngagementStatus = Literal["ENGAGED", "NOT_ENGAGED"]

Difficulty = Literal["easy", "medium", "hard"]


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


class AnalyzeResponse(BaseModel):
    # Optional: forbid extra fields to catch bugs early
    model_config = ConfigDict(extra="forbid")

    session_id: str
    feature_version: str
    model_type: str = Field(..., description="Model used: 'xgboost' or 'ebm'")

    engagement_score: float = Field(..., ge=0.0, le=1.0)
    threshold: float = Field(..., ge=0.0, le=1.0, description="Engagement threshold")
    status: EngagementStatus

    explanation_text: str

    # Safe categorical reason codes
    reason_codes: List[str] = Field(default_factory=list)

    shap_top_negative: List[ShapContributor] = Field(default_factory=list)
    shap_top_positive: List[ShapContributor] = Field(default_factory=list)




class QuizGenerateRequest(BaseModel):
    video_id: str = Field(..., description="YouTube video ID (e.g. 'dQw4w9WgXcQ')")
    difficulty: Difficulty = Field(default="medium")
    num_questions: int = Field(default=5, ge=1, le=20)


class QuizQuestion(BaseModel):
    qid: str
    type: Literal["mcq", "tf"]
    stem: str
    choices: Optional[List[str]] = None
    answer: str
    explanation: str
    difficulty: Optional[str] = "medium"


class QuizGenerateResponse(BaseModel):
    quiz_id: str
    video_id: str
    questions: List[QuizQuestion]
    debug: Optional[Dict[str, Any]] = None