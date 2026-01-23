from typing import Any, Dict, List, Literal, Optional
from pydantic import BaseModel, Field


EngagementStatus = Literal["ENGAGED", "NOT_ENGAGED"]


class AnalyzeRequest(BaseModel):
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
    session_id: str
    feature_version: str

    engagement_score: float = Field(..., ge=0.0, le=1.0)
    threshold: float = Field(..., ge=0.0, le=1.0, description="Engagement threshold, fixed at 0.85")
    status: EngagementStatus

    explanation_text: str

    shap_top_negative: List[ShapContributor] = Field(default_factory=list)
    shap_top_positive: List[ShapContributor] = Field(default_factory=list)

    counterfactual: Optional[CounterfactualResponse] = None
    debug: Optional[Dict[str, Any]] = None
git add app/api/schemas.py
git commit -m "feat(api): add request/response schemas for engagement analyze"
git push
