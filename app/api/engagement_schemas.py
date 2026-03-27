from __future__ import annotations

from datetime import datetime
from typing import Dict, List, Literal
from pydantic import BaseModel, Field, ConfigDict, model_validator


class RawEvent(BaseModel):
    model_config = ConfigDict(extra="ignore")

    event_id: str | None = None
    session_id: str | None = None
    user_id: str | None = None
    video_id: str | None = None
    video_title: str | None = None
    event_type: str | None = None
    player_state: int | None = None
    playback_rate: float | None = None
    current_time_sec: float | None = None
    video_duration_sec: float | None = None
    created_at_utc: datetime | str | None = None
    client_created_at_local: datetime | str | None = None
    client_tz_offset_min: float | None = None
    seek_from_sec: float | None = None
    seek_to_sec: float | None = None


class AnalyzeRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    session_id: str = Field(..., description="Unique session identifier from backend")
    feature_version: str = Field(..., description="Feature contract version, e.g., v1.0")

    features: Dict[str, float] | None = Field(
        default=None,
        description="Optional precomputed feature vector. Kept only for backward compatibility.",
        min_length=1,
    )
    events: List[RawEvent] | None = Field(
        default=None,
        description="Raw player events for a single session. Preferred request format.",
        min_length=1,
    )

    @model_validator(mode="after")
    def validate_input_source(self) -> "AnalyzeRequest":
        has_features = self.features is not None
        has_events = self.events is not None and len(self.events) > 0

        if has_features == has_events:
            raise ValueError("Provide exactly one of 'features' or 'events'.")

        return self


# ---------------------------------------------------------------------------
# XGBoost response (SHAP-based explanations)
# ---------------------------------------------------------------------------

class ShapContributor(BaseModel):
    """A single SHAP contributor from the XGBoost model."""
    feature: str
    shap_value: float = Field(..., description="SHAP value (Shapley approximation in log-odds space)")
    feature_value: float = Field(..., description="Raw feature value sent to the model")
    behavior_category: str = Field(..., description="Behavioral category this feature belongs to")
    reason: str = Field(..., description="Human-readable reason why this feature influenced the score")


class XGBoostAnalyzeResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    model: Literal["xgboost"] = Field(default="xgboost", description="Model identifier — always 'xgboost'")
    session_id: str
    feature_version: str

    engagement_score: float = Field(..., ge=0.0, le=1.0)

    explanation: str = Field(..., description="Human-readable explanation of the factors driving the score")

    shap_top_negative: List[ShapContributor] = Field(
        default_factory=list,
        description="Top features pushing AGAINST engagement (most negative SHAP values)",
    )
    shap_top_positive: List[ShapContributor] = Field(
        default_factory=list,
        description="Top features pushing TOWARDS engagement (most positive SHAP values)",
    )


# ---------------------------------------------------------------------------
# EBM response (native glass-box explanations)
# ---------------------------------------------------------------------------

class EBMContributor(BaseModel):
    """A single contributor from the EBM model (exact term score, not SHAP)."""
    feature: str
    contribution: float = Field(..., description="Exact EBM term score in log-odds space")
    feature_value: float = Field(..., description="Raw feature value sent to the model")
    behavior_category: str = Field(..., description="Behavioral category this feature belongs to")
    reason: str = Field(..., description="Human-readable reason why this feature influenced the score")


class EBMAnalyzeResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    model: Literal["ebm"] = Field(default="ebm", description="Model identifier — always 'ebm'")
    session_id: str
    feature_version: str

    engagement_score: float = Field(..., ge=0.0, le=1.0)

    explanation: str = Field(..., description="Human-readable explanation of the factors driving the score")

    ebm_top_negative: List[EBMContributor] = Field(
        default_factory=list,
        description="Top features pushing AGAINST engagement (most negative EBM contributions)",
    )
    ebm_top_positive: List[EBMContributor] = Field(
        default_factory=list,
        description="Top features pushing TOWARDS engagement (most positive EBM contributions)",
    )
