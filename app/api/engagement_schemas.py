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
    engagement_status: Literal["engaged", "not_engaged"] | None = Field(
        default=None,
        description="Optional backend-owned binary status. If provided, the explanation will follow this status exactly.",
    )
    engagement_threshold: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Optional backend threshold used to convert the returned score into engaged/not_engaged when no explicit engagement_status is supplied.",
    )

    features: Dict[str, float] | None = Field(
        default=None,
        description="Optional precomputed feature vector for the engagement score model. Kept only for backward compatibility.",
        min_length=1,
    )
    events: List[RawEvent] | None = Field(
        default=None,
        description="Raw player events for a single session. Preferred request format because the service computes the canonical feature row itself.",
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
    shap_value: float = Field(..., description="Local SHAP contribution for this session score")
    feature_value: float = Field(..., description="Raw feature value sent to the model")
    behavior_category: str = Field(..., description="Behavioral category this feature belongs to")
    reason: str = Field(..., description="Human-readable reason why this feature influenced the score")


class XGBoostAnalyzeResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    model: Literal["xgboost"] = Field(default="xgboost", description="Model identifier - always 'xgboost'")
    session_id: str
    feature_version: str

    engagement_score: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Continuous engagement score in [0, 1] returned directly by the API",
    )
    engagement_status: Literal["engaged", "not_engaged"] = Field(
        ...,
        description="Binary engagement status aligned with the backend rule or threshold used for this response.",
    )

    explanation: str = Field(..., description="Human-readable explanation of the factors driving the score")

    shap_top_negative: List[ShapContributor] = Field(
        default_factory=list,
        description="Top features pulling the session score downward",
    )
    shap_top_positive: List[ShapContributor] = Field(
        default_factory=list,
        description="Top features lifting the session score upward",
    )


# ---------------------------------------------------------------------------
# EBM response (native glass-box explanations)
# ---------------------------------------------------------------------------

class EBMContributor(BaseModel):
    """A single exact contributor from the EBM model."""
    feature: str
    contribution: float = Field(..., description="Exact local contribution to the final session score")
    feature_value: float = Field(..., description="Raw feature value sent to the model")
    behavior_category: str = Field(..., description="Behavioral category this feature belongs to")
    reason: str = Field(..., description="Human-readable reason why this feature influenced the score")


class EBMAnalyzeResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    model: Literal["ebm"] = Field(default="ebm", description="Model identifier - always 'ebm'")
    session_id: str
    feature_version: str

    engagement_score: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Continuous engagement score in [0, 1] returned directly by the API",
    )
    engagement_status: Literal["engaged", "not_engaged"] = Field(
        ...,
        description="Binary engagement status aligned with the backend rule or threshold used for this response.",
    )

    explanation: str = Field(..., description="Human-readable explanation of the factors driving the score")

    ebm_top_negative: List[EBMContributor] = Field(
        default_factory=list,
        description="Top features pulling the session score downward",
    )
    ebm_top_positive: List[EBMContributor] = Field(
        default_factory=list,
        description="Top features lifting the session score upward",
    )
