from __future__ import annotations

from typing import Dict, List, Literal
from pydantic import BaseModel, Field, ConfigDict


class AnalyzeRequest(BaseModel):
    # STRICT: request should not contain response fields like reason_codes
    session_id: str = Field(..., description="Unique session identifier from backend")
    feature_version: str = Field(..., description="Feature contract version, e.g., v1.0")

    features: Dict[str, float] = Field(
        ...,
        description="Feature vector computed by backend. Keys must match feature_contract.",
        min_length=1,
    )


# ---------------------------------------------------------------------------
# XGBoost response (SHAP-based explanations)
# ---------------------------------------------------------------------------

class ShapContributor(BaseModel):
    """A single SHAP contributor from the XGBoost model."""
    feature: str
    shap_value: float = Field(..., description="SHAP value (Shapley approximation in log-odds space)")
    feature_value: float = Field(..., description="Raw feature value sent to the model")
    behavior_category: str = Field(..., description="Behavioral category this feature belongs to")


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