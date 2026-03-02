from __future__ import annotations

from fastapi import APIRouter, HTTPException
from app.api.engagement_schemas import (
    AnalyzeRequest,
    XGBoostAnalyzeResponse,
    EBMAnalyzeResponse,
    ShapContributor,
    EBMContributor,
)
from app.core.logging import get_logger

from verification.engagement.contracts.contract import load_contract, validate_payload, ContractError
from verification.engagement.common.validate import FeatureValidationError
from verification.engagement.common.behavior_map import get_behavior
from verification.engagement.common.text_explainer import build_user_explanation, build_feature_reason

# XGBoost-specific
from verification.engagement.xgboost.inference.predict import predict_engagement
from verification.engagement.xgboost.explain.shap_explain import compute_local_shap, top_contributors

# EBM-specific
from verification.engagement.ebm.inference.predict import predict_engagement_ebm
from verification.engagement.ebm.explain.ebm_explain import compute_local_ebm, top_contributors_ebm

router = APIRouter(prefix="/engagement", tags=["engagement"])
log = get_logger(__name__)


# ---------------------------------------------------------------------------
# POST /engagement/analyze/xgboost
# ---------------------------------------------------------------------------
@router.post("/analyze/xgboost", response_model=XGBoostAnalyzeResponse)
def analyze_xgboost(req: AnalyzeRequest):
    """Engagement analysis using **XGBoost + SHAP** explanations."""
    try:
        # 1) Contract enforcement
        expected_cols = load_contract(req.feature_version)
        validate_payload(req.features, expected_cols)

        # 2) Predict
        pred = predict_engagement(req.features)

        # 3) SHAP explanations
        shap_rows = compute_local_shap(req.features)
        top_negative, top_positive = top_contributors(shap_rows, k=3)

        # 4) Human-readable explanation
        user_text = build_user_explanation(
            shap_top_negative=top_negative,
            shap_top_positive=top_positive,
            engagement_score=pred["engagement_score"],
        )

        # 5) Build SHAP contributor outputs
        def to_shap_contributor(row):
            return ShapContributor(
                feature=row["feature"],
                shap_value=row["shap"],
                feature_value=row["value"],
                behavior_category=get_behavior(row["feature"]),
                reason=build_feature_reason(
                    row["feature"], row["value"], row["shap"]
                ),
            )

        return XGBoostAnalyzeResponse(
            session_id=req.session_id,
            feature_version=req.feature_version,
            engagement_score=pred["engagement_score"],
            explanation=user_text,
            shap_top_negative=[to_shap_contributor(r) for r in top_negative],
            shap_top_positive=[to_shap_contributor(r) for r in top_positive],
        )

    except ContractError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    except FeatureValidationError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    except FileNotFoundError:
        raise HTTPException(status_code=500, detail="Model artifacts missing. Train the model first.")
    except Exception as e:
        log.exception("Internal error in /engagement/analyze/xgboost")
        raise HTTPException(status_code=500, detail="Internal engagement processing error.") from e


# ---------------------------------------------------------------------------
# POST /engagement/analyze/ebm
# ---------------------------------------------------------------------------
@router.post("/analyze/ebm", response_model=EBMAnalyzeResponse)
def analyze_ebm(req: AnalyzeRequest):
    """Engagement analysis using **EBM + native glass-box** explanations."""
    try:
        # 1) Contract enforcement
        expected_cols = load_contract(req.feature_version)
        validate_payload(req.features, expected_cols)

        # 2) Predict
        pred = predict_engagement_ebm(req.features)

        # 3) EBM native explanations
        ebm_rows = compute_local_ebm(req.features)
        top_negative, top_positive = top_contributors_ebm(ebm_rows, k=3)

        # 4) Human-readable explanation
        user_text = build_user_explanation(
            shap_top_negative=top_negative,
            shap_top_positive=top_positive,
            engagement_score=pred["engagement_score"],
        )

        # 5) Build EBM contributor outputs
        def to_ebm_contributor(row):
            return EBMContributor(
                feature=row["feature"],
                contribution=row["shap"],       # internal key is "shap" but exposed as "contribution"
                feature_value=row["value"],
                behavior_category=get_behavior(row["feature"]),
                reason=build_feature_reason(
                    row["feature"], row["value"], row["shap"]
                ),
            )

        return EBMAnalyzeResponse(
            session_id=req.session_id,
            feature_version=req.feature_version,
            engagement_score=pred["engagement_score"],
            explanation=user_text,
            ebm_top_negative=[to_ebm_contributor(r) for r in top_negative],
            ebm_top_positive=[to_ebm_contributor(r) for r in top_positive],
        )

    except ContractError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    except FeatureValidationError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    except FileNotFoundError:
        raise HTTPException(status_code=500, detail="Model artifacts missing. Train the model first.")
    except Exception as e:
        log.exception("Internal error in /engagement/analyze/ebm")
        raise HTTPException(status_code=500, detail="Internal engagement processing error.") from e
