from __future__ import annotations

from fastapi import APIRouter, HTTPException
from app.api.schemas import (
    AnalyzeRequest,
    AnalyzeResponse,
    ShapContributor,
)
from app.core.logging import get_logger

from ml.contracts.contract import load_contract, validate_payload, ContractError
from ml.inference.predict import predict_engagement_routed
from ml.inference.validate import FeatureValidationError
from ml.explain.shap_explain import compute_local_shap, top_contributors
from ml.explain.ebm_explain import compute_local_ebm, top_contributors_ebm
from ml.explain.behavior_map import get_behavior
from ml.explain.text_explainer import build_user_explanation

router = APIRouter(prefix="/engagement", tags=["engagement"])
log = get_logger(__name__)


# ---------------------------------------------------------------------------
# Shared implementation – keeps both endpoints DRY
# ---------------------------------------------------------------------------
def _analyze_impl(
    req: AnalyzeRequest,
    model_type: str,
) -> AnalyzeResponse:
    try:
        # 1) Contract enforcement
        expected_cols = load_contract(req.feature_version)
        validate_payload(req.features, expected_cols)

        # 2) Predict (routed by model_type)
        pred = predict_engagement_routed(req.features, model_type=model_type)

        # 3) Explanations (SHAP for XGBoost, native for EBM)
        if model_type == "ebm":
            shap_rows = compute_local_ebm(req.features)
            top_negative, top_positive = top_contributors_ebm(shap_rows, k=3)
        else:
            shap_rows = compute_local_shap(req.features)
            top_negative, top_positive = top_contributors(shap_rows, k=3)

        # 4) User explanation + reason codes
        user_text, pos_behaviors, neg_behaviors, reason_codes = build_user_explanation(
            shap_top_negative=top_negative,
            shap_top_positive=top_positive,
            status=pred["status"],
        )

        # 5) Prepare SHAP contributor outputs
        def to_contributor(row):
            return ShapContributor(
                feature=row["feature"],
                shap=row["shap"],
                value=row["value"],
                behavior=get_behavior(row["feature"]),
            )

        shap_top_negative_out = [to_contributor(r) for r in top_negative]
        shap_top_positive_out = [to_contributor(r) for r in top_positive]

        # 6) Response
        return AnalyzeResponse(
            session_id=req.session_id,
            feature_version=req.feature_version,
            model_type=model_type,
            engagement_score=pred["engagement_score"],
            threshold=pred["threshold"],
            status=pred["status"],
            explanation_text=user_text,
            reason_codes=reason_codes,
            shap_top_negative=shap_top_negative_out,
            shap_top_positive=shap_top_positive_out,
        )

    except ContractError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except FeatureValidationError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except FileNotFoundError:
        raise HTTPException(status_code=500, detail="Model artifacts missing. Train the model first.")
    except Exception as e:
        log.exception("Internal error in /engagement/analyze")
        raise HTTPException(status_code=500, detail=str(e))


# ---------------------------------------------------------------------------
# POST /engagement/analyze/xgboost
# ---------------------------------------------------------------------------
@router.post("/analyze/xgboost", response_model=AnalyzeResponse)
def analyze_xgboost(req: AnalyzeRequest):
    """Engagement analysis using **XGBoost + SHAP** explanations."""
    return _analyze_impl(req, "xgboost")


# ---------------------------------------------------------------------------
# POST /engagement/analyze/ebm
# ---------------------------------------------------------------------------
@router.post("/analyze/ebm", response_model=AnalyzeResponse)
def analyze_ebm(req: AnalyzeRequest):
    """Engagement analysis using **EBM + native** explanations."""
    return _analyze_impl(req, "ebm")