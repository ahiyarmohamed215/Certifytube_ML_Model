from __future__ import annotations

from typing import Any, Dict, List, Tuple

from fastapi import APIRouter, HTTPException

from app.api.engagement_schemas import (
    AnalyzeRequest,
    EBMAnalyzeResponse,
    EBMContributor,
    ShapContributor,
    XGBoostAnalyzeResponse,
)
from app.core.database import EngagementPersistenceError, save_engagement_analysis
from app.core.logging import get_logger
from verification.engagement.common.behavior_map import get_behavior
from verification.engagement.common.event_pipeline import (
    EventPipelineError,
    compute_features_from_events,
)
from verification.engagement.common.text_explainer import (
    build_feature_reason,
    build_user_explanation,
)
from verification.engagement.common.validate import FeatureValidationError
from verification.engagement.contracts.contract import (
    ContractError,
    load_contract,
    validate_payload,
)
from verification.engagement.ebm.explain.ebm_explain import (
    compute_local_ebm,
    top_contributors_ebm,
)
from verification.engagement.ebm.inference.predict import predict_engagement_ebm
from verification.engagement.xgboost.explain.shap_explain import (
    compute_local_shap,
    top_contributors,
)
from verification.engagement.xgboost.inference.predict import predict_engagement

router = APIRouter(prefix="/engagement", tags=["engagement"])
log = get_logger(__name__)


def _resolve_request_features(
    req: AnalyzeRequest,
) -> Tuple[Dict[str, float], List[Dict[str, Any]], str]:
    """Normalize either request format into the final feature payload."""
    expected_cols = load_contract(req.feature_version)

    if req.features is not None:
        # Backward-compatible path: backend already computed the features.
        validate_payload(req.features, expected_cols)
        return dict(req.features), [], "features"

    # Preferred path: backend sends raw session events and this service computes
    # the canonical feature row before prediction.
    event_records: List[Dict[str, Any]] = []
    for event in req.events or []:
        record = event.model_dump(mode="python")
        if not record.get("session_id"):
            record["session_id"] = req.session_id
        event_records.append(record)

    features = compute_features_from_events(
        event_records,
        expected_session_id=req.session_id,
    )
    validate_payload(features, expected_cols)
    return dict(features), event_records, "events"


# ---------------------------------------------------------------------------
# POST /engagement/analyze/xgboost
# ---------------------------------------------------------------------------
@router.post("/analyze/xgboost", response_model=XGBoostAnalyzeResponse)
def analyze_xgboost(req: AnalyzeRequest):
    """Engagement analysis using XGBoost + SHAP explanations."""
    try:
        features, event_records, input_source = _resolve_request_features(req)

        pred = predict_engagement(features)

        shap_rows = compute_local_shap(features)
        top_negative, top_positive = top_contributors(shap_rows, k=3)

        user_text = build_user_explanation(
            shap_top_negative=top_negative,
            shap_top_positive=top_positive,
            engagement_score=pred["engagement_score"],
        )

        def to_shap_contributor(row: Dict[str, Any]) -> ShapContributor:
            return ShapContributor(
                feature=row["feature"],
                shap_value=row["shap"],
                feature_value=row["value"],
                behavior_category=get_behavior(row["feature"]),
                reason=build_feature_reason(
                    row["feature"],
                    row["value"],
                    row["shap"],
                ),
            )

        shap_negative = [to_shap_contributor(row) for row in top_negative]
        shap_positive = [to_shap_contributor(row) for row in top_positive]

        save_engagement_analysis(
            session_id=req.session_id,
            feature_version=req.feature_version,
            model_name="xgboost",
            input_source=input_source,
            features=features,
            events=event_records,
            engagement_score=pred["engagement_score"],
            explanation=user_text,
            top_negative=[item.model_dump() for item in shap_negative],
            top_positive=[item.model_dump() for item in shap_positive],
        )

        return XGBoostAnalyzeResponse(
            session_id=req.session_id,
            feature_version=req.feature_version,
            engagement_score=pred["engagement_score"],
            explanation=user_text,
            shap_top_negative=shap_negative,
            shap_top_positive=shap_positive,
        )
    except ContractError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except EventPipelineError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except FeatureValidationError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except EngagementPersistenceError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except FileNotFoundError:
        raise HTTPException(status_code=500, detail="Model artifacts missing. Train the model first.")
    except Exception as exc:
        log.exception("Internal error in /engagement/analyze/xgboost")
        raise HTTPException(status_code=500, detail="Internal engagement processing error.") from exc


# ---------------------------------------------------------------------------
# POST /engagement/analyze/ebm
# ---------------------------------------------------------------------------
@router.post("/analyze/ebm", response_model=EBMAnalyzeResponse)
def analyze_ebm(req: AnalyzeRequest):
    """Engagement analysis using EBM + native glass-box explanations."""
    try:
        features, event_records, input_source = _resolve_request_features(req)

        pred = predict_engagement_ebm(features)

        ebm_rows = compute_local_ebm(features)
        top_negative, top_positive = top_contributors_ebm(ebm_rows, k=3)

        user_text = build_user_explanation(
            shap_top_negative=top_negative,
            shap_top_positive=top_positive,
            engagement_score=pred["engagement_score"],
        )

        def to_ebm_contributor(row: Dict[str, Any]) -> EBMContributor:
            return EBMContributor(
                feature=row["feature"],
                contribution=row["shap"],
                feature_value=row["value"],
                behavior_category=get_behavior(row["feature"]),
                reason=build_feature_reason(
                    row["feature"],
                    row["value"],
                    row["shap"],
                ),
            )

        ebm_negative = [to_ebm_contributor(row) for row in top_negative]
        ebm_positive = [to_ebm_contributor(row) for row in top_positive]

        save_engagement_analysis(
            session_id=req.session_id,
            feature_version=req.feature_version,
            model_name="ebm",
            input_source=input_source,
            features=features,
            events=event_records,
            engagement_score=pred["engagement_score"],
            explanation=user_text,
            top_negative=[item.model_dump() for item in ebm_negative],
            top_positive=[item.model_dump() for item in ebm_positive],
        )

        return EBMAnalyzeResponse(
            session_id=req.session_id,
            feature_version=req.feature_version,
            engagement_score=pred["engagement_score"],
            explanation=user_text,
            ebm_top_negative=ebm_negative,
            ebm_top_positive=ebm_positive,
        )
    except ContractError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except EventPipelineError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except FeatureValidationError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except EngagementPersistenceError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except FileNotFoundError:
        raise HTTPException(status_code=500, detail="Model artifacts missing. Train the model first.")
    except Exception as exc:
        log.exception("Internal error in /engagement/analyze/ebm")
        raise HTTPException(status_code=500, detail="Internal engagement processing error.") from exc
