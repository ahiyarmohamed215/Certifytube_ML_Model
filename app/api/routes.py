from __future__ import annotations

from fastapi import APIRouter, HTTPException, Query, Request
from app.api.schemas import (
    AnalyzeRequest,
    AnalyzeResponse,
    ShapContributor,
    CounterfactualResponse,
    CounterfactualSuggestion,
)
from app.core.logging import get_logger
from app.core.settings import settings

from ml.contracts.contract import load_contract, validate_payload, ContractError
from ml.inference.predict import predict_engagement
from ml.inference.validate import FeatureValidationError
from ml.explain.shap_explain import compute_local_shap, top_contributors
from ml.explain.behavior_map import get_behavior
from ml.explain.text_explainer import build_user_explanation
from ml.explain.templates import build_technical_explanation
from ml.counterfactual.generate import generate_counterfactual

router = APIRouter(prefix="/engagement", tags=["engagement"])
log = get_logger(__name__)


def _is_internal_request(request: Request) -> bool:
    """
    Minimal internal gate.
    Replace this with JWT role checks or service-to-service auth later.
    """
    key = request.headers.get("X-Internal-Key")
    return bool(key) and key == getattr(settings, "INTERNAL_API_KEY", None)


@router.post("/analyze", response_model=AnalyzeResponse)
def analyze(
    req: AnalyzeRequest,
    request: Request,
    include_debug: bool = Query(False, description="Internal use only. Enables debug output."),
    include_counterfactual: bool = Query(False, description="Internal use only. Enables counterfactual output."),
):
    """
    PUBLIC DEFAULT:
      - explanation_text (safe)
      - reason_codes (safe)
      - shap fields (consider removing publicly later)

    INTERNAL ONLY (requires X-Internal-Key):
      - debug
      - counterfactual
    """
    try:
        # ---- SECURITY GATE (critical) ----
        if not _is_internal_request(request):
            include_debug = False
            include_counterfactual = False

        # 1) Contract enforcement
        expected_cols = load_contract(req.feature_version)
        validate_payload(req.features, expected_cols)

        # 2) Predict
        pred = predict_engagement(req.features)

        # 3) SHAP
        shap_rows = compute_local_shap(req.features)
        top_negative, top_positive = top_contributors(shap_rows, k=3)

        # 4) User explanation + reason codes (Option B)
        user_text, pos_behaviors, neg_behaviors, reason_codes = build_user_explanation(
            shap_top_negative=top_negative,
            shap_top_positive=top_positive,
            status=pred["status"],
        )

        # 5) Technical explanation (internal only)
        technical_text = build_technical_explanation(
            status=pred["status"],
            pos_behaviors=pos_behaviors,
            neg_behaviors=neg_behaviors,
        )

        # 6) Prepare SHAP contributor outputs
        def to_contributor(row):
            return ShapContributor(
                feature=row["feature"],
                shap=row["shap"],
                value=row["value"],
                behavior=get_behavior(row["feature"]),
            )

        shap_top_negative_out = [to_contributor(r) for r in top_negative]
        shap_top_positive_out = [to_contributor(r) for r in top_positive]

        # 7) Counterfactual (INTERNAL ONLY, and only when NOT_ENGAGED)
        counterfactual_out = None
        if include_counterfactual and pred["status"] == "NOT_ENGAGED":
            cf = generate_counterfactual(req.features, target_threshold=pred["threshold"])
            if cf:
                counterfactual_out = CounterfactualResponse(
                    target_threshold=cf["target_threshold"],
                    suggestions=[
                        CounterfactualSuggestion(
                            feature=s["feature"],
                            current=s["current"],
                            suggested=s["suggested"],
                            action=s["action"],
                        )
                        for s in cf["suggestions"]
                    ],
                )

        # 8) Debug block (INTERNAL ONLY)
        debug_info = None
        if include_debug:
            debug_info = {
                "technical_explanation": technical_text,
                "top_positive_behaviors": pos_behaviors,
                "top_negative_behaviors": neg_behaviors,
                "shap_positive_features": [r["feature"] for r in top_positive],
                "shap_negative_features": [r["feature"] for r in top_negative],
            }

        # 9) Response
        return AnalyzeResponse(
            session_id=req.session_id,
            feature_version=req.feature_version,
            engagement_score=pred["engagement_score"],
            threshold=pred["threshold"],
            status=pred["status"],
            explanation_text=user_text,
            reason_codes=reason_codes,
            shap_top_negative=shap_top_negative_out,
            shap_top_positive=shap_top_positive_out,
            counterfactual=counterfactual_out,
            debug=debug_info,
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