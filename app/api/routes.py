from fastapi import APIRouter, HTTPException
from app.api.schemas import (
    AnalyzeRequest,
    AnalyzeResponse,
    ShapContributor,
    CounterfactualResponse,
    CounterfactualSuggestion,
)

from ml.inference.predict import predict_engagement
from ml.inference.validate import FeatureValidationError
from ml.explain.shap_explain import compute_local_shap, top_contributors
from ml.explain.behavior_map import get_behavior
from ml.explain.text_explainer import build_explanation_text
from ml.counterfactual.generate import generate_counterfactual

router = APIRouter(prefix="/engagement", tags=["engagement"])


@router.post("/analyze", response_model=AnalyzeResponse)
def analyze(req: AnalyzeRequest):
    """
    Full analysis endpoint:
    - Predict engagement score
    - Compute SHAP local explanations
    - Generate explanation text
    - If NOT engaged: generate counterfactual guidance
    """

    try:
        # 1) Predict engagement score
        pred = predict_engagement(req.features)

        # 2) SHAP local explanations
        shap_rows = compute_local_shap(req.features)
        top_negative, top_positive = top_contributors(shap_rows, k=3)

        # 3) Explanation text
        explanation_text = build_explanation_text(
            shap_top_negative=top_negative,
            shap_top_positive=top_positive,
            status=pred["status"],
        )

        # 4) Prepare SHAP outputs
        def to_contributor(row):
            return ShapContributor(
                feature=row["feature"],
                shap=row["shap"],
                value=row["value"],
                behavior=get_behavior(row["feature"]),
            )

        shap_top_negative_out = [to_contributor(r) for r in top_negative]
        shap_top_positive_out = [to_contributor(r) for r in top_positive]

        # 5) Counterfactuals only for NOT_ENGAGED
        counterfactual_out = None
        if pred["status"] == "NOT_ENGAGED":
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

    except FeatureValidationError as e:
        raise HTTPException(status_code=400, detail=str(e))

    except FileNotFoundError:
        raise HTTPException(
            status_code=500,
            detail="Model artifacts not found. Train the model before inference.",
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    return AnalyzeResponse(
        session_id=req.session_id,
        feature_version=req.feature_version,
        engagement_score=pred["engagement_score"],
        threshold=pred["threshold"],
        status=pred["status"],
        explanation_text=explanation_text,
        shap_top_negative=shap_top_negative_out,
        shap_top_positive=shap_top_positive_out,
        counterfactual=counterfactual_out,
    )
