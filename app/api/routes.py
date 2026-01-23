from fastapi import APIRouter, HTTPException
from app.api.schemas import AnalyzeRequest, AnalyzeResponse

from ml.inference.predict import predict_engagement
from ml.inference.validate import FeatureValidationError

router = APIRouter(prefix="/engagement", tags=["engagement"])


@router.post("/analyze", response_model=AnalyzeResponse)
def analyze(req: AnalyzeRequest):
    """
    Class 6:
    - Accept features from backend
    - Run XGBoost inference
    - Return score + status
    """

    try:
        result = predict_engagement(req.features)

    except FeatureValidationError as e:
        raise HTTPException(status_code=400, detail=str(e))

    except FileNotFoundError as e:
        raise HTTPException(
            status_code=500,
            detail="Model artifacts not found. Train the model before inference.",
        )

    return AnalyzeResponse(
        session_id=req.session_id,
        feature_version=req.feature_version,
        engagement_score=result["engagement_score"],
        threshold=result["threshold"],
        status=result["status"],
        explanation_text="Explanation not implemented yet",
        shap_top_negative=[],
        shap_top_positive=[],
        counterfactual=None,
    )
