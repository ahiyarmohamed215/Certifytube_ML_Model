from fastapi import APIRouter, HTTPException
from app.api.schemas import AnalyzeRequest, AnalyzeResponse

router = APIRouter(prefix="/engagement", tags=["engagement"])


@router.post("/analyze", response_model=AnalyzeResponse)
def analyze(req: AnalyzeRequest):
    """
    Class 4: API wiring only.

    Next classes will implement:
    - model loading
    - predict
    - SHAP explanation
    - counterfactual generation

    For now, we keep the endpoint live and explicit.
    """
    raise HTTPException(
        status_code=501,
        detail="ML engine not implemented yet. Complete Class 5+ (model loading + predict + SHAP).",
    )
