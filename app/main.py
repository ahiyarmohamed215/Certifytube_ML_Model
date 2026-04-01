from dotenv import load_dotenv
from fastapi import FastAPI

from app.api.engagement_routes import router as engagement_router
from app.api.quiz_routes import router as quiz_router

load_dotenv()

app = FastAPI(
    title="CertifyTube ML Service",
    version="3.1.0",
    description=(
        "Dual-verification ML service: "
        "Layer 1 - continuous engagement score regression "
        "(XGBoost + SHAP, EBM + native explainability); "
        "Layer 2 - transcript-grounded quiz generation with answers and explanations."
    ),
)

app.include_router(engagement_router)
app.include_router(quiz_router)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}
