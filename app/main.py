from fastapi import FastAPI

from app.api.routes import router as engagement_router
from app.api.quizz_routes import router as quiz_router

app = FastAPI(
    title="CertifyTube ML Service",
    version="1.0.0",
    description="Engagement (XGBoost+SHAP+Counterfactual) + Quiz (Gemini-based transcript quiz)",
)

# Routers
app.include_router(engagement_router)
app.include_router(quiz_router)


@app.get("/health")
def health():
    return {"status": "ok"}
