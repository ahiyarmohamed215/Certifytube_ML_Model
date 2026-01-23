from fastapi import FastAPI
from app.api.routes import router as engagement_router

app = FastAPI(
    title="CertifyTube Engagement ML Service",
    version="1.0.0",
    description="XGBoost + SHAP + Counterfactual explanations for engagement verification",
)

# Routers
app.include_router(engagement_router)


@app.get("/health")
def health():
    return {"status": "ok"}
