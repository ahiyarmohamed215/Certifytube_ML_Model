from fastapi import FastAPI

from dotenv import load_dotenv
load_dotenv()

from app.api.routes import router as engagement_router

app = FastAPI(
    title="CertifyTube ML Service",
    version="2.0.0",
    description="Engagement scoring with explanations (XGBoost+SHAP | EBM+NativeExplain)",
)

# Routers
app.include_router(engagement_router)


@app.get("/health")
def health():
    return {"status": "ok"}
