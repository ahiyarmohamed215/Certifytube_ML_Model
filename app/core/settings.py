from __future__ import annotations
from pydantic import BaseModel

class QuizSettings(BaseModel):
    gemini_model: str = "gemini-1.5-flash"
    quiz_pass_threshold: float = 0.80

quiz_settings = QuizSettings()

class Settings(BaseModel):
    # API / model config
    engagement_threshold: float = 0.85
    feature_version: str = "v1.0"

    # Paths
    artifacts_dir: str = "ml/artifacts"
    contracts_dir: str = "ml/contracts"


settings = Settings()
INTERNAL_API_KEY: str = "change-me"  # better: load from env
