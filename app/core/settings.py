from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict

class QuizSettings(BaseModel):
    gemini_model: str = "gemini-2.0-flash"
    quiz_pass_threshold: float = 0.80

quiz_settings = QuizSettings()

class Settings(BaseSettings):
    # tells pydantic-settings to read .env
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

    engagement_threshold: float = 0.85
    feature_version: str = "v1.0"
    artifacts_dir: str = "ml/artifacts"
    contracts_dir: str = "ml/contracts"

    # if you also have GEMINI_API_KEY in .env:
    gemini_api_key: str = Field(alias="GEMINI_API_KEY")

settings = Settings()
