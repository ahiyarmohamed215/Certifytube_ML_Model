from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    engagement_threshold: float = 0.85
    feature_version: str = "v1.0"
    engagement_artifacts_xgboost_dir: str = "verification/engagement/xgboost/artifacts"
    engagement_artifacts_ebm_dir: str = "verification/engagement/ebm/artifacts"
    contracts_dir: str = "verification/engagement/contracts"

    quiz_model: str = Field(default="deepseek/deepseek-r1", alias="QUIZ_MODEL")
    openrouter_base_url: str = Field(
        default="https://openrouter.ai/api/v1",
        alias="OPENROUTER_BASE_URL",
    )
    openrouter_api_key: str = Field(default="", alias="OPENROUTER_API_KEY")
    openrouter_site_url: str = Field(default="http://localhost", alias="OPENROUTER_SITE_URL")
    openrouter_app_name: str = Field(default="CertifyTube ML Service", alias="OPENROUTER_APP_NAME")
    llm_timeout_seconds: float = Field(default=40.0, alias="LLM_TIMEOUT_SECONDS")
    quiz_max_questions: int = Field(default=20, alias="QUIZ_MAX_QUESTIONS")
    quiz_max_attempts_per_question: int = Field(default=1, alias="QUIZ_MAX_ATTEMPTS_PER_QUESTION")
    quiz_generation_timeout_seconds: float = Field(
        default=300.0,
        alias="QUIZ_GENERATION_TIMEOUT_SECONDS",
    )

    # MySQL (transcript cache)
    db_host: str = Field(default="localhost", alias="DB_HOST")
    db_port: int = Field(default=3306, alias="DB_PORT")
    db_user: str = Field(default="root", alias="DB_USER")
    db_password: str = Field(default="", alias="DB_PASSWORD")
    db_name: str = Field(default="certifytube", alias="DB_NAME")


settings = Settings()
