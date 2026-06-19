"""Environment-based application settings."""

from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

from app.constants import (
    DEFAULT_LANGSMITH_PROJECT,
    DEFAULT_LOOKBACK_DAYS,
    MAX_REVIEW_ATTEMPTS,
    TECHNICAL_ANALYSIS_LOOKBACK_DAYS,
)


class Settings(BaseSettings):
    """Runtime settings loaded from environment variables or a local .env file."""

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    app_env: str = Field(default="local", alias="APP_ENV")
    log_level: str = Field(default="INFO", alias="LOG_LEVEL")

    openai_api_key: str | None = Field(default=None, alias="OPENAI_API_KEY")
    openai_model: str = Field(default="gpt-5-nano", alias="OPENAI_MODEL")
    tavily_api_key: str | None = Field(default=None, alias="TAVILY_API_KEY")
    finnhub_api_key: str | None = Field(default=None, alias="FINNHUB_API_KEY")
    langsmith_api_key: str | None = Field(default=None, alias="LANGSMITH_API_KEY")
    langsmith_tracing: bool = Field(default=False, alias="LANGSMITH_TRACING")
    langsmith_project: str = Field(default=DEFAULT_LANGSMITH_PROJECT, alias="LANGSMITH_PROJECT")
    sec_user_agent: str | None = Field(default=None, alias="SEC_USER_AGENT")

    max_review_attempts: int = Field(default=MAX_REVIEW_ATTEMPTS, alias="MAX_REVIEW_ATTEMPTS")
    default_lookback_days: int = Field(default=DEFAULT_LOOKBACK_DAYS, alias="DEFAULT_LOOKBACK_DAYS")
    technical_analysis_lookback_days: int = Field(
        default=TECHNICAL_ANALYSIS_LOOKBACK_DAYS,
        alias="TECHNICAL_ANALYSIS_LOOKBACK_DAYS",
    )
    tavily_timeout_seconds: float = Field(default=10.0, alias="TAVILY_TIMEOUT_SECONDS")
    tavily_max_results: int = Field(default=5, alias="TAVILY_MAX_RESULTS")
    finnhub_timeout_seconds: float = Field(default=10.0, alias="FINNHUB_TIMEOUT_SECONDS")
    finnhub_news_lookback_days: int = Field(default=30, alias="FINNHUB_NEWS_LOOKBACK_DAYS")
    sec_edgar_timeout_seconds: float = Field(default=10.0, alias="SEC_EDGAR_TIMEOUT_SECONDS")
    sec_edgar_min_request_interval_seconds: float = Field(
        default=0.1,
        alias="SEC_EDGAR_MIN_REQUEST_INTERVAL_SECONDS",
    )
    sqlite_checkpoint_db: str = Field(default="checkpoints.sqlite", alias="SQLITE_CHECKPOINT_DB")

    @property
    def missing_required_config(self) -> list[str]:
        """Return required configuration names that are not currently set."""

        required_values = {
            "OPENAI_API_KEY": self.openai_api_key,
            "TAVILY_API_KEY": self.tavily_api_key,
            "FINNHUB_API_KEY": self.finnhub_api_key,
            "LANGSMITH_API_KEY": self.langsmith_api_key,
            "SEC_USER_AGENT": self.sec_user_agent,
        }
        return [name for name, value in required_values.items() if not value]


@lru_cache
def get_settings() -> Settings:
    """Return cached application settings."""

    return Settings()
