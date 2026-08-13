from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application configuration loaded from environment variables."""

    app_name: str = Field(default="Lead Scoring ML API", alias="APP_NAME")
    app_env: Literal["local", "development", "staging", "production"] = Field(
        default="local",
        alias="APP_ENV",
    )
    app_debug: bool = Field(default=False, alias="APP_DEBUG")
    api_v1_prefix: str = Field(default="/api/v1", alias="API_V1_PREFIX")
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = Field(
        default="INFO",
        alias="LOG_LEVEL",
    )
    model_path: Path = Field(default=Path("artifacts/model.joblib"), alias="MODEL_PATH")
    metrics_path: Path = Field(default=Path("artifacts/metrics.json"), alias="METRICS_PATH")
    manifest_path: Path = Field(
        default=Path("artifacts/model_manifest.json"),
        alias="MANIFEST_PATH",
    )
    dataset_path: Path = Field(default=Path("data/leads.csv"), alias="DATASET_PATH")
    max_batch_size: int = Field(default=1000, alias="MAX_BATCH_SIZE", ge=1, le=10000)

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )


@lru_cache
def get_settings() -> Settings:
    """Return one cached settings object for the process."""

    return Settings()
