"""
TrueLens AI — Application Configuration

Centralized configuration management using Pydantic Settings.
All values can be overridden via environment variables or .env file.

Author: TrueLens AI Team
License: MIT
"""

from pydantic_settings import BaseSettings
from pydantic import Field
from typing import Optional
from functools import lru_cache


class Settings(BaseSettings):
    """Application settings with environment variable support."""

    # Application
    APP_NAME: str = "TrueLens AI"
    APP_VERSION: str = "1.0.0"
    DEBUG: bool = False
    LOG_LEVEL: str = "INFO"

    # Server
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    WORKERS: int = 1

    # CORS
    CORS_ORIGINS: str = "http://localhost:3000,http://localhost:5173"

    # ML Models
    MODEL_PATH: Optional[str] = Field(default=None, description="Path to trained model weights")
    DEVICE: str = "auto"
    IMAGE_SIZE: int = 224

    # Storage
    UPLOAD_DIR: str = "./uploads"
    RESULTS_DIR: str = "./results"
    MAX_FILE_SIZE_MB: int = 20

    # Rate Limiting
    RATE_LIMIT_PER_MINUTE: int = 30

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8", "case_sensitive": True}


@lru_cache()
def get_settings() -> Settings:
    """Get cached application settings instance."""
    return Settings()
