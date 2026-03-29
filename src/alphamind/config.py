"""Typed application configuration loaded from the environment.

Settings are resolved in this order: explicit environment variables, then a
local ``.env`` file (if present), then defaults declared on the model. A single
cached ``Settings`` instance is exposed through :func:`get_settings` so the
rest of the application can treat configuration as read-only.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

Environment = Literal["development", "test", "staging", "production"]


class Settings(BaseSettings):
    """Runtime configuration for AlphaMind."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # --- Runtime ---
    environment: Environment = Field(default="development")
    log_level: str = Field(default="INFO")

    # --- External services ---
    database_url: str = Field(
        ...,
        description="SQLAlchemy async DSN, e.g. postgresql+asyncpg://user:pass@host/db",
    )
    redis_url: str = Field(
        ...,
        description="Redis DSN, e.g. redis://host:6379/0",
    )

    # --- Third-party APIs ---
    sec_user_agent: str = Field(
        ...,
        description=(
            "Required User-Agent for SEC EDGAR requests. Must identify an "
            "application and provide a contact email per SEC fair-access policy."
        ),
    )

    @property
    def is_production(self) -> bool:
        return self.environment == "production"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the process-wide settings instance."""

    return Settings()  # type: ignore[call-arg]


__all__ = ["Environment", "Settings", "get_settings"]
