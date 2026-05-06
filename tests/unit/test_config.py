"""Unit tests for config loading."""

from __future__ import annotations

from pathlib import Path

import pytest

from alphamind.config import Settings

REQUIRED = {
    "DATABASE_URL": "postgresql+asyncpg://u:p@localhost:5432/alphamind",
    "REDIS_URL": "redis://localhost:6379/0",
    "SEC_USER_AGENT": "AlphaMind test test@example.com",
}

OPTIONAL = ("STORAGE_BACKEND", "STORAGE_LOCAL_PATH")


def _apply(monkeypatch: pytest.MonkeyPatch, env: dict[str, str]) -> None:
    for key in ("ENVIRONMENT", "LOG_LEVEL", *REQUIRED.keys(), *OPTIONAL):
        monkeypatch.delenv(key, raising=False)
    for key, value in env.items():
        monkeypatch.setenv(key, value)


def test_settings_load_from_env(monkeypatch: pytest.MonkeyPatch) -> None:
    _apply(monkeypatch, REQUIRED)

    settings = Settings(_env_file=None)  # type: ignore[call-arg]

    assert settings.database_url == REQUIRED["DATABASE_URL"]
    assert settings.redis_url == REQUIRED["REDIS_URL"]
    assert settings.sec_user_agent == REQUIRED["SEC_USER_AGENT"]
    assert settings.environment == "development"
    assert settings.log_level == "INFO"
    assert settings.is_production is False


def test_settings_reject_missing_required(monkeypatch: pytest.MonkeyPatch) -> None:
    _apply(monkeypatch, {k: v for k, v in REQUIRED.items() if k != "SEC_USER_AGENT"})

    with pytest.raises(ValueError, match="sec_user_agent"):
        Settings(_env_file=None)  # type: ignore[call-arg]


def test_production_flag(monkeypatch: pytest.MonkeyPatch) -> None:
    _apply(monkeypatch, {**REQUIRED, "ENVIRONMENT": "production"})

    settings = Settings(_env_file=None)  # type: ignore[call-arg]

    assert settings.environment == "production"
    assert settings.is_production is True


def test_rejects_invalid_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    _apply(monkeypatch, {**REQUIRED, "ENVIRONMENT": "prod"})

    with pytest.raises(ValueError):
        Settings(_env_file=None)  # type: ignore[call-arg]


def test_storage_settings_default_to_local(monkeypatch: pytest.MonkeyPatch) -> None:
    _apply(monkeypatch, REQUIRED)

    settings = Settings(_env_file=None)  # type: ignore[call-arg]

    assert settings.storage_backend == "local"
    assert settings.storage_local_path == Path("./data/storage")


def test_storage_settings_override_from_env(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _apply(
        monkeypatch,
        {**REQUIRED, "STORAGE_LOCAL_PATH": str(tmp_path)},
    )

    settings = Settings(_env_file=None)  # type: ignore[call-arg]

    assert settings.storage_local_path == tmp_path
