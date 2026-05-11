"""Unit tests for :func:`get_reranker`."""

from __future__ import annotations

import asyncio
from collections.abc import Iterator

import pytest

from alphamind.config import Settings, get_settings
from alphamind.reranking import factory
from alphamind.reranking.cross_encoder import CrossEncoderReranker
from alphamind.reranking.deterministic import DeterministicReranker


@pytest.fixture(autouse=True)
def _clear_caches() -> Iterator[None]:
    get_settings.cache_clear()
    factory.get_reranker.cache_clear()
    yield
    get_settings.cache_clear()
    factory.get_reranker.cache_clear()


def _settings(**overrides: object) -> Settings:
    base: dict[str, object] = {
        "database_url": "postgresql+asyncpg://x/y",
        "redis_url": "redis://localhost:6379/0",
        "sec_user_agent": "AlphaMind test test@example.com",
    }
    base.update(overrides)
    return Settings(**base)  # type: ignore[arg-type]


def test_returns_deterministic_when_backend_is_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(factory, "get_settings", _settings)

    reranker = factory.get_reranker()

    assert isinstance(reranker, DeterministicReranker)


def test_returns_cross_encoder_when_backend_set(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        factory,
        "get_settings",
        lambda: _settings(
            reranker_backend="cross_encoder",
            cross_encoder_model="cross-encoder/ms-marco-MiniLM-L-6-v2",
        ),
    )

    reranker = factory.get_reranker()

    # Don't trigger model load — just verify the wrapper was constructed.
    assert isinstance(reranker, CrossEncoderReranker)
    assert reranker.model_name == "cross-encoder:cross-encoder/ms-marco-MiniLM-L-6-v2"


def test_dispose_is_safe_before_construction() -> None:
    # Pre-condition: cache empty (the autouse fixture cleared it).
    asyncio.run(factory.dispose_reranker())
    assert factory.get_reranker.cache_info().currsize == 0
