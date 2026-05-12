"""Unit tests for :func:`get_embedder` and :func:`dispose_embedder`."""

from __future__ import annotations

import asyncio
from collections.abc import Iterator

import pytest

from alphamind.config import Settings, get_settings
from alphamind.retrieval.embeddings import factory
from alphamind.retrieval.embeddings.base import EmbedderError
from alphamind.retrieval.embeddings.deterministic import DeterministicHashEmbedder
from alphamind.retrieval.embeddings.gemini import GeminiEmbedder


@pytest.fixture(autouse=True)
def _clear_caches() -> Iterator[None]:
    get_settings.cache_clear()
    factory.get_embedder.cache_clear()
    yield
    get_settings.cache_clear()
    factory.get_embedder.cache_clear()


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

    embedder = factory.get_embedder()

    assert isinstance(embedder, DeterministicHashEmbedder)


def test_returns_gemini_when_backend_and_key_set(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        factory,
        "get_settings",
        lambda: _settings(embedding_backend="gemini", google_api_key="sk-test"),
    )

    embedder = factory.get_embedder()
    try:
        assert isinstance(embedder, GeminiEmbedder)
    finally:
        # GeminiEmbedder owns an httpx client; close it before the test exits.
        asyncio.run(factory.dispose_embedder())


def test_gemini_backend_without_api_key_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        factory,
        "get_settings",
        lambda: _settings(embedding_backend="gemini", google_api_key=None),
    )

    with pytest.raises(EmbedderError, match="GOOGLE_API_KEY"):
        factory.get_embedder()


def test_dispose_is_safe_before_construction() -> None:
    # Autouse fixture clears the cache; no embedder built yet.
    asyncio.run(factory.dispose_embedder())
    assert factory.get_embedder.cache_info().currsize == 0


def test_dispose_clears_cache_for_deterministic_backend(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(factory, "get_settings", _settings)

    factory.get_embedder()
    assert factory.get_embedder.cache_info().currsize == 1

    asyncio.run(factory.dispose_embedder())
    assert factory.get_embedder.cache_info().currsize == 0
