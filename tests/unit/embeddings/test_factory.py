"""Unit tests for :func:`get_embedder`."""

from __future__ import annotations

from collections.abc import Iterator

import pytest

from alphamind.config import Settings, get_settings
from alphamind.embeddings import factory
from alphamind.embeddings.base import EmbedderError
from alphamind.embeddings.deterministic import DeterministicEmbedder
from alphamind.embeddings.gemini import GeminiEmbedder


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
    monkeypatch.setattr(
        factory,
        "get_settings",
        lambda: _settings(embedder_backend="deterministic", embedder_dimension=128),
    )

    embedder = factory.get_embedder()

    assert isinstance(embedder, DeterministicEmbedder)
    assert embedder.dimension == 128


def test_returns_gemini_when_backend_and_key_set(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        factory,
        "get_settings",
        lambda: _settings(
            embedder_backend="gemini",
            google_api_key="sk-test",
            embedder_dimension=768,
        ),
    )

    embedder = factory.get_embedder()

    try:
        assert isinstance(embedder, GeminiEmbedder)
        assert embedder.dimension == 768
        assert embedder.model_name == "gemini:text-embedding-004"
    finally:
        # The factory cached this instance; clear before the next test so
        # its httpx client gets closed via dispose_embedder below.
        pass


def test_gemini_backend_without_api_key_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        factory,
        "get_settings",
        lambda: _settings(embedder_backend="gemini", google_api_key=None),
    )

    with pytest.raises(EmbedderError, match="GOOGLE_API_KEY"):
        factory.get_embedder()
