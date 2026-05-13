"""Unit tests for :func:`get_reranker` and :func:`dispose_reranker`."""

from __future__ import annotations

import asyncio
from collections.abc import Iterator

import pytest

from alphamind.config import Settings, get_settings
from alphamind.retrieval.search import reranker_factory
from alphamind.retrieval.search.cross_encoder import CrossEncoderReranker
from alphamind.retrieval.search.rerank import DeterministicReranker


@pytest.fixture(autouse=True)
def _clear_caches() -> Iterator[None]:
    get_settings.cache_clear()
    reranker_factory.get_reranker.cache_clear()
    yield
    get_settings.cache_clear()
    reranker_factory.get_reranker.cache_clear()


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
    monkeypatch.setattr(reranker_factory, "get_settings", _settings)

    reranker = reranker_factory.get_reranker()

    assert isinstance(reranker, DeterministicReranker)


def test_returns_cross_encoder_when_backend_set(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        reranker_factory,
        "get_settings",
        lambda: _settings(
            reranker_backend="cross_encoder",
            cross_encoder_model="cross-encoder/ms-marco-MiniLM-L-12-v2",
        ),
    )

    reranker = reranker_factory.get_reranker()

    # Construction must NOT load the model; just verify wrapping.
    assert isinstance(reranker, CrossEncoderReranker)


def test_dispose_is_safe_before_construction() -> None:
    asyncio.run(reranker_factory.dispose_reranker())
    assert reranker_factory.get_reranker.cache_info().currsize == 0


def test_dispose_clears_cache(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(reranker_factory, "get_settings", _settings)

    reranker_factory.get_reranker()
    assert reranker_factory.get_reranker.cache_info().currsize == 1

    asyncio.run(reranker_factory.dispose_reranker())
    assert reranker_factory.get_reranker.cache_info().currsize == 0
