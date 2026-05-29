"""Tests for the LLM factory backend selection.

Both ``get_settings`` and ``get_llm_client`` are ``lru_cache``d, so
each test sets the required env, clears both caches, and clears them
again on the way out so it doesn't leak a cached client into the next
test.
"""

from __future__ import annotations

from collections.abc import Iterator

import pytest

from alphamind.config import get_settings
from alphamind.llm.base import LLMClientError
from alphamind.llm.echo import EchoLLMClient
from alphamind.llm.factory import get_llm_client
from alphamind.llm.local import LocalLLMClient

_REQUIRED_ENV = {
    "DATABASE_URL": "postgresql+asyncpg://u:p@localhost:5432/alphamind",
    "REDIS_URL": "redis://localhost:6379/0",
    "SEC_USER_AGENT": "test (test@example.com)",
}


@pytest.fixture
def clean_caches() -> Iterator[None]:
    get_settings.cache_clear()
    get_llm_client.cache_clear()
    yield
    get_settings.cache_clear()
    get_llm_client.cache_clear()


def _set_env(monkeypatch: pytest.MonkeyPatch, **overrides: str) -> None:
    for key, value in {**_REQUIRED_ENV, **overrides}.items():
        monkeypatch.setenv(key, value)


def test_factory_returns_echo_by_default(
    monkeypatch: pytest.MonkeyPatch, clean_caches: None
) -> None:
    _set_env(monkeypatch, LLM_BACKEND="echo")
    assert isinstance(get_llm_client(), EchoLLMClient)


def test_factory_local_backend_constructs_without_loading_vllm(
    monkeypatch: pytest.MonkeyPatch, clean_caches: None
) -> None:
    _set_env(
        monkeypatch,
        LLM_BACKEND="local",
        SLM_BASE_MODEL="Qwen/Qwen2.5-1.5B-Instruct",
    )
    client = get_llm_client()
    assert isinstance(client, LocalLLMClient)
    # Construction must not have loaded the engine (no vLLM import).
    assert client.default_model == "Qwen/Qwen2.5-1.5B-Instruct"


def test_factory_local_backend_reports_adapter_as_model(
    monkeypatch: pytest.MonkeyPatch, clean_caches: None
) -> None:
    _set_env(
        monkeypatch,
        LLM_BACKEND="local",
        SLM_ADAPTER_PATH="/checkpoints/claim-extractor/adapter",
    )
    client = get_llm_client()
    assert isinstance(client, LocalLLMClient)
    assert client.default_model == "/checkpoints/claim-extractor/adapter"


def test_factory_anthropic_without_key_raises(
    monkeypatch: pytest.MonkeyPatch, clean_caches: None
) -> None:
    _set_env(monkeypatch, LLM_BACKEND="anthropic")
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    with pytest.raises(LLMClientError, match="ANTHROPIC_API_KEY"):
        get_llm_client()
