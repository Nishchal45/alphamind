"""Factory that returns the configured LLM client as a singleton."""

from __future__ import annotations

from functools import lru_cache

from alphamind.config import get_settings
from alphamind.llm.anthropic import AnthropicLLMClient
from alphamind.llm.base import LLMClient, LLMClientError
from alphamind.llm.echo import EchoLLMClient


@lru_cache(maxsize=1)
def get_llm_client() -> LLMClient:
    """Return the process-wide LLM client instance."""

    settings = get_settings()
    backend = settings.llm_backend.lower()

    if backend == "echo":
        return EchoLLMClient(default_model=settings.llm_model)

    if backend == "anthropic":
        api_key = settings.anthropic_api_key
        if not api_key:
            raise LLMClientError("ANTHROPIC_API_KEY is required when llm_backend='anthropic'")
        return AnthropicLLMClient(
            api_key=api_key,
            default_model=settings.llm_model,
        )

    if backend == "local":
        # Imported here, not at module top, so the vLLM dependency chain
        # is only touched when the local backend is actually selected.
        from alphamind.llm.local import LocalLLMClient  # noqa: PLC0415

        return LocalLLMClient(
            base_model=settings.slm_base_model,
            adapter_path=settings.slm_adapter_path,
        )

    raise LLMClientError(f"unsupported llm backend: {backend!r}")


__all__ = ["get_llm_client"]
