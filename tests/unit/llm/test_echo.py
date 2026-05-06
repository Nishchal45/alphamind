"""Tests for the echo LLM stub."""

from __future__ import annotations

import pytest

from alphamind.llm import EchoLLMClient, LLMClient
from alphamind.llm.base import (
    AssistantMessage,
    SystemMessage,
    UserMessage,
)

pytestmark = pytest.mark.asyncio


async def test_satisfies_protocol() -> None:
    assert isinstance(EchoLLMClient(), LLMClient)


async def test_echoes_last_user_message() -> None:
    client = EchoLLMClient()

    response = await client.complete(
        [
            SystemMessage("you are a research assistant"),
            UserMessage("what's NVDA's q3 risk?"),
            AssistantMessage("(thinking)"),
            UserMessage("be concise"),
        ]
    )

    assert response.content == "echo: be concise"
    assert response.stop_reason == "end_turn"
    assert response.input_tokens > 0
    assert response.output_tokens > 0


async def test_uses_provided_model_override() -> None:
    client = EchoLLMClient(default_model="echo-default")

    response = await client.complete([UserMessage("ping")], model="echo-custom")

    assert response.model == "echo-custom"


async def test_rejects_empty_message_list() -> None:
    client = EchoLLMClient()
    with pytest.raises(ValueError):
        await client.complete([])
