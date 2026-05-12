"""Tests for the Anthropic LLM adapter.

Uses a mocked ``anthropic.AsyncAnthropic`` client so no API key is needed
and no network calls are made. We're testing the adapter wiring (system-
message handling, response parsing, error mapping), not the SDK itself.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock

import anthropic
import pytest

from alphamind.llm.anthropic import AnthropicLLMClient
from alphamind.llm.base import (
    AssistantMessage,
    LLMClientError,
    SystemMessage,
    UserMessage,
)

pytestmark = pytest.mark.asyncio


def _fake_response(text: str = "stubbed answer") -> Any:
    """Build a minimal duck-typed ``anthropic.types.Message``."""

    class _Block:
        type = "text"

        def __init__(self, txt: str) -> None:
            self.text = txt

    class _Usage:
        input_tokens = 42
        output_tokens = 17

    class _Msg:
        model = "claude-sonnet-4-5"
        stop_reason = "end_turn"

        def __init__(self) -> None:
            self.content = [_Block(text)]
            self.usage = _Usage()

    return _Msg()


@pytest.fixture
def mocked_client() -> tuple[anthropic.AsyncAnthropic, AsyncMock]:
    """Returns (sdk_client, messages.create mock)."""
    create_mock = AsyncMock(return_value=_fake_response())
    sdk_client = AsyncMock(spec=anthropic.AsyncAnthropic)
    sdk_client.messages = AsyncMock()
    sdk_client.messages.create = create_mock
    return sdk_client, create_mock


async def test_complete_returns_parsed_response(
    mocked_client: tuple[anthropic.AsyncAnthropic, AsyncMock],
) -> None:
    sdk_client, _ = mocked_client
    client = AnthropicLLMClient(api_key="test-key", client=sdk_client)

    response = await client.complete([UserMessage("hello")])

    assert response.content == "stubbed answer"
    assert response.input_tokens == 42
    assert response.output_tokens == 17
    assert response.stop_reason == "end_turn"


async def test_lifts_system_messages_into_top_level_field(
    mocked_client: tuple[anthropic.AsyncAnthropic, AsyncMock],
) -> None:
    sdk_client, create_mock = mocked_client
    client = AnthropicLLMClient(api_key="test-key", client=sdk_client)

    await client.complete(
        [
            SystemMessage("you are a research assistant"),
            UserMessage("what's NVDA's q3 risk?"),
            AssistantMessage("Here's a thought."),
            UserMessage("be concise"),
        ],
        system="prepend this too",
    )

    kwargs = create_mock.call_args.kwargs
    assert kwargs["system"] == "prepend this too\n\nyou are a research assistant"
    # Only user/assistant messages remain in the messages array.
    assert all(m["role"] in ("user", "assistant") for m in kwargs["messages"])
    assert len(kwargs["messages"]) == 3


async def test_uses_default_model_when_no_override(
    mocked_client: tuple[anthropic.AsyncAnthropic, AsyncMock],
) -> None:
    sdk_client, create_mock = mocked_client
    client = AnthropicLLMClient(
        api_key="test-key",
        default_model="claude-sonnet-4-5",
        client=sdk_client,
    )

    await client.complete([UserMessage("hi")])

    assert create_mock.call_args.kwargs["model"] == "claude-sonnet-4-5"


async def test_passes_through_max_tokens_and_temperature(
    mocked_client: tuple[anthropic.AsyncAnthropic, AsyncMock],
) -> None:
    sdk_client, create_mock = mocked_client
    client = AnthropicLLMClient(api_key="test-key", client=sdk_client)

    await client.complete(
        [UserMessage("hi")],
        max_tokens=512,
        temperature=0.4,
    )

    kwargs = create_mock.call_args.kwargs
    assert kwargs["max_tokens"] == 512
    assert kwargs["temperature"] == 0.4


async def test_rejects_empty_message_list() -> None:
    client = AnthropicLLMClient(api_key="test-key", client=AsyncMock())
    with pytest.raises(ValueError):
        await client.complete([])


async def test_rejects_messages_with_only_system_role(
    mocked_client: tuple[anthropic.AsyncAnthropic, AsyncMock],
) -> None:
    sdk_client, _ = mocked_client
    client = AnthropicLLMClient(api_key="test-key", client=sdk_client)

    with pytest.raises(ValueError, match="user/assistant"):
        await client.complete([SystemMessage("only system")])


async def test_rejects_empty_api_key() -> None:
    with pytest.raises(ValueError):
        AnthropicLLMClient(api_key="")


async def test_maps_sdk_apierror_to_llmclienterror(
    mocked_client: tuple[anthropic.AsyncAnthropic, AsyncMock],
) -> None:
    sdk_client, create_mock = mocked_client

    class _BadRequest(anthropic.BadRequestError):
        def __init__(self) -> None:
            pass

        def __str__(self) -> str:
            return "bad input"

    create_mock.side_effect = _BadRequest()
    client = AnthropicLLMClient(
        api_key="test-key",
        client=sdk_client,
        max_retries=1,
    )

    with pytest.raises(LLMClientError, match="anthropic api error"):
        await client.complete([UserMessage("hi")])
