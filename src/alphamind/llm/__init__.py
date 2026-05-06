"""LLM provider integration.

The :class:`LLMClient` Protocol is the only thing call sites depend on —
agent nodes, the synthesizer, the critic. Concrete implementations:

- :class:`AnthropicLLMClient` — wraps the Anthropic Python SDK; the
  default in production.
- :class:`EchoLLMClient` — deterministic stub for tests and offline
  development. Echoes the last user message back.

The factory :func:`get_llm_client` reads ``LLM_BACKEND`` from config and
returns a singleton. Swapping providers is a config change.
"""

from __future__ import annotations

from alphamind.llm.anthropic import AnthropicLLMClient
from alphamind.llm.base import (
    AssistantMessage,
    LLMClient,
    LLMClientError,
    LLMResponse,
    Message,
    SystemMessage,
    UserMessage,
)
from alphamind.llm.echo import EchoLLMClient
from alphamind.llm.factory import get_llm_client

__all__ = [
    "AnthropicLLMClient",
    "AssistantMessage",
    "EchoLLMClient",
    "LLMClient",
    "LLMClientError",
    "LLMResponse",
    "Message",
    "SystemMessage",
    "UserMessage",
    "get_llm_client",
]
