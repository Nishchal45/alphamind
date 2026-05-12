"""Anthropic SDK adapter implementing :class:`LLMClient`.

Wraps :class:`anthropic.AsyncAnthropic`. Two reasons to keep this thin:

1. The protocol is what agent code depends on. The wrapper exists so
   provider-specific quirks (Anthropic's separate ``system`` parameter,
   its content-block response format) don't leak into call sites.
2. Cost tracking and request retries belong here, not in every agent
   node. Agents call ``complete``; this layer handles the rest.

Retries: tenacity with exponential backoff on Anthropic's rate-limit
errors. The SDK already retries some classes itself, but its defaults
are conservative; we layer our own retry on top to cover the wider set
of retryable errors observed in production.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence

import anthropic
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from alphamind.llm.base import LLMClient, LLMClientError, LLMResponse, Message

logger = logging.getLogger(__name__)


DEFAULT_MODEL = "claude-sonnet-4-5"
DEFAULT_MAX_RETRIES = 4


_RETRYABLE_EXCEPTIONS = (
    anthropic.RateLimitError,
    anthropic.APIConnectionError,
    anthropic.APITimeoutError,
    anthropic.InternalServerError,
)


class AnthropicLLMClient(LLMClient):
    """Anthropic-backed implementation of :class:`LLMClient`."""

    def __init__(
        self,
        *,
        api_key: str,
        default_model: str = DEFAULT_MODEL,
        max_retries: int = DEFAULT_MAX_RETRIES,
        client: anthropic.AsyncAnthropic | None = None,
    ) -> None:
        if not api_key:
            raise ValueError("api_key must be non-empty")
        self._client = client or anthropic.AsyncAnthropic(api_key=api_key)
        self._default_model = default_model
        self._max_retries = max_retries

    @property
    def default_model(self) -> str:
        return self._default_model

    async def complete(
        self,
        messages: Sequence[Message],
        *,
        model: str | None = None,
        max_tokens: int = 1024,
        temperature: float = 0.0,
        system: str | None = None,
    ) -> LLMResponse:
        if not messages:
            raise ValueError("messages must be non-empty")

        # Anthropic takes ``system`` as a top-level parameter, separate
        # from the messages array. If the caller passed a system message
        # inside ``messages``, lift it out so the SDK is happy.
        api_messages: list[dict[str, str]] = []
        sys_buffer: list[str] = [system] if system else []

        for msg in messages:
            if msg.role == "system":
                sys_buffer.append(msg.content)
            else:
                api_messages.append({"role": msg.role, "content": msg.content})

        if not api_messages:
            raise ValueError("messages must contain at least one user/assistant turn")

        api_system = "\n\n".join(s for s in sys_buffer if s)

        # Build kwargs dict so we can omit ``system`` entirely when empty —
        # the SDK's default sentinel is ``anthropic.Omit``, and passing
        # ``None`` or ``""`` would still send the field.
        create_kwargs: dict[str, object] = {
            "model": model or self._default_model,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "messages": api_messages,
        }
        if api_system:
            create_kwargs["system"] = api_system

        retryer = retry(
            stop=stop_after_attempt(self._max_retries),
            wait=wait_exponential(multiplier=1, min=1, max=30),
            retry=retry_if_exception_type(_RETRYABLE_EXCEPTIONS),
            reraise=True,
        )

        @retryer
        async def _do() -> anthropic.types.Message:
            # The kwargs dict is typed dict[str, object] for ergonomic
            # construction (we conditionally include ``system``), but the
            # SDK's ``create`` is fully typed via overloads. Splatting a
            # generic dict trips the overload selector even though every
            # value is the right shape — silence at the call site rather
            # than copy the SDK's typed-param dance into our wrapper.
            result: anthropic.types.Message = (
                await self._client.messages.create(**create_kwargs)  # type: ignore[call-overload]
            )
            return result

        try:
            response = await _do()
        except anthropic.APIError as exc:
            raise LLMClientError(f"anthropic api error: {exc}") from exc

        # Anthropic returns content as a list of blocks; we want the first
        # text block's text. Tool-use blocks land in later phases.
        text_parts = [block.text for block in response.content if block.type == "text"]
        content = "".join(text_parts)

        return LLMResponse(
            content=content,
            model=response.model,
            input_tokens=response.usage.input_tokens,
            output_tokens=response.usage.output_tokens,
            stop_reason=response.stop_reason,
        )
