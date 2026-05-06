"""Deterministic echo client for tests and offline development.

Echoes the last user message back as the assistant response. Useless for
real reasoning, useful for letting the agent graph and CLI run end-to-
end without an Anthropic API key. Same role as the deterministic
embedder: lets the pipeline be exercised without external services.
"""

from __future__ import annotations

from collections.abc import Sequence

from alphamind.llm.base import LLMResponse, Message


class EchoLLMClient:
    """Returns ``"echo: <last user message>"`` as the assistant reply."""

    DEFAULT_MODEL = "echo-stub-1"

    def __init__(self, *, default_model: str = DEFAULT_MODEL) -> None:
        self._default_model = default_model

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

        last_user = next(
            (m.content for m in reversed(messages) if m.role == "user"),
            "",
        )
        content = f"echo: {last_user}"

        # Approximate token counts as 1 token per 4 characters. Useless for
        # billing but lets accounting code in the agent graph run.
        prompt_chars = sum(len(m.content) for m in messages) + len(system or "")
        return LLMResponse(
            content=content,
            model=model or self._default_model,
            input_tokens=prompt_chars // 4,
            output_tokens=len(content) // 4,
            stop_reason="end_turn",
        )
