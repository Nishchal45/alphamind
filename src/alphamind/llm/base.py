"""LLM client protocol + message types shared by every backend."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal, Protocol, runtime_checkable

Role = Literal["system", "user", "assistant"]


class LLMClientError(RuntimeError):
    """Raised when the LLM backend returns a non-recoverable error."""


@dataclass(frozen=True, slots=True)
class Message:
    """A single chat message.

    ``role`` is ``"system"``, ``"user"``, or ``"assistant"``. Backends that
    fold ``system`` into a separate parameter (Anthropic) take care of that
    internally.
    """

    role: Role
    content: str

    def __post_init__(self) -> None:
        if self.role not in ("system", "user", "assistant"):
            raise ValueError(f"invalid role: {self.role!r}")


def SystemMessage(content: str) -> Message:  # noqa: N802 — intentionally constructor-like
    return Message(role="system", content=content)


def UserMessage(content: str) -> Message:  # noqa: N802
    return Message(role="user", content=content)


def AssistantMessage(content: str) -> Message:  # noqa: N802
    return Message(role="assistant", content=content)


@dataclass(frozen=True, slots=True)
class LLMResponse:
    """A single completion."""

    content: str
    model: str
    input_tokens: int
    output_tokens: int
    stop_reason: str | None


@runtime_checkable
class LLMClient(Protocol):
    """A minimal chat-completion surface.

    Implementations must be safe to call concurrently. The protocol covers
    the workflows AlphaMind needs today:

    - One-shot completion of a chat history.

    Streaming and tool-calling extend the surface and land in later
    phases when the agent graph needs them.
    """

    @property
    def default_model(self) -> str:
        """Model identifier used when ``complete()`` is called without ``model``."""
        ...

    async def complete(
        self,
        messages: Sequence[Message],
        *,
        model: str | None = None,
        max_tokens: int = 1024,
        temperature: float = 0.0,
        system: str | None = None,
    ) -> LLMResponse:
        """Run a chat completion.

        Parameters
        ----------
        messages:
            Conversation history. Backends that don't support a separate
            ``system`` parameter prepend it as a system-role message.
        system:
            Optional system prompt. Provider-specific handling — Anthropic
            takes it as a top-level field; OpenAI etc. fold it into the
            messages array.
        """
        ...
