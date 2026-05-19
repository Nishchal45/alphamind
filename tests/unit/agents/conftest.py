"""Shared helpers for agent-graph tests.

The agent graph is built on the :class:`alphamind.llm.base.LLMClient`
protocol. For deterministic tests we want a client that returns canned
responses keyed on the prompt (or just in sequence). :class:`EchoLLMClient`
is too dumb — it just echoes the user message back, so we can't test
the JSON-parsing paths.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

from alphamind.llm.base import LLMResponse, Message


class ScriptedLLMClient:
    """Returns canned responses based on the prompt.

    Three modes:

    - ``responses`` is a list -> each ``complete()`` call pops the next
      one (FIFO).
    - ``responses`` is a dict -> match the last user message's content
      against the keys (substring match) and return the corresponding
      value.
    - ``responses`` is a callable -> called with (messages, system) and
      its return value is the assistant content.
    """

    DEFAULT_MODEL = "scripted-stub-1"

    def __init__(
        self,
        responses: (list[str] | dict[str, str] | Callable[[Sequence[Message], str | None], str]),
        *,
        default_model: str = DEFAULT_MODEL,
    ) -> None:
        self._responses = responses
        self._default_model = default_model
        self.calls: list[dict[str, object]] = []

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
        last_user = next(
            (m.content for m in reversed(messages) if m.role == "user"),
            "",
        )
        self.calls.append(
            {
                "messages": list(messages),
                "system": system,
                "model": model,
                "max_tokens": max_tokens,
                "last_user": last_user,
            }
        )

        if isinstance(self._responses, list):
            if not self._responses:
                raise RuntimeError("ScriptedLLMClient: ran out of canned responses")
            content = self._responses.pop(0)
        elif isinstance(self._responses, dict):
            content = next(
                (resp for needle, resp in self._responses.items() if needle in last_user),
                "",
            )
            if not content:
                raise RuntimeError(
                    f"ScriptedLLMClient: no scripted response matched user message: {last_user!r}"
                )
        else:
            content = self._responses(messages, system)

        return LLMResponse(
            content=content,
            model=model or self._default_model,
            input_tokens=sum(len(m.content) for m in messages) // 4,
            output_tokens=len(content) // 4,
            stop_reason="end_turn",
        )
