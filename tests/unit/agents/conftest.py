"""Shared fixtures for agent-node tests."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import date
from typing import Any

import pytest

from alphamind.agents.state import Source
from alphamind.llm.base import LLMResponse, Message


@dataclass
class ScriptedLLMClient:
    """LLM client that returns canned responses in order.

    Useful for testing agent nodes: queue up the JSON the model is
    "supposed" to return, then invoke the node. ``calls`` records the
    arguments each invocation saw so tests can assert on prompt
    content.
    """

    responses: list[str] = field(default_factory=list)
    default_model: str = "scripted-stub-1"
    calls: list[dict[str, Any]] = field(default_factory=list)

    async def complete(
        self,
        messages: Sequence[Message],
        *,
        model: str | None = None,
        max_tokens: int = 1024,
        temperature: float = 0.0,
        system: str | None = None,
    ) -> LLMResponse:
        if not self.responses:
            raise AssertionError(
                f"ScriptedLLMClient ran out of responses on call #{len(self.calls) + 1}"
            )
        content = self.responses.pop(0)
        self.calls.append(
            {
                "messages": list(messages),
                "model": model,
                "max_tokens": max_tokens,
                "temperature": temperature,
                "system": system,
            }
        )
        return LLMResponse(
            content=content,
            model=model or self.default_model,
            input_tokens=sum(len(m.content) for m in messages) // 4,
            output_tokens=len(content) // 4,
            stop_reason="end_turn",
        )


@dataclass
class SystemKeyedLLMClient:
    """LLM client that selects its response by a substring of the system prompt.

    Useful for tests where multiple nodes (specialists running in
    parallel under LangGraph) may call ``complete`` in non-deterministic
    order. The :class:`ScriptedLLMClient` queue gets the wrong response
    to the wrong node if the order changes; this stub keys on the
    system prompt instead so the test stays stable.

    ``responses_by_marker`` maps a substring (e.g. ``"risk specialist"``)
    to the canned response to return when the request's ``system``
    argument contains that substring. The first match wins.
    """

    responses_by_marker: dict[str, str] = field(default_factory=dict)
    default_response: str = "{}"
    default_model: str = "system-keyed-stub-1"
    calls: list[dict[str, Any]] = field(default_factory=list)

    async def complete(
        self,
        messages: Sequence[Message],
        *,
        model: str | None = None,
        max_tokens: int = 1024,
        temperature: float = 0.0,
        system: str | None = None,
    ) -> LLMResponse:
        self.calls.append(
            {
                "messages": list(messages),
                "model": model,
                "max_tokens": max_tokens,
                "temperature": temperature,
                "system": system,
            }
        )
        content = self.default_response
        for marker, response in self.responses_by_marker.items():
            if system is not None and marker in system:
                content = response
                break
        return LLMResponse(
            content=content,
            model=model or self.default_model,
            input_tokens=sum(len(m.content) for m in messages) // 4,
            output_tokens=len(content) // 4,
            stop_reason="end_turn",
        )


@pytest.fixture
def sample_sources() -> list[Source]:
    """A small canned source pool — three chunks, one filing each."""

    return [
        Source(
            chunk_id=101,
            filing_id=1,
            ticker="NVDA",
            form="10-K",
            filing_date=date(2024, 2, 1),
            section="Item 1A",
            text="Sales to customers in China represented 17% of total revenue.",
            score=0.91,
        ),
        Source(
            chunk_id=102,
            filing_id=1,
            ticker="NVDA",
            form="10-K",
            filing_date=date(2024, 2, 1),
            section="Item 7",
            text="Gross margin expanded to 73% as Data Center mix increased.",
            score=0.84,
        ),
        Source(
            chunk_id=103,
            filing_id=2,
            ticker="NVDA",
            form="10-Q",
            filing_date=date(2024, 8, 1),
            section="Item 1A",
            text="Export controls could materially impact future China sales.",
            score=0.79,
        ),
    ]
