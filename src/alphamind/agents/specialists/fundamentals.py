"""Fundamentals specialist node.

Pulls retrieval hits, asks the LLM for 3-7 structured findings drawn
from financial-statement / MD&A / business-description material, and
writes both the source pool and the findings back to state.

The hard work — JSON parsing, citation validation, accumulator
plumbing — lives in :mod:`alphamind.agents.specialists._base`; this
module exists to bind the fundamentals prompt and name.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import Any

from alphamind.agents.prompts import FUNDAMENTALS_SYSTEM
from alphamind.agents.specialists._base import RetrievalFn, make_specialist_node
from alphamind.agents.state import ResearchState
from alphamind.llm import LLMClient

NODE_NAME = "fundamentals"


def make_fundamentals_node(
    client: LLMClient,
    retrieve: RetrievalFn,
    *,
    model: str | None = None,
    max_tokens: int = 1024,
) -> Callable[[ResearchState], Awaitable[dict[str, Any]]]:
    """Return a LangGraph node bound to ``client`` and ``retrieve``."""
    return make_specialist_node(
        specialist="fundamentals",
        system_prompt=FUNDAMENTALS_SYSTEM,
        client=client,
        retrieve=retrieve,
        model=model,
        max_tokens=max_tokens,
    )


__all__ = ["NODE_NAME", "RetrievalFn", "make_fundamentals_node"]
