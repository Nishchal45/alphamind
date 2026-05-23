"""Sentiment specialist node.

Reads the same retrieval surface as fundamentals and risk, but through
the sentiment prompt — its job is to surface qualitative posture and
tone shifts in management commentary.

The strongest sentiment signal — earnings-call transcripts — isn't in
the corpus today (the architecture map promises a transcripts adapter
but ingestion currently covers EDGAR only). Until that lands the
specialist works from MD&A narrative and 8-K event commentary, which
is honest if thin; the system prompt is explicit about the limitation.

Shape mirrors :func:`alphamind.agents.specialists.fundamentals.make_fundamentals_node`
and :func:`alphamind.agents.specialists.risk.make_risk_node` — only the
name and the bound prompt change.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import Any

from alphamind.agents.prompts import SENTIMENT_SYSTEM
from alphamind.agents.specialists._base import RetrievalFn, make_specialist_node
from alphamind.agents.state import ResearchState
from alphamind.llm import LLMClient

NODE_NAME = "sentiment"


def make_sentiment_node(
    client: LLMClient,
    retrieve: RetrievalFn,
    *,
    model: str | None = None,
    max_tokens: int = 1024,
) -> Callable[[ResearchState], Awaitable[dict[str, Any]]]:
    """Return a LangGraph node bound to ``client`` and ``retrieve``."""
    return make_specialist_node(
        specialist="sentiment",
        system_prompt=SENTIMENT_SYSTEM,
        client=client,
        retrieve=retrieve,
        model=model,
        max_tokens=max_tokens,
    )


__all__ = ["NODE_NAME", "make_sentiment_node"]
