"""Sentiment specialist node.

Reads the same retrieval surface as the other specialists but through
a tone-focused prompt — its job is to surface *how* management talks
about the business: hedging language, forward-looking statements,
uncertainty markers, defensive disclosure phrasing, and shifts in
tone on recurring topics.

The natural source for sentiment signal is earnings-call transcripts
and prepared remarks (cf. Loughran & McDonald 2011, Tetlock 2007).
Transcript ingestion has not landed yet; this specialist is wired
against filing prose in the meantime. Treat its findings as a
lower-signal substitute until that source is online.
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
