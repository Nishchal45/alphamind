"""Technical specialist node.

Technical analysis in the strict sense — price action, momentum,
volume, relative strength — requires market-data ingestion that has
not shipped yet. Until it does, this specialist runs against the
existing filing-chunks retrieval surface with a prompt focused on the
closest analogue available there: *quantitative trend signals* in the
operating data the issuer itself reports (growth rates, margin
direction, segment trajectories, KPI inflection).

When market-data ingestion lands, this specialist's prompt and
retrieval surface change; the node-factory shape does not.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import Any

from alphamind.agents.prompts import TECHNICAL_SYSTEM
from alphamind.agents.specialists._base import RetrievalFn, make_specialist_node
from alphamind.agents.state import ResearchState
from alphamind.llm import LLMClient

NODE_NAME = "technical"


def make_technical_node(
    client: LLMClient,
    retrieve: RetrievalFn,
    *,
    model: str | None = None,
    max_tokens: int = 1024,
) -> Callable[[ResearchState], Awaitable[dict[str, Any]]]:
    """Return a LangGraph node bound to ``client`` and ``retrieve``."""
    return make_specialist_node(
        specialist="technical",
        system_prompt=TECHNICAL_SYSTEM,
        client=client,
        retrieve=retrieve,
        model=model,
        max_tokens=max_tokens,
    )


__all__ = ["NODE_NAME", "make_technical_node"]
