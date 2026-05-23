"""Technical specialist node — no-data placeholder.

The technical specialist reasons about price action, momentum, support /
resistance, volatility regime, and volume dynamics. None of those
signals live in SEC filings — they need historical OHLCV bars and
corporate-action data from a market-data adapter that this project
hasn't built yet (ingestion currently covers EDGAR only).

Two alternatives were considered and rejected:

1. **Skip the specialist entirely** — drop ``"technical"`` from
   :data:`alphamind.agents.state.SpecialistName` and from the router
   prompt. This reshuffles the literal type and the router's choices
   the day the adapter lands, breaking every persisted run.
2. **Pretend by querying the filing corpus with technical-flavoured
   augmentation** — would mostly retrieve forward-looking-statement
   boilerplate and hallucinate technical signals out of legal
   disclaimers. Actively misleading.

What ships instead: a node with the same call signature as the other
specialists (so :func:`build_research_graph` doesn't need a special
case) that short-circuits before retrieval and the LLM. The node emits
nothing — no sources, no findings, no usage — and logs that it was
called. When the market-data adapter lands the override goes away and
``technical.py`` falls through to :func:`make_specialist_node` like
the others.
"""

from __future__ import annotations

import logging
from collections.abc import Awaitable, Callable
from typing import Any

from alphamind.agents.specialists._base import RetrievalFn
from alphamind.agents.state import ResearchState
from alphamind.llm import LLMClient

logger = logging.getLogger(__name__)

NODE_NAME = "technical"
NO_DATA_REASON = (
    "market-data adapter is not built yet; technical signals require "
    "OHLCV price / volume data not present in the SEC filing corpus"
)


def make_technical_node(
    client: LLMClient,
    retrieve: RetrievalFn,
    *,
    model: str | None = None,
    max_tokens: int = 1024,
) -> Callable[[ResearchState], Awaitable[dict[str, Any]]]:
    """Return a LangGraph node that short-circuits with no findings.

    Arguments are accepted for signature parity with the other
    ``make_*_node`` factories so the graph builder can pass the same
    ``(client, retrieve, model=, max_tokens=)`` tuple to every
    specialist without branching. They are unused while the stub is
    active.
    """

    async def node(state: ResearchState) -> dict[str, Any]:
        logger.info(
            "technical: %s — emitting no sources / findings / usage",
            NO_DATA_REASON,
        )
        # ``state`` is unused; the stub doesn't read the query.
        del state
        return {"sources": [], "findings": [], "usage": []}

    return node


__all__ = ["NODE_NAME", "NO_DATA_REASON", "make_technical_node"]
