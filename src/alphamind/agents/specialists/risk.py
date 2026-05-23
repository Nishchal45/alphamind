"""Risk specialist node.

Reads the same retrieval surface as the fundamentals specialist but
through a different prompt — its job is to surface the bear-side
downside: Item 1A risk factors, Item 3 legal proceedings, Item 7A
market-risk disclosures, regulatory exposure, going-concern language.

A future iteration may bias retrieval itself toward the risk-relevant
sections (Item 1A, Item 7A, Item 3) so this specialist sees a denser
source pool than fundamentals. For now both specialists share the
retrieval pool and the prompt is the only differentiator. The
synthesizer / critic see both sets of findings tagged with the
producing specialist.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import Any

from alphamind.agents.prompts import RISK_SYSTEM
from alphamind.agents.specialists._base import RetrievalFn, make_specialist_node
from alphamind.agents.state import ResearchState
from alphamind.llm import LLMClient

NODE_NAME = "risk"


def make_risk_node(
    client: LLMClient,
    retrieve: RetrievalFn,
    *,
    model: str | None = None,
    max_tokens: int = 1024,
) -> Callable[[ResearchState], Awaitable[dict[str, Any]]]:
    """Return a LangGraph node bound to ``client`` and ``retrieve``."""
    return make_specialist_node(
        specialist="risk",
        system_prompt=RISK_SYSTEM,
        client=client,
        retrieve=retrieve,
        model=model,
        max_tokens=max_tokens,
    )


__all__ = ["NODE_NAME", "make_risk_node"]
