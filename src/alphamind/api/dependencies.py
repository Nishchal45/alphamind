"""FastAPI dependency providers.

Centralised so route handlers depend on these names, not on the
underlying factories directly. Tests override these via
``app.dependency_overrides`` to inject fakes — that's the seam.
"""

from __future__ import annotations

from typing import Any, cast

from fastapi import Request
from langgraph.graph.state import CompiledStateGraph

from alphamind.agents.state import ResearchState

ResearchGraphT = CompiledStateGraph[ResearchState, Any, Any, Any]


def get_research_graph_dep(request: Request) -> ResearchGraphT:
    """Pull the process-wide compiled research graph out of app state.

    The graph is constructed once in the lifespan and reused across
    requests. Reconstructing per-request would defeat the LLM-client
    singleton caching in :mod:`alphamind.llm.factory` and the embedder
    caching in :mod:`alphamind.retrieval.embeddings.factory`.
    """

    graph = getattr(request.app.state, "research_graph", None)
    if graph is None:  # pragma: no cover — guarded by lifespan in production
        raise RuntimeError(
            "research_graph not initialised; app lifespan did not run "
            "(are you constructing the FastAPI app without create_app()?)"
        )
    return cast("ResearchGraphT", graph)


__all__ = ["ResearchGraphT", "get_research_graph_dep"]
