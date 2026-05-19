"""FastAPI application factory + lifespan.

The factory pattern (vs. a module-level ``app = FastAPI()``) is
deliberate:

- Tests construct their own app with overridden dependencies; they
  must not share global state with the production app.
- ``uvicorn --factory`` calls this function explicitly, which makes
  the contract obvious: ``create_app()`` is the only public entry
  point.

The lifespan owns the lifecycle of singletons that hold async
resources — the research graph (which in turn holds the LLM client
and the embedder), the SQL engine, the embedder's HTTP client. They
are constructed lazily on first use and disposed cleanly on shutdown.
"""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI

from alphamind.agents.graph import get_research_graph
from alphamind.api.routes import health, research
from alphamind.config import get_settings
from alphamind.db.session import dispose_engine
from alphamind.retrieval.embeddings.factory import dispose_embedder

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Build the research graph once; tear down async resources on exit.

    The graph itself is cheap to construct — the expensive bits (LLM
    client, embedder HTTP session) live behind cached factories. But
    holding the graph in ``app.state`` is what makes
    :func:`alphamind.api.dependencies.get_research_graph_dep` work
    without each request rebuilding the wiring.
    """

    settings = get_settings()
    logger.info(
        "alphamind.api starting environment=%s llm_backend=%s embedding_backend=%s",
        settings.environment,
        settings.llm_backend,
        settings.embedding_backend,
    )

    app.state.research_graph = get_research_graph()
    try:
        yield
    finally:
        logger.info("alphamind.api shutting down — disposing async resources")
        await dispose_embedder()
        await dispose_engine()


def create_app() -> FastAPI:
    """Build the AlphaMind FastAPI app."""

    app = FastAPI(
        title="AlphaMind",
        description=(
            "Agentic equity-research API. Runs the LangGraph agent team "
            "over ingested SEC filings and streams progress as "
            "Server-Sent Events."
        ),
        version="0.1.0",
        lifespan=lifespan,
    )
    app.include_router(health.router)
    app.include_router(research.router)
    return app


__all__ = ["create_app", "lifespan"]
