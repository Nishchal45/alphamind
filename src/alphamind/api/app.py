"""FastAPI application factory + lifespan.

Factory pattern (vs. a module-level ``app = FastAPI()``) is deliberate:

- Tests construct their own app with overridden dependencies; they
  must not share global state with the production app.
- ``uvicorn --factory`` calls this function explicitly, which makes
  the contract obvious — ``create_app()`` is the public entry point.

The lifespan builds the compiled research graph once, parameterised on
the LLM client + the HybridSearch-backed retrieval function. Both rely
on cached singletons under the hood; the lifespan also disposes their
async resources on shutdown.
"""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI

from alphamind.agents.graph import build_research_graph
from alphamind.api.retrieval import build_hybrid_retrieval
from alphamind.api.routes import health, research
from alphamind.config import get_settings
from alphamind.db.session import dispose_engine
from alphamind.llm.factory import get_llm_client
from alphamind.retrieval.embeddings.factory import dispose_embedder, get_embedder
from alphamind.retrieval.search import HybridSearch, get_reranker

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Build the research graph once; tear down async resources on exit."""

    settings = get_settings()
    logger.info(
        "alphamind.api starting environment=%s llm_backend=%s embedding_backend=%s",
        settings.environment,
        settings.llm_backend,
        settings.embedding_backend,
    )

    search = HybridSearch(embedder=get_embedder(), reranker=get_reranker())
    retrieve = build_hybrid_retrieval(search)
    llm = get_llm_client()

    app.state.research_graph = build_research_graph(llm=llm, retrieve=retrieve)
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
