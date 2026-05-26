"""FastAPI application for the research DAG.

The app exposes two endpoints:

- ``GET /healthz`` — liveness probe.
- ``POST /research`` — Server-Sent Events stream of the research
  pipeline. Body is :class:`alphamind.api.schemas.ResearchRequest`;
  response is ``text/event-stream`` with one ``node`` event per
  LangGraph super-step, terminating in ``done`` or ``error``.

The compiled graph is a process-wide singleton, built once in the
lifespan handler and stashed on ``app.state.graph``. Tests pass a
pre-built stub graph via :func:`create_app`'s ``graph`` parameter so
they don't have to provision Postgres or hit a real LLM.
"""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, HTTPException, Request, status
from fastapi.middleware.cors import CORSMiddleware
from langgraph.graph.state import CompiledStateGraph
from sse_starlette.sse import EventSourceResponse

from alphamind.agents import build_research_graph
from alphamind.agents.retrieval import build_filing_retrieval_fn
from alphamind.api.schemas import HealthResponse, ResearchRequest
from alphamind.api.streaming import stream_research_events
from alphamind.config import get_settings
from alphamind.db.session import dispose_engine
from alphamind.llm.factory import get_llm_client
from alphamind.retrieval.embeddings.factory import dispose_embedder, get_embedder
from alphamind.retrieval.search import HybridSearch, dispose_reranker, get_reranker

logger = logging.getLogger("alphamind.api")


def _build_production_graph() -> CompiledStateGraph[Any, Any, Any, Any]:
    """Wire the real retrieval + LLM stack into a compiled DAG.

    Mirrors ``scripts/research.py``'s startup path so the API and CLI
    speak the same graph. Errors propagate — the lifespan hook will
    surface them and the server will fail to start, which is the
    correct behaviour when retrieval or the LLM backend isn't reachable.
    """
    embedder = get_embedder()
    reranker = get_reranker()
    search = HybridSearch(embedder=embedder, reranker=reranker)
    retrieve = build_filing_retrieval_fn(search)
    llm = get_llm_client()
    return build_research_graph(llm=llm, retrieve=retrieve)


def create_app(
    *,
    graph: CompiledStateGraph[Any, Any, Any, Any] | None = None,
) -> FastAPI:
    """Build the FastAPI app.

    Parameters
    ----------
    graph:
        Optional pre-built compiled graph. Production passes ``None``
        and the factory constructs the real graph from settings.
        Tests pass a stub graph (see ``tests/unit/api/conftest.py``)
        so they don't touch Postgres or hit a real LLM.

    Graph construction happens here, not in the lifespan handler, so
    ``app.state.graph`` is available before any request handlers run
    — including under ``httpx.AsyncClient + ASGITransport``, which
    doesn't fire FastAPI's lifespan. The lifespan owns disposal only.
    """
    compiled: CompiledStateGraph[Any, Any, Any, Any]
    if graph is not None:
        compiled = graph
        logger.info("api: using injected graph (test mode)")
        # Skip the production settings read — tests don't have to set
        # DATABASE_URL et al. just to exercise the HTTP surface.
        cors_origins: list[str] = []
    else:
        logger.info("api: building production graph...")
        compiled = _build_production_graph()
        logger.info("api: production graph compiled")
        cors_origins = get_settings().cors_origins_list

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        # Disposal happens on shutdown regardless of how the graph was
        # built — the dispose hooks are no-ops when their resource was
        # never constructed (deterministic backends, test mode).
        try:
            yield
        finally:
            await dispose_embedder()
            await dispose_reranker()
            await dispose_engine()

    app = FastAPI(
        title="AlphaMind Research API",
        description="Streaming research over SEC filings via the AlphaMind agent DAG.",
        version="0.1.0",
        lifespan=lifespan,
    )
    app.state.graph = compiled

    if cors_origins:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=cors_origins,
            allow_credentials=False,
            allow_methods=["GET", "POST"],
            allow_headers=["*"],
        )

    @app.get(
        "/healthz",
        response_model=HealthResponse,
        tags=["meta"],
        summary="Liveness probe.",
    )
    async def healthz(request: Request) -> HealthResponse:
        return HealthResponse(
            status="ok",
            graph_ready=getattr(request.app.state, "graph", None) is not None,
        )

    @app.post(
        "/research",
        tags=["research"],
        summary="Stream a research run as Server-Sent Events.",
        response_class=EventSourceResponse,
    )
    async def research(request: Request, body: ResearchRequest) -> EventSourceResponse:
        compiled = getattr(request.app.state, "graph", None)
        if compiled is None:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="research graph is not initialised",
            )

        return EventSourceResponse(
            stream_research_events(
                compiled,
                query=body.query,
                as_of=body.as_of,
                top_k=body.top_k,
            ),
            media_type="text/event-stream",
        )

    return app


# Module-level ``app`` for ``uvicorn alphamind.api.app:app``. Built
# lazily on first attribute access so that test imports — which
# typically do ``from alphamind.api.app import create_app`` — don't
# trigger ``get_settings()`` and demand DATABASE_URL etc. just to read
# a symbol. The lifespan still only fires when the server actually
# starts handling requests.
_app: FastAPI | None = None


def __getattr__(name: str) -> Any:
    """Construct the production ``app`` on first attribute access."""
    if name == "app":
        global _app  # noqa: PLW0603
        if _app is None:
            _app = create_app()
        return _app
    raise AttributeError(f"module 'alphamind.api.app' has no attribute {name!r}")


__all__ = ["create_app"]
